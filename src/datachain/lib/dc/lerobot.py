import json
import logging
import os
from typing import TYPE_CHECKING, Any, cast

from datachain.query import Session

if TYPE_CHECKING:
    from datachain.lib.file import File, VideoFile

    from .datachain import DataChain

logger = logging.getLogger("datachain")

INFO = "meta/info.json"
EPISODES = "meta/episodes/"


def read_lerobot(
    path: str | os.PathLike[str],
    session: Session | None = None,
    settings: dict | None = None,
    **kwargs: Any,
) -> "DataChain":
    """Generate a chain with one row per episode of a LeRobot v3 dataset.

    Each row holds the columns of `meta/episodes/` (`episode_index`, `length`,
    `tasks`, ...; stats skipped) and a `videos` signal with one `VideoFragment` per
    camera: the episode's span of the shared mp4. Column names and camera keys are
    normalized like `read_parquet` columns, e.g. `data/file_index` becomes
    `data_file_index` and `observation.images.wrist_left` becomes
    `videos.observation_images_wrist_left`. Episodes whose video files are missing
    from storage are skipped with a warning.

    Per-frame data is not included; read it with
    `dc.read_parquet(path + "/data/")` and merge on `episode_index`.

    Many episodes share one mp4. A UDF that takes a fragment reads only that
    episode's bytes; with `settings(cache=True)` each mp4 downloads once and is
    reused by the other episodes.

    Parameters:
        path: Dataset root, e.g. `hf://datasets/lerobot/aloha_mobile_cabinet` or
            `s3://bucket/dataset/`.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        kwargs: Parameters to pass to `read_storage`, e.g. `anon=True`.

    Example:
        ```py
        import datachain as dc

        fps = 15  # "fps" in meta/info.json

        def first_frame(cam: dc.VideoFragment) -> float:
            frame = cam.video.get_frame(round(cam.start * fps))
            return float(frame.get_np().mean())

        chain = (
            dc.read_lerobot("s3://robolake/droid/droid_10k/", anon=True)
            .filter(dc.C("length") > 100)
            .limit(10)
            .map(
                brightness=first_frame,
                params=["videos.observation_images_wrist_left"],
            )
        )
        ```
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from datachain.lib.arrow import schema_to_output
    from datachain.lib.data_model import dict_to_data_model
    from datachain.lib.file import VideoFragment

    from .records import read_records
    from .storage import read_storage

    listing = read_storage(
        path, type="video", session=session, settings=settings, **kwargs
    )
    files = {f.path: f for f in cast("list[VideoFile]", listing.to_values("file"))}
    infos = [p for p in files if p == INFO or p.endswith(f"/{INFO}")]
    if len(infos) != 1:
        raise ValueError(f"Expected one {INFO} under '{path}', found {len(infos)}")
    root = infos[0].removesuffix(INFO)
    with _open(files[infos[0]]) as f:
        info = json.load(f)
    version = info.get("codebase_version")
    if not str(version).startswith("v3."):
        raise ValueError(f"'{path}' is LeRobot {version}, only v3 is supported")

    tables = []
    for p in sorted(files):
        if p.startswith(root + EPISODES) and p.endswith(".parquet"):
            with _open(files[p]) as f:
                pf = pq.ParquetFile(f)
                cols = [c for c in pf.schema_arrow.names if not c.startswith("stats/")]
                tables.append(pf.read(columns=cols))
    if not tables:
        raise ValueError(f"No {EPISODES} parquet files under '{path}'")
    table = pa.concat_tables(tables, promote_options="default")
    meta = [c for c in table.column_names if not c.startswith("videos/")]
    output, names = schema_to_output(table.select(meta).schema.remove_metadata())
    schema = dict(output)
    keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    if keys:
        schema["videos"] = dict_to_data_model("", dict.fromkeys(keys, VideoFragment))

    def fragment(row: dict, key: str) -> VideoFragment | None:
        p = info["video_path"].format(
            video_key=key,
            chunk_index=row[f"videos/{key}/chunk_index"],
            file_index=row[f"videos/{key}/file_index"],
        )
        if (video := files.get(root + p)) is None:
            return None
        return VideoFragment(
            video=video,
            start=row[f"videos/{key}/from_timestamp"],
            end=row[f"videos/{key}/to_timestamp"],
        )

    rows = table.to_pylist()
    records = []
    for row in rows:
        frags = {k: fragment(row, k) for k in keys}
        if None in frags.values():
            continue
        record = {name: row[c] for name, c in zip(output, names, strict=True)}
        if keys:
            record["videos"] = schema["videos"](**frags)
        records.append(record)
    if skipped := len(rows) - len(records):
        logger.warning(
            "Skipped %d of %d episodes with missing video files in '%s'",
            skipped,
            len(rows),
            path,
        )
    return read_records(records, schema, session=listing.session, settings=settings)


def _open(file: "File"):
    """Open a listed file, failing if it changed since listing, like `read_parquet`."""
    if file._caching_enabled:
        file.ensure_cached()
    elif not file.get_local_path():
        file._catalog.get_client(file.source).check_unchanged(file)  # type: ignore[union-attr]
    return file.open()
