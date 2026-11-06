import json
import logging
import os
from typing import TYPE_CHECKING, Any, cast

from datachain.query import Session

if TYPE_CHECKING:
    from datachain.lib.file import VideoFile

    from .datachain import DataChain

logger = logging.getLogger("datachain")

INFO = "meta/info.json"
EPISODES = "meta/episodes/"


def read_lerobot(
    path: str | os.PathLike[str],
    session: Session | None = None,
    settings: dict | None = None,
    **kwargs,
) -> "DataChain":
    """Generate a chain with one row per episode of a LeRobot v3 dataset.

    Each row holds the scalar columns of `meta/episodes/` (`episode_index`,
    `length`, `tasks`, ... with `/` turned into `_`, stats skipped) and a `videos`
    signal with one `VideoFragment` per camera: the episode's span of the shared
    mp4. Camera keys become field names with `.` turned into `_`, e.g.
    `videos.observation_images_wrist_left`. Episodes whose video files are missing
    from storage are skipped with a warning.

    Per-frame data is not included; read it with
    `dc.read_parquet(path + "/data/")` and merge on `episode_index`.

    Many episodes share one mp4: to process a set of episodes, use
    `settings(cache=True)` so each mp4 downloads once; to read a few episodes,
    use `settings(prefetch=0)` so only their bytes are read.

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
            .settings(prefetch=0)
            .map(
                brightness=first_frame,
                params=["videos.observation_images_wrist_left"],
            )
        )
        ```
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from datachain.lib.data_model import dict_to_data_model
    from datachain.lib.file import VideoFragment
    from datachain.lib.utils import normalize_col_names

    from .storage import read_storage
    from .values import read_values

    listing = read_storage(
        path, type="video", session=session, settings=settings, **kwargs
    )
    videos = cast("list[VideoFile]", listing.to_values("file"))
    files = {f.path: f for f in videos}
    infos = [p for p in files if p == INFO or p.endswith(f"/{INFO}")]
    if len(infos) != 1:
        raise ValueError(f"Expected one {INFO} under '{path}', found {len(infos)}")
    root = infos[0].removesuffix(INFO)
    info = json.loads(files[infos[0]].read_text())
    version = info.get("codebase_version")
    if not str(version).startswith("v3."):
        raise ValueError(f"'{path}' is LeRobot {version}, only v3 is supported")

    keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    tables = []
    for p in sorted(files):
        if p.startswith(root + EPISODES) and p.endswith(".parquet"):
            with files[p].open() as f:
                pf = pq.ParquetFile(f)
                cols = [c for c in pf.schema_arrow.names if not c.startswith("stats/")]
                tables.append(pf.read(columns=cols))
    if not tables:
        raise ValueError(f"No {EPISODES} parquet files under '{path}'")
    table = pa.concat_tables(tables)
    meta = [c for c in table.column_names if not c.startswith("videos/")]

    def fragments(row: dict) -> dict[str, VideoFragment] | None:
        frags = {}
        for k in keys:
            p = info["video_path"].format(
                video_key=k,
                chunk_index=row[f"videos/{k}/chunk_index"],
                file_index=row[f"videos/{k}/file_index"],
            )
            if (video := files.get(root + p)) is None:
                return None
            frags[k] = VideoFragment(
                video=video,
                start=row[f"videos/{k}/from_timestamp"],
                end=row[f"videos/{k}/to_timestamp"],
            )
        return frags

    rows = table.to_pylist()
    episodes = [(row, frags) for row in rows if (frags := fragments(row)) is not None]
    if skipped := len(rows) - len(episodes):
        logger.warning(
            "Skipped %d of %d episodes with missing video files in '%s'",
            skipped,
            len(rows),
            path,
        )
    columns: dict[str, Any] = {
        name: [row[c] for row, _ in episodes]
        for name, c in normalize_col_names(meta).items()
    }
    if keys:
        Videos = dict_to_data_model("", dict.fromkeys(keys, VideoFragment))  # noqa: N806
        columns["videos"] = [Videos(**frags) for _, frags in episodes]
    return read_values(session=listing.session, settings=settings, **columns)
