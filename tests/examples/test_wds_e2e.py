import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

import datachain as dc
from datachain.lib.file import File
from datachain.lib.webdataset import WDSAllFile, process_webdataset
from datachain.lib.webdataset_laion import Laion, WDSLaion
from tests.examples.wds_data import WDS_META, WDS_TAR_SHARDS
from tests.utils import make_tar


@pytest.fixture
def webdataset_tars(tmp_path):
    """
    Creates tar file with webdataset data (.json, .txt and .jpg files in it)
    Returns path to a directory of tar file
    """
    data_path = tmp_path / "datacomp-sample"
    shards_path = data_path / "shards"

    fh = io.BytesIO()
    with tarfile.open(fileobj=fh, mode="w") as tar:
        for idx, rec in enumerate(WDS_TAR_SHARDS):
            # json file
            data = json.dumps(rec).encode()
            info = tarfile.TarInfo(f"{idx}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

            # image file
            data = b"123"  # some dummy data
            info = tarfile.TarInfo(f"{idx}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

            # txt file
            data = rec["caption"].encode()
            info = tarfile.TarInfo(f"{idx}.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    shards_path.mkdir(parents=True, exist_ok=True)
    with open(shards_path / "00000000.tar", "wb") as f:
        f.write(fh.getvalue())

    return shards_path


@pytest.fixture
def webdataset_metadata(tmp_path):
    """
    Creates webdataset metadata parquet file which goes with webdataset_tars
    fixture
    Returns path to a directory of parquet file
    """
    data_path = tmp_path / "datacomp-sample"
    metadata_path = data_path / "metadata"

    metadata_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_dict(WDS_META)
    df.to_parquet(metadata_path / "00000000.parquet")

    return metadata_path


def test_wds(test_session, webdataset_tars):
    res = dc.read_storage(Path(webdataset_tars).as_uri(), session=test_session).gen(
        laion=process_webdataset(spec=WDSLaion), params="file"
    )

    num_rows = 0
    for laion_wds in res.to_values("laion"):
        num_rows += 1
        assert isinstance(laion_wds, WDSLaion)
        idx, data = next(
            (i, d)
            for i, d in enumerate(WDS_TAR_SHARDS)
            if d["uid"] == laion_wds.json.uid
        )

        assert laion_wds.txt == data["caption"]
        assert laion_wds.file.location
        assert laion_wds.file.source == Path(webdataset_tars).as_uri()
        assert laion_wds.file.parent
        assert laion_wds.file.name == f"{idx}.jpg"
        assert laion_wds.file.location
        assert laion_wds.json.model_dump() == Laion(**data).model_dump()

    assert num_rows == len(WDS_TAR_SHARDS)


def test_wds_merge_with_parquet_meta(
    test_session, webdataset_tars, webdataset_metadata
):
    wds = dc.read_storage(Path(webdataset_tars).as_uri(), session=test_session).gen(
        laion=process_webdataset(spec=WDSLaion), params="file"
    )

    meta = dc.read_parquet(Path(webdataset_metadata).as_uri())

    res = wds.merge(meta, on="laion.json.uid", right_on="uid")

    num_rows = 0
    for r in res.to_values("laion"):
        num_rows += 1
        assert isinstance(r, WDSLaion)
        assert isinstance(r.file, File)
        assert isinstance(r.json, Laion)
        data = next(d for d in WDS_TAR_SHARDS if d["uid"] == r.json.uid)
        assert r.txt == data["caption"]
        assert r.json.uid == data["uid"]

    assert num_rows == len(WDS_TAR_SHARDS)

    meta_res = res.to_list(*WDS_META.keys())

    for field_name_idx, rows_values in enumerate(WDS_META.values()):
        assert sorted(rows_values.values()) == sorted(
            [r[field_name_idx] for r in meta_res]
        )

    # validate correct merge
    for laion_uid, uid in res.to_iter("laion.json.uid", "uid"):
        assert laion_uid == uid
    for caption, text in res.to_iter("laion.json.caption", "text"):
        assert caption == text


def test_read_webdataset_matches_gen(test_session, webdataset_tars):
    uri = Path(webdataset_tars).as_uri()
    expected = (
        dc.read_storage(uri, session=test_session)
        .gen(laion=process_webdataset(spec=WDSLaion))
        .order_by("laion.file.path")
        .to_values("laion")
    )

    res = dc.read_webdataset(uri, spec=WDSLaion, column="laion", session=test_session)

    assert res.schema == {"laion": WDSLaion}
    assert res.order_by("laion.file.path").to_values("laion") == expected
    assert len(expected) == len(WDS_TAR_SHARDS)
    assert {r.file.read() for r in res.to_values("laion")} == {b"123"}


def test_read_webdataset_defaults_skip_non_tar_files(test_session, webdataset_tars):
    shard = (webdataset_tars / "00000000.tar").read_bytes()
    (webdataset_tars / "00000001.tar").write_bytes(shard)
    (webdataset_tars / "00000002.tar.gz").write_bytes(shard)
    (webdataset_tars / "00000000_stats.json").write_text("{}")

    res = dc.read_webdataset(Path(webdataset_tars).as_uri(), session=test_session)

    assert res.schema == {"wds": WDSAllFile}
    rows = res.to_list("wds.txt", "wds.file.path")
    assert sorted(txt for txt, _ in rows) == sorted(
        2 * [d["caption"] for d in WDS_TAR_SHARDS]
    )
    assert {path.split("/")[0] for _, path in rows} == {"00000000.tar", "00000001.tar"}


@pytest.mark.parametrize(
    "options,column,values",
    [
        ({"encoding": "latin-1"}, "wds.txt", ["Ã©"]),
        ({"core_extensions": ("txt",)}, "wds.file.path", ["shard.tar/0.txt"]),
    ],
)
def test_read_webdataset_save_after_option_change_rebuilds(
    test_session, tmp_path, options, column, values
):
    (tmp_path / "shard.tar").write_bytes(make_tar({"0.cap": "c", "0.txt": "é"}))
    base = {"core_extensions": ("cap",), "session": test_session}
    dc.read_webdataset(tmp_path.as_uri(), **base).save("samples")

    dc.read_webdataset(tmp_path.as_uri(), **(base | options)).save("samples")

    assert dc.read_dataset("samples", session=test_session).to_values(column) == values


def test_read_webdataset_ignores_core_extensions_changed_after_the_call(
    test_session, tmp_path
):
    (tmp_path / "shard.tar").write_bytes(make_tar({"0.cap": "c", "0.txt": "t"}))
    extensions = ["cap"]
    chain = dc.read_webdataset(
        tmp_path.as_uri(), core_extensions=extensions, session=test_session
    )

    extensions[0] = "txt"

    assert chain.to_values("wds.file.path") == ["shard.tar/0.cap"]
