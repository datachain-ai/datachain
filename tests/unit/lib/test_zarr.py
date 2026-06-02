from types import SimpleNamespace

import numpy as np
import pytest

import datachain as dc

zarr = pytest.importorskip("zarr")


def _make_store(root, who, n=4):
    g = zarr.open_group(str(root), mode="w")
    g.attrs["who"] = who
    g.create_array("images", shape=(n, 8, 8), chunks=(1, 8, 8), dtype="uint8")
    g["images"][:] = np.arange(n * 8 * 8).reshape(n, 8, 8).astype("uint8")
    labels = g.create_group("labels")
    labels.create_array("y", shape=(n,), dtype="int64")
    labels["y"][:] = np.arange(n)
    return g


def _single_store(tmp_dir, test_session, name="s"):
    _make_store(tmp_dir / f"{name}.zarr", name)
    chain = dc.read_zarr((tmp_dir / f"{name}.zarr").as_uri(), session=test_session)
    (store,) = next(iter(chain.to_iter("zarr")))
    return store


def test_read_zarr_one_row_per_store(tmp_dir, test_session):
    for name in ("scan001", "scan002", "scan003"):
        _make_store(tmp_dir / f"{name}.zarr", name)

    chain = dc.read_zarr(tmp_dir.as_uri(), session=test_session)
    stores = [s for (s,) in chain.order_by("zarr.file.path").to_iter("zarr")]

    assert [s.path for s in stores] == ["scan001.zarr", "scan002.zarr", "scan003.zarr"]


def test_read_zarr_discovers_stores_at_any_depth(tmp_dir, test_session):
    _make_store(tmp_dir / "top.zarr", "top")
    _make_store(tmp_dir / "a" / "b" / "deep.zarr", "deep")

    chain = dc.read_zarr(tmp_dir.as_uri(), session=test_session)
    paths = sorted(s.path for (s,) in chain.to_iter("zarr"))

    # Both the top-level and the nested store are found, and nested array
    # metadata (e.g. ".../images/zarr.json") is never reported as a store.
    assert paths == ["a/b/deep.zarr", "top.zarr"]


def test_read_zarr_directory(tmp_dir, test_session):
    store = _single_store(tmp_dir, test_session, name="only")

    assert store.path == "only.zarr"
    assert store.source.startswith("file://")


def test_zarr_store_get_info(tmp_dir, test_session):
    store = _single_store(tmp_dir, test_session)
    info = store.get_info()

    assert info.zarr_format == 3
    assert sorted(info.arrays) == ["images", "labels/y"]
    assert info.attrs == {"who": "s"}


def test_zarr_store_get_arrays(tmp_dir, test_session):
    store = _single_store(tmp_dir, test_session)
    arrays = {a.path: a for a in store.get_arrays()}

    assert sorted(arrays) == ["images", "labels/y"]
    assert arrays["images"].shape == [4, 8, 8]
    assert arrays["images"].chunks == [1, 8, 8]
    assert arrays["images"].dtype == "uint8"
    assert arrays["labels/y"].shape == [4]


def test_zarr_array_read(tmp_dir, test_session):
    store = _single_store(tmp_dir, test_session)

    data = store.get_array("images").read()
    assert data.shape == (4, 8, 8)
    assert int(data.flat[0]) == 0

    one = store.get_array("images").read(np.s_[0])
    assert one.shape == (8, 8)

    labels = store.get_array("labels/y").read()
    assert labels.tolist() == [0, 1, 2, 3]


def test_zarr_store_get_array_not_an_array(tmp_dir, test_session):
    store = _single_store(tmp_dir, test_session)

    with pytest.raises(ValueError, match="not a Zarr array"):
        store.get_array("labels")


def test_read_zarr_custom_column(tmp_dir, test_session):
    _make_store(tmp_dir / "s.zarr", "s")

    chain = dc.read_zarr(
        (tmp_dir / "s.zarr").as_uri(), column="store", session=test_session
    )
    (store,) = next(iter(chain.to_iter("store")))

    assert store.path == "s.zarr"


def test_zarr_array_only_store(tmp_dir, test_session):
    zarr.open_array(str(tmp_dir / "arr.zarr"), mode="w", shape=(2, 3), dtype="uint8")

    chain = dc.read_zarr((tmp_dir / "arr.zarr").as_uri(), session=test_session)
    (store,) = next(iter(chain.to_iter("zarr")))
    arrays = store.get_arrays()

    assert [a.path for a in arrays] == [""]
    assert store.get_array().shape == [2, 3]


def test_zarr_selection_read(tmp_dir, test_session):
    store = _single_store(tmp_dir, test_session)

    selection = store.get_array("images").select(0)
    assert selection.index == [0]
    assert selection.read().shape == (8, 8)

    block = store.get_array("images").select([1])
    assert block.read().shape == (8, 8)


def test_zarr_selection_read_bytes_image(tmp_dir, test_session):
    store = _single_store(tmp_dir, test_session)

    selection = store.get_array("images").select(0, media="image")
    content = selection.read_bytes()

    assert content[:8] == b"\x89PNG\r\n\x1a\n"


def test_zarr_selection_read_bytes_rejects_non_image(tmp_dir, test_session):
    store = _single_store(tmp_dir, test_session)

    selection = store.get_array("images").select(0, media="audio")

    with pytest.raises(ValueError, match="supports image media"):
        selection.read_bytes()


def test_zarr_selection_read_bytes_converts_non_uint8(tmp_dir, test_session):
    g = zarr.open_group(str(tmp_dir / "f.zarr"), mode="w")
    g.create_array("img", shape=(2, 4, 4), chunks=(1, 4, 4), dtype="float32")
    g["img"][:] = np.zeros((2, 4, 4), dtype="float32")

    chain = dc.read_zarr((tmp_dir / "f.zarr").as_uri(), session=test_session)
    (store,) = next(iter(chain.to_iter("zarr")))
    content = store.get_array("img").select(0, media="image").read_bytes()

    assert content[:8] == b"\x89PNG\r\n\x1a\n"


def test_zarr_store_open_passes_remote_storage_options(
    tmp_dir, test_session, monkeypatch
):
    from datachain.lib.file import File
    from datachain.lib.zarr import ZarrStore

    _make_store(tmp_dir / "s.zarr", "s")
    captured = {}
    real_open = zarr.open

    def fake_open(url, mode="r", storage_options=None):
        captured["url"] = url
        captured["storage_options"] = storage_options
        return real_open(str(tmp_dir / "s.zarr"), mode=mode)

    monkeypatch.setattr("datachain.lib.zarr.zarr.open", fake_open)

    file = File(source="s3://bucket", path="s.zarr")
    file._catalog = SimpleNamespace(client_config={"key": "secret"})
    info = ZarrStore(file=file).get_info()

    assert captured["storage_options"] == {"key": "secret"}
    assert info.attrs == {"who": "s"}


@pytest.mark.parametrize(
    "source,path,expected_source,expected_path",
    [
        # Discovered marker nested under the listing root.
        ("gs://bucket", "data/scan.zarr/zarr.json", "gs://bucket", "data/scan.zarr"),
        # Concrete store path: the store root is the source itself.
        ("gs://bucket/scan.zarr", "zarr.json", "gs://bucket", "scan.zarr"),
        # Whole bucket is a single store (no path segment to split off).
        ("s3://bucket", ".zgroup", "s3://bucket", ""),
        ("s3://bucket/", ".zgroup", "s3://bucket", ""),
    ],
)
def test_file_to_store_root_split(source, path, expected_source, expected_path):
    from datachain.lib.file import File
    from datachain.lib.zarr import file_to_store

    store = file_to_store(File(source=source, path=path))

    assert store.source == expected_source
    assert store.path == expected_path
