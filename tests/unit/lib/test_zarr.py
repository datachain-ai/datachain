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
