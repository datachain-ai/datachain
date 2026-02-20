import gc
import os

import pytest

from datachain.cache import Cache
from datachain.lib.file import File
from datachain.lib.pytorch import PytorchDataset
from datachain.lib.settings import Settings


@pytest.mark.parametrize(
    "cache,prefetch", [(True, 0), (True, 10), (False, 10), (False, 0)]
)
def test_cache(catalog, cache, prefetch):
    settings = Settings(cache=cache, prefetch=prefetch)
    ds = PytorchDataset("fake", 1, catalog, dc_settings=settings)
    assert ds.cache == cache
    assert ds.prefetch == prefetch

    if cache or not prefetch:
        assert catalog.cache is ds._cache
        return

    assert catalog.cache is not ds._cache
    head, tail = os.path.split(ds._cache.cache_dir)
    assert head == catalog.cache.tmp_dir
    assert tail.startswith("prefetch-")


@pytest.mark.parametrize("cache", [True, False])
def test_close(mocker, catalog, cache):
    spy = mocker.spy(Cache, "destroy")
    ds = PytorchDataset(
        "fake", 1, catalog, dc_settings=Settings(cache=cache, prefetch=10)
    )

    ds.close()

    if cache:
        spy.assert_not_called()
    else:
        spy.assert_called_once()


def test_prefetched_files_are_removed_after_yield(tmp_dir, mocker, catalog, cache):
    files = []
    for name in "abc":
        (tmp_dir / name).write_text(name, encoding="utf-8")
        file = File(path=name, source=f"file://{tmp_dir}")
        file._set_stream(catalog)
        files.append((file,))

    ds = PytorchDataset(
        "fake",
        1,
        catalog,
        dc_settings=Settings(prefetch=10),
        remove_prefetched=True,
    )
    mocker.patch.object(ds, "_row_iter", return_value=iter(files))

    seen = []
    for (file,) in ds._iter_with_prefetch():
        # previously prefetched files should have been removed by now
        for f in seen:
            assert not f._catalog.cache.contains(f)
            assert not f.get_local_path()
        seen.append(file)

        assert file._catalog.cache.contains(file)
        assert file.get_local_path()


@pytest.mark.parametrize("cache", [True, False])
def test_prefetch_cache_gets_destroyed_on_gc(mocker, catalog, cache):
    spy = mocker.patch.object(Cache, "destroy")
    ds = PytorchDataset(
        "fake", 1, catalog, dc_settings=Settings(cache=cache, prefetch=10)
    )

    del ds
    gc.collect()

    if cache:
        spy.assert_not_called()
    else:
        spy.assert_called_once()
