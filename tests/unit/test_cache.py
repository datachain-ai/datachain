import asyncio
import errno
import os
import stat
from contextlib import nullcontext
from pathlib import Path

import pytest
from dvc_objects.fs.utils import remove
from fsspec.asyn import get_loop, sync

from datachain.cache import Cache, get_temp_cache, temporary_cache
from datachain.lib.file import File


@pytest.fixture
def cache(tmp_path):
    return Cache(str(tmp_path / "cache"), str(tmp_path / "tmp"))


def test_simple(cache):
    uid = File(source="s3://foo", path="data/bar", etag="xyz", size=3, location=None)
    data = b"foo"
    assert not cache.contains(uid)

    cache.store_data(uid, data)
    assert cache.contains(uid)
    with open(cache.get_path(uid), mode="rb") as f:
        assert f.read() == data

    cache.clear()
    assert not cache.contains(uid)


def test_get_total_size(cache):
    file_info = [
        ("file1", b"foo"),
        ("file2", b"bar"),
        ("file3", b"some file data"),
        ("file4", b"more file data " * 1024),
    ]
    expected_total = sum(len(d) for _, d in file_info)
    for name, data in file_info:
        uid = File(
            source="s3://foo", path=f"data/{name}", etag="xyz", size=3, location=None
        )
        cache.store_data(uid, data)
    total = cache.get_total_size()
    assert total == expected_total

    cache.clear()
    empty_total = cache.get_total_size()
    assert empty_total == 0


def test_remove(cache):
    uid = File(
        source="s3://bkt42", path="dir1/dir2/file84", etag="abc", size=3, location=None
    )
    cache.store_data(uid, b"some random string 679")

    assert cache.contains(uid)
    cache.remove(uid)
    assert not cache.contains(uid)


def test_destroy(cache: Cache):
    file = File(source="s3://foo", path="data/bar", etag="xyz", size=3, location=None)
    cache.store_data(file, b"foo")
    assert cache.contains(file)

    cache.destroy()
    assert not os.path.exists(cache.cache_dir)


def test_get_temp_cache(tmp_path):
    temp = get_temp_cache(tmp_path, prefix="test-")
    assert os.path.isdir(temp.cache_dir)
    assert isinstance(temp, Cache)
    head, tail = os.path.split(temp.cache_dir)
    assert head == str(tmp_path)
    assert tail.startswith("test-")


def test_temporary_cache(tmp_path):
    with temporary_cache(tmp_path, prefix="test-") as temp:
        assert os.path.isdir(temp.cache_dir)
    assert not os.path.exists(temp.cache_dir)


@pytest.mark.parametrize("method", ["remove", "store_data", "clear", "destroy"])
def test_readonly_blocks_sync_mutating_methods(cache, method):
    file = File(source="s3://foo", path="data/bar", etag="xyz", size=3, location=None)
    cache.store_data(file, b"foo")
    ro = cache.as_readonly()

    args: tuple = ()
    if method == "remove":
        args = (file,)
    elif method == "store_data":
        args = (file, b"foo")

    with pytest.raises(RuntimeError, match=f"cannot call {method}\\(\\)"):
        getattr(ro, method)(*args)


def test_readonly_blocks_download(cache):
    file = File(source="s3://foo", path="data/bar", etag="xyz", size=3, location=None)
    ro = cache.as_readonly()
    with pytest.raises(RuntimeError, match=r"cannot call download\(\)"):
        sync(get_loop(), ro.download, file, None)


def test_cache_download_reserved_chars_in_path(cloud_test_catalog_upload):
    rel_path = "dir #% percent/hello.txt"

    catalog = cloud_test_catalog_upload.catalog
    source = cloud_test_catalog_upload.src_uri
    client = catalog.get_client(source)
    client.upload(b"hi", rel_path)

    file = File(path=rel_path, source=source, size=-1)
    file._set_stream(catalog, False)

    sync(get_loop(), catalog.cache.download, file, client)

    cached_path = catalog.cache.get_path(file)
    assert cached_path is not None
    contents = Path(cached_path).read_text(encoding="utf-8")
    assert contents == "hi"


@pytest.mark.parametrize("error", [None, ConnectionResetError, asyncio.CancelledError])
def test_download_cleans_temp_and_protects_cache(cache, error, monkeypatch, mocker):
    class Client:
        async def get_file(self, src, dst, callback, version_id=None):
            Path(dst).write_bytes(b"foo")  # noqa: ASYNC240
            if error:
                raise error

    def remove_temp(path):
        # Windows cleanup clears read-only state shared by hardlinked paths.
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        remove(path)

    monkeypatch.setattr("datachain.cache.remove", remove_temp)
    mocker.patch.object(
        cache.odb.fs, "reflink", side_effect=OSError(errno.ENOTSUP, "unsupported")
    )
    file = File(source="s3://foo", path="data/bar", etag="xyz", size=3)
    os.makedirs(cache.tmp_dir)
    with pytest.raises(error) if error else nullcontext():
        asyncio.run(cache.download(file, Client()))

    assert os.listdir(cache.tmp_dir) == []
    assert cache.contains(file) is (error is None)
    if error is None:
        assert cache.odb.is_protected(cache.get_path(file))


@pytest.mark.asyncio
@pytest.mark.parametrize("same_configuration", [True, False])
async def test_concurrent_downloads_share_only_the_same_client_configuration(
    cache, same_configuration
):
    from types import SimpleNamespace

    from datachain.client.fsspec import Client

    file = File(source="s3://bucket", path="shared", size=4)
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def put(file, **kwargs):
        calls.append(file.path)
        started.set()
        await release.wait()
        cache.store_data(file, b"data")

    fs = object()
    first = SimpleNamespace(cache=cache, fs=fs, _put_in_cache=put)
    second = SimpleNamespace(
        cache=cache, fs=fs if same_configuration else object(), _put_in_cache=put
    )
    tasks = [asyncio.create_task(Client._download(c, file)) for c in (first, second)]
    await started.wait()
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*tasks)
    assert len(calls) == (1 if same_configuration else 2)
    assert not cache._download_locks


@pytest.mark.asyncio
async def test_cancelled_download_waiter_does_not_cancel_the_transfer(cache):
    from types import SimpleNamespace

    from datachain.client.fsspec import Client

    file = File(source="s3://bucket", path="shared", size=4)
    started, release = asyncio.Event(), asyncio.Event()

    async def put(file, **kwargs):
        started.set()
        await release.wait()
        cache.store_data(file, b"data")

    client = SimpleNamespace(cache=cache, fs=object(), _put_in_cache=put)
    owner = asyncio.create_task(Client._download(client, file))
    await started.wait()
    waiter = asyncio.create_task(Client._download(client, file))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not owner.done()
    release.set()
    await owner
    assert cache.contains(file)
    assert not cache._download_locks


@pytest.mark.asyncio
async def test_failed_shared_download_can_be_retried(cache):
    from types import SimpleNamespace

    from datachain.client.fsspec import Client

    file = File(source="s3://bucket", path="shared", size=4)
    calls = []

    async def put(file, **kwargs):
        calls.append(1)
        await asyncio.sleep(0)
        if len(calls) == 1:
            raise OSError("transfer failed")
        cache.store_data(file, b"data")

    client = SimpleNamespace(cache=cache, fs=object(), _put_in_cache=put)
    results = await asyncio.gather(
        Client._download(client, file),
        Client._download(client, file),
        return_exceptions=True,
    )
    assert isinstance(results[0], OSError)
    assert results[1] is None
    assert cache.contains(file)
    assert not cache._download_locks


def test_cache_pickle_does_not_copy_runtime_owners(cache):
    import pickle

    file = File(source="s3://bucket", path="shared", size=4)
    cache.store_data(file, b"data")
    oid = cache._retain_prefetched(file)
    restored = pickle.loads(pickle.dumps(cache))  # noqa: S301
    assert restored.contains(file)
    assert not restored._prefetch_refs
    cache._release_prefetched(oid)


def test_prefetch_keeps_shared_archive_until_the_run_ends(cache, catalog, monkeypatch):
    from contextlib import closing

    from datachain.lib.udf import _prefetch_inputs

    parent = File(source="s3://bucket", path="archive.tar", size=4)
    members = [
        File(
            source="s3://bucket",
            path=f"member{i}",
            size=1,
            location=[{"vtype": "tar", "parent": parent.model_dump()}],
        )
        for i in range(2)
    ]
    catalog.cache = cache
    for member in members:
        member._set_stream(catalog)

    async def prefetch(file, download_cb=None):
        target = file._prefetch_target()
        if not cache.contains(target):
            cache.store_data(target, b"data")
        return True

    monkeypatch.setattr(File, "_prefetch", prefetch)
    with closing(
        _prefetch_inputs([[f] for f in members], prefetch=2, remove_prefetched=True)
    ) as rows:
        next(rows)
        assert cache.contains(parent)
        next(rows)
        assert cache.contains(parent)
    # a later row may read another member; the run's temporary cache removes it
    assert cache.contains(parent)
    assert not cache._prefetch_refs


def test_prefetch_close_releases_unconsumed_rows_and_original_identity(
    cache, catalog, monkeypatch
):
    from contextlib import closing

    from datachain.lib.udf import _prefetch_inputs

    catalog.cache = cache
    file = File(source="s3://bucket", path="original", size=4)
    original = file.model_copy()
    file._set_stream(catalog)

    async def prefetch(file, download_cb=None):
        cache.store_data(file, b"data")
        return True

    monkeypatch.setattr(File, "_prefetch", prefetch)
    with closing(
        _prefetch_inputs([[file]], prefetch=1, remove_prefetched=True)
    ) as rows:
        next(rows)
        file.path = "changed-by-udf"
    assert not cache.contains(original)
    assert not cache._prefetch_refs
