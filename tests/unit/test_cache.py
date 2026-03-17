import os
from pathlib import Path

import pytest
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
