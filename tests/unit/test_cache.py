import asyncio
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientPayloadError
from fsspec.asyn import get_loop, sync
from fsspec.callbacks import Callback

from datachain.cache import Cache, get_temp_cache, temporary_cache
from datachain.client.http import HTTPClient
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


@pytest.mark.parametrize("error_type", [OSError, asyncio.CancelledError])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("own_callback", [False, True])
def test_download_error_cleans_temporary_file(
    cache, monkeypatch, error_type, partial, own_callback
):
    file = File(source="s3://bucket", path="file.bin", size=100)
    error = error_type("interrupted")
    callback = Mock(spec=Callback)
    monkeypatch.setattr("datachain.cache.TqdmCallback", Mock(return_value=callback))

    def get_file(lpath, rpath, **kwargs):
        if partial:
            Path(rpath).parent.mkdir(parents=True, exist_ok=True)
            Path(rpath).write_bytes(b"partial")
        raise error

    client = Mock()
    client.get_file = AsyncMock(side_effect=get_file)
    add = Mock()
    monkeypatch.setattr(cache.odb, "add", add)

    with pytest.raises(error_type) as exc:
        asyncio.run(cache.download(file, client, None if own_callback else callback))

    assert exc.value is error
    assert not list(Path(cache.tmp_dir).glob("*"))
    assert not cache.contains(file)
    add.assert_not_called()
    if own_callback:
        callback.close.assert_called_once_with()
    else:
        callback.close.assert_not_called()


@pytest.mark.parametrize("add_fails", [False, True])
@pytest.mark.parametrize("own_callback", [False, True])
def test_download_cleans_temporary_file_after_transfer(
    cache, monkeypatch, add_fails, own_callback
):
    file = File(source="s3://bucket", path="file.bin", size=4)
    callback = Mock(spec=Callback)
    monkeypatch.setattr("datachain.cache.TqdmCallback", Mock(return_value=callback))

    def get_file(lpath, rpath, **kwargs):
        Path(rpath).parent.mkdir(parents=True, exist_ok=True)
        Path(rpath).write_bytes(b"data")

    client = Mock()
    client.get_file = AsyncMock(side_effect=get_file)
    if add_fails:
        error = OSError("cache full")
        monkeypatch.setattr(cache.odb, "add", Mock(side_effect=error))
        with pytest.raises(OSError) as exc:
            asyncio.run(
                cache.download(file, client, None if own_callback else callback)
            )
        assert exc.value is error
        assert not cache.contains(file)
    else:
        asyncio.run(cache.download(file, client, None if own_callback else callback))
        assert Path(cache.get_path(file)).read_bytes() == b"data"

    assert not list(Path(cache.tmp_dir).glob("*"))
    if own_callback:
        callback.close.assert_called_once_with()
    else:
        callback.close.assert_not_called()


@pytest.mark.parametrize("error_type", [OSError, asyncio.CancelledError])
def test_download_cleanup_failure_preserves_transfer_error(
    cache, monkeypatch, error_type
):
    file = File(source="s3://bucket", path="file.bin", size=100)
    error = error_type("interrupted")

    def get_file(lpath, rpath, **kwargs):
        Path(rpath).parent.mkdir(parents=True, exist_ok=True)
        Path(rpath).write_bytes(b"partial")
        raise error

    client = Mock()
    client.get_file = AsyncMock(side_effect=get_file)
    cache_os = Mock(wraps=os)
    cache_os.unlink.side_effect = PermissionError("file is still open")
    monkeypatch.setattr("datachain.cache.os", cache_os)

    with pytest.raises(error_type) as exc:
        asyncio.run(cache.download(file, client, Callback()))

    assert exc.value is error
    cache_os.unlink.assert_called_once()
    assert not cache.contains(file)


def test_download_cleanup_failure_after_success_is_reported(cache, monkeypatch):
    file = File(source="s3://bucket", path="file.bin", size=4)

    def get_file(lpath, rpath, **kwargs):
        Path(rpath).parent.mkdir(parents=True, exist_ok=True)
        Path(rpath).write_bytes(b"data")

    client = Mock()
    client.get_file = AsyncMock(side_effect=get_file)
    error = PermissionError("file is still open")
    cache_os = Mock(wraps=os)
    cache_os.unlink.side_effect = error
    monkeypatch.setattr("datachain.cache.os", cache_os)

    with pytest.raises(PermissionError) as exc:
        asyncio.run(cache.download(file, client, Callback()))

    assert exc.value is error
    cache_os.unlink.assert_called_once()
    assert Path(cache.get_path(file)).read_bytes() == b"data"


def test_download_interrupted_http_response_cleans_temporary_file(cache):
    body = b"x" * (128 * 1024)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body) * 2))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HTTPClient(f"127.0.0.1:{server.server_port}", {}, cache)
            file = File(source=client.uri, path="file.bin", size=len(body) * 2)
            Path(cache.tmp_dir).mkdir(parents=True, exist_ok=True)

            with pytest.raises(ClientPayloadError):
                sync(get_loop(), cache.download, file, client, Callback())

            assert not list(Path(cache.tmp_dir).glob("*"))
            assert not cache.contains(file)
        finally:
            server.shutdown()
            thread.join()
