import os
from datetime import UTC, datetime
from unittest.mock import Mock, mock_open, patch

import pytest

from datachain.cache import Cache
from datachain.client.fsspec import Client
from datachain.client.http import HTTPClient, HTTPSClient
from datachain.lib.file import File, FileError


def test_protocol_detection_https():
    client_class = Client.get_implementation("https://example.com/file.txt")
    assert client_class == HTTPSClient

    client_class = Client.get_implementation("http://example.com/file.txt")
    assert client_class == HTTPClient


def test_split_url():
    domain, path = HTTPClient.split_url("https://example.com/path/to/file.txt")
    assert domain == "example.com"
    assert path == "path/to/file.txt"

    # URL with query and fragment
    domain, path = HTTPClient.split_url("https://example.com/api?key=123#anchor")
    assert domain == "example.com"
    assert path == "api?key=123#anchor"


def test_is_root_url():
    assert HTTPClient.is_root_url("https://example.com")
    assert HTTPClient.is_root_url("https://example.com/")
    assert not HTTPClient.is_root_url("https://example.com/path")
    assert not HTTPClient.is_root_url("https://example.com?query=1")
    assert not HTTPClient.is_root_url("https://example.com#fragment")


def test_from_name_with_https():
    cache = Mock(spec=Cache)
    client = HTTPSClient.from_name("https://example.com/path", cache, {})
    assert client.protocol == "https"
    assert client.name == "example.com/path"
    assert client.PREFIX == "https://"


def test_from_name_with_http():
    cache = Mock(spec=Cache)
    client = HTTPClient.from_name("http://example.com/path", cache, {})
    assert client.protocol == "http"
    assert client.name == "example.com/path"
    assert client.PREFIX == "http://"


def test_get_uri_http():
    cache = Mock(spec=Cache)
    client = HTTPClient("example.com:8080", {}, cache)

    assert client.get_uri("file.txt") == "http://example.com:8080/file.txt"


def test_upload_raises_not_implemented():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    with pytest.raises(NotImplementedError, match="HTTP/HTTPS client is read-only"):
        client.upload(b"data", "path/to/file.txt")


def test_create_fs():
    from fsspec.implementations.http import HTTPFileSystem

    fs = HTTPClient.create_fs()
    assert isinstance(fs, HTTPFileSystem)

    fs = HTTPClient.create_fs(version_aware=True)
    assert isinstance(fs, HTTPFileSystem)
    assert not hasattr(fs, "version_aware")

    fs = HTTPClient.create_fs(timeout=30, headers={"User-Agent": "test"})
    assert isinstance(fs, HTTPFileSystem)

    fs = HTTPClient.create_fs()
    assert isinstance(fs, HTTPFileSystem)


def test_info_to_file():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    info = {
        "size": 1024,
        "ETag": '"abc123"',
        "last_modified": "Wed, 12 Oct 2024 07:28:00 GMT",
    }
    file = client.info_to_file(info, "path/to/file.txt")
    assert file.path == "path/to/file.txt"
    assert file.size == 1024
    assert file.etag == "abc123"

    # Test with int timestamp
    info = {"last_modified": 1700000000}
    file = client.info_to_file(info, "file1.txt")
    expected = datetime.fromtimestamp(1700000000, UTC)
    assert file.last_modified == expected

    # Test with float timestamp
    info = {"last_modified": 1700000000.5}
    file = client.info_to_file(info, "file2.txt")
    expected = datetime.fromtimestamp(1700000000.5, UTC)
    assert file.last_modified == expected

    # Test with invalid date string (triggers ValueError)
    info = {"last_modified": "invalid"}
    file = client.info_to_file(info, "file3.txt")
    assert isinstance(file.last_modified, datetime)  # Falls back to current time


def test_open_object(monkeypatch):
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    file = File(source="https://example.com", path="path/to/file.txt")

    # Test with cache hit
    cache.get_path.return_value = "/cache/path/file.txt"
    with patch("builtins.open", mock_open(read_data=b"cached content")) as mock_file:
        client.open_object(file, use_cache=True)
        mock_file.assert_called_once_with("/cache/path/file.txt", mode="rb")

    # Test without cache (cache miss)
    cache.get_path.return_value = None
    monkeypatch.setattr(client.fs, "open", Mock())
    client.open_object(file, use_cache=True)
    client.fs.open.assert_called_once()

    # Test with use_cache=False (bypass cache)
    cache.get_path.return_value = "/cache/path/file.txt"
    monkeypatch.setattr(client.fs, "open", Mock())
    client.open_object(file, use_cache=False)
    client.fs.open.assert_called_once()  # Should fetch from remote, not cache


def test_read_tar_member_of_uncached_archive_returns_its_data(
    http_server, tmp_path, catalog, small_tar_member_data, small_tar_member
):
    data, raw = small_tar_member_data
    (tmp_path / "archive.tar").write_bytes(raw)
    member = small_tar_member(catalog.get_client(http_server))

    assert member.read() == data


def test_url():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)
    assert client.url("file.txt") == "https://example.com/file.txt"


def test_get_file_info(monkeypatch):
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)
    monkeypatch.setattr(client.fs, "info", Mock(return_value={"size": 2048}))

    file = client.get_file_info("file.txt")
    assert file.path == "file.txt"
    assert file.size == 2048


@pytest.mark.asyncio
async def test_fetch_dir():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    with pytest.raises(NotImplementedError):
        await client._fetch_dir("prefix", None, None)


@pytest.mark.asyncio
async def test_get_file(monkeypatch):
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    async def mock_get_file(lpath, rpath, callback):
        return "data"

    monkeypatch.setattr(client.fs, "_get_file", mock_get_file)
    result = await client.get_file("/local", "/remote", None)
    assert result == "data"


def test_storage_uri():
    assert str(HTTPSClient.storage_uri("https://example.com")) == "https://example.com"
    assert str(HTTPSClient.storage_uri("example.com")) == "https://example.com"


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("operation", ["prefix", "metadata", "full", "save"])
def test_http_reads_reject_overwritten_source(
    http_server, tmp_path, catalog, cache, operation
):
    from PIL import Image

    if operation == "metadata":
        pytest.importorskip("torch")

    path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(path)
    client = catalog.get_client(http_server)
    file = client.get_file_info("image.png")
    assert file.etag
    file._set_stream(catalog, caching_enabled=cache)
    mtime = path.stat().st_mtime + 1
    Image.new("RGB", (3, 3)).save(path)
    os.utime(path, (mtime, mtime))

    error = FileError if operation == "metadata" else FileNotFoundError
    destination = tmp_path / "saved.png"
    destination.write_bytes(b"previous destination")
    with pytest.raises(error) as exc:
        if operation == "prefix":
            file.read_bytes(3)
        elif operation == "metadata":
            file.as_image_file().get_info()
        elif operation == "save":
            file.save(str(destination))
        else:
            file.read_bytes()
    cause = exc.value.__cause__ if operation == "metadata" else exc.value
    assert isinstance(cause, FileNotFoundError)
    assert "changed on the source" in str(cause)
    assert not catalog.cache.contains(file)
    assert destination.read_bytes() == b"previous destination"


def test_http_cache_hit_reads_saved_content_after_source_changes(
    http_server, tmp_path, catalog
):
    path = tmp_path / "content.bin"
    path.write_bytes(b"original")
    file = catalog.get_client(http_server).get_file_info("content.bin")
    file._set_stream(catalog, caching_enabled=True)
    file.ensure_cached()
    path.unlink()
    assert file.read_bytes(3) == b"ori"
    assert file.read_bytes() == b"original"
    file.save(str(tmp_path / "saved.bin"))
    assert (tmp_path / "saved.bin").read_bytes() == b"original"
