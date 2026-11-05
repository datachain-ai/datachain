import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, mock_open, patch
from urllib.parse import unquote

import pytest

import datachain as dc
from datachain.cache import Cache
from datachain.client.fsspec import Client
from datachain.client.http import HTTPClient, HTTPSClient
from datachain.lib.file import File


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


@pytest.mark.parametrize("client_class", [HTTPClient, HTTPSClient])
@pytest.mark.parametrize(
    "path,encoded_path",
    [
        ("v1.0/file.txt", "v1.0/file.txt"),
        ("files.example.com/file.txt", "files.example.com/file.txt"),
        (
            "release-1.2.3/file with spaces.txt?download=1#section",
            "release-1.2.3/file%20with%20spaces.txt?download=1#section",
        ),
        (
            "v1.0/file%20name.txt?token=a%2Fb#section",
            "v1.0/file%20name.txt?token=a%2Fb#section",
        ),
        ("nested/v1.0/file.txt", "nested/v1.0/file.txt"),
        (".well-known/file.txt", ".well-known/file.txt"),
    ],
)
def test_relative_paths_keep_the_clients_host(client_class, path, encoded_path):
    client = client_class("example.com:8080", {}, Mock(spec=Cache))
    expected = f"{client_class.PREFIX}example.com:8080/{encoded_path}"

    assert client.get_uri(path) == expected
    assert client.url(path) == expected


@pytest.mark.parametrize("client_class", [HTTPClient, HTTPSClient])
@pytest.mark.parametrize("scheme_in_name", [False, True])
def test_scheme_only_source_accepts_host_in_path(client_class, scheme_in_name):
    name = client_class.PREFIX if scheme_in_name else ""
    client = client_class(name, {}, Mock(spec=Cache))

    assert (
        client.get_uri("example.com/data/file.txt")
        == f"{client_class.PREFIX}example.com/data/file.txt"
    )


@pytest.mark.parametrize("client_class", [HTTPClient, HTTPSClient])
@pytest.mark.parametrize(
    "path", ["example.com:8080/data/file.txt", "127.0.0.1:8000/data/file.txt"]
)
def test_scheme_only_from_name_preserves_host_ports(client_class, path):
    client = client_class.from_name(client_class.PREFIX, Mock(spec=Cache), {})

    assert client.name == ""
    assert client.get_uri(path) == f"{client_class.PREFIX}{path}"


@pytest.mark.parametrize("client_class", [HTTPClient, HTTPSClient])
def test_full_url_source_preserves_its_base_path(client_class):
    client = client_class(
        f"{client_class.PREFIX}example.com/base", {}, Mock(spec=Cache)
    )

    assert (
        client.get_uri("v1.0/file.txt")
        == f"{client_class.PREFIX}example.com/base/v1.0/file.txt"
    )


def test_read_storage_http_metadata_matches_the_requested_path(test_session):
    expected_body = b"the requested file contents"
    other_body = b"other"
    paths = []
    expected_path = ""

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_content(False)

        def do_GET(self):
            self.send_content(True)

        def send_content(self, send_body):
            path = unquote(self.path)
            paths.append(path)
            body = expected_body if path == expected_path else other_body
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "ETag", '"expected"' if body == expected_body else '"other"'
            )
            self.end_headers()
            if send_body:
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host = f"127.0.0.1:{server.server_port}"
            expected_path = f"/{host}/file.txt"
            uri = f"http://{host}{expected_path}"
            control = File.at(uri, session=test_session).resolve()
            assert control.size == len(expected_body)

            scheme_client = HTTPClient.from_name(
                "http://", test_session.catalog.cache, {}
            )
            scheme_metadata = scheme_client.get_file_info(f"{host}{expected_path}")
            assert scheme_metadata.size == len(expected_body)
            assert scheme_metadata.etag == "expected"

            file = dc.read_storage(uri, session=test_session).to_values("file")[0]

            assert file.source == f"http://{host}"
            assert file.path == f"{host}/file.txt"
            assert file.size == len(expected_body)
            assert file.etag == "expected"
            assert file.read() == expected_body
            assert set(paths) == {expected_path}
        finally:
            server.shutdown()
            thread.join()


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
async def test_get_file():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    async def mock_get_file(lpath, rpath, callback):
        return "data"

    client.fs._get_file = mock_get_file
    result = await client.get_file("/local", "/remote", None)
    assert result == "data"


def test_storage_uri():
    assert str(HTTPSClient.storage_uri("https://example.com")) == "https://example.com"
    assert str(HTTPSClient.storage_uri("example.com")) == "https://example.com"
