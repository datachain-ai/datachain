import asyncio
import functools
import gzip
import io
import threading
from datetime import timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from dateutil.parser import isoparse
from fsspec.asyn import sync
from fsspec.callbacks import DEFAULT_CALLBACK
from gcsfs.core import GCSFile
from gcsfs.extended_gcsfs import BucketType

from datachain.asyn import get_loop
from datachain.cache import Cache
from datachain.client import Client
from datachain.client.gcs import GCSClient, _ListedGCSFile
from datachain.client.writeconfig import WriteConfig
from datachain.lib.file import File


@pytest.mark.parametrize("block_size", [None, 256 * 1024])
def test_tar_member_versioned_gcs_read_ahead(
    catalog,
    monkeypatch,
    small_tar_member_data,
    small_tar_member,
    tar_member_range_requests,
    block_size,
):
    data, raw = small_tar_member_data
    client = GCSClient(
        "bucket",
        {"token": "anon", "default_block_size": 2 * 1024 * 1024, "version_aware": True},
        catalog.cache,
    )
    fs = client.fs
    monkeypatch.setattr(fs, "info", lambda *a, **kw: {"size": len(raw)})
    open_file = fs.open

    def open_readahead(path, **kwargs):
        return open_file(path, cache_type="readahead", **kwargs)

    monkeypatch.setattr(fs, "open", open_readahead)
    member = small_tar_member(client, version="123")
    # a listed member with a version opens as _ListedGCSFile, which checks etags
    requests = tar_member_range_requests(_ListedGCSFile)

    with member.open(block_size=block_size) as stream:
        assert stream.read() == data
        assert stream.read() == b""

    for file, _, _ in requests:
        assert isinstance(file, GCSFile)
        assert file.path == "bucket/archive.tar#123"
        assert file.generation == "123"
    assert requests[0][1] == member.location[0]["offset"] > 0
    fetched = sum(end - start for _, start, end in requests)
    expected_block = len(data) if block_size is None else block_size
    assert len(data) <= fetched <= len(data) + expected_block + io.DEFAULT_BUFFER_SIZE
    if block_size is not None:
        assert fetched >= block_size


@pytest.mark.parametrize("streaming", [True, False])
def test_write_kwargs_maps_fields(streaming):
    # gcsfs takes content_type directly and the other content settings via
    # fixed_key_metadata; custom metadata goes to metadata.
    cfg = WriteConfig(
        content_type="application/pdf",
        content_disposition="attachment",
        cache_control="max-age=3600",
        content_encoding="gzip",
        metadata={"a": "b"},
    )
    assert GCSClient._write_kwargs(cfg, streaming=streaming) == {
        "content_type": "application/pdf",
        "metadata": {"a": "b"},
        "fixed_key_metadata": {
            "content_disposition": "attachment",
            "cache_control": "max-age=3600",
            "content_encoding": "gzip",
        },
    }


def test_write_kwargs_empty_config_maps_nothing():
    assert GCSClient._write_kwargs(WriteConfig(), streaming=False) == {}


def test_write_kwargs_rejects_write_options():
    # gcsfs has no raw write-kwargs passthrough, so the escape hatch is rejected.
    cfg = WriteConfig(write_options={"foo": "bar"})
    with pytest.raises(NotImplementedError, match="write_options"):
        GCSClient._write_kwargs(cfg, streaming=False)


_FAKE_SAS = "https://storage.googleapis.com/foo/bar?x-goog-signature=abc"
_VER = "1234567890"
_INFO = {
    "name": "gs://foo/blob.txt",
    "size": 42,
    "etag": "abc123",
    "updated": "2024-01-01T00:00:00Z",
}


def _make_client() -> GCSClient:
    client = GCSClient("foo", {}, MagicMock())
    client._fs = MagicMock()
    client._fs.storage_options = {}  # non-anon
    client._fs.sign.return_value = _FAKE_SAS
    client._fs._info = AsyncMock(return_value=_INFO)
    client._fs._get_file = AsyncMock(return_value=None)
    return client


def test_anon_url():
    client = Client.get_client("gs://foo", None, anon=True)
    assert client.url("bar") == "https://storage.googleapis.com/foo/bar"


def test_anon_versioned_url():
    client = Client.get_client("gs://foo", None, anon=True)
    assert (
        client.url("bar", version_id="1234566")
        == "https://storage.googleapis.com/foo/bar?generation=1234566"
    )


def test_anon_url_hash_in_key():
    client = Client.get_client("gs://foo", None, anon=True)
    assert client.url("dir/file#variant.txt") == (
        "https://storage.googleapis.com/foo/dir/file%23variant.txt"
    )


def test_url_versioned_sign_called_with_hash_generation():
    client = _make_client()
    client.url("blob.txt", version_id=_VER)
    assert f"#{_VER}" in client._fs.sign.call_args[0][0]


def test_url_unversioned_sign_called_with_clean_path():
    client = _make_client()
    client.url("blob.txt")
    assert "#" not in client._fs.sign.call_args[0][0]


# '#' and '?' are valid GCS object key characters, but gcsfs split_path uses
# urlsplit() to extract the generation fragment, which misparses keys that
# contain those characters.  Until that is fixed upstream we raise early with
# a clear error.  The tests below document the *desired* behavior and are
# marked xfail so they become regular passes once the upstream is fixed and our
# guard is removed.
_xfail_special_char_versioned = pytest.mark.xfail(
    reason="gcsfs split_path breaks with '#'/'?' in versioned key names",
    strict=True,
)


@_xfail_special_char_versioned
def test_url_versioned_hash_in_key():
    client = _make_client()
    client.url("dir/file#variant.txt", version_id=_VER)


@_xfail_special_char_versioned
def test_url_versioned_qmark_in_key():
    client = _make_client()
    client.url("dir/file?variant.txt", version_id=_VER)


@_xfail_special_char_versioned
def test_get_file_info_hash_in_key_with_version():
    client = _make_client()
    client.get_file_info("blob#file.txt", version_id=_VER)


@_xfail_special_char_versioned
def test_get_file_info_qmark_in_key_with_version():
    client = _make_client()
    client.get_file_info("blob?file.txt", version_id=_VER)


@_xfail_special_char_versioned
def test_get_size_hash_in_key_with_version():
    client = _make_client()
    sync(
        get_loop(),
        client.get_size,
        File(source="gs://foo", path="blob#file.txt", version=_VER),
    )


@_xfail_special_char_versioned
def test_get_size_qmark_in_key_with_version():
    client = _make_client()
    sync(
        get_loop(),
        client.get_size,
        File(source="gs://foo", path="blob?file.txt", version=_VER),
    )


@_xfail_special_char_versioned
def test_get_current_etag_hash_in_key_with_version():
    client = _make_client()
    sync(
        get_loop(),
        client.get_current_etag,
        File(source="gs://foo", path="blob#file.txt", version=_VER),
    )


@_xfail_special_char_versioned
def test_get_current_etag_qmark_in_key_with_version():
    client = _make_client()
    sync(
        get_loop(),
        client.get_current_etag,
        File(source="gs://foo", path="blob?file.txt", version=_VER),
    )


def test_get_file_with_version():
    client = _make_client()
    sync(
        get_loop(),
        client.get_file,
        "gs://foo/blob.txt",
        "/dev/null",
        DEFAULT_CALLBACK,
        version_id=_VER,
    )
    args, kwargs = client._fs._get_file.call_args
    assert args[0] == f"gs://foo/blob.txt#{_VER}"
    assert kwargs.get("generation") is None


@_xfail_special_char_versioned
def test_get_file_hash_in_key_with_version():
    client = _make_client()
    sync(
        get_loop(),
        client.get_file,
        "gs://foo/blob#file.txt",
        "/dev/null",
        DEFAULT_CALLBACK,
        version_id=_VER,
    )
    args, kwargs = client._fs._get_file.call_args
    assert args[0] == f"gs://foo/blob#file.txt#{_VER}"
    assert kwargs.get("generation") is None


@_xfail_special_char_versioned
def test_get_file_qmark_in_key_with_version():
    client = _make_client()
    sync(
        get_loop(),
        client.get_file,
        "gs://foo/blob?file.txt",
        "/dev/null",
        DEFAULT_CALLBACK,
        version_id=_VER,
    )
    args, kwargs = client._fs._get_file.call_args
    assert args[0] == f"gs://foo/blob?file.txt#{_VER}"
    assert kwargs.get("generation") is None


@pytest.mark.parametrize(
    "value",
    [
        "2026-09-30T23:00:00.123Z",
        "2026-09-30T23:00:00.12Z",
        "2026-09-30T23:00:00.123456789Z",
        "2026-09-30T23:00:00Z",
        "2026-09-30T23:00:00+02:00",
    ],
)
def test_parse_timestamp_matches_isoparse(value):
    parsed = GCSClient.parse_timestamp(value)
    assert parsed == isoparse(value)
    assert isinstance(parsed.tzinfo, timezone)


class _Recording(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)


_PLAIN = b"hello world " * 400
_GZIPPED = gzip.compress(_PLAIN)


@pytest.fixture
def gcs_stub_url():
    async def metadata(request):
        return web.json_response(
            {
                "kind": "storage#object",
                "bucket": "b",
                "name": "data.txt",
                "generation": _VER,
                "size": str(len(_GZIPPED)),
                "etag": "e1",
                "contentEncoding": "gzip",
                "updated": "2026-10-28T00:00:00Z",
            }
        )

    async def media(request):
        if "gzip" not in request.headers.get("Accept-Encoding", ""):
            return web.Response(body=_PLAIN)
        start, end = 0, len(_GZIPPED) - 1
        if rng := request.headers.get("Range"):
            first, last = rng.removeprefix("bytes=").split("-")
            start, end = int(first), int(last or end)
        return web.Response(
            status=206 if rng else 200,
            body=_GZIPPED[start : end + 1],
            headers={"Content-Encoding": "gzip", "ETag": "e1"},
        )

    app = web.Application()
    app.router.add_get("/storage/v1/b/b/o/data.txt", metadata)
    app.router.add_get("/download/storage/v1/b/b/o/data.txt", media)
    runner = web.AppRunner(app)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    run = functools.partial(asyncio.run_coroutine_threadsafe, loop=loop)
    try:
        run(runner.setup()).result()
        run(web.TCPSite(runner, "127.0.0.1", 0).start()).result()
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        run(runner.cleanup()).result()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


@pytest.fixture
def stub_client(request, gcs_stub_url, tmp_path, mocker):
    mocker.patch(
        "gcsfs.extended_gcsfs.ExtendedGcsFileSystem._get_bucket_type",
        AsyncMock(return_value=BucketType.UNKNOWN),
    )
    client = GCSClient(
        "b",
        {
            "endpoint_url": gcs_stub_url,
            "token": "anon",
            "skip_instance_cache": True,
            "session_kwargs": getattr(request, "param", None),
        },
        Cache(str(tmp_path / "cache"), str(tmp_path / "tmp")),
    )
    yield client
    if client.fs._session is not None:
        sync(get_loop(), client.fs._session.close)


_STUB_FILE = File(
    source="gs://b", path="data.txt", version=_VER, etag="e1", size=len(_GZIPPED)
)


@pytest.mark.parametrize(
    "stub_client",
    [
        None,
        {"headers": None},
        {"headers": [("X-Caller", "1")]},
        {"headers": {"accept-encoding": "identity"}},
    ],
    ids=["default", "no-headers", "header-pairs", "caller-accept-encoding"],
    indirect=True,
)
def test_cached_read_of_gzip_encoded_object_keeps_stored_bytes(stub_client):
    stub_client.put_in_cache(_STUB_FILE)

    assert Path(stub_client.cache.get_path(_STUB_FILE)).read_bytes() == _GZIPPED


@pytest.mark.parametrize("offset", [0, 10], ids=["from-start", "seek"])
def test_uncached_read_of_gzip_encoded_object_keeps_stored_bytes(stub_client, offset):
    with stub_client.open_object(_STUB_FILE, use_cache=False, block_size=16) as f:
        f.seek(offset)
        data = f.read()

    assert data == _GZIPPED[offset:]


@pytest.mark.parametrize("version_aware", [False, True])
def test_listing_asks_for_every_field_it_reads(version_aware):
    client = _make_client()
    client._fs.version_aware = version_aware
    item = _Recording(
        name="a", size="1", generation="7", etag="e", updated="2026-10-01T00:00:00Z"
    )
    client._fs._call = AsyncMock(return_value={"items": [item]})

    sync(get_loop(), client._list_page, "", "", None)

    fields = client._fs._call.call_args.kwargs["fields"]
    assert fields == GCSClient.LIST_FIELDS
    assert item.read <= set(fields.removeprefix("items(").split(")")[0].split(","))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version, etag", [("", "abc123"), (_VER, ""), (_VER, "abc123")]
)
async def test_download_listed_file_checks_response_identity(tmp_path, version, etag):
    client = _make_client()
    client._fs.consistency = "size"
    client._fs._call = AsyncMock(return_value=({"ETag": '"abc123"'}, b"woof"))
    file = File(source="gs://foo", path="listed", size=4, etag=etag, version=version)
    callback = MagicMock()
    path = tmp_path / "tmp" / "file"
    await client._download_file(file, str(path), callback)
    assert path.read_bytes() == b"woof"
    client._fs._info.assert_not_called()
    client._fs.url.assert_called_once_with(
        f"gs://foo/listed#{version}" if version else "gs://foo/listed"
    )
    callback.set_size.assert_called_once_with(4)
    callback.relative_update.assert_called_once_with(4)


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{"ETag": '"changed"'}, {}])
async def test_download_listed_file_rejects_changed_or_missing_etag(tmp_path, headers):
    client = _make_client()
    client._fs._call = AsyncMock(return_value=(headers, b"woof"))
    file = File(source="gs://foo", path="listed", size=4, etag="abc123")
    path = tmp_path / "file"
    with pytest.raises(FileNotFoundError, match="update=True"):
        await client._download_file(file, str(path), MagicMock())
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("consistency", ["size", "md5"])
async def test_download_listed_file_rejects_corrupt_data(tmp_path, consistency):
    from gcsfs.retry import ChecksumError

    client = _make_client()
    client._fs.consistency = consistency
    client._fs._call = AsyncMock(
        return_value=({"ETag": "abc123", "X-Goog-Hash": "md5=invalid"}, b"woof")
    )
    file = File(source="gs://foo", path="listed", size=5, etag="abc123")
    path = tmp_path / "file"
    with pytest.raises((OSError, ChecksumError)):
        await client._download_file(file, str(path), MagicMock())
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [-1, 5 * 1024 * 1024 + 1])
async def test_download_without_small_file_metadata_uses_get_file(tmp_path, size):
    client = _make_client()
    file = File(source="gs://foo", path="listed", size=size, etag="abc123")
    await client._download_file(file, str(tmp_path / "file"), DEFAULT_CALLBACK)
    client._fs._get_file.assert_awaited_once()
