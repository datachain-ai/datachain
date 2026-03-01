from unittest.mock import AsyncMock, MagicMock

import pytest
from fsspec.asyn import sync
from fsspec.callbacks import DEFAULT_CALLBACK

from datachain.asyn import get_loop
from datachain.client import Client
from datachain.client.gcs import GCSClient
from datachain.lib.file import File

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
    assert args[0] == "gs://foo/blob#file.txt"
    assert f"#{_VER}" not in args[0]
    assert kwargs.get("generation") == _VER


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
    assert args[0] == "gs://foo/blob?file.txt"
    assert f"#{_VER}" not in args[0]
    assert kwargs.get("generation") == _VER
