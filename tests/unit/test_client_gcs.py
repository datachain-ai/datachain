from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Barrier, Event
from unittest.mock import AsyncMock, MagicMock

import pytest
from fsspec.asyn import sync
from fsspec.callbacks import DEFAULT_CALLBACK
from gcsfs import GCSFileSystem

from datachain.asyn import get_loop
from datachain.client import Client
from datachain.client.gcs import GCSClient
from datachain.client.writeconfig import WriteConfig
from datachain.lib.file import File


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
    client._fs.version_aware = False
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


def test_hash_generation_suffix_is_literal_for_unversioned_reads(tmp_path, monkeypatch):
    path = "gs://foo/dir/#0"
    client = _make_client()
    client.fs_kwargs = {"token": "anon", "version_aware": True}
    split_path = GCSFileSystem._split_path(path, version_aware=True)
    assert split_path == ("foo", "dir/", "0")
    client._fs.version_aware = True
    client._fs.split_path.return_value = split_path

    unversioned_fs = MagicMock()
    unversioned_fs._get_file = AsyncMock(return_value=None)
    unversioned_fs.open.return_value = BytesIO(b"object contents")
    create_fs = MagicMock(return_value=unversioned_fs)
    monkeypatch.setattr(client, "create_fs", create_fs)
    client.cache.get_path.return_value = None

    sync(
        get_loop(),
        client.get_file,
        path,
        str(tmp_path / "download"),
        DEFAULT_CALLBACK,
    )
    with client.open_object(
        File(source="gs://foo", path="dir/#0"), use_cache=False
    ) as f:
        assert f.read() == b"object contents"

    create_fs.assert_called_once_with(token="anon", version_aware=False)  # noqa: S106
    unversioned_fs._get_file.assert_awaited_once_with(
        path, str(tmp_path / "download"), callback=DEFAULT_CALLBACK
    )
    unversioned_fs.open.assert_called_once_with(path, block_size=None)


def test_unversioned_filesystem_is_cached_during_concurrent_reads(monkeypatch):
    path = "gs://foo/dir/#0"
    client = _make_client()
    client.fs_kwargs = {"token": "anon", "version_aware": True}
    client._fs.version_aware = True
    split_barrier = Barrier(8)

    def split_path(_):
        split_barrier.wait(timeout=5)
        return "foo", "dir/", "0"

    client._fs.split_path.side_effect = split_path

    unversioned_fs = MagicMock()
    create_started = Event()
    release_create = Event()

    def create_unversioned_fs(**kwargs):
        create_started.set()
        if not release_create.wait(timeout=5):
            raise TimeoutError("filesystem creation was not released")
        return unversioned_fs

    create_fs = MagicMock(side_effect=create_unversioned_fs)
    monkeypatch.setattr(client, "create_fs", create_fs)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(client._fs_for_path, path, None) for _ in range(8)]
        try:
            assert create_started.wait(timeout=5)
        finally:
            release_create.set()
        filesystems = [future.result(timeout=5) for future in futures]

    assert all(fs is unversioned_fs for fs in filesystems)
    create_fs.assert_called_once_with(token="anon", version_aware=False)  # noqa: S106


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
