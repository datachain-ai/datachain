import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from datachain.client import Client
from datachain.client.gcs import GCSClient
from datachain.client.local import FileClient
from datachain.lib.file import File


def test_bad_protocol():
    with pytest.raises(NotImplementedError):
        Client.get_implementation("bogus://bucket")


def test_win_paths_are_recognized():
    if sys.platform != "win32":
        pytest.skip()

    assert Client.get_implementation("file://C:/bucket") == FileClient
    assert Client.get_implementation("file://C:\\bucket") == FileClient
    assert Client.get_implementation("file://\\bucket") == FileClient
    assert Client.get_implementation("file:///bucket") == FileClient
    assert Client.get_implementation("C://bucket") == FileClient
    assert Client.get_implementation("C:\\bucket") == FileClient
    assert Client.get_implementation("\bucket") == FileClient


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_path_ends_with_slash(cloud_type):
    uri, rel_part = Client.parse_url("./animals/".replace("/", os.sep))
    assert uri == (Path().absolute() / Path("animals")).as_uri()
    assert rel_part == ""


@pytest.fixture
def _clear_anon_cache():
    Client._anon_buckets.clear()
    yield
    Client._anon_buckets.clear()


def _gcs_client(bucket: str = "foo", **fs_kwargs) -> GCSClient:
    return GCSClient(bucket, fs_kwargs, MagicMock())


def _info_ok():
    return AsyncMock(
        return_value={
            "name": "gs://foo/x.txt",
            "size": 1,
            "etag": "e",
            "updated": "2024-01-01T00:00:00Z",
        }
    )


def test_anon_fallback_no_error_no_retry(monkeypatch, _clear_anon_cache):
    auth_fs = MagicMock()
    auth_fs._info = _info_ok()
    create_fs = MagicMock(return_value=auth_fs)
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client = _gcs_client()
    client.get_file_info("x.txt")

    create_fs.assert_called_once()
    assert create_fs.call_args.kwargs.get("anon") is None
    assert GCSClient._bucket_needs_anon("foo") is None


def test_anon_fallback_retry_succeeds_marks_bucket(monkeypatch, _clear_anon_cache):
    auth_fs = MagicMock()
    auth_fs._info = AsyncMock(side_effect=PermissionError)
    anon_fs = MagicMock()
    anon_fs._info = _info_ok()
    create_fs = MagicMock(side_effect=[auth_fs, anon_fs])
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client = _gcs_client()
    client.get_file_info("x.txt")

    assert GCSClient._bucket_needs_anon("foo") is True
    assert create_fs.call_count == 2
    assert create_fs.call_args_list[1].kwargs.get("anon") is True


def test_anon_fallback_retry_also_fails_marks_as_failed(monkeypatch, _clear_anon_cache):
    auth_fs = MagicMock()
    auth_fs._info = AsyncMock(side_effect=PermissionError)
    anon_fs = MagicMock()
    anon_fs._info = AsyncMock(side_effect=PermissionError)
    create_fs = MagicMock(side_effect=[auth_fs, anon_fs])
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client = _gcs_client()
    with pytest.raises(PermissionError):
        client.get_file_info("x.txt")

    assert GCSClient._bucket_needs_anon("foo") is False
    assert create_fs.call_count == 2


def test_anon_fallback_cached_as_failed_skips_retry(monkeypatch, _clear_anon_cache):
    GCSClient._mark_bucket_anon("foo", False)
    auth_fs = MagicMock()
    auth_fs._info = AsyncMock(side_effect=PermissionError)
    create_fs = MagicMock(return_value=auth_fs)
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client = _gcs_client()
    with pytest.raises(PermissionError):
        client.get_file_info("x.txt")

    create_fs.assert_called_once()
    assert create_fs.call_args.kwargs.get("anon") is None


def test_anon_fallback_cached_bucket_uses_anon_directly(monkeypatch, _clear_anon_cache):
    GCSClient._mark_bucket_anon("foo", True)
    create_fs = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    _ = _gcs_client().fs._info

    create_fs.assert_called_once()
    assert create_fs.call_args.kwargs.get("anon") is True


def test_anon_fallback_explicit_anon_no_retry(monkeypatch, _clear_anon_cache):
    auth_fs = MagicMock()
    auth_fs._info = AsyncMock(side_effect=PermissionError)
    create_fs = MagicMock(return_value=auth_fs)
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client = _gcs_client(anon=True)
    with pytest.raises(PermissionError):
        client.get_file_info("x.txt")

    create_fs.assert_called_once()
    assert GCSClient._bucket_needs_anon("foo") is None


def test_anon_fallback_open_object_retry_succeeds(monkeypatch, _clear_anon_cache):
    auth_fs = MagicMock()
    auth_fs.open = MagicMock(side_effect=PermissionError)
    anon_fs = MagicMock()
    anon_fs.open = MagicMock(return_value=MagicMock())
    create_fs = MagicMock(side_effect=[auth_fs, anon_fs])
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client = _gcs_client()
    client.cache.get_path = MagicMock(return_value=None)
    client.open_object(File(source="gs://foo", path="x.txt"))

    assert GCSClient._bucket_needs_anon("foo") is True


def test_anon_fallback_explicit_creds_ignore_cache(monkeypatch, _clear_anon_cache):
    # A previous no-creds caller cached the bucket as anon-needed.
    GCSClient._mark_bucket_anon("foo", True)

    # New client with explicit creds: should NOT use anon directly, should
    # try with creds first, and should NOT overwrite the cache.
    auth_fs = MagicMock()
    auth_fs._info = _info_ok()
    anon_fs = MagicMock()
    create_fs = MagicMock(side_effect=[auth_fs, anon_fs])
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client = _gcs_client(token="explicit-token")  # noqa: S106
    client.get_file_info("x.txt")

    # Only the auth fs was built, anon was never instantiated.
    create_fs.assert_called_once()
    assert create_fs.call_args.kwargs.get("anon") is None
    # Cache was not touched - still True from the earlier no-creds caller.
    assert GCSClient._bucket_needs_anon("foo") is True


def test_anon_fallback_explicit_creds_anon_retry_does_not_cache(
    monkeypatch, _clear_anon_cache
):
    # Client with explicit creds whose creds fail - anon retry succeeds,
    # but the success must NOT pollute the shared cache.
    auth_fs = MagicMock()
    auth_fs._info = AsyncMock(side_effect=PermissionError)
    anon_fs = MagicMock()
    anon_fs._info = _info_ok()
    create_fs = MagicMock(side_effect=[auth_fs, anon_fs])
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client = _gcs_client(token="explicit-token")  # noqa: S106
    client.get_file_info("x.txt")

    # Cache untouched.
    assert GCSClient._bucket_needs_anon("foo") is None


def test_anon_fallback_write_open_no_retry_no_cache_poisoning(
    monkeypatch, _clear_anon_cache
):
    # A write-mode open that fails with PermissionError must NOT trigger
    # an anon retry (anon can never write) and must NOT poison the cache
    # with False - otherwise future legitimate reads would lose fallback.
    auth_fs = MagicMock()
    auth_fs.open = MagicMock(side_effect=PermissionError)
    create_fs = MagicMock(return_value=auth_fs)
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client = _gcs_client()
    with pytest.raises(PermissionError):
        client.fs.open("gs://foo/x.txt", "wb")

    # No anon fs was built (no retry attempted).
    create_fs.assert_called_once()
    assert create_fs.call_args.kwargs.get("anon") is None
    # Cache is clean - subsequent reads can still fall back.
    assert GCSClient._bucket_needs_anon("foo") is None
