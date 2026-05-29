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
    Client._ANON_BUCKETS.clear()
    yield
    Client._ANON_BUCKETS.clear()


def _gcs_client(bucket: str = "foo", **fs_kwargs) -> GCSClient:
    return GCSClient(bucket, fs_kwargs, MagicMock())


def test_anon_fallback_no_error_no_retry(monkeypatch, _clear_anon_cache):
    client = _gcs_client()
    client._fs = MagicMock()
    client._fs._info = AsyncMock(
        return_value={
            "name": "gs://foo/x.txt",
            "size": 1,
            "etag": "e",
            "updated": "2024-01-01T00:00:00Z",
        }
    )
    create_fs = MagicMock()
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    client.get_file_info("x.txt")

    create_fs.assert_not_called()
    assert not GCSClient._bucket_needs_anon("foo")


def test_anon_fallback_retry_succeeds_marks_bucket(monkeypatch, _clear_anon_cache):
    client = _gcs_client()
    auth_fs = MagicMock()
    auth_fs._info = AsyncMock(side_effect=PermissionError)
    client._fs = auth_fs

    anon_fs = MagicMock()
    anon_fs._info = AsyncMock(
        return_value={
            "name": "gs://foo/x.txt",
            "size": 1,
            "etag": "e",
            "updated": "2024-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(GCSClient, "create_fs", MagicMock(return_value=anon_fs))

    client.get_file_info("x.txt")

    assert GCSClient._bucket_needs_anon("foo")
    assert client._fs is anon_fs
    assert GCSClient.create_fs.call_args.kwargs.get("anon") is True


def test_anon_fallback_retry_also_fails_marks_as_failed(monkeypatch, _clear_anon_cache):
    client = _gcs_client()
    auth_fs = MagicMock()
    auth_fs._info = AsyncMock(side_effect=PermissionError)
    client._fs = auth_fs

    anon_fs = MagicMock()
    anon_fs._info = AsyncMock(side_effect=PermissionError)
    monkeypatch.setattr(GCSClient, "create_fs", MagicMock(return_value=anon_fs))

    with pytest.raises(PermissionError):
        client.get_file_info("x.txt")

    assert GCSClient._bucket_needs_anon("foo") is False
    assert client._fs is auth_fs


def test_anon_fallback_cached_as_failed_skips_retry(monkeypatch, _clear_anon_cache):
    GCSClient._mark_bucket_anon("foo", False)
    client = _gcs_client()
    auth_fs = MagicMock()
    auth_fs._info = AsyncMock(side_effect=PermissionError)
    client._fs = auth_fs
    create_fs = MagicMock()
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    with pytest.raises(PermissionError):
        client.get_file_info("x.txt")

    create_fs.assert_not_called()


def test_anon_fallback_cached_bucket_uses_anon_directly(monkeypatch, _clear_anon_cache):
    GCSClient._mark_bucket_anon("foo", True)
    create_fs = MagicMock()
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    _ = _gcs_client().fs

    create_fs.assert_called_once()
    assert create_fs.call_args.kwargs.get("anon") is True


def test_anon_fallback_explicit_anon_no_retry(monkeypatch, _clear_anon_cache):
    client = _gcs_client(anon=True)
    client._fs = MagicMock()
    client._fs._info = AsyncMock(side_effect=PermissionError)
    create_fs = MagicMock()
    monkeypatch.setattr(GCSClient, "create_fs", create_fs)

    with pytest.raises(PermissionError):
        client.get_file_info("x.txt")

    create_fs.assert_not_called()
    assert not GCSClient._bucket_needs_anon("foo")


def test_anon_fallback_open_object_retry_succeeds(monkeypatch, _clear_anon_cache):
    client = _gcs_client()
    auth_fs = MagicMock()
    auth_fs.open = MagicMock(side_effect=PermissionError)
    client._fs = auth_fs
    client.cache.get_path = MagicMock(return_value=None)

    anon_fs = MagicMock()
    anon_fs.open = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(GCSClient, "create_fs", MagicMock(return_value=anon_fs))

    client.open_object(File(source="gs://foo", path="x.txt"))

    assert GCSClient._bucket_needs_anon("foo")
