import pickle
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fsspec.asyn import sync
from huggingface_hub import hf_file_system
from huggingface_hub.hf_file_system import HfFileSystemFile

from datachain.asyn import get_loop
from datachain.cache import Cache
from datachain.client.hf import HfClient
from datachain.lib.file import File


def test_info_to_file_uses_last_commit_metadata_when_present():
    cache = Mock(spec=Cache)
    client = HfClient("datasets/org/repo", {}, cache)
    commit_date = datetime(2024, 10, 12, 7, 28, tzinfo=UTC)

    file = client.info_to_file(
        {
            "size": 1024,
            "blob_id": "abc123",
            "last_commit": SimpleNamespace(oid="commit123", date=commit_date),
        },
        "path/to/file.parquet",
    )

    assert file.path == "path/to/file.parquet"
    assert file.size == 1024
    assert file.etag == "abc123"
    assert file.version == "commit123"
    assert file.last_modified == commit_date


def test_info_to_file_requires_last_commit_metadata():
    cache = Mock(spec=Cache)
    client = HfClient("datasets/org/repo", {}, cache)

    with pytest.raises(RuntimeError, match="without last_commit"):
        client.info_to_file(
            {
                "size": 1024,
                "blob_id": "abc123",
                "last_commit": None,
            },
            "path/to/file.parquet",
        )


def test_ls_dir_requests_expanded_info():
    cache = Mock(spec=Cache)
    client = HfClient("datasets/org/repo", {}, cache)
    client._fs = Mock()
    client._fs._ls = AsyncMock(return_value=[])

    sync(get_loop(), client.ls_dir, "datasets/org/repo/path")

    client._fs._ls.assert_awaited_once_with(
        "datasets/org/repo/path", detail=True, expand_info=True
    )


def test_get_file_info_requests_expanded_info():
    cache = Mock(spec=Cache)
    client = HfClient("datasets/org/repo", {}, cache)
    commit_date = datetime(2024, 10, 12, 7, 28, tzinfo=UTC)
    client._fs = Mock()
    client._fs._info = AsyncMock(
        return_value={
            "size": 1024,
            "blob_id": "abc123",
            "last_commit": SimpleNamespace(oid="commit123", date=commit_date),
        }
    )

    file = client.get_file_info("path/to/file.parquet")

    client._fs._info.assert_awaited_once_with(
        "hf://datasets/org/repo/path/to/file.parquet", expand_info=True
    )
    assert file.version == "commit123"
    assert file.last_modified == commit_date


def test_upload_requests_expanded_info_for_returned_file():
    cache = Mock(spec=Cache)
    client = HfClient("datasets/org/repo", {}, cache)
    commit_date = datetime(2024, 10, 12, 7, 28, tzinfo=UTC)
    client._fs = Mock()
    client._fs.info.return_value = {
        "size": 4,
        "blob_id": "abc123",
        "last_commit": SimpleNamespace(oid="commit123", date=commit_date),
    }

    file = client.upload(b"data", "path/to/file.parquet")

    client._fs.makedirs.assert_called_once_with(
        "hf://datasets/org/repo/path/to", exist_ok=True
    )
    client._fs.pipe_file.assert_called_once_with(
        "hf://datasets/org/repo/path/to/file.parquet", b"data"
    )
    client._fs.info.assert_called_once_with(
        "hf://datasets/org/repo/path/to/file.parquet", expand_info=True
    )
    assert file.version == "commit123"
    assert file.last_modified == commit_date


@pytest.fixture
def hf_archive_client(catalog, monkeypatch, small_tar_member_data):
    _, raw = small_tar_member_data
    client = HfClient("datasets/fictional/fixture", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    assert fs.block_size is None
    resolved_path_type = getattr(
        hf_file_system,
        "HfFileSystemResolvedRepositoryPath",
        hf_file_system.HfFileSystemResolvedPath,
    )
    monkeypatch.setattr(
        fs,
        "resolve_path",
        lambda *a, **kw: resolved_path_type(
            repo_type="dataset",
            repo_id="fictional/fixture",
            revision="main",
            path_in_repo="archive.tar",
        ),
    )
    monkeypatch.setattr(fs, "info", lambda *a, **kw: {"size": len(raw), "type": "file"})
    return client


def test_tar_member_huggingface_read_ahead(
    hf_archive_client,
    small_tar_member_data,
    small_tar_member,
    tar_member_range_requests,
):
    data, _ = small_tar_member_data
    member = small_tar_member(hf_archive_client)
    requests = tar_member_range_requests(HfFileSystemFile)

    with member.open() as stream:
        assert stream.read() == data
        assert stream.read() == b""

    assert all(isinstance(file, HfFileSystemFile) for file, _, _ in requests)
    fetched = sum(end - start for _, start, end in requests)
    assert len(data) <= fetched <= 2 * len(data)


def test_open_object_huggingface_block_size_zero_reads_only_asked_bytes(
    hf_archive_client, small_tar_member_data, tar_member_range_requests
):
    _, raw = small_tar_member_data
    requests = tar_member_range_requests(HfFileSystemFile)
    file = File(source=hf_archive_client.uri, path="archive.tar", size=len(raw))

    with hf_archive_client.open_object(file, use_cache=False, block_size=0) as f:
        f.seek(len(raw) - 10)
        assert f.read(5) == raw[-10:-5]

    assert [(start, end) for _, start, end in requests] == [
        (len(raw) - 10, len(raw) - 5)
    ]


def test_pinned_version_is_read_by_revision():
    client = HfClient("datasets/org/repo", {}, Mock(spec=Cache))
    client._fs = Mock()
    client._fs._get_file = AsyncMock()

    sync(get_loop(), client.get_file, "org/repo/a.txt", "local/a", None, "commit123")

    client._fs._get_file.assert_awaited_once_with(
        "org/repo/a.txt", "local/a", callback=None, revision="commit123"
    )


@pytest.mark.parametrize(
    "path",
    [
        "org/repo@main/archive.tar",
        "org/repo@refs/pr/1/archive.tar",
        "org/repo@refs%2Fpr%2F1/archive.tar",
        "org/repo@v1/archive.tar",
        "org/repo/archive.tar",
    ],
)
@pytest.mark.parametrize("block_size", [None, 0])
@pytest.mark.parametrize("repo_type", ["datasets", "models"])
def test_open_reads_listed_commit_with_revision_in_uri(
    catalog,
    monkeypatch,
    small_tar_member_data,
    tar_member_range_requests,
    path,
    block_size,
    repo_type,
):
    from huggingface_hub import HfFileSystem

    monkeypatch.setattr(
        HfFileSystem, "_repo_and_revision_exist", lambda *a, **kw: (True, None)
    )
    client = HfClient(repo_type, {"token": False}, catalog.cache)
    _, raw = small_tar_member_data
    monkeypatch.setattr(client.fs.sync_fs, "info", lambda *a, **kw: {"size": len(raw)})
    tar_member_range_requests(HfFileSystemFile)
    file = File(source=client.uri, path=path, size=len(raw), version="commit123")
    with client.open_object(file, use_cache=False, block_size=block_size) as stream:
        assert stream.fileobj.resolved_path.revision == "commit123"
        assert stream.read(5) == raw[:5]
    assert file.path == path


def test_unpinned_uri_keeps_branch_and_at_in_filename(catalog, monkeypatch):
    from huggingface_hub import HfFileSystem

    monkeypatch.setattr(
        HfFileSystem, "_repo_and_revision_exist", lambda *a, **kw: (True, None)
    )
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    assert fs.resolve_path("datasets/org/repo@main/a@b.txt").revision == "main"
    from datachain.client.hf_file_system import _drop_path_revision

    assert (
        _drop_path_revision("datasets/org/repo/a@b.txt") == "datasets/org/repo/a@b.txt"
    )


def test_hf_filesystem_is_pickleable(catalog):
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    restored = pickle.loads(pickle.dumps(fs))  # noqa: S301
    assert type(restored) is type(fs)
    assert restored.storage_options == fs.storage_options
