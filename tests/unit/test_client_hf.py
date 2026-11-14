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


SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def repo_client():
    client = HfClient("datasets", {}, Mock(spec=Cache))
    client._fs = Mock()
    fs = client._fs.sync_fs
    fs.endpoint = "https://huggingface.co"
    fs.resolve_path = lambda path: SimpleNamespace(
        repo_id="org/repo",
        repo_type="dataset",
        revision=path.split("@")[1].split("/")[0],
        path_in_repo="data/a.json",
    )
    client._fs._get_file = AsyncMock()
    return client


class FakeStream:
    def __init__(self, fs, path):
        self.data = [b"ab", b"c"]

    def read(self, size):
        assert self.url() == (
            f"https://huggingface.co/datasets/org/repo/resolve/{SHA}/data/a.json"
        )
        return self.data.pop(0) if self.data else b""

    def url(self):
        raise AssertionError("lists the folder")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def test_get_file_is_one_get_of_the_resolve_url(repo_client, monkeypatch, tmp_path):
    monkeypatch.setattr(hf_file_system, "HfFileSystemStreamFile", FakeStream)
    callback = Mock()

    sync(
        get_loop(),
        repo_client.get_file,
        f"hf://datasets/org/repo@{SHA}/data/a.json",
        str(tmp_path / "a"),
        callback,
    )

    assert (tmp_path / "a").read_bytes() == b"abc"
    assert [c.args for c in callback.relative_update.call_args_list] == [(2,), (1,)]
    repo_client._fs._get_file.assert_not_awaited()


def test_get_file_from_a_bucket_uses_the_filesystem(repo_client, monkeypatch):
    monkeypatch.setattr(hf_file_system, "HfFileSystemStreamFile", None)
    repo_client._fs.sync_fs.resolve_path = lambda path: SimpleNamespace(path="a")

    sync(get_loop(), repo_client.get_file, "hf://buckets/org/b/a", "local", Mock())

    repo_client._fs._get_file.assert_awaited_once()


@pytest.mark.parametrize("revision,checked", [(SHA, False), ("main", True)])
def test_files_at_a_commit_are_not_checked_for_changes(repo_client, revision, checked):
    commit = SimpleNamespace(oid="c", date=datetime(2024, 10, 12, tzinfo=UTC))
    repo_client._fs._info = AsyncMock(
        return_value={"size": 4, "blob_id": "e", "last_commit": commit}
    )
    file = File(source="hf://datasets", path=f"org/repo@{revision}/a", etag="e")

    repo_client.check_unchanged(file)

    assert repo_client._fs._info.await_count == checked
