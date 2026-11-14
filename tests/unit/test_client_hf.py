from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fsspec.asyn import sync
from huggingface_hub import hf_file_system
from huggingface_hub.hf_file_system import HfFileSystemFile

from datachain.asyn import get_loop
from datachain.cache import Cache
from datachain.client import hf
from datachain.client.hf import HfClient
from datachain.lib.file import File


def test_get_file_info_and_upload_give_no_version_and_the_blob_id():
    client = HfClient("datasets", {}, Mock(spec=Cache))
    client._fs = Mock()
    info = {"size": 4, "blob_id": "abc123", "type": "file"}
    client._fs._info = AsyncMock(return_value=info)
    client._fs.info.return_value = info

    listed = client.get_file_info("org/repo/a.txt")
    uploaded = client.upload(b"data", "org/repo/a.txt")

    client._fs._info.assert_awaited_once_with("hf://datasets/org/repo/a.txt")
    client._fs.info.assert_called_once_with("hf://datasets/org/repo/a.txt")
    for file in (listed, uploaded):
        assert (file.path, file.size, file.etag, file.version) == (
            "org/repo/a.txt",
            4,
            "abc123",
            "",
        )


def _listing(client, prefix):
    async def run():
        return [
            (f.path, f.size, f.etag)
            async for page in client.scandir(prefix)
            for f in page
        ]

    return sync(get_loop(), run)


def _repo_path(path):
    """A resolved repository path, as old huggingface_hub versions give it:
    ``unresolve()`` drops ``@main``."""
    sub = path.partition("@main")[2].strip("/")
    return SimpleNamespace(
        repo_id="org/repo",
        repo_type="dataset",
        revision="main",
        path_in_repo=sub,
        unresolve=lambda: f"datasets/org/repo/{sub}".rstrip("/"),
    )


@pytest.fixture
def client():
    client = HfClient("datasets", {}, Mock(spec=Cache))
    client._fs = Mock()
    fs = client._fs.sync_fs
    fs.resolve_path = _repo_path
    fs._api.repo_info.return_value = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename=name, size=size, blob_id=f"b{size}")
            for name, size in [
                ("README.md", 1),
                ("data/a.json", 2),
                ("data/x/b.json", 3),
            ]
        ]
    )
    fs.find.return_value = {
        "datasets/org/repo/README.md": {"size": 1, "blob_id": "b1"},
        "datasets/org/repo/data/a.json": {"size": 2, "blob_id": "b2"},
        "datasets/org/repo/data/x/b.json": {"size": 3, "blob_id": "b3"},
    }
    return client


DATA = [
    ("org/repo@main/data/a.json", 2, "b2"),
    ("org/repo@main/data/x/b.json", 3, "b3"),
]


def test_listing_takes_the_files_from_the_repository_info(client):
    assert _listing(client, "org/repo@main/data") == DATA
    client._fs.sync_fs._api.repo_info.assert_called_once_with(
        "org/repo", repo_type="dataset", revision="main", files_metadata=True
    )
    client._fs.sync_fs.find.assert_not_called()


def test_listing_walks_the_tree_when_the_repository_info_is_capped(client, monkeypatch):
    monkeypatch.setattr(hf, "SIBLINGS_LIMIT", 3)
    client._fs.sync_fs.find.return_value = {
        k: v for k, v in client._fs.sync_fs.find.return_value.items() if "/data/" in k
    }

    assert _listing(client, "org/repo@main/data") == DATA
    client._fs.sync_fs.find.assert_called_once_with(
        "datasets/org/repo@main/data", detail=True
    )


def test_listing_the_repository_root_returns_every_file(client):
    assert len(_listing(client, "org/repo@main")) == 3


@pytest.mark.parametrize("limit", [hf.SIBLINGS_LIMIT, 0])
@pytest.mark.parametrize("prefix", ["org/repo@main/nope", "org/repo@main/README.md"])
def test_listing_a_missing_folder_raises(client, monkeypatch, limit, prefix):
    monkeypatch.setattr(hf, "SIBLINGS_LIMIT", limit)
    if prefix.endswith("README.md"):  # find gives a file itself without metadata
        client._fs.sync_fs.find.return_value = {"datasets/org/repo/README.md": {}}
    else:
        client._fs.sync_fs.find.return_value = {}

    with pytest.raises(FileNotFoundError, match=prefix):
        _listing(client, prefix)


def test_listing_an_empty_repository_is_empty(client):
    client._fs.sync_fs._api.repo_info.return_value = SimpleNamespace(siblings=None)

    assert _listing(client, "org/repo@main") == []


@pytest.mark.parametrize(
    "found,expected",
    [
        ({"buckets/org/b/data/a.json": {"size": 2}}, [("org/b/data/a.json", 2, "")]),
        ({}, FileNotFoundError),
    ],
)
def test_listing_a_bucket_walks_the_tree(found, expected):
    client = HfClient("buckets", {}, Mock(spec=Cache))
    client._fs = Mock()
    fs = client._fs.sync_fs
    fs.resolve_path = lambda path: SimpleNamespace(
        bucket_id="org/b", path="data", unresolve=lambda: "buckets/org/b/data"
    )
    fs.find.return_value = found

    if expected is FileNotFoundError:
        with pytest.raises(FileNotFoundError):
            _listing(client, "org/b/data")
    else:
        assert _listing(client, "org/b/data") == expected
    fs._api.repo_info.assert_not_called()


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
