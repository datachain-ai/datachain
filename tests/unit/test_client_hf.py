import pickle
from contextlib import aclosing, contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fsspec.asyn import sync
from huggingface_hub import hf_file_system
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.hf_file_system import HfFileSystemFile
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    _pagination,
)

from datachain.asyn import get_loop
from datachain.cache import Cache
from datachain.client.hf import HfClient
from datachain.lib.file import File


@pytest.fixture
def hf_repositories(monkeypatch):
    calls = []

    def exists(self, repo_type, repo_id, revision):
        calls.append((repo_type, repo_id, revision))
        if (
            repo_id in {"org/repo", "models/repo", "buckets/org", "gpt2"}
            and revision != "missing"
        ):
            return True, None
        return False, HFValidationError("Unknown test repository")

    monkeypatch.setattr(hf_file_system.HfFileSystem, "_repo_and_revision_exist", exists)
    return calls


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


def test_recursive_listing_preserves_versions_without_walking_directories(
    catalog, monkeypatch, hf_repositories
):
    client = HfClient("datasets", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    date = datetime(2024, 10, 12, tzinfo=UTC)
    monkeypatch.setattr(
        fs,
        "find",
        Mock(
            return_value={
                "datasets/org/repo@main/data/a.json": {
                    "size": 3,
                    "blob_id": "blob-a",
                    "last_commit": SimpleNamespace(oid="old-commit", date=date),
                },
                "datasets/org/repo@main/data/nested/b.json": {
                    "size": 4,
                    "blob_id": "blob-b",
                    "last_commit": SimpleNamespace(oid="other-commit", date=date),
                },
            }
        ),
    )
    ls_spy = Mock(side_effect=AssertionError("recursive listing must not call ls"))
    monkeypatch.setattr(fs, "ls", ls_spy)
    rows = sync(get_loop(), lambda: _collect_hf_listing(client, "org/repo@main/data"))
    assert [(r.path, r.size, r.etag, r.version, r.last_modified) for r in rows] == [
        ("org/repo@main/data/a.json", 3, "blob-a", "old-commit", date),
        ("org/repo@main/data/nested/b.json", 4, "blob-b", "other-commit", date),
    ]
    fs.find.assert_called_once_with(
        "datasets/org/repo@main/data", detail=True, expand_info=True
    )
    ls_spy.assert_not_called()


async def _collect_hf_listing(client, prefix):
    return [file async for page in client.scandir(prefix) for file in page]


def test_recursive_listing_accepts_single_files_and_empty_directories(
    catalog, monkeypatch, hf_repositories
):
    client = HfClient("datasets", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    date = datetime(2024, 10, 12, tzinfo=UTC)
    monkeypatch.setattr(
        fs,
        "find",
        lambda *a, **kw: {"datasets/org/repo@main/file": {}},
    )
    info_calls = []

    def fake_info(path, *a, **kw):
        info_calls.append((path, kw))
        return {
            "size": 4,
            "blob_id": "blob",
            "last_commit": SimpleNamespace(oid="c1", date=date),
        }

    monkeypatch.setattr(fs, "info", fake_info)
    rows = sync(get_loop(), lambda: _collect_hf_listing(client, "org/repo@main/file"))
    assert [(r.path, r.version) for r in rows] == [("org/repo@main/file", "c1")]
    assert info_calls[-1] == ("datasets/org/repo@main/file", {"expand_info": True})

    monkeypatch.setattr(fs, "find", lambda *a, **kw: {})
    info_calls.clear()
    monkeypatch.setattr(
        fs,
        "info",
        lambda path, *a, **kw: (info_calls.append((path, kw)), {"type": "directory"})[
            1
        ],
    )
    assert (
        sync(get_loop(), lambda: _collect_hf_listing(client, "org/repo@main/file"))
        == []
    )
    assert info_calls == [("datasets/org/repo@main/file", {})]

    monkeypatch.setattr(
        fs,
        "info",
        Mock(side_effect=FileNotFoundError("missing")),
    )
    with pytest.raises(FileNotFoundError):
        sync(get_loop(), lambda: _collect_hf_listing(client, "org/repo@main/file"))


def test_recursive_listing_uses_sdk_pagination(catalog, monkeypatch, hf_repositories):
    client = HfClient("datasets", {"token": False}, catalog.cache)
    next_url = "https://huggingface.co/api/datasets/org/repo/tree/main?cursor=page2"
    page1 = httpx.Response(
        200,
        headers={"Link": f'<{next_url}>; rel="next"'},
        json=[
            {"type": "directory", "path": "data", "oid": "tree", "size": 0},
            {
                "type": "file",
                "path": "data/a.txt",
                "size": 4,
                "oid": "blob-c1",
                "lastCommit": {
                    "id": "c1",
                    "title": "first",
                    "date": "2024-10-12T00:00:00.000Z",
                },
            },
        ],
        request=httpx.Request(
            "GET", "https://huggingface.co/api/datasets/org/repo/tree/main"
        ),
    )
    page2 = httpx.Response(
        200,
        json=[
            {
                "type": "file",
                "path": "data/deep/b.txt",
                "size": 7,
                "oid": "blob-c2",
                "lastCommit": {
                    "id": "c2",
                    "title": "second",
                    "date": "2024-10-13T00:00:00.000Z",
                },
            }
        ],
        request=httpx.Request("GET", next_url),
    )
    responses = [page1, page2]
    requests = []

    def fake_http_backoff(method, url, **kwargs):
        requests.append((method, url, kwargs.get("params")))
        return responses.pop(0)

    monkeypatch.setattr(_pagination, "http_backoff", fake_http_backoff)
    rows = sync(get_loop(), lambda: _collect_hf_listing(client, "org/repo@main"))
    assert [(r.path, r.version, r.size, r.etag) for r in rows] == [
        ("org/repo@main/data/a.txt", "c1", 4, "blob-c1"),
        ("org/repo@main/data/deep/b.txt", "c2", 7, "blob-c2"),
    ]
    assert len(requests) == 2
    assert requests[1][1] == next_url


def test_recursive_listing_bucket_preserves_xet_hash_and_mtime_and_skips_siblings(
    catalog, monkeypatch, hf_repositories
):
    client = HfClient("buckets", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    monkeypatch.setattr(
        hf_file_system.HfFileSystem,
        "_bucket_exists",
        lambda self, bid: (True, None),
        raising=False,
    )
    date = datetime(2025, 2, 3, tzinfo=UTC)
    if hasattr(fs._api, "list_bucket_tree"):
        bucket_file_cls = getattr(hf_file_system, "BucketFile", SimpleNamespace)

        def fake_bucket_tree(bucket_id, prefix=None, recursive=False, **kwargs):
            yield bucket_file_cls(
                type="file",
                path="logs/a.parquet",
                size=100,
                xetHash="xet-abc",
                uploadedAt="2025-02-03T00:00:00.000Z",
            )
            yield bucket_file_cls(
                type="file",
                path="logs_old/b.parquet",
                size=200,
                xetHash="xet-sibling",
                uploadedAt="2025-02-03T00:00:00.000Z",
            )

        monkeypatch.setattr(fs._api, "list_bucket_tree", fake_bucket_tree)
    else:
        monkeypatch.setattr(
            fs,
            "find",
            lambda *a, **kw: {
                "buckets/org/bucket/logs/a.parquet": {
                    "size": 100,
                    "xet_hash": "xet-abc",
                    "uploaded_at": date,
                },
                "buckets/org/bucket/logs_old/b.parquet": {
                    "size": 200,
                    "xet_hash": "xet-sibling",
                    "uploaded_at": date,
                },
            },
        )
    rows = sync(get_loop(), lambda: _collect_hf_listing(client, "org/bucket/logs"))
    assert [(r.path, r.size, r.etag, r.version, r.last_modified) for r in rows] == [
        ("org/bucket/logs/a.parquet", 100, "xet-abc", "", date)
    ]


def test_recursive_listing_streams_pages_and_stops_on_cancel(
    catalog, monkeypatch, hf_repositories
):
    client = HfClient("datasets", {"token": False}, catalog.cache)
    client.LIST_PAGE_SIZE = 1
    fs = client.fs.sync_fs
    date = datetime(2024, 10, 12, tzinfo=UTC)
    commit = SimpleNamespace(oid="c1", date=date)
    monkeypatch.setattr(
        fs,
        "find",
        lambda *a, **kw: {
            "datasets/org/repo/a.txt": {
                "size": 1,
                "blob_id": "b1",
                "last_commit": commit,
            },
            "datasets/org/repo/b.txt": {
                "size": 2,
                "blob_id": "b2",
                "last_commit": commit,
            },
            "datasets/org/repo/c.txt": {
                "size": 3,
                "blob_id": "b3",
                "last_commit": commit,
            },
        },
    )

    async def collect_pages():
        return [[row.path for row in page] async for page in client.scandir("org/repo")]

    assert sync(get_loop(), collect_pages) == [
        ["org/repo/a.txt"],
        ["org/repo/b.txt"],
        ["org/repo/c.txt"],
    ]

    async def cancel_after_first_page():
        seen = []
        async with aclosing(client.scandir("org/repo")) as gen:
            async for page in gen:
                seen.append([row.path for row in page])
                break
        return seen

    assert sync(get_loop(), cancel_after_first_page) == [["org/repo/a.txt"]]


def test_recursive_listing_requires_last_commit(catalog, monkeypatch, hf_repositories):
    repo_client = HfClient("datasets", {"token": False}, catalog.cache)
    monkeypatch.setattr(
        repo_client.fs.sync_fs,
        "find",
        lambda *args, **kwargs: {
            "datasets/org/repo/a.txt": {"size": 4, "last_commit": None}
        },
    )
    with pytest.raises(RuntimeError, match="without last_commit"):
        sync(get_loop(), lambda: _collect_hf_listing(repo_client, "org/repo"))


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


@pytest.mark.parametrize("revision", ["c1", None])
def test_cache_fill_with_branch_uri_downloads_selected_revision(
    catalog, monkeypatch, hf_repositories, revision
):
    client = HfClient("datasets", {"token": False}, catalog.cache)
    monkeypatch.setattr(client.fs, "_info", AsyncMock(return_value={"size": 4}))
    monkeypatch.setattr(
        client.fs.sync_fs, "info", lambda *a, **kw: {"size": 4, "type": "file"}
    )
    urls = []

    def download(url, temp_file, **kwargs):
        urls.append(url)
        temp_file.write(b"data")

    monkeypatch.setattr(hf_file_system, "http_get", download)
    file = File(
        source=client.uri, path="org/repo@main/a.txt", size=4, version=revision or ""
    )

    client.put_in_cache(file)

    assert [url.split("/resolve/")[1] for url in urls] == [
        f"{revision or 'main'}/a.txt"
    ]
    with client.open_object(file) as stream:
        assert stream.read() == b"data"


@pytest.mark.parametrize(
    "path",
    [
        "repo@main/archive.tar",
        "repo@refs/pr/1/archive.tar",
        "repo@refs%2Fpr%2F1/archive.tar",
        "repo@v1/archive.tar",
        "repo/archive.tar",
    ],
)
@pytest.mark.parametrize("block_size", [None, 0])
@pytest.mark.parametrize("client_name,prefix", [("datasets", "org/"), ("org", "")])
def test_open_reads_listed_commit_with_revision_in_uri(
    catalog,
    monkeypatch,
    small_tar_member_data,
    tar_member_range_requests,
    path,
    block_size,
    client_name,
    prefix,
    hf_repositories,
):
    client = HfClient(client_name, {"token": False}, catalog.cache)
    _, raw = small_tar_member_data
    commit = "a" * 40
    monkeypatch.setattr(client.fs, "_info", AsyncMock(return_value={"size": len(raw)}))
    monkeypatch.setattr(
        client.fs.sync_fs,
        "info",
        Mock(
            side_effect=AssertionError(
                "open_object must not call fs.info when size is known"
            )
        ),
    )
    tar_member_range_requests(HfFileSystemFile)
    rel_path = f"{prefix}{path}"
    file = File(source=client.uri, path=rel_path, size=len(raw), version=commit)
    with client.open_object(file, use_cache=False, block_size=block_size) as stream:
        assert stream.fileobj.resolved_path.revision == commit
        assert stream.read(5) == raw[:5]
    assert file.path == rel_path


@pytest.mark.parametrize(
    "path,revision,expected",
    [
        ("datasets/org/repo@main/a@b.txt", None, ("main", "a@b.txt")),
        ("datasets/org/repo@main/a@b.txt", "c1", ("c1", "a@b.txt")),
        ("datasets/org/repo/a@b.txt", "c1", ("c1", "a@b.txt")),
        ("datasets/org/repo/@file.txt", "c1", ("c1", "@file.txt")),
    ],
)
def test_resolve_path_keeps_at_sign_in_file_name(
    catalog, hf_repositories, path, revision, expected
):
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    resolved = fs.resolve_path(path, revision=revision)
    assert (resolved.revision, resolved.path_in_repo) == expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("datasets/org/repo@main/a@b.txt", "a@b.txt"),
        ("datasets/org/repo/a@b.txt", "a@b.txt"),
        ("datasets/org/repo@refs/pr/1/a.txt", "a.txt"),
        ("org/repo/a.txt", "a.txt"),
        ("datasets/org/repo/@a.txt", "@a.txt"),
    ],
)
def test_pinned_paths_without_sdk_uri_parser(
    catalog, monkeypatch, hf_repositories, path, expected
):
    orig_parse = getattr(hf_file_system, "parse_hf_uri", None)
    orig_resolve = hf_file_system.HfFileSystem.resolve_path

    def resolve_without_sdk_parser(self, path, revision=None):
        monkeypatch.setattr(hf_file_system, "parse_hf_uri", orig_parse, raising=False)
        try:
            return orig_resolve(self, path, revision=revision)
        finally:
            monkeypatch.setattr(hf_file_system, "parse_hf_uri", None, raising=False)

    monkeypatch.setattr(hf_file_system, "parse_hf_uri", None, raising=False)
    monkeypatch.setattr(
        hf_file_system.HfFileSystem, "resolve_path", resolve_without_sdk_parser
    )
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    resolved = fs.resolve_path(path, revision="c1")
    assert (resolved.repo_id, resolved.revision, resolved.path_in_repo) == (
        "org/repo",
        "c1",
        expected,
    )


@pytest.mark.parametrize(
    "path,revision,expected",
    [
        ("gpt2@main/config.json", "c1", "config.json"),
        ("gpt2/config@v1.json", "c1", "config@v1.json"),
        ("gpt2@refs/pr/1/config@v1.json", "c1", "config@v1.json"),
        ("gpt2@refs%2Fpr%2F1/config.json", "c1", "config.json"),
        ("gpt2@main/config.json", "missing", None),
    ],
)
def test_resolve_short_repo_id_uses_sdk_rules(
    catalog, hf_repositories, path, revision, expected
):
    upstream = hf_file_system.HfFileSystem(token=False)
    try:
        upstream.resolve_path("gpt2@main/config.json")
    except (ValueError, FileNotFoundError):
        pytest.skip("This SDK no longer supports single-component repository IDs")
    fs = HfClient("models", {"token": False}, catalog.cache).fs.sync_fs
    if expected is None:
        with pytest.raises(FileNotFoundError):
            fs.resolve_path(path, revision=revision)
        return
    resolved = fs.resolve_path(path, revision=revision)
    assert (resolved.repo_id, resolved.revision, resolved.path_in_repo) == (
        "gpt2",
        "c1",
        expected,
    )


@pytest.mark.parametrize("revision", ["c1", "main"])
def test_cache_fill_directory_check_uses_pinned_commit(
    catalog, monkeypatch, hf_repositories, revision
):
    client = HfClient("datasets", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    checks = []

    def info(path, **kwargs):
        resolved = fs.resolve_path(path, revision=kwargs.get("revision"))
        checks.append(resolved.revision)
        return {
            "size": 4,
            "type": "file" if resolved.revision == revision else "directory",
        }

    monkeypatch.setattr(fs, "info", info)
    monkeypatch.setattr(client.fs, "_info", AsyncMock(side_effect=info))
    monkeypatch.setattr(
        hf_file_system,
        "http_get",
        lambda url, temp_file, **kw: temp_file.write(b"data"),
    )
    file = File(
        source=client.uri,
        path="org/repo@moved-branch/a@b.txt",
        size=4,
        version=revision,
    )
    client.put_in_cache(file)
    with client.open_object(file) as stream:
        assert stream.read() == b"data"
    assert checks and set(checks) == {revision}


def test_pinned_missing_namespaced_repo_does_not_fall_back_to_short_repo(
    catalog, hf_repositories
):
    fs = HfClient("models", {"token": False}, catalog.cache).fs.sync_fs
    with pytest.raises(FileNotFoundError):
        fs.resolve_path("gpt2/missing@main/file.txt", revision="c1")


@pytest.mark.parametrize("path", ["org/repo/a.txt", "datasets/org/repo/a.txt"])
def test_unpinned_paths_use_sdk_repo_type(catalog, hf_repositories, path):
    client = HfClient("org", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    resolved = fs.resolve_path(path)
    assert resolved.repo_id == "org/repo"
    assert resolved.path_in_repo == "a.txt"
    if not path.startswith("datasets/"):
        rel = client.rel_path(resolved.unresolve())
        client.validate_file_path(rel)
        assert rel == "repo/a.txt"


def test_special_ref_path_matches_sdk_parsing(catalog, hf_repositories):
    path = "datasets/org/repo@refs/convert/parquet-v2/a.parquet"
    expected = hf_file_system.HfFileSystem(token=False).resolve_path(path)
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    resolved = fs.resolve_path(path, revision="c1")
    assert resolved.path_in_repo == expected.path_in_repo
    assert resolved.revision == "c1"


def test_pinned_commit_does_not_require_uri_branch(catalog, monkeypatch):
    calls = []

    def exists(self, repo_type, repo_id, revision):
        calls.append(revision)
        return revision == "c1", None

    monkeypatch.setattr(hf_file_system.HfFileSystem, "_repo_and_revision_exist", exists)
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    resolved = fs.resolve_path("org/repo@deleted-branch/a@b.txt", revision="c1")
    assert resolved.revision == "c1"
    assert resolved.path_in_repo == "a@b.txt"
    assert calls and set(calls) == {"c1"}


def test_pinned_unchanged_check_reuses_sdk_directory_cache(
    catalog, monkeypatch, hf_repositories
):
    commit = "a" * 40
    client = HfClient("datasets", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    fs.dircache[f"datasets/org/repo@{commit}"] = [
        {
            "name": f"datasets/org/repo@{commit}/a.txt",
            "size": 4,
            "type": "file",
            "last_commit": None,
        }
    ]
    fetch = Mock(
        side_effect=AssertionError("Cached size must not fetch expanded metadata")
    )
    monkeypatch.setattr(fs._api, "get_paths_info", fetch)
    file = File(
        source=client.uri,
        path="org/repo@main/a.txt",
        version=commit,
        etag="blob",
        size=4,
    )
    client.check_unchanged(file)
    fetch.assert_not_called()


def test_mutable_branch_version_checks_etag_on_unchanged(
    catalog, monkeypatch, hf_repositories
):
    client = HfClient("datasets", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    fs.dircache["datasets/org/repo@main"] = [
        {
            "name": "datasets/org/repo@main/a.txt",
            "size": 4,
            "type": "file",
            "blob_id": "new-blob",
        }
    ]
    file = File(
        source=client.uri,
        path="org/repo/a.txt",
        version="main",
        etag="old-blob",
        size=4,
    )
    with pytest.raises(FileNotFoundError, match="etag was old-blob, now new-blob"):
        client.check_unchanged(file)

    fs.dircache["datasets/org/repo@main"] = [
        {
            "name": "datasets/org/repo@main/a.txt",
            "size": 4,
            "type": "file",
            "last_commit": None,
        }
    ]
    monkeypatch.setattr(
        fs._api,
        "get_paths_info",
        lambda *a, **kw: [RepoFile(path="a.txt", size=4, oid="new-blob")],
    )
    with pytest.raises(FileNotFoundError, match="etag was old-blob, now new-blob"):
        client.check_unchanged(file)


def test_version_only_file_without_size_does_not_fail_size_check(
    catalog, monkeypatch, hf_repositories
):
    commit = "a" * 40
    client = HfClient("datasets", {"token": False}, catalog.cache)
    monkeypatch.setattr(client.fs, "_info", AsyncMock(return_value={"size": 4}))
    for rev in (commit, "v1"):
        client.check_unchanged(
            File(source=client.uri, path="org/repo/a.txt", version=rev)
        )
    with pytest.raises(FileNotFoundError, match="size was 0, now 4"):
        client.check_unchanged(
            File(
                source=client.uri,
                path="org/repo/a.txt",
                version=commit,
                etag="empty-blob",
                size=0,
            )
        )


def test_bucket_metadata_and_resolve_path_with_revision(
    catalog, monkeypatch, hf_repositories
):
    cache = Mock(spec=Cache)
    client = HfClient("buckets", {}, cache)
    date = datetime(2025, 1, 2, tzinfo=UTC)
    file = client.info_to_file(
        {"size": 12, "xet_hash": "xet-123", "uploaded_at": date},
        "org/bucket/file.bin",
    )
    assert (file.path, file.size, file.etag, file.version, file.last_modified) == (
        "org/bucket/file.bin",
        12,
        "xet-123",
        "",
        date,
    )

    fs = HfClient("buckets", {"token": False}, catalog.cache).fs.sync_fs
    monkeypatch.setattr(
        hf_file_system.HfFileSystem,
        "_bucket_exists",
        lambda self, bid: (True, None),
        raising=False,
    )
    resolved = fs.resolve_path("buckets/org/bucket/file.bin", revision="c1")
    assert getattr(resolved, "bucket_id", getattr(resolved, "repo_id", None)) in {
        "org/bucket",
        "buckets/org",
    }


def test_anon_translates_to_token_false_and_credentials_detected():
    assert HfClient.has_explicit_credentials({"token": "hf_secret"})
    assert not HfClient.has_explicit_credentials({"anon": True})
    fs = HfClient.create_fs(anon=True)
    assert fs.sync_fs.token is False
    fs_override = HfClient.create_fs(anon=True, token="hf_secret")  # noqa: S106
    assert fs_override.sync_fs.token is False


def test_stream_file_close_releases_exit_stack(catalog, monkeypatch, hf_repositories):
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    closed = Mock()

    @contextmanager
    def fake_stream(*args, **kwargs):
        try:
            yield SimpleNamespace(
                status_code=200,
                is_error=False,
                headers={"Content-Length": "4"},
                iter_bytes=lambda chunk_size=None: iter([b"data"]),
                iter_content=lambda chunk_size=None: iter([b"data"]),
                raise_for_status=lambda: None,
                close=Mock(),
            )
        finally:
            closed()

    monkeypatch.setattr(hf_file_system, "http_stream_backoff", fake_stream)
    monkeypatch.setattr(
        hf_file_system,
        "_get_file_length_from_http_response",
        lambda response: 4,
        raising=False,
    )
    with fs.open("datasets/org/repo@main/a.txt", "rb", block_size=0) as stream:
        assert stream.read() == b"data"
    closed.assert_called_once()


def test_auth_http_error_is_also_permission_error(catalog, monkeypatch):
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    req = httpx.Request("GET", "https://huggingface.co/api/datasets/org/repo")
    resp_401 = httpx.Response(401, request=req)
    resp_403 = httpx.Response(403, request=req)
    resp_500 = httpx.Response(500, request=req)

    monkeypatch.setattr(
        fs._api,
        "repo_info",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            GatedRepoError("403 Forbidden", response=resp_403)
        ),
    )
    with pytest.raises(PermissionError) as gated_exc:
        fs.resolve_path("datasets/org/gated/a.txt")
    restored_gated = pickle.loads(pickle.dumps(gated_exc.value))  # noqa: S301
    assert isinstance(restored_gated, FileNotFoundError)
    assert isinstance(restored_gated, PermissionError)

    monkeypatch.setattr(
        hf_file_system.HfFileSystem,
        "resolve_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            HfHubHTTPError("401 Unauthorized", response=resp_401)
        ),
    )
    with pytest.raises(PermissionError) as exc_info:
        fs.resolve_path("datasets/org/repo/a.txt")
    assert isinstance(exc_info.value, HfHubHTTPError)
    restored_401 = pickle.loads(pickle.dumps(exc_info.value))  # noqa: S301
    assert isinstance(restored_401, HfHubHTTPError)
    assert isinstance(restored_401, PermissionError)

    monkeypatch.setattr(
        hf_file_system.HfFileSystem,
        "resolve_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            HfHubHTTPError("500 Internal Server Error", response=resp_500)
        ),
    )
    with pytest.raises(HfHubHTTPError) as exc_500:
        fs.resolve_path("datasets/org/repo/a.txt")
    assert not isinstance(exc_500.value, PermissionError)


def test_hf_filesystem_is_pickleable(catalog):
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    restored = pickle.loads(pickle.dumps(fs))  # noqa: S301
    assert type(restored) is type(fs)
    assert restored.storage_options == fs.storage_options
