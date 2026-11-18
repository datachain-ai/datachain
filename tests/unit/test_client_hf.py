import pickle
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fsspec.asyn import sync
from huggingface_hub import hf_file_system
from huggingface_hub.hf_file_system import HfFileSystemFile
from huggingface_hub.utils import HFValidationError

from datachain.asyn import get_loop
from datachain.cache import Cache
from datachain.client.hf import HfClient
from datachain.client.hf_file_system import PinnedHfFileSystemFile
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
    requests = tar_member_range_requests(PinnedHfFileSystemFile)

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
    requests = tar_member_range_requests(PinnedHfFileSystemFile)
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

    def read(stream, size):
        if stream.loc:
            return b""
        urls.append(stream.url())
        stream.loc += 4
        return b"data"

    monkeypatch.setattr(hf_file_system.HfFileSystemStreamFile, "read", read)
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
    tar_member_range_requests(PinnedHfFileSystemFile)
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

    def read(stream, size):
        if stream.loc:
            return b""
        stream.loc += 4
        return b"data"

    monkeypatch.setattr(hf_file_system.HfFileSystemStreamFile, "read", read)
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
    client = HfClient("datasets", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    fs.dircache["datasets/org/repo@c1"] = [
        {
            "name": "datasets/org/repo@c1/a.txt",
            "size": 4,
            "type": "file",
            "blob_id": "blob",
            "last_commit": None,
        }
    ]
    fetch = Mock(
        side_effect=AssertionError("Cached size must not fetch expanded metadata")
    )
    monkeypatch.setattr(fs._api, "get_paths_info", fetch)
    file = File(
        source=client.uri, path="org/repo@main/a.txt", version="c1", etag="blob", size=4
    )
    client.check_unchanged(file)
    fetch.assert_not_called()


def test_mutable_branch_version_checks_etag_on_unchanged():
    client = HfClient("datasets", {}, Mock(spec=Cache))
    client._fs = Mock()
    client._fs._info = AsyncMock(return_value={"size": 4, "blob_id": "new-blob"})
    file = File(
        source=client.uri,
        path="org/repo/a.txt",
        version="main",
        etag="old-blob",
        size=4,
    )
    with pytest.raises(FileNotFoundError, match="etag was old-blob, now new-blob"):
        client.check_unchanged(file)
    client._fs.sync_fs.resolve_path.assert_not_called()

    client._fs._info = AsyncMock(
        side_effect=[{"size": 4}, {"size": 4, "blob_id": "new-blob"}]
    )
    with pytest.raises(FileNotFoundError, match="etag was old-blob, now new-blob"):
        client.check_unchanged(file)


def test_bucket_metadata_and_resolve_path_with_revision(catalog, monkeypatch):
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
        hf_file_system.HfFileSystem, "_bucket_exists", lambda self, bid: (True, None)
    )
    resolved = fs.resolve_path("buckets/org/bucket/file.bin", revision="main")
    assert getattr(resolved, "bucket_id", None) == "org/bucket"

    import asyncio

    from fsspec.callbacks import DEFAULT_CALLBACK

    monkeypatch.setattr(
        hf_file_system.HfFileSystem, "get_file", lambda self, rpath, *a, **kw: rpath
    )
    assert (
        fs.get_file("buckets/org/bucket/file.bin", "dst")
        == "buckets/org/bucket/file.bin"
    )
    assert (
        fs.get_file("buckets/org/bucket/file.bin", "dst", revision="main")
        == "buckets/org/bucket/file.bin"
    )
    bclient = HfClient("buckets", {"token": False}, catalog.cache)
    bclient._fs = Mock()
    bclient._fs.sync_fs = fs
    bclient._fs._get_file = AsyncMock(return_value=None)
    asyncio.run(
        bclient.get_file("buckets/org/bucket/file.bin", "dst", DEFAULT_CALLBACK)
    )

    commit = "a" * 40
    rclient = HfClient("datasets", {"token": False}, catalog.cache)
    monkeypatch.setattr(
        rclient.fs.sync_fs,
        "resolve_path",
        lambda path, revision=None: SimpleNamespace(revision=commit),
    )
    rclient.check_unchanged(
        File(
            source=rclient.uri,
            path=f"org/repo@{commit}/a.txt",
            version="",
            etag="blob",
            size=4,
        )
    )


def test_anon_translates_to_token_false_and_credentials_detected():
    assert HfClient.has_explicit_credentials({"token": "hf_secret"})
    assert not HfClient.has_explicit_credentials({"anon": True})
    fs = HfClient.create_fs(anon=True)
    assert fs.sync_fs.token is False


def test_stream_file_close_releases_exit_stack(catalog, monkeypatch, hf_repositories):
    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    stream = fs.open("datasets/org/repo@main/a.txt", "rb", block_size=0)
    monkeypatch.setattr(
        hf_file_system.HfFileSystemStreamFile,
        "_open_connection",
        lambda self: None,
    )
    stream._open_connection()
    closed = Mock()
    stream._exit_stack.callback(closed)
    stream.close()
    closed.assert_called_once()


def test_auth_http_error_is_also_permission_error(catalog, monkeypatch):
    import httpx
    from huggingface_hub.utils import HfHubHTTPError

    fs = HfClient("datasets", {"token": False}, catalog.cache).fs.sync_fs
    resp_401 = httpx.Response(
        401,
        request=httpx.Request("GET", "https://huggingface.co/api/datasets/org/repo"),
    )
    resp_500 = httpx.Response(
        500,
        request=httpx.Request("GET", "https://huggingface.co/api/datasets/org/repo"),
    )

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


@pytest.mark.parametrize(
    "path", ["org/repo@main/a.txt", "org/repo@v1/a.txt", "org/repo/a.txt"]
)
def test_download_url_uses_explicit_commit_without_folder_metadata(
    catalog, monkeypatch, hf_repositories, path
):
    client = HfClient("datasets", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    monkeypatch.setattr(
        fs, "info", Mock(side_effect=AssertionError("Unexpected metadata lookup"))
    )
    assert (
        fs.url("datasets/" + path, revision="c1")
        == "https://huggingface.co/datasets/org/repo/resolve/c1/a.txt"
    )


def test_immutable_listed_commit_needs_no_metadata_lookup():
    client = HfClient("datasets", {}, Mock(spec=Cache))
    client._fs = Mock()
    file = File(source=client.uri, path="org/repo@main/a.txt", version="a" * 40, size=4)
    client.check_unchanged(file)
    client._fs._info.assert_not_called()
    client._fs.sync_fs.resolve_path.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellations", [1, 2])
async def test_cancelled_download_waits_for_stream_cleanup(
    monkeypatch, tmp_path, cancellations
):
    import asyncio
    import threading

    client = HfClient("datasets", {}, Mock(spec=Cache))
    client._fs = Mock()
    client._fs.sync_fs.resolve_path.return_value = SimpleNamespace(
        repo_id="org/repo",
        unresolve=lambda: "datasets/org/repo@c1/a.txt",
    )
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    stopped_on_cancel = False

    def stream(path, dst, callback, stop=None):
        nonlocal stopped_on_cancel
        started.set()
        assert release.wait(2)
        stopped_on_cancel = bool(stop and stop.is_set())
        finished.set()

    monkeypatch.setattr(client, "_stream", stream)
    task = asyncio.create_task(
        client.get_file(
            "hf://datasets/org/repo/a.txt", str(tmp_path / "out"), Mock(), "c1"
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 1)
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set()
    assert stopped_on_cancel


def test_stream_stops_on_cancel_and_closes_exit_stack(tmp_path):
    import threading

    client = HfClient("datasets", {}, Mock(spec=Cache))
    client._fs = Mock()
    stop = threading.Event()
    exit_stack = Mock()
    reads = 0

    def read_chunk(size):
        nonlocal reads
        reads += 1
        stop.set()
        return b"chunk"

    src = MagicMock()
    src.__enter__.return_value = src
    src.read.side_effect = read_chunk
    src._exit_stack = exit_stack
    client._fs.sync_fs.open.return_value = src

    dst = tmp_path / "out"
    client._stream("datasets/org/repo@c1/a.txt", str(dst), Mock(), stop)
    assert reads == 1
    assert dst.read_bytes() == b"chunk"
    exit_stack.close.assert_called_once()


def test_bucket_url_keeps_sdk_signature(catalog, monkeypatch):
    client = HfClient("buckets", {"token": False}, catalog.cache)
    fs = client.fs.sync_fs
    monkeypatch.setattr(
        fs, "resolve_path", lambda *a, **kw: SimpleNamespace(bucket_id="org/bucket")
    )
    monkeypatch.setattr(
        hf_file_system.HfFileSystem, "url", lambda self, path: "bucket-url"
    )
    assert fs.url("buckets/org/bucket/file") == "bucket-url"


RETRY_CODES = (408, 429, 500, 502, 503, 504)
COMMIT = "a" * 40
RESOLVE_URL = f"https://huggingface.co/datasets/org/repo/resolve/{COMMIT}/a.parquet"


@pytest.fixture
def hf_range(catalog, monkeypatch, hf_repositories):
    client = HfClient("datasets", {"token": "hf_secret"}, catalog.cache)
    monkeypatch.setattr(client.fs.sync_fs, "info", lambda *a, **kw: {"size": 16})
    file = File(
        source=client.uri, path="org/repo@main/a.parquet", size=16, version=COMMIT
    )
    responses, calls = [], []

    def backoff(method, url, *, headers, retry_on_status_codes=None, **kw):
        calls.append((method, url, dict(headers), retry_on_status_codes))
        resp = responses.pop(0)
        final_url, status, req_headers = (
            (resp, 206, {}) if isinstance(resp, str) else (*resp, {})[:3]
        )
        start, end = map(int, headers["range"][6:].split("-"))
        return SimpleNamespace(
            url=final_url,
            status_code=status,
            content=b"0123456789abcdef"[start : end + 1],
            request=SimpleNamespace(headers=req_headers),
            raise_for_status=Mock(
                side_effect=RuntimeError(f"HTTP {status}") if status >= 400 else None
            ),
        )

    monkeypatch.setattr(hf_file_system, "http_backoff", backoff)
    return client, file, responses, calls


@pytest.mark.parametrize(
    "first_resp,next_url,next_auth",
    [
        ("https://cdn.hf.co/signed", "https://cdn.hf.co/signed", None),
        (
            f"https://huggingface.co/datasets/renamed/repo/resolve/{COMMIT}/a.parquet",
            RESOLVE_URL,
            "Bearer hf_secret",
        ),
        (RESOLVE_URL, RESOLVE_URL, "Bearer hf_secret"),
        (
            ("https://cdn.hf.co/signed", 206, {"Authorization": "Bearer hf_secret"}),
            RESOLVE_URL,
            "Bearer hf_secret",
        ),
        (
            "https://cdn.hf.co/signed?X-Xet-Signed-Range=bytes%3D0-3",
            RESOLVE_URL,
            "Bearer hf_secret",
        ),
    ],
)
def test_range_redirect_caching_policy(hf_range, first_resp, next_url, next_auth):
    client, file, responses, calls = hf_range
    responses.extend([first_resp, "https://cdn.hf.co/signed"])
    with client.open_object(file, use_cache=False, block_size=0) as stream:
        assert (stream.read(4), stream.read(4)) == (b"0123", b"4567")

    assert [(m, u, h["range"], h.get("authorization"), r) for m, u, h, r in calls] == [
        ("GET", RESOLVE_URL, "bytes=0-3", "Bearer hf_secret", RETRY_CODES),
        ("GET", next_url, "bytes=4-7", next_auth, RETRY_CODES),
    ]


def test_range_redirect_per_open_file_and_case_insensitive_auth(hf_range, monkeypatch):
    client, file, responses, calls = hf_range
    monkeypatch.setattr(
        client.fs.sync_fs._api,
        "_build_hf_headers",
        lambda: {"Authorization": "Bearer hf_secret"},
    )
    responses.extend(["https://cdn.hf.co/1"] * 2 + ["https://cdn.hf.co/2"] * 2)
    for _ in range(2):
        with client.open_object(file, use_cache=False, block_size=0) as stream:
            assert (stream.read(4), stream.read(4)) == (b"0123", b"4567")

    assert [(m, u, h["range"], h.get("Authorization"), r) for m, u, h, r in calls] == [
        ("GET", RESOLVE_URL, "bytes=0-3", "Bearer hf_secret", RETRY_CODES),
        ("GET", "https://cdn.hf.co/1", "bytes=4-7", None, RETRY_CODES),
        ("GET", RESOLVE_URL, "bytes=0-3", "Bearer hf_secret", RETRY_CODES),
        ("GET", "https://cdn.hf.co/2", "bytes=4-7", None, RETRY_CODES),
    ]


@pytest.mark.parametrize("expired_status", [400, 401, 403, 404, 410])
def test_range_read_expired_cached_url_refreshes(hf_range, expired_status):
    client, file, responses, calls = hf_range
    responses.extend(
        [
            "https://cdn.hf.co/expired",
            ("https://cdn.hf.co/expired", expired_status),
            "https://cdn.hf.co/fresh",
            "https://cdn.hf.co/fresh",
            ("https://cdn.hf.co/fresh", expired_status),
            ("https://cdn.hf.co/dead", expired_status),
        ]
    )
    with client.open_object(file, use_cache=False, block_size=0) as stream:
        assert [stream.read(4) for _ in range(3)] == [b"0123", b"4567", b"89ab"]
        with pytest.raises(RuntimeError, match=f"HTTP {expired_status}"):
            stream.read(4)

    assert [(u, h["range"], h.get("authorization")) for _, u, h, _ in calls] == [
        (RESOLVE_URL, "bytes=0-3", "Bearer hf_secret"),
        ("https://cdn.hf.co/expired", "bytes=4-7", None),
        (RESOLVE_URL, "bytes=4-7", "Bearer hf_secret"),
        ("https://cdn.hf.co/fresh", "bytes=8-11", None),
        ("https://cdn.hf.co/fresh", "bytes=12-15", None),
        (RESOLVE_URL, "bytes=12-15", "Bearer hf_secret"),
    ]


def test_streaming_writes_and_buckets_bypass_pinned_range_file(
    catalog, monkeypatch, hf_repositories
):
    fs = HfClient("datasets", {"token": "hf_secret"}, catalog.cache).fs.sync_fs
    monkeypatch.setattr(fs._api, "upload_file", Mock())
    with fs.open("datasets/org/repo/a.txt", "rb", block_size=0) as stream:
        assert not isinstance(stream, PinnedHfFileSystemFile)
    with fs.open("datasets/org/repo/a.txt", "wb") as stream:
        assert type(stream) is HfFileSystemFile

    parent_fetch = Mock(return_value=b"0123")
    monkeypatch.setattr(HfFileSystemFile, "_fetch_range", parent_fetch)
    monkeypatch.setattr(
        fs,
        "resolve_path",
        lambda *a, **kw: SimpleNamespace(
            bucket_id="org/bucket", unresolve=lambda: "buckets/org/bucket/file"
        ),
    )
    monkeypatch.setattr(fs, "info", lambda *a, **kw: {"size": 4})
    with fs.open("buckets/org/bucket/file", "rb", cache_type="none") as stream:
        assert stream.read(4) == b"0123"
    parent_fetch.assert_called_once_with(0, 4)
