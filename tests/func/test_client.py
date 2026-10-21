import asyncio
import os
import sys
from contextlib import aclosing
from datetime import UTC
from pathlib import Path
from uuid import uuid4

import pytest
from fsspec.asyn import sync
from fsspec.callbacks import DEFAULT_CALLBACK
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tqdm import tqdm

from datachain.asyn import get_loop
from datachain.client import Client
from datachain.client.fsspec import RangeListedClient
from datachain.client.ranges import RangeListing
from datachain.lib.file import File, FileError, FileRow
from tests.data import ENTRIES

_non_null_text = st.text(
    alphabet=st.characters(blacklist_categories=["Cc", "Cs"]), min_size=1
)


@pytest.fixture
def client(cloud_server, cloud_server_credentials, mocker):
    mocker.patch("datachain.client.listing_worker.LIST_PROCESSES", 0)
    uri = cloud_server.src_uri
    return Client.get_implementation(uri).from_source(
        uri, cache=None, **cloud_server.client_config
    )


def normalize_entries(entries):
    return {e.path for e in entries}


def match_entries(result, expected):
    assert len(result) == len(expected)
    assert normalize_entries(result) == normalize_entries(expected)


async def find(client, prefix, method="default"):
    results = []
    async for entries in client.scandir(prefix, method=method):
        results.extend(entries)
    return results


def scandir(client, prefix, method="default"):
    return sync(get_loop(), find, client, prefix, method)


def test_scandir_error(client):
    with pytest.raises(FileNotFoundError):
        scandir(client, "bogus")


@pytest.mark.parametrize("tree", [{}], indirect=True)
def test_scandir_empty_bucket(client):
    results = scandir(client, "")
    match_entries(results, [])


@pytest.mark.xfail
def test_scandir_not_dir(client):
    with pytest.raises(FileNotFoundError):
        scandir(client, "description")


def test_scandir_success(client):
    results = scandir(client, "")
    match_entries(results, ENTRIES)
    row = FileRow if isinstance(client, RangeListedClient) else File
    assert {type(e) for e in results} == {row}


def _paths(tree, prefix=""):
    for name, value in tree.items():
        if isinstance(value, dict):
            yield from _paths(value, f"{prefix}{name}/")
        else:
            yield prefix + name


RANGE_TREES = {
    "skewed": {"a": "1", "big": {f"{i:03}": "x" for i in range(80)}, "z": "1"},
    "flat": {f"{i * 7919 % 4096:03x}": "x" for i in range(80)},
    "siblings": {"a": {"b": "x"}, "a-b": "x", "a.c": "x", "a0": "x", "ab": {"c": "x"}},
}


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize(
    "tree", list(RANGE_TREES.values()), ids=list(RANGE_TREES), indirect=True
)
@pytest.mark.parametrize("prefix", ["", "big"])
def test_scandir_ranges_are_exact(client, tree, prefix, mocker):
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 3)
    ranges = mocker.spy(client, "_list_page")
    expected = [p for p in _paths(tree) if p.startswith(prefix)]
    if not expected:
        with pytest.raises(FileNotFoundError):
            scandir(client, prefix)
        return

    paths = [e.path for e in scandir(client, prefix)]

    assert sorted(paths) == sorted(expected)
    assert ranges.call_count > 1


@pytest.mark.parametrize("cloud_type", ["s3", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [{f"{i:03}": "x" for i in range(200)}], indirect=True)
@pytest.mark.parametrize("workers", [2, 16])
@pytest.mark.parametrize("young_pages", [4, 10**6])
def test_scandir_young_and_old_ranges_are_exact(
    client, tree, workers, young_pages, mocker
):
    # a range split off another inherits its age, so old ranges split twice
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch("datachain.client.ranges.LIST_WORKERS", workers)
    mocker.patch("datachain.client.ranges.YOUNG_PAGES", young_pages)
    split = mocker.spy(RangeListing, "_split")

    assert sorted(e.path for e in scandir(client, "")) == sorted(tree)
    assert split.call_count


@pytest.mark.parametrize("cloud_type", ["s3", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [{f"{i:03}": "x" for i in range(200)}], indirect=True)
def test_scandir_splits_past_the_data_are_few(client, tree, mocker):
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 2)
    calls = mocker.spy(client, "_list_page")

    assert sorted(e.path for e in scandir(client, "")) == sorted(tree)
    ranges = [c.args[1] for c in calls.call_args_list if c.args[2] is None]
    assert sum(lo > "199" for lo in ranges) <= 4


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [{f"{i:03}": "x" for i in range(200)}], indirect=True)
@pytest.mark.parametrize("processes", [False, True])
def test_scandir_is_exact_when_the_reader_is_slow(client, tree, processes, mocker):
    mocker.patch("datachain.client.listing_worker.LIST_PROCESSES", 2 * processes)
    # pages wait for room in the result queue while other ranges split
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)

    async def slow_reader():
        paths = []
        async for entries in client.scandir(""):
            paths.extend(e.path for e in entries)
            await asyncio.sleep(0.02)
        return paths

    assert sorted(sync(get_loop(), slow_reader)) == sorted(tree)


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
@pytest.mark.parametrize("tree", [{"a": "x", "z": "x"}], indirect=True)
def test_a_page_reads_to_a_delete_marker(client, tree):
    client.fs.rm_file(f"{client.uri}/z")

    assert sync(get_loop(), client._list_page, "", "m", None).last == "z"
    assert sync(get_loop(), client._list_page, "", "z", None).last is None


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
# more pages than the result queue holds: an open listing would still be waiting
@pytest.mark.parametrize("tree", [{f"{i:03}": "x" for i in range(200)}], indirect=True)
def test_scandir_stops_listing_when_closed_early(client, tree, mocker):
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)

    async def first_page():
        async with aclosing(client.scandir("")) as entries:
            async for page in entries:
                return page

    async def pending_listing_tasks():
        tasks = [
            t
            for t in asyncio.all_tasks()
            if t.get_coro().__qualname__.startswith(("RangeListing.", "Client.scandir"))
        ]
        if tasks:
            await asyncio.wait(tasks, timeout=5)
        return [t for t in tasks if not t.done()]

    assert sync(get_loop(), first_page)
    assert not sync(get_loop(), pending_listing_tasks)


UNLISTED_TREE = {"a": {f"{i:02}": "x" for i in range(40)}, "z": {"0": "x"}}


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [UNLISTED_TREE], indirect=True)
def test_scandir_ranges_see_past_unlisted_keys(client, tree, mocker):
    for i in range(60):  # folder markers: listed, never rows
        client.fs.pipe_file(f"{client.uri}/m/{i:02}/", b"")
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 3)
    calls = mocker.spy(client, "_list_page")

    expected = sorted(_paths(tree))
    assert sorted(e.path for e in scandir(client, "")) == expected
    assert calls.call_count <= 2 * (len(expected) + 60) // 4


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [RANGE_TREES["flat"]], indirect=True)
def test_scandir_builds_no_files_past_range_ends(client, tree, mocker):
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 3)
    bounded, past_hi = [], []
    list_page = type(client)._list_page

    async def recorded(self, prefix, start_after, token, hi=None, **kwargs):
        page = await list_page(self, prefix, start_after, token, hi, **kwargs)
        if hi is not None:
            bounded.append(hi)
            order = self.key_order
            past_hi.extend(r.path for r in page.rows if order(r.path) > order(hi))
        return page

    mocker.patch.object(type(client), "_list_page", recorded)

    assert len(scandir(client, "")) == len(tree)
    assert bounded
    assert not past_hi


@pytest.mark.parametrize("cloud_type", ["gs"], indirect=True)
@pytest.mark.parametrize("tree", [RANGE_TREES["flat"]], indirect=True)
def test_gcs_listing_ends_at_hi(client, tree, mocker):
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    keys = sorted(tree)
    lo, hi = keys[10], keys[20]
    read = []
    call = client.fs._call

    async def counted(*args, **kwargs):
        page = await call(*args, **kwargs)
        read.extend(item["name"] for item in page.get("items", []))
        return page

    mocker.patch.object(client.fs, "_call", counted)

    async def listed():
        page = await client._list_page("", lo, None, hi)
        while page.token is not None:
            page = await client._list_page("", lo, page.token, hi)

    sync(get_loop(), listed)
    assert read == keys[10:21]  # startOffset includes the start key


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [RANGE_TREES["flat"]], indirect=True)
def test_scandir_out_of_key_order_is_one_range(client, tree, mocker):
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch.object(type(client), "_lists_in_key_order", return_value=False)
    ranges = mocker.spy(client, "_list_page")

    paths = [e.path for e in scandir(client, "")]

    assert sorted(paths) == sorted(_paths(tree))
    assert [c.args[2] for c in ranges.call_args_list].count(None) == 1


PREFIXED_TREE = {
    "n1": {f"{uuid}.jpg": "x" for uuid in ["0a", "3f", "5c", "9d", "b2", "e7"]},
    **{f"n{i}": {f"{i:02}{j}.jpg": "x" for j in range(9)} for i in (2, 3)},
}


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize(
    "tree",
    [
        PREFIXED_TREE,
        {"n2": {f"{i:03}.jpg": "x" for i in range(60)}, "z": "x"},
        {"n2": {f"{c}.jpg": "x" for c in "abcdefghi"}, "n3": {"a.jpg": "x"}},
    ],
    indirect=True,
)
@pytest.mark.parametrize("processes", [False, True])
def test_scandir_requests_stay_inside_the_prefix(client, tree, processes, mocker):
    mocker.patch("datachain.client.listing_worker.LIST_PROCESSES", 2 * processes)
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    calls = mocker.spy(client, "_list_page")

    async def listed():
        paths = []
        async for entries in client.scandir("n2"):
            paths.extend(e.path for e in entries)
        return paths

    expected = [p for p in _paths(tree) if p.startswith("n2/")]
    assert sorted(sync(get_loop(), listed)) == sorted(expected)
    starts = [c.args[1] for c in calls.call_args_list]
    assert len(starts) > 1
    assert all(s.startswith("n2/") for s in starts if s)


# fake-gcs-server repeats a key's versions after a page token
@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
@pytest.mark.parametrize("tree", [{"d": {"b.txt": "0"}, "t": "x"}], indirect=True)
def test_scandir_versions_of_one_key_stay_inside_the_prefix(client, tree, mocker):
    for i in range(1, 10):
        client.fs.pipe_file(f"{client.uri}/d/b.txt", str(i).encode())
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    calls = mocker.spy(client, "_list_page")

    assert len(scandir(client, "d")) == 10
    assert all(c.args[1].startswith("d/") for c in calls.call_args_list if c.args[1])


EMOJI_TREE = {
    "img": {
        **{f"\U0001f600_{i:02}.png": "x" for i in range(25)},
        **{f"\uff08{i:02}\uff09.png": "x" for i in range(25)},
    }
}


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [EMOJI_TREE], indirect=True)
@pytest.mark.parametrize("processes", [False, True])
def test_scandir_lists_keys_past_the_basic_multilingual_plane(
    client, tree, processes, mocker
):
    mocker.patch("datachain.client.listing_worker.LIST_PROCESSES", 2 * processes)
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)

    async def listed():
        paths = []
        async for entries in client.scandir(""):
            paths.extend(e.path for e in entries)
        return paths

    assert sorted(sync(get_loop(), listed)) == sorted(_paths(tree))


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
@pytest.mark.parametrize(
    "tree", [{"dm": {"a": "x", "b": "x"}, "z": "x"}], indirect=True
)
def test_prefix_of_only_delete_markers_is_not_found(client, tree):
    client.fs.rm([f"{client.uri}/dm/a", f"{client.uri}/dm/b"])
    listed = client.fs.call_s3("list_object_versions", Bucket=client.name, Prefix="dm/")
    for version in listed["Versions"]:
        client.fs.call_s3(
            "delete_object",
            Bucket=client.name,
            Key=version["Key"],
            VersionId=version["VersionId"],
        )

    with pytest.raises(FileNotFoundError):
        scandir(client, "dm")


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
def test_scandir_one_page_is_one_range(client, mocker):
    ranges = mocker.spy(client, "_list_page")
    assert scandir(client, "")
    assert ranges.call_count == 1


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
def test_listing_parses_timestamps_without_dateutil(client):
    assert {e.last_modified.tzinfo for e in scandir(client, "")} == {UTC}


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [{"a": "x"}], indirect=True)
def test_scandir_lists_double_slash_keys_but_not_leading_slash_or_folder_markers(
    client,
):
    keys = ["d//e.txt", "/lead.txt", "folder/"]
    for key in keys:
        body = b"" if key.endswith("/") else b"x"
        client.fs.pipe_file(f"{client.name}/{key}", body)
    try:
        paths = sorted(e.path for e in scandir(client, ""))
        assert paths == ["a", "d//e.txt"]
    finally:
        for key in keys:
            client.fs.rm_file(f"{client.name}/{key}")
        client.fs.invalidate_cache()


def test_scandir_alternate(client):
    results = scandir(client, "", method="nested")
    match_entries(results, ENTRIES)


@pytest.mark.parametrize("cloud_type", ["gs"], indirect=True)
def test_gcs_object_close_is_idempotent_on_event_loop(client):
    file = client.get_file_info("dogs/dog1")
    with client.open_object(file, use_cache=False) as stream:
        assert stream.read() == b"woof"

    async def close():
        stream.close()

    sync(get_loop(), close)
    assert stream.closed


def test_gcs_client_gets_credentials_from_env(monkeypatch, mocker):
    from datachain.client.gcs import GCSClient

    monkeypatch.setenv(
        "DATACHAIN_GCP_CREDENTIALS", '{"token": "test-credentials-token"}'
    )
    init = mocker.patch(
        "datachain.client.gcs.GCSFileSystem.__init__", return_value=None
    )
    mocker.patch(
        "datachain.client.gcs.GCSFileSystem.invalidate_cache", return_value=None
    )

    GCSClient.create_fs()

    init.assert_called_once_with(
        token={"token": "test-credentials-token"}, version_aware=True
    )


@pytest.mark.parametrize("tree", [{}], indirect=True)
def test_fetch_dir_does_not_return_self(client, cloud_type):
    if cloud_type == "file":
        pytest.skip()

    client.fs.touch(f"{client.uri}/directory//file")

    subdirs = sync(
        get_loop(), client._fetch_dir, "directory/", tqdm(disable=True), asyncio.Queue()
    )

    assert "directory" not in subdirs


@pytest.mark.parametrize(
    "rel_suffix, expected_uri_suffix, expected_rel",
    [
        # ---- single segment: URI stays at base ----
        ("animals", "", "animals"),
        ("file.txt", "", "file.txt"),
        # ---- multiple segments: URI absorbs all but last ----
        ("a/b", "/a", "b"),
        ("deep/nested/path/file.txt", "/deep/nested/path", "file.txt"),
        # ---- trailing slash → directory semantics: rel is empty ----
        ("animals/", "/animals", ""),
        ("a/b/c/", "/a/b/c", ""),
        # ---- special characters preserved as-is ----
        ("v1.0-release", "", "v1.0-release"),
        ("path with spaces", "", "path with spaces"),
        ("100%done", "", "100%done"),
        ("café", "", "café"),
        # ---- backslash: literal filename char on Unix, separator on Windows ----
        pytest.param(
            "dir\\file",
            "/dir" if sys.platform == "win32" else "",
            "file" if sys.platform == "win32" else "dir\\file",
            id="backslash",
        ),
    ],
)
def test_parse_url_file(tmp_path, rel_suffix, expected_uri_suffix, expected_rel):
    base_uri = tmp_path.as_uri()
    url = f"{base_uri}/{rel_suffix}"

    uri, rel_part = Client.parse_url(url)

    assert uri == f"{base_uri}{expected_uri_suffix}"
    assert rel_part == expected_rel


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
def test_parse_url_cloud(cloud_test_catalog):
    base_uri = cloud_test_catalog.src_uri
    cases = [
        # (rel_suffix, expected_rel)
        ("animals", "animals"),
        ("path/to/file.txt", "path/to/file.txt"),
        ("animals/", "animals/"),
        ("", ""),
    ]
    for rel_suffix, expected_rel in cases:
        url = f"{base_uri}/{rel_suffix}"
        uri, rel_part = Client.parse_url(url)
        assert uri == base_uri, f"uri mismatch for {rel_suffix!r}"
        assert rel_part == expected_rel, f"rel mismatch for {rel_suffix!r}"


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rel_path=_non_null_text)
def test_get_client(cloud_test_catalog, rel_path, cloud_type):
    catalog = cloud_test_catalog.catalog
    bucket_uri = cloud_test_catalog.src_uri
    url = f"{bucket_uri}/{rel_path}"
    client = Client.get_client(url, catalog.cache)
    assert client
    assert client.uri


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_absolute_path_without_protocol(cloud_test_catalog):
    working_dir = Path().absolute()
    uri, rel_part = Client.parse_url(str(working_dir / Path("animals")))
    assert uri == working_dir.as_uri()
    assert rel_part == "animals"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX root-path behavior")
@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
@pytest.mark.parametrize(
    "url,expected_uri,expected_rel",
    [
        ("/tmp", "file:////", "tmp"),  # noqa: S108
        ("/tmp.txt", "file:////", "tmp.txt"),  # noqa: S108
        ("file:///tmp", "file:////", "tmp"),
        ("file:///tmp.txt", "file:////", "tmp.txt"),
        ("/tmp/", "file:///tmp", ""),  # noqa: S108
        ("file:///tmp/", "file:///tmp", ""),
    ],
    ids=[
        "plain-dir",
        "plain-file",
        "uri-dir",
        "uri-file",
        "plain-trailing-slash",
        "uri-trailing-slash",
    ],
)
def test_parse_file_absolute_posix_root_paths(
    cloud_test_catalog, url, expected_uri, expected_rel
):
    uri, rel_part = Client.parse_url(url)

    assert uri == expected_uri
    assert rel_part == expected_rel


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_relative_path_multiple_dirs_back(cloud_test_catalog):
    uri, rel_part = Client.parse_url("../../animals".replace("/", os.sep))
    assert uri == Path().absolute().parents[1].as_uri()
    assert rel_part == "animals"


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
@pytest.mark.parametrize("url", ["./animals".replace("/", os.sep), "animals"])
def test_parse_file_relative_path_working_dir(cloud_test_catalog, url):
    uri, rel_part = Client.parse_url(url)
    assert uri == Path().absolute().as_uri()
    assert rel_part == "animals"


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_relative_path_home_dir(cloud_test_catalog):
    if sys.platform == "win32":
        # home dir shortcut is not available on windows
        pytest.skip()
    uri, rel_part = Client.parse_url("~/animals")
    assert uri == Path().home().as_uri()
    assert rel_part == "animals"


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_parse_cloud_path_ends_with_slash(cloud_test_catalog):
    uri = f"{cloud_test_catalog.src_uri}/animals/"
    uri, rel_part = Client.parse_url(uri)
    assert uri == cloud_test_catalog.src_uri
    assert rel_part == "animals/"


@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_get_size_roundtrip_file_and_cloud(
    cloud_test_catalog_upload,
    cloud_type,
    version_aware,
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)

    rel_path = f"client-get-size/{uuid4().hex}.bin"

    data_v1 = b"v1"
    client.upload(data_v1, rel_path)
    assert sync(
        get_loop(), client.get_size, File(source=ctc.src_uri, path=rel_path)
    ) == len(data_v1)


@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_get_size_roundtrip_versioned_selects_version(
    cloud_test_catalog_upload,
    cloud_type,
    version_aware,
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)

    rel_path = f"client-get-size/{uuid4().hex}.bin"

    data_v1 = b"v1"
    uploaded_v1 = client.upload(data_v1, rel_path)

    data_v2 = b"version-2"
    uploaded_v2 = client.upload(data_v2, rel_path)

    assert uploaded_v1.version
    assert uploaded_v2.version

    size_v1 = sync(
        get_loop(),
        client.get_size,
        File(source=ctc.src_uri, path=rel_path, version=uploaded_v1.version),
    )
    size_v2 = sync(
        get_loop(),
        client.get_size,
        File(source=ctc.src_uri, path=rel_path, version=uploaded_v2.version),
    )

    assert size_v1 == len(data_v1)
    assert size_v2 == len(data_v2)


def test_get_size_missing_raises_filenotfound(
    cloud_test_catalog,
):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    missing_rel_path = f"client-get-size-missing/{uuid4().hex}.bin"

    with pytest.raises(FileNotFoundError):
        sync(
            get_loop(),
            client.get_size,
            File(source=ctc.src_uri, path=missing_rel_path),
        )


def test_get_size_directory_path_raises_fileerror(
    cloud_test_catalog_upload,
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)

    rel_dir = f"client-get-size-dir/{uuid4().hex}/"
    with pytest.raises(FileError, match=r"must not be a directory"):
        sync(get_loop(), client.get_size, File(source=ctc.src_uri, path=rel_dir))


@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_get_file_roundtrip_file_and_cloud(
    cloud_test_catalog_upload,
    cloud_type,
    version_aware,
    tmp_path,
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)

    rel_path = f"client-get-file/{uuid4().hex}.bin"
    data_v1 = b"v1"
    client.upload(data_v1, rel_path)

    dst_v1 = tmp_path / "out_v1.bin"
    sync(
        get_loop(),
        client.get_file,
        File(source=ctc.src_uri, path=rel_path).get_fs_path(),
        dst_v1.as_posix(),
        callback=DEFAULT_CALLBACK,
        version_id=None,
    )
    assert dst_v1.read_bytes() == data_v1


@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_get_file_roundtrip_versioned_selects_version(
    cloud_test_catalog_upload,
    cloud_type,
    version_aware,
    tmp_path,
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)

    rel_path = f"client-get-file/{uuid4().hex}.bin"
    from_path = File(source=ctc.src_uri, path=rel_path).get_fs_path()

    data_v1 = b"v1"
    uploaded_v1 = client.upload(data_v1, rel_path)

    data_v2 = b"version-2"
    uploaded_v2 = client.upload(data_v2, rel_path)

    assert uploaded_v1.version
    assert uploaded_v2.version

    dst_old = tmp_path / "out_old.bin"
    dst_new = tmp_path / "out_new.bin"

    sync(
        get_loop(),
        client.get_file,
        from_path,
        dst_old.as_posix(),
        callback=DEFAULT_CALLBACK,
        version_id=uploaded_v1.version,
    )
    sync(
        get_loop(),
        client.get_file,
        from_path,
        dst_new.as_posix(),
        callback=DEFAULT_CALLBACK,
        version_id=uploaded_v2.version,
    )

    assert dst_old.read_bytes() == data_v1
    assert dst_new.read_bytes() == data_v2


def test_get_file_missing_raises_filenotfound(
    cloud_test_catalog,
    tmp_path,
):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)

    missing_rel_path = f"client-get-file-missing/{uuid4().hex}.bin"
    dst = tmp_path / "out_missing.bin"
    from_path = File(source=ctc.src_uri, path=missing_rel_path).get_fs_path()

    with pytest.raises(FileNotFoundError):
        sync(
            get_loop(),
            client.get_file,
            from_path,
            dst.as_posix(),
            callback=DEFAULT_CALLBACK,
            version_id=None,
        )


@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_get_etag_roundtrip_file_and_cloud(
    cloud_test_catalog_upload,
    cloud_type,
    version_aware,
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)

    rel_path = f"client-get-etag/{uuid4().hex}.bin"
    data_v1 = b"v1"
    uploaded_v1 = client.upload(data_v1, rel_path)

    etag_v1 = sync(
        get_loop(),
        client.get_current_etag,
        File(source=ctc.src_uri, path=rel_path),
    )
    assert etag_v1
    assert etag_v1 == uploaded_v1.etag


@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_get_etag_roundtrip_versioned_selects_version(
    cloud_test_catalog_upload,
    cloud_type,
    version_aware,
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)

    rel_path = f"client-get-etag/{uuid4().hex}.bin"
    data_v1 = b"v1"
    uploaded_v1 = client.upload(data_v1, rel_path)

    data_v2 = b"version-2"
    uploaded_v2 = client.upload(data_v2, rel_path)

    assert uploaded_v1.version
    assert uploaded_v2.version

    etag_old = sync(
        get_loop(),
        client.get_current_etag,
        File(source=ctc.src_uri, path=rel_path, version=uploaded_v1.version),
    )
    etag_new = sync(
        get_loop(),
        client.get_current_etag,
        File(source=ctc.src_uri, path=rel_path, version=uploaded_v2.version),
    )

    assert etag_old == uploaded_v1.etag
    assert etag_new == uploaded_v2.etag
    assert etag_old != etag_new


@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_get_etag_missing_raises_filenotfound(
    cloud_test_catalog,
):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    missing_rel_path = f"client-get-etag-missing/{uuid4().hex}.bin"

    with pytest.raises(FileNotFoundError):
        sync(
            get_loop(),
            client.get_current_etag,
            File(source=ctc.src_uri, path=missing_rel_path),
        )


_xfail_special_char_versioned = pytest.mark.xfail(
    reason="versioned keys containing '#' or '?' are not supported on this backend",
    strict=False,
)

# Flat (cloud_type, char) params so per-combo xfail works cleanly.
_VERSIONED_SPECIAL_CHAR_PARAMS = [
    pytest.param("s3", "#", id="s3-hash"),
    pytest.param("s3", "?", id="s3-qmark"),
    pytest.param("gs", "#", id="gs-hash", marks=_xfail_special_char_versioned),
    pytest.param("gs", "?", id="gs-qmark", marks=_xfail_special_char_versioned),
    pytest.param("azure", "#", id="azure-hash"),
    pytest.param("azure", "?", id="azure-qmark", marks=_xfail_special_char_versioned),
]

_special_char_versioned = pytest.mark.parametrize(
    "cloud_type,char",
    _VERSIONED_SPECIAL_CHAR_PARAMS,
    indirect=["cloud_type"],
)


@pytest.mark.parametrize("version_aware", [True], indirect=True)
@_special_char_versioned
def test_get_file_info_versioned_special_char_in_key(
    cloud_test_catalog_upload, cloud_type, version_aware, char
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)
    rel_path = f"special-key/{uuid4().hex}{char}v.bin"

    up1 = client.upload(b"v1", rel_path)
    up2 = client.upload(b"version-2", rel_path)
    assert up1.version and up2.version

    f1 = client.get_file_info(rel_path, version_id=up1.version)
    f2 = client.get_file_info(rel_path, version_id=up2.version)
    assert f1.size == len(b"v1")
    assert f2.size == len(b"version-2")


@pytest.mark.parametrize("version_aware", [True], indirect=True)
@_special_char_versioned
def test_get_size_versioned_special_char_in_key(
    cloud_test_catalog_upload, cloud_type, version_aware, char
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)
    rel_path = f"special-key/{uuid4().hex}{char}v.bin"

    up1 = client.upload(b"v1", rel_path)
    up2 = client.upload(b"version-2", rel_path)
    assert up1.version and up2.version

    s1 = sync(
        get_loop(),
        client.get_size,
        File(source=ctc.src_uri, path=rel_path, version=up1.version),
    )
    s2 = sync(
        get_loop(),
        client.get_size,
        File(source=ctc.src_uri, path=rel_path, version=up2.version),
    )
    assert s1 == len(b"v1")
    assert s2 == len(b"version-2")


@pytest.mark.parametrize("version_aware", [True], indirect=True)
@_special_char_versioned
def test_get_etag_versioned_special_char_in_key(
    cloud_test_catalog_upload, cloud_type, version_aware, char
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)
    rel_path = f"special-key/{uuid4().hex}{char}v.bin"

    up1 = client.upload(b"v1", rel_path)
    up2 = client.upload(b"version-2", rel_path)
    assert up1.version and up2.version

    etag1 = sync(
        get_loop(),
        client.get_current_etag,
        File(source=ctc.src_uri, path=rel_path, version=up1.version),
    )
    etag2 = sync(
        get_loop(),
        client.get_current_etag,
        File(source=ctc.src_uri, path=rel_path, version=up2.version),
    )
    assert etag1 == up1.etag
    assert etag2 == up2.etag
    assert etag1 != etag2


@pytest.mark.parametrize("version_aware", [True], indirect=True)
@_special_char_versioned
def test_get_file_versioned_special_char_in_key(
    cloud_test_catalog_upload, cloud_type, version_aware, char, tmp_path
):
    ctc = cloud_test_catalog_upload
    client = ctc.catalog.get_client(ctc.src_uri)
    rel_path = f"special-key/{uuid4().hex}{char}v.bin"
    from_path = File(source=ctc.src_uri, path=rel_path).get_fs_path()

    up1 = client.upload(b"v1", rel_path)
    up2 = client.upload(b"version-2", rel_path)
    assert up1.version and up2.version

    dst1 = tmp_path / "out1.bin"
    dst2 = tmp_path / "out2.bin"
    sync(
        get_loop(),
        client.get_file,
        from_path,
        dst1.as_posix(),
        callback=DEFAULT_CALLBACK,
        version_id=up1.version,
    )
    sync(
        get_loop(),
        client.get_file,
        from_path,
        dst2.as_posix(),
        callback=DEFAULT_CALLBACK,
        version_id=up2.version,
    )
    assert dst1.read_bytes() == b"v1"
    assert dst2.read_bytes() == b"version-2"
