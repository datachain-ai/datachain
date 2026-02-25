import asyncio
import os
import sys
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
from datachain.lib.file import File, FileError
from tests.data import ENTRIES

_non_null_text = st.text(
    alphabet=st.characters(blacklist_categories=["Cc", "Cs"]), min_size=1
)


@pytest.fixture
def client(cloud_server, cloud_server_credentials):
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


def test_scandir_alternate(client):
    results = scandir(client, "", method="nested")
    match_entries(results, ENTRIES)


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
