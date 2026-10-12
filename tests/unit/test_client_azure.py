import os
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree.ElementTree import fromstring

import pytest
from adlfs import AzureBlobFileSystem
from fsspec.asyn import sync
from fsspec.callbacks import DEFAULT_CALLBACK

from datachain.asyn import get_loop
from datachain.client.azure import AzureClient
from datachain.lib.file import File

_FAKE_SAS = "https://account.blob.core.windows.net/mycontainer/blob.txt?sv=x&sig=y"
_VER = "ver-abc-123"
_INFO = {
    "name": "az://mycontainer/blob.txt",
    "size": 42,
    "etag": '"abc123"',
    "last_modified": datetime(2024, 1, 1, tzinfo=UTC),
}


@pytest.fixture(autouse=True)
def _one_process(mocker):
    mocker.patch("datachain.client.listing_worker.LIST_PROCESSES", 0)  # mocks stay here


def _make_client() -> AzureClient:
    client = AzureClient("mycontainer", {}, MagicMock())
    client._fs = MagicMock()
    client._fs.sign.return_value = _FAKE_SAS
    client._fs._info = AsyncMock(return_value=_INFO)
    client._fs._get_file = AsyncMock(return_value=None)
    return client


def test_url_versioned_versionid_exactly_once():
    result = _make_client().url("blob.txt", version_id=_VER)
    assert result.count("versionid=") == 1
    assert result.endswith(f"&versionid={_VER}")


def test_url_versioned_sign_called_with_embedded_versionid():
    client = _make_client()
    client.url("blob.txt", version_id=_VER)
    assert f"?versionid={_VER}" in client._fs.sign.call_args[0][0]


def test_url_unversioned_no_versionid_in_result():
    assert "versionid" not in _make_client().url("blob.txt")


def test_url_unversioned_sign_called_with_clean_path():
    client = _make_client()
    client.url("blob.txt")
    assert "?" not in client._fs.sign.call_args[0][0]


@pytest.mark.parametrize("char", ["?", "#", "?versionid="])
def test_url_key_forwarded_to_sign(char):
    client = _make_client()
    client.url(f"blob{char}file.txt", version_id=_VER)
    path_arg = client._fs.sign.call_args[0][0]
    assert f"blob{char}file.txt" in path_arg
    assert f"?versionid={_VER}" in path_arg


# Runs real adlfs._url; only generate_blob_sas is mocked.
@pytest.mark.parametrize(
    "char",
    [
        "?",
        "#",
        pytest.param(
            "?versionid=",
            marks=pytest.mark.xfail(
                reason="adlfs.split_path truncates key at '?versionid=' (FOLLOW-UP §4)",
                strict=True,
            ),
        ),
    ],
)
def test_url_blob_name_and_version_reach_sdk(char):
    client, _service_client, bc = _make_client_sdk()
    encoded = char.replace("#", "%23").replace("?", "%3F").replace("=", "%3D")
    bc.url = f"https://account.blob.core.windows.net/mycontainer/blob{encoded}file.txt"

    def _real_sign(path, expiration=3600, **kwargs):
        return sync(
            get_loop(), AzureBlobFileSystem._url, client._fs, path, expiration, **kwargs
        )

    client._fs.sign = _real_sign

    with patch("adlfs.spec.generate_blob_sas", return_value="sv=x&sig=y") as mock_sas:
        client.url(f"blob{char}file.txt", version_id=_VER)
        assert mock_sas.call_args[1]["blob_name"] == f"blob{char}file.txt"
        assert mock_sas.call_args[1].get("version_id") == _VER


_xfail_versionid_in_key = pytest.mark.xfail(
    reason="adlfs.split_path misparses '?versionid=' in key name",
    strict=False,
)

_DETAILS_MOCK = [
    {
        "name": "mycontainer/blob.txt",
        "size": 42,
        "type": "file",
        "etag": '"abc123"',
        "last_modified": datetime(2024, 1, 1, tzinfo=UTC),
    }
]


def _make_client_sdk():
    """AzureClient whose _info/_get_file run real adlfs code with
    service_client mocked at the Azure SDK boundary."""
    bc = AsyncMock()
    bc.__aenter__ = AsyncMock(return_value=bc)
    bc.__aexit__ = AsyncMock(return_value=False)
    blob_props = MagicMock()
    blob_props.has_key.side_effect = lambda k: (
        k in {"container", "name", "size", "etag", "last_modified"}
    )
    blob_props.container = "mycontainer"
    blob_props.name = "blob.txt"
    blob_props.size = 42
    blob_props.etag = '"abc123"'
    blob_props.last_modified = datetime(2024, 1, 1, tzinfo=UTC)
    blob_props.metadata = None
    bc.get_blob_properties = AsyncMock(return_value=blob_props)
    stream = AsyncMock()
    stream.readinto = AsyncMock()
    bc.download_blob = AsyncMock(return_value=stream)

    service_client = MagicMock()
    service_client.get_blob_client.return_value = bc

    mock_fs = MagicMock()
    mock_fs.version_aware = True
    mock_fs.dircache = {}
    mock_fs._ls_from_cache = MagicMock(return_value=None)
    mock_fs._strip_protocol.side_effect = AzureBlobFileSystem._strip_protocol
    mock_fs.service_client = service_client
    mock_fs._details = AsyncMock(return_value=_DETAILS_MOCK)
    mock_fs.split_path = lambda path, **kw: AzureBlobFileSystem.split_path(
        mock_fs, path, **kw
    )
    mock_fs.connection_string = None
    mock_fs.account_name = "testaccount"
    mock_fs.account_key = "dGVzdA=="
    mock_fs._timeout_kwargs = {}

    async def _real_info(path, refresh=False, **kwargs):
        return await AzureBlobFileSystem._info(mock_fs, path, refresh=refresh, **kwargs)

    async def _real_get_file(
        rpath,
        lpath,
        recursive=False,
        delimiter="/",
        callback=None,
        max_concurrency=None,
        **kwargs,
    ):
        return await AzureBlobFileSystem._get_file(
            mock_fs,
            rpath,
            lpath,
            recursive=recursive,
            delimiter=delimiter,
            callback=callback,
            max_concurrency=max_concurrency,
            **kwargs,
        )

    mock_fs._info = _real_info
    mock_fs._get_file = _real_get_file

    client = AzureClient("mycontainer", {}, MagicMock())
    client._fs = mock_fs
    return client, service_client, bc


@pytest.mark.parametrize(
    "char",
    [
        "?",
        "#",
        pytest.param("?versionid=", marks=_xfail_versionid_in_key),
    ],
)
def test_get_file_info_key_and_version(char):
    client, service_client, bc = _make_client_sdk()
    client.get_file_info(f"blob{char}file.txt", version_id=_VER)
    container, path = service_client.get_blob_client.call_args[0]
    assert container == "mycontainer"
    assert path == f"blob{char}file.txt"
    assert bc.get_blob_properties.call_args[1].get("version_id") == _VER


@pytest.mark.parametrize(
    "char",
    [
        "?",
        "#",
        pytest.param("?versionid=", marks=_xfail_versionid_in_key),
    ],
)
def test_get_size_key_and_version(char):
    client, service_client, bc = _make_client_sdk()
    sync(
        get_loop(),
        client.get_size,
        File(source="az://mycontainer", path=f"blob{char}file.txt", version=_VER),
    )
    container, path = service_client.get_blob_client.call_args[0]
    assert container == "mycontainer"
    assert path == f"blob{char}file.txt"
    assert bc.get_blob_properties.call_args[1].get("version_id") == _VER


@pytest.mark.parametrize(
    "char",
    [
        "?",
        "#",
        pytest.param("?versionid=", marks=_xfail_versionid_in_key),
    ],
)
def test_get_current_etag_key_and_version(char):
    client, service_client, bc = _make_client_sdk()
    sync(
        get_loop(),
        client.get_current_etag,
        File(source="az://mycontainer", path=f"blob{char}file.txt", version=_VER),
    )
    container, path = service_client.get_blob_client.call_args[0]
    assert container == "mycontainer"
    assert path == f"blob{char}file.txt"
    assert bc.get_blob_properties.call_args[1].get("version_id") == _VER


@pytest.mark.parametrize(
    "char",
    [
        "?",
        "#",
        pytest.param("?versionid=", marks=_xfail_versionid_in_key),
    ],
)
def test_get_file_key_and_version(char):
    client, service_client, bc = _make_client_sdk()
    sync(
        get_loop(),
        client.get_file,
        f"az://mycontainer/blob{char}file.txt",
        os.devnull,
        DEFAULT_CALLBACK,
        version_id=_VER,
    )
    container, path = service_client.get_blob_client.call_args[0]
    assert container == "mycontainer"
    assert path == f"blob{char}file.txt"
    assert bc.download_blob.call_args[1].get("version_id") == _VER


def test_blob_name_is_decoded():
    client = AzureClient("container", {}, None)
    client._fs = MagicMock(version_aware=True)
    blob = fromstring(  # noqa: S314
        "<Blob><Name Encoded='true'>a%01b</Name><VersionId>v1</VersionId>"
        "<IsCurrentVersion>true</IsCurrentVersion><Properties>"
        "<Last-Modified>Wed, 30 Sep 2026 23:00:00 GMT</Last-Modified>"
        "<Etag>0x8DC</Etag><Content-Length>7</Content-Length></Properties></Blob>"
    )
    assert client._blob_name(blob) == "a\x01b"
    assert client._xml_to_file(blob, "a\x01b") == File(
        source="az://container",
        path="a\x01b",
        etag="0x8DC",
        version="v1",
        last_modified=datetime(2026, 9, 30, 23, tzinfo=UTC),
        size=7,
    )


@pytest.mark.parametrize(
    "version_aware,include",
    [(True, ["metadata", "versions"]), (False, ["metadata"])],
)
def test_listing_asks_for_versions_only_when_version_aware(version_aware, include):
    _, container, _ = _listed("", version_aware)

    assert container.list_blob_names.call_args.kwargs["include"] == include


def _blob(name: str, resource_type: str = "", metadata: str = "") -> str:
    return (
        f"<Blob><Name>{name}</Name><Properties>"
        "<Last-Modified>Wed, 30 Sep 2026 23:00:00 GMT</Last-Modified>"
        f"<Etag>0x8DC</Etag><Content-Length>1</Content-Length>{resource_type}"
        f"</Properties><Metadata>{metadata}</Metadata></Blob>"
    )


def _listed(blobs: str, version_aware: bool = False) -> tuple[AzureClient, Any, list]:
    page = fromstring(  # noqa: S314
        f"<EnumerationResults><Blobs>{blobs}</Blobs></EnumerationResults>"
    )

    class Pages:
        _response = page
        continuation_token = None
        done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return page

    container = MagicMock()
    container.__aenter__ = AsyncMock(return_value=container)
    container.__aexit__ = AsyncMock(return_value=False)
    container.list_blob_names.return_value.by_page.return_value = Pages()
    client = AzureClient("mycontainer", {}, MagicMock())
    client._fs = MagicMock(version_aware=version_aware)
    client._fs.service_client.get_container_client.return_value = container

    async def list_all():
        return [f.path async for page in client.scandir("") for f in page]

    return client, container, sync(get_loop(), list_all)


@pytest.mark.parametrize("resource_type", ["", "<ResourceType>file</ResourceType>"])
def test_hierarchical_namespace_lists_out_of_key_order(resource_type):
    client, _, _ = _listed(_blob("a", resource_type))

    assert client._lists_in_key_order() is (not resource_type)


def test_hierarchical_namespace_directories_are_not_files():
    directory = "<ResourceType>directory</ResourceType>"
    file = "<ResourceType>file</ResourceType>"
    blobs = [
        _blob("dir", directory),
        _blob("dir/f1", file),
        _blob("dir/sub", directory),
        _blob("dir/sub/f2", file),
    ]

    _, _, paths = _listed("".join(blobs))

    assert paths == ["dir/f1", "dir/sub/f2"]


def test_hadoop_folder_markers_are_not_files():
    marker = "<hdi_isfolder>true</hdi_isfolder>"
    blobs = [
        _blob("year=2008", metadata=marker),
        _blob("year=2008/month=1", metadata=marker),
        _blob("year=2008/month=1/part-0.parquet"),
    ]

    _, _, paths = _listed("".join(blobs))

    assert paths == ["year=2008/month=1/part-0.parquet"]
