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
from datachain.client import Client
from datachain.client.azure import AzureClient
from datachain.dataset import StorageURI
from datachain.lib.file import File, FileRow


@pytest.fixture(autouse=True)
def _no_env_connection_string(monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    AzureBlobFileSystem.clear_instance_cache()
    yield
    AzureBlobFileSystem.clear_instance_cache()


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


def test_name_without_account():
    client = AzureClient("mycontainer", {}, MagicMock())
    assert client.name == "mycontainer"
    assert client.container == "mycontainer"
    assert client.uri == "az://mycontainer"
    assert "account_name" not in client.fs_kwargs


def test_name_with_account():
    client = AzureClient("mycontainer@myaccount", {}, MagicMock())
    assert client.name == "mycontainer@myaccount"
    assert client.container == "mycontainer"
    assert client.uri == "az://mycontainer@myaccount"
    assert client.fs_kwargs["account_name"] == "myaccount"


def test_uri_account_overrides_client_config():
    client = AzureClient(
        "mycontainer@myaccount", {"account_name": "other"}, MagicMock()
    )
    assert client.fs_kwargs["account_name"] == "myaccount"


def test_client_config_account_kept_without_uri_account():
    client = AzureClient("mycontainer", {"account_name": "other"}, MagicMock())
    assert client.fs_kwargs["account_name"] == "other"


_CONN_STR = (
    "DefaultEndpointsProtocol=https;AccountName=myaccount;"
    "AccountKey=dGVzdA==;EndpointSuffix=core.windows.net"
)
_SAS = "SharedAccessSignature=sv=x&sig=y"
_ENDPOINT_SAS = f"BlobEndpoint=https://myaccount.blob.core.windows.net;{_SAS}"
_CHINA_CONN_STR = _CONN_STR.replace("core.windows.net", "chinacloudapi.cn")
_DUP_CONN_STR = "AccountName=other;AccountName=myaccount;AccountKey=dGVzdA=="
_DOMAIN_CONN_STR = f"BlobEndpoint=https://files.example.com;{_SAS}"


@pytest.mark.parametrize(
    "fs_kwargs",
    [
        {"connection_string": _CONN_STR},
        {"connection_string": _CHINA_CONN_STR},
        {"connection_string": _ENDPOINT_SAS},
        {"account_host": "myaccount.blob.core.windows.net"},
    ],
    ids=["named", "sovereign-suffix", "endpoint-sas", "account-host"],
)
def test_uri_account_matching_endpoint_ok(fs_kwargs):
    client = AzureClient("mycontainer@myaccount", fs_kwargs, MagicMock())
    assert client.fs.service_client.account_name == "myaccount"


@pytest.mark.parametrize(
    "fs_kwargs",
    [
        {"connection_string": _DOMAIN_CONN_STR},
        {"account_host": "files.example.com"},
        {"account_host": "myaccount.blob.local.azurestack.external"},
        {"account_host": "myaccount.z1.blob.storage.azure.net"},
    ],
    ids=["custom-domain-conn", "custom-domain-host", "azure-stack", "dns-zone"],
)
def test_uri_account_unidentifiable_endpoint_trusted(fs_kwargs):
    client = AzureClient("mycontainer@myaccount", fs_kwargs, MagicMock())
    assert client.fs.service_client.account_name is None
    assert client.fs_kwargs["account_name"] == "myaccount"


@pytest.mark.parametrize(
    "fs_kwargs,target",
    [
        ({"connection_string": _CONN_STR}, "account 'myaccount'"),
        ({"connection_string": _ENDPOINT_SAS}, "account 'myaccount'"),
        (
            {
                "connection_string": "AccountName=other;"
                f"BlobEndpoint=https://myaccount.blob.core.windows.net;{_SAS}"
            },
            "account 'myaccount'",
        ),
        ({"connection_string": _DUP_CONN_STR}, "account 'myaccount'"),
        (
            {"connection_string": "UseDevelopmentStorage=true"},
            "account 'devstoreaccount1'",
        ),
        ({"account_host": "myaccount.blob.core.windows.net"}, "account 'myaccount'"),
    ],
    ids=[
        "named",
        "endpoint-sas",
        "name-vs-endpoint",
        "duplicate-name",
        "dev-storage",
        "account-host",
    ],
)
def test_uri_account_conflicting_endpoint_raises(fs_kwargs, target):
    with pytest.raises(
        ValueError, match=f"conflicts with the configured endpoint for {target}"
    ):
        AzureClient("mycontainer@other", fs_kwargs, MagicMock())


def test_uri_account_conflicting_env_connection_string_raises(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", _CONN_STR)
    with pytest.raises(ValueError, match="conflicts with"):
        AzureClient("mycontainer@other", {}, MagicMock())


def test_devstoreaccount_development_storage_ok():
    client = AzureClient(
        "mycontainer@devstoreaccount1",
        {"connection_string": "UseDevelopmentStorage=true"},
        MagicMock(),
    )
    assert client.fs.service_client.account_name == "devstoreaccount1"


def test_no_uri_account_ignores_connection_string_account():
    client = AzureClient("mycontainer", {"connection_string": _CONN_STR}, MagicMock())
    assert "account_name" not in client.fs_kwargs
    assert client.fs.service_client.account_name == "myaccount"


@pytest.mark.parametrize("service", ["blob", "dfs"])
def test_name_with_full_account_host(service):
    client = AzureClient(
        f"mycontainer@myaccount.{service}.core.windows.net", {}, MagicMock()
    )
    assert client.name == f"mycontainer@myaccount.{service}.core.windows.net"
    assert client.container == "mycontainer"
    assert client.fs_kwargs["account_name"] == "myaccount"


@pytest.mark.parametrize(
    "netloc",
    [
        "mycontainer@myaccount.blob.core.chinacloudapi.cn",
        "mycontainer@myaccount.z1.blob.storage.azure.net",
        "mycontainer@myaccount.blob.core.windows.net:443",
        "mycontainer@files.example.com",
        "mycontainer@MyAccount",
        "mycontainer@my-account",
        "mycontainer@ab",
        "mycontainer@",
    ],
)
def test_unsupported_account_raises(netloc):
    with pytest.raises(ValueError, match="Unsupported Azure storage account"):
        AzureClient(netloc, {}, MagicMock())


def test_parse_url_with_account():
    uri, rel_path = Client.parse_url("az://mycontainer@myaccount/dir/blob.txt")
    assert uri == "az://mycontainer@myaccount"
    assert rel_path == "dir/blob.txt"


def test_get_client_with_account():
    client = Client.get_client("az://mycontainer@myaccount/dir/blob.txt", MagicMock())
    assert isinstance(client, AzureClient)
    assert client.uri == "az://mycontainer@myaccount"
    assert client.container == "mycontainer"
    assert client.fs_kwargs["account_name"] == "myaccount"


def test_from_source_preserves_account():
    client = AzureClient.from_source(
        StorageURI("az://mycontainer@myaccount"), MagicMock()
    )
    assert client.name == "mycontainer@myaccount"
    assert client.container == "mycontainer"
    assert client.fs_kwargs["account_name"] == "myaccount"


def test_from_source_with_path_parses_netloc_account():
    client = AzureClient.from_source(
        StorageURI("az://mycontainer@myaccount/exports/"), MagicMock()
    )
    assert client.name == "mycontainer@myaccount/exports"
    assert client.container == "mycontainer"
    assert client.fs_kwargs["account_name"] == "myaccount"


def test_get_uri_with_account():
    client = AzureClient("mycontainer@myaccount", {}, MagicMock())
    assert client.get_uri("dir/blob.txt") == "az://mycontainer@myaccount/dir/blob.txt"


def test_create_fs_receives_account_name():
    with patch.object(AzureClient, "FS_CLASS") as mock_fs_cls:
        mock_fs_cls.return_value.service_client.account_name = "myaccount"
        AzureClient("mycontainer@myaccount", {}, MagicMock())
    assert mock_fs_cls.call_args[1]["account_name"] == "myaccount"


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


def _make_client_sdk(name="mycontainer"):
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

    client = AzureClient(name, {}, MagicMock())
    client._fs = mock_fs
    return client, service_client, bc


def test_get_file_info_with_account_in_name():
    client, service_client, _bc = _make_client_sdk("mycontainer@myaccount")
    file = client.get_file_info("dir/blob.txt", version_id=_VER)
    container, path = service_client.get_blob_client.call_args[0]
    assert container == "mycontainer"
    assert path == "dir/blob.txt"
    assert file.source == "az://mycontainer@myaccount"
    assert file.path == "dir/blob.txt"


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


def test_put_in_cache_pinned_version_still_checks_etag():
    client, _, bc = _make_client_sdk()
    file = File(
        source="az://mycontainer", path="blob.txt", version=_VER, etag="stale-etag"
    )

    with pytest.raises(FileNotFoundError, match="changed on the source"):
        client.put_in_cache(file)
    bc.download_blob.assert_not_called()


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
    assert client._file_row(blob, "a\x01b") == FileRow(
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


def _listed(
    blobs: str, version_aware: bool = False, name: str = "mycontainer"
) -> tuple[AzureClient, Any, list]:
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
    client = AzureClient(name, {}, MagicMock())
    client._fs = MagicMock(version_aware=version_aware)
    client._fs.service_client.get_container_client.return_value = container

    async def list_all():
        return [f.path async for page in client.scandir("") for f in page]

    return client, container, sync(get_loop(), list_all)


def test_listing_opens_container_without_account():
    client, _, paths = _listed(_blob("a"), name="mycontainer@myaccount")

    client.fs.service_client.get_container_client.assert_called_once_with(
        container="mycontainer"
    )
    assert paths == ["a"]


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


@pytest.mark.parametrize(
    "marker",
    [
        "<hdi_isfolder>true</hdi_isfolder>",
        "<Hdi_isfolder>true</Hdi_isfolder>",
        "<is_directory>true</is_directory>",
    ],
    ids=["hadoop", "hadoop-capitalized", "adlfs"],
)
def test_folder_markers_are_not_files(marker):
    blobs = [
        _blob("year=2008", metadata=marker),
        _blob("year=2008/month=1", metadata=marker),
        _blob("year=2008/month=1/part-0.parquet"),
    ]

    _, _, paths = _listed("".join(blobs))

    assert paths == ["year=2008/month=1/part-0.parquet"]
