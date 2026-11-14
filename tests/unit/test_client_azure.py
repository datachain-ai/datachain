import os
import subprocess
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree.ElementTree import fromstring

import pytest
from adlfs import AzureBlobFileSystem
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    CredentialUnavailableError,
)
from fsspec.asyn import sync
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.config import conf

from datachain.asyn import get_loop
from datachain.client.azure import AzureClient, _cannot_authenticate
from datachain.lib.file import File, FileRow

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


@pytest.fixture
def no_azure_env(monkeypatch):
    for var in list(os.environ):
        if var.startswith("AZURE_"):
            monkeypatch.delenv(var)
    monkeypatch.setattr("datachain.client.azure._CREDENTIAL_CHECKS", {})
    AzureClient.FS_CLASS.clear_instance_cache()
    yield
    AzureClient.FS_CLASS.clear_instance_cache()


class _Credential:
    def __init__(self, error=None):
        self.error = error

    def get_token(self, *scopes, **kwargs):
        if self.error:
            raise self.error
        return "token"


def _chain(*credentials):
    return ChainedTokenCredential(*credentials)


@pytest.mark.parametrize(
    "credentials, found",
    [
        (
            [
                CredentialUnavailableError("no env"),
                CredentialUnavailableError("no cli"),
            ],
            False,
        ),
        ([CredentialUnavailableError("no env"), None], True),
        ([ClientAuthenticationError("bad secret"), None], False),
    ],
    ids=["nothing-configured", "identity", "failing-identity"],
)
def test_default_credentials_unavailable_after_authentication_failure(
    credentials, found
):
    chain = _chain(*map(_Credential, credentials))
    assert _cannot_authenticate(chain) is not found


def test_cli_timeout_allows_automatic_anonymous_access(mocker):
    mocker.patch(
        "azure.identity._credentials.azure_cli.shutil.which", return_value="az"
    )
    mocker.patch(
        "azure.identity._credentials.azure_cli.subprocess.check_output",
        side_effect=subprocess.TimeoutExpired("az", 1),
    )
    assert _cannot_authenticate(AzureCliCredential())


@pytest.mark.parametrize("anon", [None, False])
def test_failed_environment_credentials_respect_anonymous_mode(
    no_azure_env, monkeypatch, mocker, anon
):
    for name, value in {
        "AZURE_CLIENT_ID": "client",
        "AZURE_TENANT_ID": "tenant",
        "AZURE_CLIENT_SECRET": "secret",
    }.items():
        monkeypatch.setenv(name, value)
    mocker.patch(
        "azure.identity.ClientSecretCredential.get_token",
        side_effect=ClientAuthenticationError("bad secret"),
    )
    fs = AzureClient.create_fs(account_name="failed", anon=anon)
    assert fs.anon is (anon is None)
    if anon is False:
        with pytest.raises(ClientAuthenticationError, match="bad secret"):
            fs.sync_credential.get_token("https://storage.azure.com/.default")


@pytest.mark.parametrize("found", [True, False])
def test_create_fs_anon_when_credentials_unavailable(no_azure_env, found):
    with patch(
        "datachain.client.azure._cannot_authenticate", return_value=not found
    ) as probe:
        fs = AzureClient.create_fs(account_name=f"acc{found}")
        AzureClient.create_fs(account_name=f"acc{found}")
    assert fs.anon is not found
    probe.assert_called_once()


def test_create_fs_asks_again_after_credentials_unavailable(no_azure_env, mocker):
    probe = mocker.patch(
        "datachain.client.azure._cannot_authenticate", side_effect=[True, False]
    )
    now = mocker.patch("datachain.client.azure.time.monotonic", return_value=0.0)
    assert AzureClient.create_fs(account_name="later").anon

    now.return_value = 61.0  # then az login
    assert not AzureClient.create_fs(account_name="later").anon
    assert probe.call_count == 2


def test_identity_probe_rechecks_authenticated_identity_after_ttl(no_azure_env, mocker):
    probe = mocker.patch(
        "datachain.client.azure._cannot_authenticate", side_effect=[False, True]
    )
    now = mocker.patch("datachain.client.azure.time.monotonic", return_value=0.0)
    assert not AzureClient.create_fs(account_name="first").anon
    now.return_value = 61.0
    assert AzureClient.create_fs(account_name="second").anon
    assert probe.call_count == 2


def test_identity_probe_does_not_share_different_credential_selections(
    no_azure_env, mocker
):
    probe = mocker.patch(
        "datachain.client.azure._cannot_authenticate", side_effect=[True, False]
    )
    assert AzureClient.create_fs(
        account_name="first", exclude_environment_credential=True
    ).anon
    assert not AzureClient.create_fs(account_name="second").anon
    assert AzureClient.create_fs(
        account_name="first", exclude_environment_credential=True
    ).anon
    assert not AzureClient.create_fs(account_name="second").anon
    assert probe.call_count == 2


@pytest.mark.parametrize("absent", [False, True])
def test_identity_probe_is_shared_across_accounts_and_listing_modes(
    no_azure_env, mocker, absent
):
    probe = mocker.patch(
        "datachain.client.azure._cannot_authenticate", return_value=absent
    )
    for account, version_aware in [
        ("first", False),
        ("first", True),
        ("second", False),
    ]:
        assert (
            AzureClient.create_fs(
                account_name=account, version_aware=version_aware
            ).anon
            is absent
        )
    probe.assert_called_once()


def test_identity_probe_rechecks_after_configuration_changes(
    no_azure_env, mocker, monkeypatch
):
    probe = mocker.patch(
        "datachain.client.azure._cannot_authenticate", side_effect=[True, False]
    )
    assert AzureClient.create_fs(account_name="first").anon
    monkeypatch.setenv("AZURE_CLIENT_ID", "configured-client")
    assert not AzureClient.create_fs(account_name="second").anon
    assert probe.call_count == 2


@pytest.mark.parametrize("env", [{}, {"AZURE_STORAGE_ANON": "false"}])
def test_create_fs_keeps_explicit_authenticated_mode(no_azure_env, monkeypatch, env):
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    kwargs = {} if env else {"anon": False}
    with patch(
        "datachain.client.azure._cannot_authenticate", return_value=True
    ) as probe:
        fs = AzureClient.create_fs(account_name=f"explicit{len(env)}", **kwargs)
    probe.assert_not_called()
    assert not fs.anon


@pytest.mark.parametrize("anon", [False, True])
def test_create_fs_keeps_fsspec_anonymous_setting(no_azure_env, monkeypatch, anon):
    monkeypatch.setitem(conf, "abfs", {"anon": anon})
    with patch(
        "datachain.client.azure._cannot_authenticate", return_value=True
    ) as probe:
        fs = AzureClient.create_fs(account_name="configured")
    assert fs.anon is anon
    probe.assert_not_called()


def test_create_fs_reloads_environment_credentials_for_same_account(
    no_azure_env, monkeypatch, mocker
):
    from azure.core.credentials import AccessToken

    options = {
        "account_name": "same",
        "exclude_managed_identity_credential": True,
        "exclude_workload_identity_credential": True,
        "exclude_shared_token_cache_credential": True,
        "exclude_visual_studio_code_credential": True,
        "exclude_cli_credential": True,
        "exclude_powershell_credential": True,
        "exclude_developer_cli_credential": True,
        "exclude_interactive_browser_credential": True,
        "exclude_broker_credential": True,
    }
    mocker.patch(
        "azure.identity.ClientSecretCredential.get_token",
        return_value=AccessToken("token", 2**31),
    )
    assert AzureClient.create_fs(**options).anon
    for name, value in {
        "AZURE_CLIENT_ID": "client",
        "AZURE_TENANT_ID": "tenant",
        "AZURE_CLIENT_SECRET": "secret",
    }.items():
        monkeypatch.setenv(name, value)

    first = AzureClient.create_fs(**options)
    assert not first.anon
    assert first.sync_credential.credentials[0]._credential is not None
    assert AzureClient.create_fs(**options) is first

    monkeypatch.setenv("AZURE_CLIENT_SECRET", "new-secret")
    second = AzureClient.create_fs(**options)
    assert not second.anon
    assert second is not first

    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "a2V5")
    keyed = AzureClient.create_fs(**options)
    assert not keyed.anon
    assert keyed.account_key == "a2V5"
    assert keyed is not second


@pytest.mark.parametrize("anonymous_default", [False, True])
def test_bucket_status_refreshes_authenticated_environment(
    no_azure_env, monkeypatch, mocker, anonymous_default
):
    if anonymous_default:
        monkeypatch.setenv("AZURE_STORAGE_ANON", "true")
    service = mocker.patch("datachain.client.azure.BlobServiceClient")
    container = service.return_value.get_container_client.return_value
    container.get_container_properties.side_effect = ClientAuthenticationError(
        "private container"
    )
    filesystems = []

    async def info(fs, name):
        assert not fs.anon
        filesystems.append(fs)
        if not fs.account_key:
            raise ClientAuthenticationError("no credentials")
        return {}

    mocker.patch.object(AzureClient.FS_CLASS, "_info", new=info)
    mocker.patch(
        "datachain.client.azure._cannot_authenticate",
        side_effect=AssertionError("Authenticated probes must not fall back"),
    )
    assert AzureClient.bucket_status("private", account_name="same").access == "denied"
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "a2V5")
    assert (
        AzureClient.bucket_status("private", account_name="same").access
        == "authenticated"
    )
    assert filesystems[0] is not filesystems[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anon": False, "credential": "c"},
        {"account_key": "k"},
        {"sas_token": "s"},
        {
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=acc;"
            "AccountKey=a2V5;EndpointSuffix=core.windows.net"
        },
    ],
)
def test_create_fs_explicit_config_skips_credential_check(no_azure_env, kwargs):
    with patch("datachain.client.azure._cannot_authenticate") as probe:
        fs = AzureClient.create_fs(account_name="acc", **kwargs)
    probe.assert_not_called()
    assert not fs.anon


@pytest.mark.parametrize(
    "var,value", [("AZURE_STORAGE_ACCOUNT_KEY", "a2V5"), ("AZURE_STORAGE_ANON", "1")]
)
def test_create_fs_env_config_skips_credential_check(
    no_azure_env, monkeypatch, var, value
):
    monkeypatch.setenv(var, value)
    with patch("datachain.client.azure._cannot_authenticate") as probe:
        AzureClient.create_fs(account_name="acc")
    probe.assert_not_called()


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


def test_identity_probe_cache_evicts_old_configurations(no_azure_env, mocker):
    probe = mocker.patch(
        "datachain.client.azure._cannot_authenticate", return_value=True
    )
    for timeout in range(1, 18):
        assert AzureClient.create_fs(
            account_name="account", process_timeout=timeout
        ).anon
    assert probe.call_count == 17

    assert AzureClient.create_fs(account_name="account", process_timeout=1).anon
    assert probe.call_count == 18
    assert AzureClient.create_fs(account_name="account", process_timeout=17).anon
    assert probe.call_count == 18
