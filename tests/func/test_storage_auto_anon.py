from unittest.mock import patch

import pytest

import datachain as dc
from datachain.cache import Cache
from datachain.client import Client
from datachain.client.azure import AzureClient
from datachain.client.fsspec import AnonFallbackFS
from datachain.client.gcs import GCSClient
from datachain.client.s3 import ClientS3

_CREDENTIAL_KEYS = (
    ClientS3.CREDENTIAL_KEYS | GCSClient.CREDENTIAL_KEYS | AzureClient.CREDENTIAL_KEYS
)


def _config_without_credentials(cloud_server) -> dict:
    # Test conftest puts creds into cloud_server.client_config so general
    # tests skip the auto-anon probe; strip them here so the probe runs.
    return {
        k: v for k, v in cloud_server.client_config.items() if k not in _CREDENTIAL_KEYS
    }


@pytest.mark.parametrize(
    "cloud_type",
    [
        "s3",
        "gs",
        pytest.param(
            "azure",
            marks=pytest.mark.xfail(
                reason=(
                    "Azure bucket_status probe hardcodes blob.core.windows.net; "
                    "the Azurite mock isn't reachable that way."
                ),
                strict=True,
            ),
        ),
    ],
    indirect=True,
)
def test_read_storage_auto_anon_on_public_cloud_bucket(cloud_server):
    """Public bucket, no creds in client_config → auto-anon kicks in."""
    chain = dc.read_storage(
        cloud_server.src_uri, client_config=_config_without_credentials(cloud_server)
    )
    assert chain.session.catalog.client_config.get("anon") is True


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_read_storage_no_auto_anon_when_credentials_in_config(cloud_server):
    """Creds in client_config → credential gate short-circuits, anon NOT set."""
    config = {
        **_config_without_credentials(cloud_server),
        "key": "fake",
        "secret": "fake",
    }
    chain = dc.read_storage(cloud_server.src_uri, client_config=config)
    assert "anon" not in chain.session.catalog.client_config


def test_auto_anon_unsupported_backend_yields_no_anon(tmp_dir, catalog):
    # FileClient doesn't implement bucket_status → helper returns False
    # without patching, so anon stays unset.
    chain = dc.read_storage(tmp_dir.as_uri())
    assert "anon" not in chain.session.catalog.client_config


@pytest.mark.parametrize("explicit", [True, False])
@patch("datachain.lib.dc.storage._all_buckets_anonymous")
def test_explicit_anon_skips_auto_detect(probe, explicit, tmp_dir, catalog):
    chain = dc.read_storage(tmp_dir.as_uri(), anon=explicit)
    probe.assert_not_called()
    assert chain.session.catalog.client_config.get("anon") is explicit


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
def test_anon_fallback_proxy_integration(
    cloud_server, cloud_server_credentials, tmp_path
):
    cache = Cache(str(tmp_path / "cache"), str(tmp_path / "tmp"))
    client = Client.get_client(
        cloud_server.src_uri, cache, **cloud_server.client_config
    )

    # Building fs returns the proxy for cloud clients.
    _ = client.fs
    assert isinstance(client._fs, AnonFallbackFS)

    # End-to-end op through proxy → real fsspec → emulator.
    info = client.get_file_info("description")
    assert info.path == "description"
    assert info.size and info.size > 0
