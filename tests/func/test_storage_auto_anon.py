from unittest.mock import patch

import pytest

import datachain as dc
from datachain.client.azure import AzureClient
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
    catalog = chain.session.catalog
    assert catalog.client_config_for(cloud_server.src_uri).get("anon") is True
    # The per-call config is registered per source, never written into the
    # session-wide default.
    assert "anon" not in catalog.client_config


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_read_storage_no_auto_anon_when_credentials_in_config(cloud_server):
    """Creds in client_config → credential gate short-circuits, anon NOT set."""
    config = {
        **_config_without_credentials(cloud_server),
        "key": "fake",
        "secret": "fake",
    }
    chain = dc.read_storage(cloud_server.src_uri, client_config=config)
    assert "anon" not in chain.session.catalog.client_config_for(cloud_server.src_uri)


def test_auto_anon_unsupported_backend_yields_no_anon(tmp_dir, catalog):
    # FileClient doesn't implement bucket_status → helper returns False
    # without patching, so anon stays unset.
    chain = dc.read_storage(tmp_dir.as_uri())
    assert "anon" not in chain.session.catalog.client_config_for(tmp_dir.as_uri())


@pytest.mark.parametrize("explicit", [True, False])
@patch("datachain.lib.dc.storage._all_buckets_anonymous")
def test_explicit_anon_skips_auto_detect(probe, explicit, tmp_dir, catalog):
    chain = dc.read_storage(tmp_dir.as_uri(), anon=explicit)
    probe.assert_not_called()
    catalog = chain.session.catalog
    assert catalog.client_config_for(tmp_dir.as_uri()).get("anon") is explicit
    assert "anon" not in catalog.client_config
