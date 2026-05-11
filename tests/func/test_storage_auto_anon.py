from unittest.mock import patch

import pytest

import datachain as dc


@patch("datachain.lib.dc.storage._all_buckets_anonymous", return_value=True)
def test_auto_anon_sets_true_when_all_buckets_anonymous(_probe, tmp_dir, catalog):
    chain = dc.read_storage(tmp_dir.as_uri())
    assert chain.session.catalog.client_config.get("anon") is True


@patch("datachain.lib.dc.storage._all_buckets_anonymous", return_value=False)
def test_auto_anon_leaves_unset_when_helper_says_no(_probe, tmp_dir, catalog):
    chain = dc.read_storage(tmp_dir.as_uri())
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
