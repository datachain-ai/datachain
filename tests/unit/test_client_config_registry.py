import pytest

from datachain.catalog.catalog import AUTO_ANON_CLIENT_CONFIG


def test_registry_keys_are_rstripped(catalog):
    catalog.register_client_config("s3://bkt/", {"anon": True})
    assert catalog.source_client_configs == {"s3://bkt": {"anon": True}}


def test_lookup_is_exact_by_source(catalog):
    catalog.register_client_config("s3://bkt", {"anon": True})

    assert catalog.client_config_for("s3://bkt") == {"anon": True}
    assert catalog.client_config_for("s3://bkt/") == {"anon": True}
    # Lookup is exact: anything that is not a registered source falls back
    # to the catalog-wide default.
    assert catalog.client_config_for("s3://bkt/dir/x.csv") == catalog.client_config
    assert catalog.client_config_for("s3://elsewhere") == catalog.client_config


def test_distinct_sources_coexist(catalog):
    catalog.register_client_config("s3://bkt-a", {"key": "a"})
    catalog.register_client_config("s3://bkt-b", {"key": "b"})
    catalog.register_client_config("file:///data/photos", {"use_symlinks": True})

    assert catalog.client_config_for("s3://bkt-a") == {"key": "a"}
    assert catalog.client_config_for("s3://bkt-b") == {"key": "b"}
    assert catalog.client_config_for("file:///data/photos") == {"use_symlinks": True}


def test_reregister_same_config_is_noop(catalog):
    catalog.register_client_config("s3://bkt", {"anon": True})
    catalog.register_client_config("s3://bkt/", {"anon": True})
    assert catalog.source_client_configs == {"s3://bkt": {"anon": True}}


def test_conflicting_register_for_same_source_raises(catalog):
    catalog.register_client_config("s3://bkt", {"aws_endpoint_url": "http://a"})
    with pytest.raises(ValueError, match="different client_config"):
        catalog.register_client_config("s3://bkt", {"aws_endpoint_url": "http://b"})
    # A different source is its own entry.
    catalog.register_client_config("s3://other", {"aws_endpoint_url": "http://b"})


def test_explicit_config_upgrades_auto_anon(catalog):
    catalog.register_client_config("s3://bkt", dict(AUTO_ANON_CLIENT_CONFIG))
    catalog.register_client_config("s3://bkt", {"key": "k", "secret": "s"})
    assert catalog.client_config_for("s3://bkt") == {"key": "k", "secret": "s"}


def test_registered_config_is_copied(catalog):
    cfg = {"client_kwargs": {"endpoint_url": "http://a"}}
    catalog.register_client_config("s3://bkt", cfg)
    cfg["client_kwargs"]["endpoint_url"] = "http://mutated"
    assert catalog.client_config_for("s3://bkt") == {
        "client_kwargs": {"endpoint_url": "http://a"}
    }


def test_get_client_precedence(catalog):
    """Explicit per-call kwargs > registered source config > catalog default."""
    catalog.client_config = {"anon": False}
    catalog.register_client_config("s3://bkt", {"anon": True})

    assert catalog.get_client("s3://bkt").fs_kwargs == {"anon": True}
    assert catalog.get_client("s3://other").fs_kwargs == {"anon": False}
    assert catalog.get_client("s3://bkt", anon=False).fs_kwargs == {"anon": False}


def test_init_params_ship_registry(catalog):
    catalog.register_client_config("s3://bkt", {"anon": True})
    params = catalog.get_init_params()
    assert params["source_client_configs"] == {"s3://bkt": {"anon": True}}
