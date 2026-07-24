import pytest

from datachain.catalog.catalog import AUTO_ANON_CLIENT_CONFIG


def test_storage_uri_for_normalizes_to_storage_root(catalog):
    assert catalog.storage_uri_for("s3://bkt/dir/file.csv") == "s3://bkt"
    assert catalog.storage_uri_for("s3://bkt") == "s3://bkt"
    assert catalog.storage_uri_for("gs://other/x") == "gs://other"
    # Local: the storage root is the directory.
    assert catalog.storage_uri_for("file:///tmp/data/") == "file:///tmp/data"
    assert catalog.storage_uri_for("file:///tmp/data/f.csv") == "file:///tmp/data"


def test_register_and_lookup(catalog):
    catalog.register_client_config("s3://bkt/dir/x", {"anon": True})
    assert catalog.client_config_for("s3://bkt/other.csv") == {"anon": True}
    assert catalog.client_config_for("s3://bkt") == {"anon": True}
    # Unregistered source falls back to the catalog-wide default.
    assert catalog.client_config_for("s3://elsewhere/x") == catalog.client_config


def test_lookup_matches_dir_registered_with_trailing_slash(catalog):
    # Listings canonicalize local directories with a trailing slash; a lookup
    # by the slash-less form must still resolve.
    catalog.register_client_config("file:///tmp/data/", {"use_symlinks": True})
    assert catalog.client_config_for("file:///tmp/data") == {"use_symlinks": True}
    assert catalog.client_config_for("file:///tmp/data/") == {"use_symlinks": True}


def test_reregister_same_config_is_noop(catalog):
    catalog.register_client_config("s3://bkt", {"anon": True})
    catalog.register_client_config("s3://bkt/sub/path", {"anon": True})
    assert catalog.source_client_configs == {"s3://bkt": {"anon": True}}


def test_conflicting_register_raises(catalog):
    catalog.register_client_config("s3://bkt", {"aws_endpoint_url": "http://a"})
    with pytest.raises(ValueError, match="different client_config"):
        catalog.register_client_config("s3://bkt/x", {"aws_endpoint_url": "http://b"})
    # A different bucket is unaffected.
    catalog.register_client_config("s3://other", {"aws_endpoint_url": "http://b"})


def test_explicit_config_upgrades_auto_anon(catalog):
    catalog.register_client_config("s3://bkt", dict(AUTO_ANON_CLIENT_CONFIG))
    catalog.register_client_config("s3://bkt", {"key": "k", "secret": "s"})
    assert catalog.client_config_for("s3://bkt/f") == {"key": "k", "secret": "s"}


def test_registered_config_is_copied(catalog):
    cfg = {"client_kwargs": {"endpoint_url": "http://a"}}
    catalog.register_client_config("s3://bkt", cfg)
    cfg["client_kwargs"]["endpoint_url"] = "http://mutated"
    assert catalog.client_config_for("s3://bkt/f") == {
        "client_kwargs": {"endpoint_url": "http://a"}
    }


def test_get_client_precedence(catalog):
    """Explicit per-call kwargs > registry entry > catalog default."""
    catalog.client_config = {"anon": False}
    catalog.register_client_config("s3://bkt", {"anon": True})

    assert catalog.get_client("s3://bkt/f").fs_kwargs == {"anon": True}
    assert catalog.get_client("s3://other/f").fs_kwargs == {"anon": False}
    assert catalog.get_client("s3://bkt/f", anon=False).fs_kwargs == {"anon": False}


def test_init_params_ship_registry(catalog):
    catalog.register_client_config("s3://bkt", {"anon": True})
    params = catalog.get_init_params()
    assert params["source_client_configs"] == {"s3://bkt": {"anon": True}}
