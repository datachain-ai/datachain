import pytest

from datachain.catalog.catalog import AUTO_ANON_CLIENT_CONFIG


def test_registry_keys_are_rstripped_uris(catalog):
    catalog.register_client_config("s3://bkt/dir/", {"anon": True})
    catalog.register_client_config("file:///tmp/data/", {"use_symlinks": True})
    assert catalog.source_client_configs == {
        "s3://bkt/dir": {"anon": True},
        "file:///tmp/data": {"use_symlinks": True},
    }


def test_lookup_by_longest_prefix(catalog):
    catalog.register_client_config("s3://bkt", {"anon": True})
    catalog.register_client_config("s3://bkt/team-b", {"key": "b"})

    assert catalog.client_config_for("s3://bkt/x.csv") == {"anon": True}
    assert catalog.client_config_for("s3://bkt/team-b/x.csv") == {"key": "b"}
    assert catalog.client_config_for("s3://bkt/team-b") == {"key": "b"}
    # Unregistered source falls back to the catalog-wide default.
    assert catalog.client_config_for("s3://elsewhere/x") == catalog.client_config


def test_prefix_matches_on_path_boundary(catalog):
    catalog.register_client_config("s3://bkt/team-a", {"key": "a"})
    assert catalog.client_config_for("s3://bkt/team-a/f") == {"key": "a"}
    # Similar names must not match across a path-segment boundary.
    assert catalog.client_config_for("s3://bkt/team-ab/f") == catalog.client_config
    # A parent of the registered prefix is not covered by it.
    assert catalog.client_config_for("s3://bkt") == catalog.client_config


def test_lookup_ignores_trailing_slash(catalog):
    catalog.register_client_config("file:///tmp/data/", {"use_symlinks": True})
    assert catalog.client_config_for("file:///tmp/data") == {"use_symlinks": True}
    assert catalog.client_config_for("file:///tmp/data/") == {"use_symlinks": True}


def test_reregister_same_config_is_noop(catalog):
    catalog.register_client_config("s3://bkt", {"anon": True})
    catalog.register_client_config("s3://bkt/", {"anon": True})
    assert catalog.source_client_configs == {"s3://bkt": {"anon": True}}


def test_conflicting_register_for_same_prefix_raises(catalog):
    catalog.register_client_config("s3://bkt", {"aws_endpoint_url": "http://a"})
    with pytest.raises(ValueError, match="different client_config"):
        catalog.register_client_config("s3://bkt", {"aws_endpoint_url": "http://b"})
    # A different prefix — nested or not — is its own entry.
    catalog.register_client_config("s3://bkt/sub", {"aws_endpoint_url": "http://b"})
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
    """Explicit per-call kwargs > registered prefix > catalog default."""
    catalog.client_config = {"anon": False}
    catalog.register_client_config("s3://bkt", {"anon": True})

    assert catalog.get_client("s3://bkt/f").fs_kwargs == {"anon": True}
    assert catalog.get_client("s3://other/f").fs_kwargs == {"anon": False}
    assert catalog.get_client("s3://bkt/f", anon=False).fs_kwargs == {"anon": False}


def test_init_params_ship_registry(catalog):
    catalog.register_client_config("s3://bkt", {"anon": True})
    params = catalog.get_init_params()
    assert params["source_client_configs"] == {"s3://bkt": {"anon": True}}


def test_client_config_for_file_matches_uri_resolution(catalog):
    catalog.register_client_config("s3://bkt/team-a", {"key": "a"})
    catalog.register_client_config("s3://bkt", {"anon": True})

    assert catalog.client_config_for_file("s3://bkt", "team-a/f.csv") == {"key": "a"}
    assert catalog.client_config_for_file("s3://bkt", "team-ab/f.csv") == {"anon": True}
    assert catalog.client_config_for_file("s3://bkt", "x.csv") == {"anon": True}
    assert (
        catalog.client_config_for_file("s3://other", "x.csv") == catalog.client_config
    )
    # A prefix registered above the source covers all of it.
    catalog.register_client_config("file:///data", {"use_symlinks": True})
    assert catalog.client_config_for_file("file:///data/photos", "f.jpg") == {
        "use_symlinks": True
    }


def test_client_config_for_file_memo_invalidated_on_register(catalog):
    assert catalog.client_config_for_file("s3://bkt", "f.csv") == {}
    catalog.register_client_config("s3://bkt", {"anon": True})
    assert catalog.client_config_for_file("s3://bkt", "f.csv") == {"anon": True}
