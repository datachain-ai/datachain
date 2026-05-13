import os
import sys

import pytest

import datachain as dc
from datachain.lib.file import File
from datachain.lib.listing import (
    _sanitize_ds_name,
    calc_fingerprint,
    get_listing,
    list_bucket,
    parse_listing_uri,
)
from tests.data import ENTRIES


def test_listing_generator(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    catalog = ctc.catalog

    uri = f"{ctc.src_uri}/cats"

    chain = dc.read_records([{"uri": uri}], schema={"uri": str}).gen(
        file=list_bucket(uri, catalog.cache, client_config=catalog.client_config)
    )
    assert chain.count() == 2

    entries = sorted(
        [e for e in ENTRIES if e.path.startswith("cats/")], key=lambda e: e.path
    )
    files = chain.order_by("file.path").to_values("file")

    for cat_file, cat_entry in zip(files, entries, strict=False):
        assert cat_file.source == ctc.src_uri
        assert cat_file.path == cat_entry.path
        assert cat_file.size == cat_entry.size
        assert cat_file.etag is not None
        # If version_aware is not passed it is enforced to be True internally
        if catalog.client_config.get("version_aware", True):
            assert cat_file.version is not None
        else:
            assert cat_file.version == ""
        assert cat_file.is_latest == cat_entry.is_latest
        assert cat_file.location is None


def test_read_storage_percent_encoding_is_opaque_across_backends(
    cloud_test_catalog_upload,
):
    ctc = cloud_test_catalog_upload
    src_uri = ctc.src_uri
    catalog = ctc.catalog

    prefix = "dir with space"
    payloads = {
        "hello#.txt": b"hi#",
        "file with space.txt": b"h i",
        "file%20literal.txt": b"hi %20",
    }
    # Windows does not allow '?' in file names; include only for non-file backends.
    if not str(src_uri).startswith("file://") or sys.platform != "win32":
        payloads["he?llo.txt"] = b"hi"

    for name, content in payloads.items():
        File.upload(content, f"{src_uri}/{prefix}/{name}", catalog)

    # Reading via the literal (non-encoded) storage URI works.
    listed = dc.read_storage(f"{src_uri}/{prefix}", session=ctc.session).to_values(
        "file"
    )
    listed_paths = {f.path for f in listed}
    expected_paths = (
        set(payloads)
        if str(src_uri).startswith("file://")
        else {f"{prefix}/{name}" for name in payloads}
    )
    assert listed_paths == expected_paths

    # Percent-encoding is treated literally: it addresses a different prefix.
    encoded_prefix = "dir%20with%20space"
    with pytest.raises(FileNotFoundError, match=r"dir%20with%20space"):
        dc.read_storage(
            f"{src_uri}/{encoded_prefix}",
            session=ctc.session,
        ).to_values("file")

    # And addressing a specific key with %20 in the path fails.
    for name in payloads:
        with pytest.raises(FileNotFoundError):
            File.at(
                f"{src_uri}/{encoded_prefix}/{name}",
                session=ctc.session,
            ).read()


def test_parse_listing_uri(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    dataset_name, listing_uri, listing_path = parse_listing_uri(f"{ctc.src_uri}/dogs")
    assert dataset_name == _sanitize_ds_name(f"lst__{ctc.src_uri}/dogs/")
    assert listing_uri == f"{ctc.src_uri}/dogs/"
    if cloud_type == "file":
        assert listing_path == ""
    else:
        assert listing_path == "dogs/"


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_parse_listing_uri_with_glob(cloud_test_catalog):
    ctc = cloud_test_catalog
    dataset_name, listing_uri, listing_path = parse_listing_uri(f"{ctc.src_uri}/dogs/*")
    assert dataset_name == _sanitize_ds_name(f"lst__{ctc.src_uri}/dogs/")
    assert listing_uri == f"{ctc.src_uri}/dogs"
    assert listing_path == "dogs/*"


def test_listing_update_unchanged_keeps_old_version(test_session, tmp_dir):
    """update=True with unchanged files should keep old listing version and UUID."""
    (tmp_dir / "a.txt").write_text("hello")
    (tmp_dir / "b.txt").write_text("world")
    uri = tmp_dir.as_uri()

    dc.read_storage(uri, session=test_session).exec()

    ds_name, _, _, _ = get_listing(uri, test_session)
    catalog = test_session.catalog
    ds = catalog.get_dataset(ds_name, versions=None)
    assert ds.latest_version == "1.0.0"
    original_uuid = ds.get_version("1.0.0").uuid
    original_finished_at = ds.get_version("1.0.0").finished_at

    # Re-list with update=True — same files
    dc.read_storage(uri, session=test_session, update=True).exec()

    ds = catalog.get_dataset(ds_name, versions=None)
    assert ds.latest_version == "1.0.0"
    assert ds.get_version("1.0.0").uuid == original_uuid
    assert len(ds.versions) == 1
    # finished_at is bumped so the TTL window is refreshed on no-op re-save
    assert ds.get_version("1.0.0").finished_at > original_finished_at


def test_listing_refresh_when_expired_after_construction(
    test_session, tmp_dir, monkeypatch
):
    """Regression test for ``AssertionError: self.listing_fn`` raised from
    ``DatasetQuery.resolve_listing`` when the listing dataset exists at chain
    construction time but is detected as expired at execution time. The
    listing function must be attached unconditionally so the refresh path can
    invoke it.
    """
    (tmp_dir / "a.txt").write_text("hello")
    uri = tmp_dir.as_uri()

    dc.read_storage(uri, session=test_session).exec()

    chain = dc.read_storage(uri, session=test_session)
    monkeypatch.setattr("datachain.lib.listing.LISTING_TTL", -1)

    assert chain.count() == 1


def test_listing_update_changed_creates_new_version(test_session, tmp_dir):
    """update=True with changed files should create a new listing version."""
    (tmp_dir / "a.txt").write_text("hello")
    uri = tmp_dir.as_uri()

    dc.read_storage(uri, session=test_session).exec()

    ds_name, _, _, _ = get_listing(uri, test_session)
    catalog = test_session.catalog
    ds = catalog.get_dataset(ds_name, versions=None)
    original_uuid = ds.get_version("1.0.0").uuid

    # Add a file and re-list
    (tmp_dir / "b.txt").write_text("world")
    dc.read_storage(uri, session=test_session, update=True).exec()

    ds = catalog.get_dataset(ds_name, versions=None)
    assert ds.latest_version == "2.0.0"
    assert ds.get_version("2.0.0").uuid != original_uuid


def test_listing_update_unchanged_preserves_chain_hash(test_session, tmp_dir):
    """Chain hash should stay stable when listing content hasn't changed."""
    (tmp_dir / "a.txt").write_text("hello")
    uri = tmp_dir.as_uri()

    chain1 = dc.read_storage(uri, session=test_session)
    chain1._query.resolve_listing()
    hash1 = chain1._query.hash()

    # Re-list with update=True — same content
    chain2 = dc.read_storage(uri, session=test_session, update=True)
    chain2._query.resolve_listing()
    hash2 = chain2._query.hash()

    assert hash1 == hash2


def test_calc_fingerprint_order_independent(test_session):
    """Fingerprint should be the same regardless of row insertion order."""
    catalog = test_session.catalog
    ns = catalog.metastore.system_namespace_name
    proj = catalog.metastore.listing_project_name

    files1 = [File(path="a.jpg", etag="e1"), File(path="b.jpg", etag="e2")]
    files2 = [File(path="b.jpg", etag="e2"), File(path="a.jpg", etag="e1")]

    dc.read_values(file=files1, session=test_session).settings(
        namespace=ns, project=proj
    ).save("lst__fp-order1", listing=True)
    dc.read_values(file=files2, session=test_session).settings(
        namespace=ns, project=proj
    ).save("lst__fp-order2", listing=True)

    ds1 = catalog.get_dataset("lst__fp-order1", versions=None)
    ds2 = catalog.get_dataset("lst__fp-order2", versions=None)

    fp1 = calc_fingerprint(test_session, ds1, ds1.latest_version)
    fp2 = calc_fingerprint(test_session, ds2, ds2.latest_version)
    assert fp1 == fp2


def test_read_storage_single_local_file_deterministic_hash(test_session, tmp_dir):
    (tmp_dir / "a.txt").write_text("hello")
    uri = (tmp_dir / "a.txt").as_uri()

    h1 = dc.read_storage(uri, session=test_session)._query.hash()
    h2 = dc.read_storage(uri, session=test_session)._query.hash()
    assert h1 == h2


def test_read_storage_single_local_file_hash_changes_on_edit(test_session, tmp_dir):
    f = tmp_dir / "a.txt"
    f.write_text("hello")
    uri = f.as_uri()

    h1 = dc.read_storage(uri, session=test_session)._query.hash()

    f.write_text("hello world")
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10**9))

    h2 = dc.read_storage(uri, session=test_session)._query.hash()
    assert h1 != h2


def test_read_storage_single_file_deterministic_hash_across_clouds(
    cloud_test_catalog,
):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/description"

    h1 = dc.read_storage(uri, session=ctc.session)._query.hash()
    h2 = dc.read_storage(uri, session=ctc.session)._query.hash()
    assert h1 == h2


def test_read_storage_multiple_single_files_order_independent(test_session, tmp_dir):
    (tmp_dir / "a.txt").write_text("hello")
    (tmp_dir / "b.txt").write_text("world")
    uri_a = (tmp_dir / "a.txt").as_uri()
    uri_b = (tmp_dir / "b.txt").as_uri()

    h1 = dc.read_storage([uri_a, uri_b], session=test_session)._query.hash()
    h2 = dc.read_storage([uri_b, uri_a], session=test_session)._query.hash()
    assert h1 == h2


def test_read_storage_mixed_single_file_and_dir_deterministic(test_session, tmp_dir):
    sub = tmp_dir / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("hello")
    (sub / "b.txt").write_text("world")
    standalone = tmp_dir / "standalone.txt"
    standalone.write_text("solo")

    dir_uri = sub.as_uri()
    file_uri = standalone.as_uri()

    chain1 = dc.read_storage([dir_uri, file_uri], session=test_session)
    chain2 = dc.read_storage([dir_uri, file_uri], session=test_session)
    chain1._query.resolve_all_listings()
    chain2._query.resolve_all_listings()
    assert chain1._query.hash() == chain2._query.hash()
