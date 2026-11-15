import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.client import Client
from datachain.lib import listing
from datachain.lib.file import File
from datachain.lib.listing import (
    LISTING_BATCH_SIZE,
    _sanitize_ds_name,
    get_listing,
    list_pages,
    parse_listing_uri,
    rows_digest,
)
from tests.data import ENTRIES


def list_bucket(uri, cache, client_config=None):
    """A gen function of the rows of ``list_pages``."""

    def list_func() -> Iterator[File]:
        for page in list_pages(uri, cache, client_config):
            yield from page

    return list_func


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


def _digest(storage: str, *files: File) -> int:
    with_etag = not Client.get_implementation(storage).PINNED_VERSION_IS_IMMUTABLE
    return rows_digest([listing.file_row(f) for f in files], with_etag)


def test_listing_digest_counts_duplicates_in_any_order():
    a, b = File(path="a.jpg", etag="e1"), File(path="b.jpg", etag="e2")
    assert _digest("s3://bucket", a, a, b) == _digest("s3://bucket", b, a, a)
    assert _digest("s3://bucket", a, a, b) != _digest("s3://bucket", a, b, b)
    assert _digest("s3://bucket", a, a, b) != _digest("s3://bucket", a, b)


def _save_listing(session, uri: str, file: File):
    """Save ``file`` as a listing of ``uri``; return the listing dataset."""
    name, _, _ = parse_listing_uri(uri)

    def pages(uri, cache, client_config, encode):
        return [encode([listing.file_row(file)])]

    with patch.object(listing, "list_pages", side_effect=pages):
        listing.write_listing(uri, name, session, {})
    return session.catalog.get_dataset(name, versions=None)


def _etags(session, name: str, version: str) -> list:
    metastore = session.catalog.metastore
    chain = dc.read_dataset(
        name,
        namespace=metastore.system_namespace_name,
        project=metastore.listing_project_name,
        version=version,
        session=session,
    )
    return sorted(chain.to_values("file.etag"))


@pytest.mark.parametrize(
    "storage,etag_counts",
    [("s3://bucket", False), ("gs://bucket", False), ("az://container", True)],
    ids=["s3", "gs", "azure"],
)
def test_listing_digest_etag_counts_only_where_versions_change(storage, etag_counts):
    fp1 = _digest(storage, File(path="x.jpg", version="v1", etag="e1"))
    fp2 = _digest(storage, File(path="x.jpg", version="v1", etag="e2"))
    assert (fp1 != fp2) == etag_counts


def test_listing_digest_changes_when_version_stops_being_latest():
    fp1 = _digest("s3://bucket", File(path="x.jpg", version="v1"))
    fp2 = _digest("s3://bucket", File(path="x.jpg", version="v1", is_latest=False))
    assert fp1 != fp2


@pytest.mark.parametrize(
    "storage",
    ["s3://bucket", "gs://bucket", "az://container"],
    ids=["s3", "gs", "azure"],
)
def test_listing_digest_changes_when_size_changes(storage):
    fp1 = _digest(storage, File(path="x.jpg", version="v1", etag="e1", size=10))
    fp2 = _digest(storage, File(path="x.jpg", version="v1", etag="e1", size=20))
    assert fp1 != fp2


def test_listing_digest_counts_the_source():
    file = File(path="x.jpg", version="v1", etag="e1")
    a, b = File(source="s3://a", **file.model_dump(include={"path", "version"})), file
    assert _digest("s3://bucket", a) != _digest("s3://bucket", b)


def test_listing_without_digest_is_not_reused(test_session):
    uri = "s3://bucket/data/"
    file = File(path="x.jpg", version="v1", etag="e1")
    ds = _save_listing(test_session, uri, file)
    assert ds.get_version("1.0.0").content_hash.startswith(listing.LISTING_DIGEST)
    test_session.catalog.metastore.update_dataset_version(
        ds, "1.0.0", content_hash=None
    )

    ds = _save_listing(test_session, uri, file)

    assert [v.version for v in ds.versions] == ["1.0.0", "2.0.0"]


def test_long_listing_keeps_its_reservation_young(test_session):
    # gc removes a jobless CREATED version older than an hour: a listing that
    # runs longer restarts its version's age as it goes
    uri = "s3://bucket/data/"
    name, _, _ = parse_listing_uri(uri)
    metastore = test_session.catalog.metastore
    clock = [0.0]
    hours_ago = datetime.now(UTC) - timedelta(hours=2)

    def pages(uri, cache, client_config, encode):
        yield encode([listing.file_row(File(path="a.jpg", etag="e1"))])
        ds = test_session.catalog.get_dataset(name, versions=None)
        dv = metastore._datasets_versions
        metastore.db.execute(  # two hours pass
            dv.update().where(dv.c.id == ds.versions[0].id).values(created_at=hours_ago)
        )
        clock[0] += 2 * 3600
        yield encode([listing.file_row(File(path="b.jpg", etag="e2"))])

    with (
        patch.object(listing, "list_pages", side_effect=pages),
        patch.object(listing.time, "monotonic", side_effect=lambda: clock[0]),
    ):
        listing.write_listing(uri, name, test_session, {})

    version = test_session.catalog.get_dataset(name, versions=None).versions[0]
    assert version.created_at > hours_ago + timedelta(hours=1)
    assert _etags(test_session, name, version.version) == ["e1", "e2"]


def test_listing_creates_no_job(test_session, tmp_dir):
    (tmp_dir / "a.txt").write_text("a")
    metastore = test_session.catalog.metastore

    def jobs() -> int:
        count = sa.select(sa.func.count()).select_from(metastore._jobs)
        return next(iter(metastore.db.execute(count)))[0]

    before = jobs()  # a test session may run inside a job already
    for update in (False, True):
        dc.read_storage(tmp_dir.as_uri(), session=test_session, update=update).exec()

    assert jobs() == before


@pytest.mark.parametrize("base", [True, False], ids=["refresh", "first"])
def test_delayed_listing_does_not_publish_over_a_newer_one(test_session, base):
    # A starts over 1.0.0 (or over nothing) and lists `older`; B publishes
    # `newer` before A ends. A's version is dropped: B's stays the latest.
    uri = "s3://bucket/data/"
    name, _, _ = parse_listing_uri(uri)
    older, newer = File(path="old.jpg", etag="e1"), File(path="new.jpg", etag="e2")
    if base:
        _save_listing(test_session, uri, File(path="base.jpg", etag="e0"))

    def delayed(uri, cache, client_config, encode):
        yield encode([listing.file_row(older)])
        _save_listing(test_session, uri, newer)

    with patch.object(listing, "list_pages", side_effect=delayed):
        listing.write_listing(uri, name, test_session, {})

    ds = test_session.catalog.get_dataset(name, versions=None)
    latest = ds.latest_complete_version
    assert len(ds.versions) == (2 if base else 1)  # A's version is gone
    assert _etags(test_session, name, latest) == ["e2"]


@pytest.mark.parametrize("base", [True, False], ids=["refresh", "first"])
def test_listing_publish_races_a_newer_one(test_session, base):
    # B publishes `newer` after A has listed `older` and decided to publish,
    # right before A's status UPDATE runs: A's guarded UPDATE must not apply.
    uri = "s3://bucket/data/"
    name, _, _ = parse_listing_uri(uri)
    older, newer = File(path="old.jpg", etag="e1"), File(path="new.jpg", etag="e2")
    if base:
        _save_listing(test_session, uri, File(path="base.jpg", etag="e0"))
    metastore = test_session.catalog.metastore
    publish = metastore.update_dataset_status
    raced = []

    def racing_publish(*args, **kwargs):
        if not raced:
            raced.append(True)
            _save_listing(test_session, uri, newer)  # B, between check and publish
        return publish(*args, **kwargs)

    with patch.object(metastore, "update_dataset_status", racing_publish):
        _save_listing(test_session, uri, older)  # A

    ds = test_session.catalog.get_dataset(name, versions=None)
    assert raced
    assert len(ds.versions) == (2 if base else 1)  # A's version is gone
    assert _etags(test_session, name, ds.latest_complete_version) == ["e2"]


def test_unchanged_refresh_leaves_a_version_saved_meanwhile(test_session):
    # A starts with 1.0.0 as the latest; B saves 2.0.0 before A ends. A equals
    # 1.0.0: it is dropped, and neither version is touched.
    uri = "s3://bucket/data/"
    name, _, _ = parse_listing_uri(uri)
    old, new = File(path="x.jpg", etag="e1"), File(path="y.jpg", etag="e2")
    first = _save_listing(test_session, uri, old).get_version("1.0.0")

    def racing(uri, cache, client_config, encode):
        _save_listing(test_session, uri, new)
        return iter([encode([listing.file_row(old)])])

    with patch.object(listing, "list_pages", side_effect=racing):
        listing.write_listing(uri, name, test_session, {})

    ds = test_session.catalog.get_dataset(name, versions=None)
    assert [v.version for v in ds.versions] == ["1.0.0", "2.0.0"]
    assert ds.get_version("1.0.0").finished_at == first.finished_at
    assert _etags(test_session, name, "2.0.0") == ["e2"]


@pytest.mark.parametrize(
    "storage,version,reused",
    [
        ("s3://bucket", "v1", True),
        ("gs://bucket", "v1", True),
        ("az://container", "v1", False),
        ("s3://bucket", "", False),
    ],
    ids=["s3", "gs", "azure", "unversioned"],
)
def test_listing_save_reuses_version_only_when_etag_does_not_count(
    test_session, storage, version, reused
):
    uri = f"{storage}/data/"
    name, _, _ = parse_listing_uri(uri)
    ds = _save_listing(
        test_session, uri, File(path="x.jpg", version=version, etag="e1")
    )
    original_uuid = ds.get_version("1.0.0").uuid

    ds = _save_listing(
        test_session, uri, File(path="x.jpg", version=version, etag="e2")
    )

    assert len(ds.versions) == (1 if reused else 2)
    assert ds.get_version("1.0.0").uuid == original_uuid
    assert _etags(test_session, name, "1.0.0") == ["e1"]


@pytest.mark.parametrize(
    "tree", [{"deleted": {"gone": "one", "kept": "two"}}], indirect=True
)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_listing_update_drops_deleted_file_on_versioned_storage(cloud_test_catalog):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/deleted/"
    dc.read_storage(uri, session=ctc.session).exec()

    client = ctc.catalog.get_client(ctc.src_uri)
    client.fs.rm(client.get_uri("deleted/gone"))

    paths = dc.read_storage(uri, session=ctc.session, update=True).to_values(
        "file.path"
    )
    assert paths == ["deleted/kept"]


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


@pytest.mark.parametrize("settings", [None, {"batch_size": 7}])
def test_listing_insert_batch_size(test_session, tmp_dir, mocker, settings):
    # the user's batch_size tunes their UDFs, not the listing
    (tmp_dir / "a.txt").write_text("hello")
    spy = mocker.spy(test_session.catalog.warehouse, "insert_stored_rows")

    dc.read_storage(tmp_dir.as_uri(), session=test_session, settings=settings).exec()

    assert spy.call_args.args[-1] == LISTING_BATCH_SIZE
