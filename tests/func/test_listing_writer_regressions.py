import os

import pytest

import datachain as dc
from datachain.dataset import DatasetStatus
from datachain.error import ConcurrentDatasetModificationError, DatasetNotFoundError
from datachain.lib import listing


@pytest.fixture
def folder(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    return folder


def _refresh(session, folder, name):
    """Leave only the file ``name`` in ``folder`` and refresh its listing."""
    for file in folder.iterdir():
        file.unlink()
    (folder / name).write_text(name)
    os.utime(folder / name, ns=(1_700_000_000_000_000_000,) * 2)
    chain = dc.read_storage(folder.as_uri(), session=session, update=True)
    return chain.to_values("file.path")


def _listed(session, folder):
    return dc.read_storage(folder.as_uri(), session=session).to_values("file.path")


@pytest.mark.parametrize("newer", ["new.txt", "base.txt"], ids=["changed", "unchanged"])
def test_refresh_that_ends_during_an_older_scan_stays_latest(
    test_session, folder, monkeypatch, newer
):
    _refresh(test_session, folder, "base.txt")
    list_pages = listing.list_pages
    returned = []

    def scan_then_refresh(*args, **kwargs):
        pages = list(list_pages(*args, **kwargs))
        monkeypatch.setattr(listing, "list_pages", list_pages)
        returned.extend(_refresh(test_session, folder, newer))
        return pages

    monkeypatch.setattr(listing, "list_pages", scan_then_refresh)
    _refresh(test_session, folder, "old.txt")

    assert returned == [newer]
    assert _listed(test_session, folder) == [newer]


@pytest.mark.parametrize("newer", ["new.txt", "base.txt"], ids=["changed", "unchanged"])
def test_refresh_that_ends_before_an_older_one_publishes_stays_latest(
    test_session, folder, monkeypatch, newer
):
    _refresh(test_session, folder, "base.txt")
    catalog = test_session.catalog
    complete = catalog.complete_dataset_version
    returned = []

    def refresh_then_complete(*args, **kwargs):
        monkeypatch.setattr(catalog, "complete_dataset_version", complete)
        returned.extend(_refresh(test_session, folder, newer))
        return complete(*args, **kwargs)

    monkeypatch.setattr(catalog, "complete_dataset_version", refresh_then_complete)
    _refresh(test_session, folder, "old.txt")

    assert returned == [newer]
    assert _listed(test_session, folder) == [newer]


@pytest.mark.parametrize("newer", ["new.txt", "base.txt"], ids=["changed", "unchanged"])
def test_refresh_stays_latest_after_an_older_one_publishes_during_it(
    test_session, folder, monkeypatch, newer
):
    _refresh(test_session, folder, "base.txt")
    catalog = test_session.catalog
    complete = catalog.complete_dataset_version
    store = catalog.warehouse.insert_stored_rows
    returned = []

    def refresh_and_complete_during_it(*args, **kwargs):
        def store_then_complete_older(*store_args, **store_kwargs):
            store(*store_args, **store_kwargs)
            monkeypatch.setattr(catalog.warehouse, "insert_stored_rows", store)
            complete(*args, **kwargs)

        monkeypatch.setattr(catalog, "complete_dataset_version", complete)
        monkeypatch.setattr(
            catalog.warehouse, "insert_stored_rows", store_then_complete_older
        )
        returned.extend(_refresh(test_session, folder, newer))

    monkeypatch.setattr(
        catalog, "complete_dataset_version", refresh_and_complete_during_it
    )
    _refresh(test_session, folder, "old.txt")

    assert returned == [newer]
    assert _listed(test_session, folder) == [newer]


@pytest.mark.parametrize("stage", ["create", "finalize"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_listing_failure_releases_reservation(
    test_session, tmp_path, monkeypatch, stage, error_type
):
    (tmp_path / "a.txt").write_text("a")
    catalog = test_session.catalog
    name, *_ = listing.get_listing(tmp_path.as_uri(), test_session, update=True)
    warehouse = catalog.warehouse
    if stage == "create":
        create = warehouse.create_dataset_rows_table

        def fail_create(table_name, *args, **kwargs):
            if "lst__" in table_name:
                raise error_type("database operation failed")
            return create(table_name, *args, **kwargs)

        monkeypatch.setattr(warehouse, "create_dataset_rows_table", fail_create)
    else:
        complete = catalog.complete_dataset_version

        def fail_complete(dataset, *args, **kwargs):
            if dataset.name == name:
                raise error_type("database operation failed")
            return complete(dataset, *args, **kwargs)

        monkeypatch.setattr(catalog, "complete_dataset_version", fail_complete)
    with pytest.raises(error_type, match="database operation failed"):
        dc.read_storage(tmp_path, session=test_session, update=True).to_values(
            "file.path"
        )
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, versions=None, include_incomplete=True)


def test_failed_refresh_does_not_return_older_cache(
    test_session, tmp_path, monkeypatch
):
    (tmp_path / "a.txt").write_text("a")
    dc.read_storage(tmp_path, session=test_session).to_values("file.path")
    (tmp_path / "b.txt").write_text("b")
    catalog = test_session.catalog
    complete = catalog.complete_dataset_version

    def removed(dataset, version, *args, **kwargs):
        other = catalog.get_dataset(dataset.name, versions=[version])
        catalog.remove_dataset_version(other, version, keep_metadata=False)
        return complete(dataset, version, *args, **kwargs)

    monkeypatch.setattr(catalog, "complete_dataset_version", removed)
    with pytest.raises(ConcurrentDatasetModificationError):
        dc.read_storage(tmp_path, session=test_session, update=True).to_values(
            "file.path"
        )


def test_first_page_failure_leaves_no_reservation(test_session, tmp_path, monkeypatch):
    events = []
    name, *_ = listing.get_listing(tmp_path.as_uri(), test_session, update=True)

    def missing(*args, **kwargs):
        try:
            raise FileNotFoundError("prefix disappeared")
            yield
        finally:
            events.append("closed")

    monkeypatch.setattr(listing, "list_pages", missing)
    with pytest.raises(FileNotFoundError, match="prefix disappeared"):
        dc.read_storage(tmp_path, session=test_session, update=True).to_values(
            "file.path"
        )
    assert events == ["closed"]
    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.get_dataset(name, versions=None, include_incomplete=True)


def test_read_storage_after_insert_failure_stops_listing(
    test_session, tmp_path, monkeypatch
):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    original_pages = listing.list_pages
    insert = test_session.catalog.warehouse.insert_block
    events = []

    def pages(uri, cache, client_config=None, encode=None):
        rows = [
            row for page in original_pages(uri, cache, client_config) for row in page
        ]
        events.append("open")
        try:
            for row in rows:
                yield encode([row]) if encode else [row]
        finally:
            events.append("closed")

    def failed(*args, **kwargs):
        if events == ["open"]:
            raise RuntimeError("database insertion failed")
        return insert(*args, **kwargs)

    monkeypatch.setattr(listing, "list_pages", pages)
    monkeypatch.setattr(listing, "LISTING_BATCH_SIZE", 1)
    monkeypatch.setattr(test_session.catalog.warehouse, "insert_block", failed)
    with pytest.raises(RuntimeError, match="database insertion failed"):
        dc.read_storage(tmp_path, session=test_session).to_values("file.path")
    assert events == ["open", "closed"]


def test_failed_listing_does_not_release_another_owner(
    test_session, tmp_path, monkeypatch
):
    (tmp_path / "a.txt").write_text("a")
    catalog = test_session.catalog
    owner = test_session.get_or_create_job().id
    replacement = catalog.metastore.create_job("replacement", "")
    name, *_ = listing.get_listing(tmp_path.as_uri(), test_session, update=True)

    def replaced(*args, **kwargs):
        dataset = catalog.get_dataset(name, versions=None)
        version = dataset.versions[0].version
        changed = catalog.metastore.update_dataset_version(
            dataset,
            version,
            expected_status=DatasetStatus.CREATED,
            expected_job_id=owner,
            job_id=replacement,
        )
        assert changed is not None
        raise RuntimeError("former owner failed")

    monkeypatch.setattr(catalog.warehouse, "insert_stored_rows", replaced)
    with pytest.raises(RuntimeError, match="former owner failed"):
        dc.read_storage(tmp_path, session=test_session, update=True).to_values(
            "file.path"
        )
    version = catalog.get_dataset(name, versions=None).versions[0]
    assert version.status == DatasetStatus.CREATED
    assert version.job_id == replacement


def test_unchanged_refresh_with_removed_reservation_raises(
    test_session, tmp_path, monkeypatch
):
    (tmp_path / "a.txt").write_text("a")
    dc.read_storage(tmp_path, session=test_session).to_values("file.path")
    catalog = test_session.catalog
    stored = catalog.warehouse.insert_stored_rows
    name, *_ = listing.get_listing(tmp_path.as_uri(), test_session, update=True)

    def remove_after_store(table, *args, **kwargs):
        res = stored(table, *args, **kwargs)
        if "lst__" in table.name:
            ds = catalog.get_dataset(name, versions=None)
            created = [v for v in ds.all_versions if v.status == DatasetStatus.CREATED]
            for v in created:
                catalog.remove_dataset_version(ds, v.version, keep_metadata=False)
        return res

    monkeypatch.setattr(catalog.warehouse, "insert_stored_rows", remove_after_store)
    with pytest.raises(ConcurrentDatasetModificationError):
        dc.read_storage(tmp_path, session=test_session, update=True).to_values(
            "file.path"
        )


def test_heartbeat_and_mid_scan_removal_abort_cleanly(
    test_session, tmp_path, monkeypatch
):
    (tmp_path / "a.txt").write_text("a")
    catalog = test_session.catalog
    name, *_ = listing.get_listing(tmp_path.as_uri(), test_session, update=True)
    monkeypatch.setattr(listing, "RESERVATION_TOUCH_INTERVAL", -1)

    # 1. Heartbeat touch succeeds when reservation is active
    assert dc.read_storage(tmp_path, session=test_session, update=True).to_values(
        "file.path"
    ) == ["a.txt"]

    # 2. Heartbeat touch raises ConcurrentDatasetModificationError if reservation was removed
    orig_pages = listing.list_pages

    def remove_before_touch(*args, **kwargs):
        pages = list(orig_pages(*args, **kwargs))
        ds = catalog.get_dataset(name, versions=None)
        for v in [v for v in ds.all_versions if v.status == DatasetStatus.CREATED]:
            catalog.remove_dataset_version(ds, v.version, keep_metadata=False)
        return pages

    monkeypatch.setattr(listing, "list_pages", remove_before_touch)
    with pytest.raises(ConcurrentDatasetModificationError, match="was removed"):
        dc.read_storage(tmp_path, session=test_session, update=True).to_values(
            "file.path"
        )
