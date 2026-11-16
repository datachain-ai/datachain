import os
from unittest.mock import patch

import pytest

import datachain as dc
from datachain.dataset import DatasetStatus
from datachain.error import ConcurrentDatasetModificationError, DatasetNotFoundError
from datachain.lib import listing


def _interleaved_refresh(session, folder, scenario):
    folder.mkdir()
    catalog = session.catalog
    warehouse = catalog.warehouse
    uri = folder.as_uri()
    name, *_ = listing.get_listing(uri, session, update=True)
    original_pages = listing.list_pages
    complete = catalog.complete_dataset_version
    stored = warehouse.insert_stored_rows
    roles = []
    returned = {}
    pending = None
    finished = False
    scheduled = False

    def invoke(role):
        roles.append(role)
        try:
            filename = {
                "base": "base.txt",
                "A": "old.txt",
                "B": "base.txt" if "unchanged" in scenario else "new.txt",
            }[role]
            for file in folder.iterdir():
                file.unlink()
            file = folder / filename
            file.write_text(filename)
            os.utime(file, ns=(1_700_000_000_000_000_000,) * 2)
            returned[role] = dc.read_storage(
                uri, session=session, update=True
            ).to_values("file.path")
        finally:
            roles.pop()

    def pages(uri, cache, client_config=None, encode=None):
        rows = [
            row for page in original_pages(uri, cache, client_config) for row in page
        ]
        if roles[-1] == "A" and scenario.startswith("slow_first"):
            invoke("B")
        yield encode(rows) if encode else rows

    def finish_pending():
        nonlocal finished
        if not finished:
            dataset, version, args, kwargs = pending
            complete(dataset, version, *args, **kwargs)
            finished = True

    def after_data(table):
        if (
            roles[-1] == "B"
            and "lst__" in table.name
            and scenario.endswith("a_first")
            and pending is not None
        ):
            finish_pending()

    def store(*args, **kwargs):
        result = stored(*args, **kwargs)
        after_data(args[0])
        return result

    def finalize(dataset, version, *args, **kwargs):
        nonlocal pending, scheduled
        if (
            dataset.name == name
            and roles[-1] == "A"
            and not scenario.startswith("slow_first")
            and not scheduled
        ):
            pending = dataset, version, args, kwargs
            scheduled = True
            invoke("B")
            finish_pending()
            return
        return complete(dataset, version, *args, **kwargs)

    with (
        patch.object(listing, "list_pages", pages),
        patch.object(catalog, "complete_dataset_version", finalize),
        patch.object(warehouse, "insert_stored_rows", store),
    ):
        invoke("base")
        invoke("A")
    dataset = catalog.get_dataset(name, versions=None)
    final = dc.read_dataset(
        name,
        version=dataset.latest_complete_version,
        namespace=dataset.project.namespace.name,
        project=dataset.project.name,
        session=session,
    ).to_values("file.path")
    return returned["B"], final


@pytest.mark.parametrize(
    "scenario, expected",
    [
        ("changed_a_first", "new.txt"),
        ("unchanged_a_first", "base.txt"),
        ("changed_b_first", "new.txt"),
        ("unchanged_b_first", "base.txt"),
        ("slow_first_changed", "new.txt"),
        ("slow_first_unchanged", "base.txt"),
    ],
)
def test_newer_refresh_remains_latest(test_session, tmp_path, scenario, expected):
    returned, final = _interleaved_refresh(test_session, tmp_path / "input", scenario)
    assert returned == [expected]
    assert final == [expected]


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


def test_listing_error_closes_pages_before_traceback_release(
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
    with pytest.raises(RuntimeError, match="database insertion failed") as error:
        dc.read_storage(tmp_path, session=test_session).to_values("file.path")
    assert error.value.__traceback__ is not None
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


def test_superseded_older_writer_does_not_extend_newer_snapshot_ttl(
    test_session, tmp_path, monkeypatch
):
    (tmp_path / "a.txt").write_text("a")
    dc.read_storage(tmp_path, session=test_session).to_values("file.path")
    catalog = test_session.catalog
    stored = catalog.warehouse.insert_stored_rows
    nested = False
    newer_finished_at = None
    name, *_ = listing.get_listing(tmp_path.as_uri(), test_session, update=True)

    def interleave(table, *args, **kwargs):
        nonlocal nested, newer_finished_at
        res = stored(table, *args, **kwargs)
        if "lst__" in table.name and not nested:
            nested = True
            dc.read_storage(tmp_path, session=test_session, update=True).to_values(
                "file.path"
            )
            ds = catalog.get_dataset(name, versions=None)
            newer_finished_at = ds.get_version(ds.latest_complete_version).finished_at
        return res

    monkeypatch.setattr(catalog.warehouse, "insert_stored_rows", interleave)
    dc.read_storage(tmp_path, session=test_session, update=True).to_values("file.path")
    ds = catalog.get_dataset(name, versions=None)
    assert ds.get_version(ds.latest_complete_version).finished_at == newer_finished_at
