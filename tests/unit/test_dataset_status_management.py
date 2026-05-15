"""Tests for dataset status management and failed version cleanup."""

from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.data_storage import JobStatus
from datachain.dataset import DatasetRecord, DatasetStatus
from datachain.error import DatasetNotFoundError
from datachain.job import Job
from datachain.lib.dc.datasets import (
    datasets,
    delete_dataset,
    move_dataset,
    read_dataset,
)
from datachain.lib.listing import LISTING_PREFIX
from datachain.query.session import Session
from datachain.sql.types import String


@pytest.fixture
def job(test_session) -> Job:
    return test_session.get_or_create_job()


@pytest.fixture
def dataset_created(test_session, job) -> DatasetRecord:
    # Create a dataset version with CREATED status
    return test_session.catalog.create_dataset(
        "ds_created", columns=(sa.Column("name", String),), job_id=job.id
    )


@pytest.fixture
def dataset_failed(test_session, job) -> DatasetRecord:
    # Create a dataset version with FAILED status
    dataset = test_session.catalog.create_dataset(
        "ds_failed", columns=(sa.Column("name", String),), job_id=job.id
    )
    return test_session.catalog.metastore.update_dataset_status(
        dataset, DatasetStatus.FAILED, version=dataset.latest_version
    )


@pytest.fixture
def dataset_complete(test_session, job) -> DatasetRecord:
    # Create a dataset version with COMPLETE status
    ds = dc.read_values(value=["val1", "val2"], session=test_session).save(
        "ds_complete"
    )
    return ds.dataset  # type: ignore[return-value]


def test_mark_job_dataset_versions_as_failed(test_session, job, dataset_created):
    """Test that mark_job_dataset_versions_as_failed marks versions as FAILED."""
    # Verify initial status is CREATED
    dataset = test_session.catalog.get_dataset(dataset_created.name, versions=None)
    dataset_version = dataset.get_version(dataset.latest_version)
    assert dataset_version.status == DatasetStatus.CREATED
    assert dataset_version.job_id == job.id

    # Mark dataset versions as failed
    test_session.catalog.metastore.mark_job_dataset_versions_as_failed(job.id)

    # Verify status is now FAILED
    dataset = test_session.catalog.get_dataset(dataset_created.name, versions=None)
    dataset_version = dataset.get_version(dataset.latest_version)
    assert dataset_version.status == DatasetStatus.FAILED
    assert dataset_version.finished_at is not None


def test_mark_job_dataset_versions_as_failed_skips_complete(
    test_session, job, dataset_complete
):
    """Test that mark_job_dataset_versions_as_failed skips COMPLETE versions."""
    # Verify initial status is COMPLETE
    dataset = test_session.catalog.get_dataset(dataset_complete.name, versions=None)
    dataset_version = dataset.get_version(dataset_complete.latest_version)
    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.job_id == job.id

    # Mark dataset versions as failed
    test_session.catalog.metastore.mark_job_dataset_versions_as_failed(job.id)

    # Verify COMPLETE status is unchanged
    dataset = test_session.catalog.get_dataset(dataset_complete.name, versions=None)
    dataset_version = dataset.get_version(dataset_complete.latest_version)
    assert dataset_version.status == DatasetStatus.COMPLETE


def test_finalize_job_as_failed_removes_incomplete_dataset_versions(
    test_session, job, dataset_created, dataset_failed, dataset_complete
):
    """
    Test that _finalize_job_as_failed marks dataset versions as FAILED and removes
    them right away.
    """
    from datachain.query.session import Session

    # Set up Session state as if job is running
    Session._CURRENT_JOB = job
    Session._OWNS_JOB = True
    Session._JOB_STATUS = JobStatus.RUNNING

    # Simulate job failure
    try:
        raise RuntimeError("test error")
    except RuntimeError as e:
        test_session._finalize_job_as_failed(type(e), e, e.__traceback__)

    # Verify job is marked as FAILED
    db_job = test_session.catalog.metastore.get_job(job.id)
    assert db_job.status == JobStatus.FAILED

    # Verify dataset version is marked as FAILED and removed
    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.get_dataset(dataset_failed.name)

    # Verify dataset version is marked as FAILED and removed
    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.get_dataset(dataset_created.name)

    # Verify dataset version is left since it's completed
    test_session.catalog.get_dataset(dataset_complete.name)


def test_status_filtering_hides_non_complete_versions(
    test_session, job, dataset_created, dataset_failed, dataset_complete
):
    """Test that non-COMPLETE dataset versions are hidden from queries."""
    # Test with include_incomplete=False (what public API/CLI uses)
    datasets = list(test_session.catalog.ls_datasets())
    dataset_names = {d.name for d in datasets}

    # Only COMPLETE dataset should be visible
    assert dataset_complete.name in dataset_names
    assert dataset_created.name not in dataset_names
    assert dataset_failed.name not in dataset_names


def test_get_dataset_versions_to_clean(
    test_session, job, dataset_created, dataset_failed, dataset_complete
):
    """Test get_dataset_versions_to_clean."""
    # Mark job as failed
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    # Get failed versions to clean
    to_clean = test_session.catalog.metastore.get_dataset_versions_to_clean()

    # Should return CREATED and FAILED datasets, not COMPLETE
    cleaned_names = {dataset.name for dataset, _ in to_clean}
    assert dataset_created.name in cleaned_names
    assert dataset_failed.name in cleaned_names
    assert dataset_complete.name not in cleaned_names

    # Verify each tuple contains dataset and version
    for dataset, version in to_clean:
        assert version is not None
        assert len(dataset.versions) == 1


def test_get_dataset_versions_to_clean_skips_running_jobs(
    test_session, job, dataset_created
):
    """Test that gc skips versions whose job is still running."""
    # Job is RUNNING — its versions should NOT be returned
    to_clean = test_session.catalog.metastore.get_dataset_versions_to_clean()
    assert dataset_created.name not in {ds.name for ds, _ in to_clean}

    # Mark job as complete — now it should be returned
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.COMPLETE)
    to_clean = test_session.catalog.metastore.get_dataset_versions_to_clean()
    assert dataset_created.name in {ds.name for ds, _ in to_clean}


def test_get_dataset_versions_to_clean_scoped_to_job(
    test_session, job, dataset_created
):
    """Test that get_dataset_versions_to_clean with job_id scopes to that job."""
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)
    to_clean = test_session.catalog.metastore.get_dataset_versions_to_clean(
        job_id=job.id
    )
    assert dataset_created.name in {ds.name for ds, _ in to_clean}

    # Non-existent job_id returns nothing
    to_clean = test_session.catalog.metastore.get_dataset_versions_to_clean(
        job_id="nonexistent"
    )
    assert len(to_clean) == 0


def test_remove_dataset_versions_bulk(
    test_session, job, dataset_created, dataset_failed
):
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)
    ds_c = test_session.catalog.get_dataset(dataset_created.name, versions=None)
    ds_f = test_session.catalog.get_dataset(dataset_failed.name, versions=None)
    id_c = ds_c.get_version(ds_c.latest_version).id
    id_f = ds_f.get_version(ds_f.latest_version).id

    n = test_session.catalog.remove_dataset_versions(version_ids=[id_c, id_f])
    assert n == 2
    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.get_dataset(dataset_created.name)
    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.get_dataset(dataset_failed.name)


def test_remove_dataset_versions_job_id_filter(test_session, job, dataset_created):
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)
    ds = test_session.catalog.get_dataset(dataset_created.name, versions=None)
    vid = ds.get_version(ds.latest_version).id

    assert (
        test_session.catalog.remove_dataset_versions(
            version_ids=[vid], job_id="wrong-job-id"
        )
        == 0
    )
    test_session.catalog.get_dataset(dataset_created.name)

    assert (
        test_session.catalog.remove_dataset_versions(version_ids=[vid], job_id=job.id)
        == 1
    )
    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.get_dataset(dataset_created.name)


def test_get_dataset_versions_to_clean_finds_no_job_id(test_session):
    """Test that gc finds stale versions with no job_id (e.g. from pull_dataset)."""

    dataset = test_session.catalog.create_dataset(
        "ds_orphaned_pull",
        columns=(sa.Column("name", String),),
        job_id=None,  # Not enough alone, can be taken from DATACHAIN_JOB_ID also
    )

    # Force job_id to NULL in the DB — bypasses the `or os.getenv()` fallback
    # that may fill it in when DATACHAIN_JOB_ID is set in the test env.
    dv = test_session.catalog.metastore._datasets_versions
    test_session.catalog.metastore.db.execute(
        dv.update().where(dv.c.dataset_id == dataset.id).values(job_id=None)
    )

    # Freshly created — should NOT be returned (protects in-flight pulls)
    to_clean = test_session.catalog.metastore.get_dataset_versions_to_clean()
    assert dataset.name not in {ds.name for ds, _ in to_clean}

    # Backdate created_at to exceed the staleness threshold
    test_session.catalog.metastore.db.execute(
        dv.update()
        .where(dv.c.dataset_id == dataset.id)
        .values(created_at=datetime.now(timezone.utc) - timedelta(hours=2))
    )

    # Now it should be returned
    to_clean = test_session.catalog.metastore.get_dataset_versions_to_clean()
    assert dataset.name in {ds.name for ds, _ in to_clean}


def test_cleanup_dataset_versions(test_session, job, dataset_failed):
    """Test cleanup_dataset_versions removes datasets and returns IDs."""
    # Mark job as failed
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    # Cleanup failed versions
    num_removed = test_session.catalog.cleanup_dataset_versions()

    # Should return the cleaned version ID
    assert num_removed == 1

    # Verify dataset version is removed
    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.get_dataset(dataset_failed.name)


def test_save_sets_complete_status_at_end(test_session, dataset_complete):
    """Test that save() sets COMPLETE status only after all operations."""
    # Verify status is COMPLETE
    dataset_version = dataset_complete.get_version(dataset_complete.latest_version)
    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.finished_at is not None

    # Verify all operations completed (num_objects set, etc.)
    assert dataset_version.num_objects == 2


def test_public_api_datasets_filters_non_complete(
    test_session, dataset_created, dataset_failed, dataset_complete
):
    """Test that dc.datasets() filters out non-COMPLETE datasets."""
    ds_chain = datasets(session=test_session, column="dataset")
    dataset_names = {ds.name for (ds,) in ds_chain.to_iter("dataset")}

    assert dataset_complete.name in dataset_names, "COMPLETE dataset should be visible"
    assert dataset_created.name not in dataset_names, "CREATED dataset should be hidden"
    assert dataset_failed.name not in dataset_names, "FAILED dataset should be hidden"


@pytest.mark.parametrize("is_studio", [True])
def test_public_api_read_dataset_rejects_non_complete(test_session, studio_job):
    """Test that dc.read_dataset() rejects non-COMPLETE datasets."""
    ds_created = test_session.catalog.create_dataset(
        "ds_created_read", columns=(sa.Column("name", String),), job_id=studio_job
    )
    ds_failed = test_session.catalog.create_dataset(
        "ds_failed_read", columns=(sa.Column("name", String),), job_id=studio_job
    )
    test_session.catalog.metastore.update_dataset_status(
        ds_failed, DatasetStatus.FAILED, version=ds_failed.latest_version
    )

    # Should raise error for CREATED dataset
    with pytest.raises(DatasetNotFoundError):
        read_dataset(ds_created.name, session=test_session)

    # Should raise error for FAILED dataset
    with pytest.raises(DatasetNotFoundError):
        read_dataset(ds_failed.name, session=test_session)


def test_public_api_delete_dataset_rejects_non_complete(
    test_session, dataset_created, dataset_failed
):
    """Test that dc.delete_dataset() rejects non-COMPLETE datasets."""
    # Should raise error for CREATED dataset
    with pytest.raises(DatasetNotFoundError):
        delete_dataset(dataset_created.name, session=test_session)

    # Should raise error for FAILED dataset
    with pytest.raises(DatasetNotFoundError):
        delete_dataset(dataset_failed.name, session=test_session)


def test_public_api_move_dataset_rejects_non_complete(
    test_session, dataset_created, dataset_failed
):
    """Test that dc.move_dataset() rejects non-COMPLETE datasets."""
    # Should raise error for CREATED dataset
    with pytest.raises(DatasetNotFoundError):
        move_dataset(dataset_created.name, "new_name_created", session=test_session)

    # Should raise error for FAILED dataset
    with pytest.raises(DatasetNotFoundError):
        move_dataset(dataset_failed.name, "new_name_failed", session=test_session)


@pytest.mark.parametrize(
    "job_status, should_clean",
    [
        (JobStatus.FAILED, True),
        (JobStatus.RUNNING, False),
    ],
    ids=["finished-job", "running-job"],
)
def test_cleanup_session_dataset_versions(test_session, job, job_status, should_clean):
    """Test that cleanup_dataset_versions also cleans session_* datasets."""
    ds = dc.read_values(value=["a", "b"], session=test_session).save(
        "session_test_abc123"
    )
    test_session.catalog.metastore.set_job_status(job.id, job_status)

    num_removed = test_session.catalog.cleanup_dataset_versions()

    if should_clean:
        assert num_removed >= 1
        with pytest.raises(DatasetNotFoundError):
            test_session.catalog.get_dataset(ds.dataset.name)
    else:
        assert num_removed == 0
        test_session.catalog.get_dataset(ds.dataset.name)  # still exists


@pytest.fixture
def dataset_marked_for_removal(test_session, job) -> DatasetRecord:
    ds = dc.read_values(value=["v1"], session=test_session).save(
        "ds_marked_for_removal"
    )
    dataset = ds.dataset
    assert dataset is not None
    dv = test_session.catalog.metastore._datasets_versions
    test_session.catalog.metastore.db.execute(
        dv.update()
        .where(dv.c.dataset_id == dataset.id)
        .values(status=DatasetStatus.REMOVING)
    )
    return test_session.catalog.get_dataset(dataset.name, include_incomplete=True)


def test_get_dataset_versions_to_clean_includes_marked_for_removal(
    test_session, job, dataset_marked_for_removal
):
    to_clean = test_session.catalog.metastore.get_dataset_versions_to_clean()
    assert dataset_marked_for_removal.name not in {ds.name for ds, _ in to_clean}

    test_session.catalog.metastore.set_job_status(job.id, JobStatus.COMPLETE)
    to_clean = test_session.catalog.metastore.get_dataset_versions_to_clean()
    assert dataset_marked_for_removal.name in {ds.name for ds, _ in to_clean}


def test_cleanup_dataset_versions_removes_marked_for_removal(
    test_session, job, dataset_marked_for_removal
):
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.COMPLETE)

    test_session.catalog.cleanup_dataset_versions()

    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.get_dataset(dataset_marked_for_removal.name)


# ---------------------------------------------------------------------------
# Soft-delete (REMOVED) behavior
# ---------------------------------------------------------------------------


def _find_removed(ds: DatasetRecord, display_version: str):
    """Find a REMOVED tombstone by its original semver. The actual `version`
    column was mangled with REMOVED_VERSION_SUFFIX to free the slot."""
    for v in ds.versions:
        if v.status == DatasetStatus.REMOVED and v.display_version == display_version:
            return v
    return None


def test_soft_delete_keeps_version_row_and_drops_rows_table(
    test_session, dataset_complete
):
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    version = dataset_complete.latest_version
    rows_table = warehouse.dataset_table_name(dataset_complete, version)
    assert warehouse.db.has_table(rows_table)

    catalog.remove_dataset_version(dataset_complete, version)

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    removed = _find_removed(ds, version)
    assert removed is not None
    assert removed.status == DatasetStatus.REMOVED
    assert removed.removed_at is not None
    assert not warehouse.db.has_table(rows_table)


def test_soft_delete_preserves_dependencies(test_session, dataset_complete):
    catalog = test_session.catalog
    metastore = catalog.metastore

    src_version = dataset_complete.latest_version
    dep_chain = dc.read_dataset(dataset_complete.name, session=test_session).save(
        "ds_dependent"
    )
    dep_ds = dep_chain.dataset
    assert dep_ds is not None
    dep_version = dep_ds.latest_version
    assert len(metastore.get_direct_dataset_dependencies(dep_ds, dep_version)) == 1

    catalog.remove_dataset_version(dataset_complete, src_version)

    deps = metastore.get_direct_dataset_dependencies(dep_ds, dep_version)
    assert len(deps) == 1
    assert deps[0] is not None


def test_soft_delete_is_idempotent(test_session, dataset_complete):
    catalog = test_session.catalog
    version = dataset_complete.latest_version

    catalog.remove_dataset_version(dataset_complete, version)
    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    first = _find_removed(ds, version)
    assert first is not None
    first_removed_at = first.removed_at

    # Second call sees no live version "1.0.0" (it was mangled) → no-op.
    catalog.remove_dataset_version(ds, version)
    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    assert len([v for v in ds.versions if v.status == DatasetStatus.REMOVED]) == 1
    assert _find_removed(ds, version).removed_at == first_removed_at


def test_save_after_soft_delete_reuses_version_slot(test_session, dataset_complete):
    """Once a version is soft-deleted, its (dataset_id, version) slot is
    freed via the mangle suffix so the next save can reclaim the same
    semver."""
    catalog = test_session.catalog
    name = dataset_complete.name
    first_version = dataset_complete.latest_version

    catalog.remove_dataset_version(dataset_complete, first_version)

    new_chain = dc.read_values(value=["new1"], session=test_session).save(name)
    assert new_chain.dataset is not None
    assert new_chain.dataset.latest_version == first_version

    # The old data lives on as a tombstone for lineage.
    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    assert _find_removed(ds, first_version) is not None


def test_latest_version_skips_removed(test_session, dataset_complete):
    catalog = test_session.catalog
    name = dataset_complete.name
    v1 = dataset_complete.latest_version

    dc.read_values(value=["v2-a", "v2-b"], session=test_session).save(name)
    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    v2 = ds.latest_version
    assert v2 != v1

    catalog.remove_dataset_version(ds, v2)

    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    assert ds.latest_version == v1


def test_listing_excludes_removed_only_dataset(test_session, dataset_complete):
    catalog = test_session.catalog
    name = dataset_complete.name

    catalog.remove_dataset_version(dataset_complete, dataset_complete.latest_version)

    ds_chain = datasets(session=test_session, column="dataset")
    assert name not in {ds.name for (ds,) in ds_chain.to_iter("dataset")}


def test_read_dataset_after_soft_delete_raises(
    test_session, dataset_complete, no_studio_dataset
):
    catalog = test_session.catalog
    name = dataset_complete.name

    catalog.remove_dataset_version(dataset_complete, dataset_complete.latest_version)

    with pytest.raises(DatasetNotFoundError):
        read_dataset(name, session=test_session)


def test_janitor_still_hard_deletes_created_version(test_session, job, dataset_created):
    """The cleanup path must still hard-delete non-COMPLETE versions — we
    don't want REMOVED tombstones piling up for failed/abandoned saves."""
    catalog = test_session.catalog
    catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    catalog.cleanup_dataset_versions()

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(dataset_created.name, include_incomplete=True)


def test_remove_non_complete_version_is_hard_delete(test_session, dataset_failed):
    catalog = test_session.catalog
    name = dataset_failed.name
    catalog.remove_dataset_version(dataset_failed, dataset_failed.latest_version)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def _make_completed_dataset(catalog, name: str, project=None):
    listing = name.startswith(LISTING_PREFIX)
    ds = catalog.create_dataset(
        name,
        project=project,
        columns=(sa.Column("name", String),),
        listing=listing,
    )
    catalog.metastore.update_dataset_version(
        ds, ds.latest_version, status=DatasetStatus.COMPLETE
    )
    return catalog.get_dataset(
        name,
        namespace_name=project.namespace.name if project else None,
        project_name=project.name if project else None,
        versions=None,
        include_incomplete=True,
    )


def test_listing_dataset_stays_on_hard_delete(test_session):
    """`lst__*` listing datasets must never become tombstones — they're
    internal cache, soft-deleting them serves nobody and wastes rows."""
    catalog = test_session.catalog
    metastore = catalog.metastore
    listing_project = metastore.get_project(
        metastore.listing_project_name, metastore.system_namespace_name
    )
    name = f"{LISTING_PREFIX}internal_test"
    ds = _make_completed_dataset(catalog, name, project=listing_project)

    catalog.remove_dataset_version(ds, ds.latest_version)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def test_session_dataset_stays_on_hard_delete(test_session):
    """`session_*` intermediates are throwaway too — hard delete only."""
    catalog = test_session.catalog
    name = f"{Session.DATASET_PREFIX}internal_test"
    ds = _make_completed_dataset(catalog, name)

    catalog.remove_dataset_version(ds, ds.latest_version)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)
