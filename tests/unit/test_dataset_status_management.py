"""Tests for dataset status management and failed version cleanup."""

from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain import semver
from datachain.data_storage import JobStatus
from datachain.dataset import (
    SESSION_DATASET_PREFIX,
    DatasetRecord,
    DatasetStatus,
)
from datachain.error import (
    ConcurrentDatasetModificationError,
    DataChainError,
    DatasetInvalidVersionError,
    DatasetNotFoundError,
)
from datachain.job import Job
from datachain.lib.dc.datasets import (
    datasets,
    delete_dataset,
    move_dataset,
    read_dataset,
)
from datachain.lib.listing import LISTING_PREFIX
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


def test_mark_job_dataset_versions_as_failed_preserves_tombstones(
    test_session, job, dataset_complete
):
    """A REMOVED tombstone created inside the failing job must survive
    `mark_job_dataset_versions_as_failed`. Otherwise the tombstone flips to
    FAILED and later gets dropped by GC, losing the removed record."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    catalog.remove_dataset(dataset_complete.name, version=version, keep_metadata=True)

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    assert _find_removed(ds, version) is not None

    catalog.metastore.mark_job_dataset_versions_as_failed(job.id)

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    tombstone = _find_removed(ds, version)
    assert tombstone is not None
    assert tombstone.status == DatasetStatus.REMOVED


def test_finalize_job_as_failed_removes_incomplete_dataset_versions(
    test_session, job, dataset_created, dataset_failed, dataset_complete
):
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
        assert len(dataset.all_versions) == 1


def test_get_dataset_versions_to_clean_skips_running_jobs(
    test_session, job, dataset_created
):
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


def test_remove_dataset_versions_explicit_keep_metadata_tombstones(
    test_session, dataset_complete
):
    """Bulk delete with ``keep_metadata=True`` must leave COMPLETE versions
    as tombstones and not remove the row - overrides the GC inference path."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = catalog.get_dataset(dataset_complete.name, versions=None)
    vid = ds.get_version(version).id

    n = catalog.remove_dataset_versions(version_ids=[vid], keep_metadata=True)
    assert n == 1

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    assert _find_removed(ds, version) is not None


def test_remove_dataset_version_already_tombstoned_returns_false(
    test_session, dataset_complete
):
    """Returning False on a no-op lets bulk callers count only versions
    they actually removed."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    catalog.remove_dataset(dataset_complete.name, version=version, keep_metadata=True)

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    assert catalog.remove_dataset_version(ds, version, keep_metadata=True) is False


def test_remove_dataset_version_returns_false_on_state_race(
    test_session, dataset_complete
):
    """When the DB status changed since the caller loaded the dataset, the
    guarded UPDATE inside `_claim_and_remove` refuses to overwrite and the
    call returns False - the concurrent caller wins the claim."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    stale_ds = catalog.get_dataset(dataset_complete.name, versions=None)

    catalog.metastore.update_dataset_version(
        dataset_complete, version, status=DatasetStatus.REMOVED
    )

    assert (
        catalog.remove_dataset_version(stale_ds, version, keep_metadata=True) is False
    )


@pytest.mark.parametrize("drop_rows_first", [False, True])
def test_remove_dataset_version_resumes_stuck_tombstone(
    test_session, dataset_complete, drop_rows_first
):
    """Same-intent retry on a version stuck at REMOVING + pending_metadata_drop=False
    finishes the tombstone (flips to REMOVED) instead of silently no-op'ing."""
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    version = dataset_complete.latest_version
    rows_table = warehouse.dataset_table_name(dataset_complete, version)
    if drop_rows_first:
        warehouse.drop_dataset_rows_table(dataset_complete, version)
    assert warehouse.db.has_table(rows_table) is not drop_rows_first

    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVING)

    assert catalog.remove_dataset_version(ds, version, keep_metadata=True) is True

    assert not warehouse.db.has_table(rows_table)
    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    finalized = _find_removed(ds, version)
    assert finalized is not None
    assert finalized.status == DatasetStatus.REMOVED


def test_create_dataset_explicit_removed_version_rejected(
    test_session, dataset_complete
):
    """The catalog-level guard must reject an explicit version that matches
    a REMOVED tombstone, independently of the higher-level dc.save check."""
    catalog = test_session.catalog
    name = dataset_complete.name
    version = dataset_complete.latest_version

    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=True)

    with pytest.raises(
        DatasetInvalidVersionError,
        match=r"was removed\. Pick a different version",
    ):
        catalog.create_dataset(
            name, version=version, columns=(sa.Column("name", String),)
        )


def test_remove_dataset_version_missing_raises(test_session, dataset_complete):
    """remove_dataset_version on a non-existent version raises
    DatasetVersionNotFoundError so typos aren't silently swallowed."""
    from datachain.error import DatasetVersionNotFoundError

    catalog = test_session.catalog
    with pytest.raises(DatasetVersionNotFoundError):
        catalog.remove_dataset_version(dataset_complete, "99.0.0", keep_metadata=True)


def test_remove_keep_metadata_refused_when_status_not_removable(
    test_session, dataset_failed
):
    """keep_metadata=True is only defined for COMPLETE/REMOVING/REMOVED
    versions - anything else (FAILED, PENDING, CREATED, STALE) must be
    refused with a clear message instead of tombstoning garbage state."""
    catalog = test_session.catalog
    version = dataset_failed.latest_version

    with pytest.raises(DataChainError, match="expected COMPLETE, REMOVING or REMOVED"):
        catalog.remove_dataset_version(dataset_failed, version, keep_metadata=True)


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


def test_remove_versions_swallows_per_version_error(test_session, dataset_complete):
    """A per-version failure in the bulk loop must not abort the whole
    batch - the exception is logged and the loop moves on. Verify with a
    phantom version that raises alongside a real one that succeeds."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    pairs = [(dataset_complete, "99.99.99"), (dataset_complete, version)]

    n = catalog._remove_versions(pairs, keep_metadata=False)
    assert n == 1
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(dataset_complete.name)


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
    # Verify status is COMPLETE
    dataset_version = dataset_complete.get_version(dataset_complete.latest_version)
    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.finished_at is not None

    # Verify all operations completed (num_objects set, etc.)
    assert dataset_version.num_objects == 2


def test_public_api_datasets_filters_non_complete(
    test_session, dataset_created, dataset_failed, dataset_complete
):
    ds_chain = datasets(session=test_session, column="dataset")
    dataset_names = {ds.name for (ds,) in ds_chain.to_iter("dataset")}

    assert dataset_complete.name in dataset_names, "COMPLETE dataset should be visible"
    assert dataset_created.name not in dataset_names, "CREATED dataset should be hidden"
    assert dataset_failed.name not in dataset_names, "FAILED dataset should be hidden"


@pytest.mark.parametrize("is_studio", [True])
def test_public_api_read_dataset_rejects_non_complete(test_session, studio_job):
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
    # Should raise error for CREATED dataset
    with pytest.raises(DatasetNotFoundError):
        delete_dataset(dataset_created.name, session=test_session)

    # Should raise error for FAILED dataset
    with pytest.raises(DatasetNotFoundError):
        delete_dataset(dataset_failed.name, session=test_session)


def test_public_api_move_dataset_rejects_non_complete(
    test_session, dataset_created, dataset_failed
):
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
        .values(status=DatasetStatus.REMOVING, pending_metadata_drop=True)
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


def _find_removed(ds: DatasetRecord, version: str):
    """Find a REMOVED version by its semver."""
    for v in ds.all_versions:
        if v.status == DatasetStatus.REMOVED and v.version == version:
            return v
    return None


def test_remove_keeps_version_row_and_drops_rows_table(test_session, dataset_complete):
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    version = dataset_complete.latest_version
    rows_table = warehouse.dataset_table_name(dataset_complete, version)
    assert warehouse.db.has_table(rows_table)

    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=True)

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    removed = _find_removed(ds, version)
    assert removed is not None
    assert removed.status == DatasetStatus.REMOVED
    assert removed.removed_at is not None
    assert not warehouse.db.has_table(rows_table)


def test_remove_preserves_dependencies(test_session, dataset_complete):
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

    catalog.remove_dataset_version(dataset_complete, src_version, keep_metadata=True)

    deps = metastore.get_direct_dataset_dependencies(dep_ds, dep_version)
    assert len(deps) == 1
    assert deps[0] is not None


def test_remove_is_idempotent_on_already_removed(test_session, dataset_complete):
    catalog = test_session.catalog
    version = dataset_complete.latest_version

    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=True)
    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    first = _find_removed(ds, version)
    assert first is not None
    first_removed_at = first.removed_at

    # Second call finds the same row already REMOVED → no-op.
    catalog.remove_dataset_version(ds, version, keep_metadata=True)
    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    assert len([v for v in ds.all_versions if v.status == DatasetStatus.REMOVED]) == 1
    assert _find_removed(ds, version).removed_at == first_removed_at


def test_save_after_remove_skips_removed_version(test_session, dataset_complete):
    """A removed semver stays reserved while its record is kept - the next
    save auto-bumps past it instead of reclaiming the slot."""
    catalog = test_session.catalog
    name = dataset_complete.name
    first_version = dataset_complete.latest_version

    catalog.remove_dataset_version(dataset_complete, first_version, keep_metadata=True)

    new_chain = dc.read_values(value=["new1"], session=test_session).save(name)
    assert new_chain.dataset is not None
    assert new_chain.dataset.latest_version != first_version
    assert semver.value(new_chain.dataset.latest_version) > semver.value(first_version)

    # The old row lives on as a REMOVED record for lineage.
    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    assert _find_removed(ds, first_version) is not None


def test_remove_keep_metadata_false_drops_already_removed_version(
    test_session, dataset_complete
):
    """keep_metadata=False on a REMOVED tombstone deletes the version row
    (and the dataset row too if it was the last version)."""
    catalog = test_session.catalog
    name = dataset_complete.name
    version = dataset_complete.latest_version

    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=True)
    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    assert _find_removed(ds, version) is not None

    catalog.remove_dataset_version(ds, version, keep_metadata=False)

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def test_remove_keep_metadata_false_drops_live_complete_version(
    test_session, dataset_complete
):
    """keep_metadata=False on a fresh COMPLETE version removes it without
    leaving a record behind."""
    catalog = test_session.catalog
    name = dataset_complete.name
    version = dataset_complete.latest_version

    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=False)

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def test_save_explicit_removed_version_rejected(test_session, dataset_complete):
    """Saving with an explicit version that matches a REMOVED one must fail."""
    catalog = test_session.catalog
    name = dataset_complete.name
    version = dataset_complete.latest_version

    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=True)

    with pytest.raises(RuntimeError, match=r"was removed\. Pick a different version"):
        dc.read_values(value=["new1"], session=test_session).save(name, version=version)


def test_latest_version_skips_removed(test_session, dataset_complete):
    catalog = test_session.catalog
    name = dataset_complete.name
    v1 = dataset_complete.latest_version

    dc.read_values(value=["v2-a", "v2-b"], session=test_session).save(name)
    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    v2 = ds.latest_version
    assert v2 != v1

    catalog.remove_dataset_version(ds, v2, keep_metadata=True)

    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    assert ds.latest_version == v1


def test_listing_excludes_removed_only_dataset(test_session, dataset_complete):
    catalog = test_session.catalog
    name = dataset_complete.name

    catalog.remove_dataset_version(
        dataset_complete, dataset_complete.latest_version, keep_metadata=True
    )

    ds_chain = datasets(session=test_session, column="dataset")
    assert name not in {ds.name for (ds,) in ds_chain.to_iter("dataset")}


def test_read_dataset_after_remove_raises(
    test_session, dataset_complete, no_studio_dataset
):
    catalog = test_session.catalog
    name = dataset_complete.name

    catalog.remove_dataset_version(
        dataset_complete, dataset_complete.latest_version, keep_metadata=True
    )

    with pytest.raises(DatasetNotFoundError):
        read_dataset(name, session=test_session)


def test_janitor_fully_removes_created_version(test_session, job, dataset_created):
    """The cleanup path must fully remove non-COMPLETE versions - we don't
    want REMOVED rows piling up for failed or abandoned saves."""
    catalog = test_session.catalog
    catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    catalog.cleanup_dataset_versions()

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(dataset_created.name, include_incomplete=True)


def test_remove_non_complete_version_removes_row(test_session, dataset_failed):
    catalog = test_session.catalog
    name = dataset_failed.name
    catalog.remove_dataset_version(
        dataset_failed, dataset_failed.latest_version, keep_metadata=False
    )
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


def test_listing_dataset_never_keeps_metadata(test_session):
    """`lst__*` listing datasets must never keep metadata on remove — they're
    internal cache, keeping a record serves nobody and wastes rows."""
    catalog = test_session.catalog
    metastore = catalog.metastore
    listing_project = metastore.get_project(
        metastore.listing_project_name, metastore.system_namespace_name
    )
    name = f"{LISTING_PREFIX}internal_test"
    ds = _make_completed_dataset(catalog, name, project=listing_project)

    with pytest.raises(DataChainError, match="while keeping metadata"):
        catalog.remove_dataset_version(ds, ds.latest_version, keep_metadata=True)

    catalog.remove_dataset_version(ds, ds.latest_version, keep_metadata=False)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def test_session_dataset_never_keeps_metadata(test_session):
    """`session_*` intermediates are throwaway too — always fully removed."""
    catalog = test_session.catalog
    name = f"{SESSION_DATASET_PREFIX}internal_test"
    ds = _make_completed_dataset(catalog, name)

    with pytest.raises(DataChainError, match="while keeping metadata"):
        catalog.remove_dataset_version(ds, ds.latest_version, keep_metadata=True)

    catalog.remove_dataset_version(ds, ds.latest_version, keep_metadata=False)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def test_remove_dataset_versions_keep_metadata_downgrades_for_internal(test_session):
    """Bulk remove with `keep_metadata=True` silently fully removes internal
    datasets (`lst__*` / `session_*`) - they have no semver/lineage to
    preserve, so the flag is meaningless there. User-facing versions in the
    same batch still get the tombstone."""
    catalog = test_session.catalog
    user_ds = _make_completed_dataset(catalog, "user_ds")
    internal_ds = _make_completed_dataset(
        catalog, f"{SESSION_DATASET_PREFIX}internal_in_batch"
    )

    user_vid = user_ds.get_version(user_ds.latest_version).id
    internal_vid = internal_ds.get_version(internal_ds.latest_version).id

    catalog.remove_dataset_versions(
        version_ids=[user_vid, internal_vid], keep_metadata=True
    )

    user_after = catalog.get_dataset("user_ds", versions=None, include_incomplete=True)
    assert _find_removed(user_after, user_ds.latest_version) is not None

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(internal_ds.name, include_incomplete=True)


def test_save_skips_reserved_semver_after_delete(test_session):
    """After save -> save -> remove-with-keep-metadata latest -> save, the
    new save must skip the REMOVED slot and auto-bump past it - the semver
    stays reserved while its record is kept."""
    catalog = test_session.catalog
    dc.read_values(value=["a"], session=test_session).save("reserve_test")
    second = dc.read_values(value=["b"], session=test_session).save("reserve_test")
    removed_version = second.dataset.latest_version
    catalog.remove_dataset("reserve_test", version=removed_version, keep_metadata=True)

    third = dc.read_values(value=["c"], session=test_session).save("reserve_test")
    assert third.dataset.latest_version != removed_version
    assert semver.value(third.dataset.latest_version) > semver.value(removed_version)


def test_dependency_removed_flag(test_session):
    """A dataset dependency pointing at a REMOVED version is returned with
    ``removed=True`` so delta-style consumers can filter it without a
    separate query."""
    catalog = test_session.catalog
    source = dc.read_values(value=["a", "b"], session=test_session).save("dep_source")
    dc.read_dataset("dep_source", session=test_session).save("dep_target")
    catalog.remove_dataset(source.dataset.name, version="1.0.0", keep_metadata=True)

    deps = catalog.get_dataset_dependencies("dep_target", "1.0.0")
    assert len(deps) == 1
    assert deps[0] is not None
    assert deps[0].name == "dep_source"
    assert deps[0].removed is True


def test_dependency_after_full_remove_returns_none(test_session):
    """When the source version is fully removed (``keep_metadata=False``),
    the dependency row is left with a broken reference and the lineage
    view surfaces it as ``None`` - contrasts with the tombstone path covered
    by [[test_dependency_removed_flag]]."""
    catalog = test_session.catalog
    source = dc.read_values(value=["a", "b"], session=test_session).save("wipe_source")
    dc.read_dataset("wipe_source", session=test_session).save("wipe_target")
    catalog.remove_dataset(source.dataset.name, version="1.0.0", keep_metadata=False)

    deps = catalog.get_dataset_dependencies("wipe_target", "1.0.0")
    assert deps == [None]


def test_export_dataset_table_refuses_tombstone(test_session, dataset_complete):
    """Exporting a REMOVED tombstone must raise upfront instead of running
    and crashing later against the dropped rows table."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    catalog.remove_dataset(dataset_complete.name, version=version, keep_metadata=True)

    with pytest.raises(DatasetNotFoundError):
        catalog.export_dataset_table(
            bucket="s3://does-not-matter",
            name=dataset_complete.name,
            version=version,
            base_file_name="export",
        )


def test_bulk_remove_single_version_does_not_cascade_dataset_row(
    test_session, dataset_complete
):
    """A GC-shaped remove of a single version (whose in-memory DatasetRecord
    only carries that one version) must not delete the dataset row when
    other versions still exist in DB - otherwise FK cascade takes REMOVED
    tombstones and live versions down with it."""
    catalog = test_session.catalog
    # Add a second version, then mark it FAILED so GC will pick it up.
    columns = tuple(
        sa.Column(name, typ) for name, typ in dataset_complete.schema.items()
    )
    second, _ = catalog.create_dataset_version(
        dataset_complete, "2.0.0", columns=columns
    )
    catalog.metastore.update_dataset_version(
        second, "2.0.0", status=DatasetStatus.FAILED
    )

    # Remove v1.0.0 with keep_metadata=True so we have a tombstone we expect
    # to survive.
    catalog.remove_dataset(dataset_complete.name, version="1.0.0", keep_metadata=True)

    # GC removes v2.0.0 via the bulk path - this builds a single-version
    # DatasetRecord per row internally.
    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    v2_id = ds.get_version("2.0.0").id
    catalog.remove_dataset_versions(version_ids=[v2_id])

    # Dataset row must survive; tombstone must survive.
    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    assert _find_removed(ds, "1.0.0") is not None
    assert not ds.has_version("2.0.0")


def test_rename_dataset_across_tombstone(test_session, dataset_complete):
    """Renaming a dataset that has a REMOVED tombstone version must succeed:
    the tombstone has no rows table, so its rename is skipped, and live
    versions are renamed normally."""
    catalog = test_session.catalog
    second = dc.read_values(value=["v2a", "v2b"], session=test_session).save(
        dataset_complete.name
    )
    live_version = second.dataset.latest_version

    catalog.remove_dataset(dataset_complete.name, version="1.0.0", keep_metadata=True)

    catalog.edit_dataset(dataset_complete.name, new_name="ds_complete_renamed")

    renamed = dc.read_dataset(
        "ds_complete_renamed", version=live_version, session=test_session
    )
    assert sorted(renamed.to_values("value")) == ["v2a", "v2b"]


def test_remove_dataset_force_keep_metadata_mixed_versions(
    test_session, dataset_complete
):
    """`remove_dataset(force=True, keep_metadata=True)` on a mixed-status
    dataset leaves COMPLETE versions as tombstones and fully removes the
    rest - keep_metadata is meaningful only where there is semver/lineage
    worth preserving."""
    catalog = test_session.catalog
    columns = tuple(
        sa.Column(name, typ) for name, typ in dataset_complete.schema.items()
    )
    updated, _ = catalog.create_dataset_version(
        dataset_complete, "2.0.0", columns=columns
    )
    catalog.metastore.update_dataset_version(
        updated, "2.0.0", status=DatasetStatus.FAILED
    )

    catalog.remove_dataset(dataset_complete.name, force=True, keep_metadata=True)

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    # COMPLETE version -> tombstone; FAILED version -> row removed.
    assert _find_removed(ds, "1.0.0") is not None
    assert not ds.has_version("2.0.0")


def test_remove_dataset_force_keep_metadata_internal_downgrades(test_session):
    """Internal datasets (`lst__*`, `session_*`) have no semver/lineage to
    preserve, so `keep_metadata=True` transparently downgrades to a full
    remove."""
    catalog = test_session.catalog
    name = f"{SESSION_DATASET_PREFIX}force_test"
    ds = _make_completed_dataset(catalog, name)

    catalog.remove_dataset(ds.name, force=True, keep_metadata=True)

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(ds.name, include_incomplete=True)


def _force_status(catalog, dataset: DatasetRecord, version: str, status: int, **extra):
    """Put a version into a specific status directly. Simulates a mid-flight
    removal that crashed, or another caller having claimed the transition."""
    catalog.metastore.update_dataset_version(dataset, version, status=status, **extra)
    return catalog.get_dataset(dataset.name, versions=None, include_incomplete=True)


def test_gc_skips_finalized_tombstones(test_session, dataset_complete):
    """A finalized REMOVED tombstone is outside the GC sweep set entirely -
    GC must not touch it."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVED)
    vid = ds.get_version(version).id

    catalog.remove_dataset_versions(version_ids=[vid])

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    assert _find_removed(ds, version) is not None


@pytest.mark.parametrize("drop_rows_first", [False, True])
def test_gc_resumes_interrupted_full_remove(
    test_session, dataset_complete, drop_rows_first
):
    """A version stuck in REMOVING + pending_metadata_drop=True (a
    keep_metadata=False remove that crashed before the version row was
    dropped) is finished by the GC path - whether or not the rows table
    was already dropped before the crash."""
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    version = dataset_complete.latest_version
    name = dataset_complete.name
    rows_table = warehouse.dataset_table_name(dataset_complete, version)
    if drop_rows_first:
        warehouse.drop_dataset_rows_table(dataset_complete, version)
    assert warehouse.db.has_table(rows_table) is not drop_rows_first

    ds = _force_status(
        catalog,
        dataset_complete,
        version,
        DatasetStatus.REMOVING,
        pending_metadata_drop=True,
    )
    vid = ds.get_version(version).id

    catalog.remove_dataset_versions(version_ids=[vid])

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)
    assert not warehouse.db.has_table(rows_table)


@pytest.mark.parametrize("drop_rows_first", [False, True])
def test_gc_resumes_interrupted_keep_metadata(
    test_session, dataset_complete, drop_rows_first
):
    """A version stuck in REMOVING + pending_metadata_drop=False (a
    keep-metadata remove that crashed mid-drop) gets the rows table cleaned
    up and the status flipped to REMOVED by the GC path - whether or not
    the rows table was already dropped before the crash."""
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    version = dataset_complete.latest_version
    rows_table = warehouse.dataset_table_name(dataset_complete, version)
    if drop_rows_first:
        warehouse.drop_dataset_rows_table(dataset_complete, version)
    assert warehouse.db.has_table(rows_table) is not drop_rows_first

    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVING)
    vid = ds.get_version(version).id

    catalog.remove_dataset_versions(version_ids=[vid])

    assert not warehouse.db.has_table(rows_table)
    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    finalized = _find_removed(ds, version)
    assert finalized is not None
    assert finalized.status == DatasetStatus.REMOVED


def test_dataset_ls_include_removed_marks_output(
    capsys, test_session, dataset_complete
):
    """`dataset ls --include-removed` yields tombstoned versions and the
    CLI output tags them with '(removed)'."""
    from datachain.cli.commands.datasets import list_datasets, list_datasets_local

    catalog = test_session.catalog
    version = dataset_complete.latest_version
    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=True)

    default = list(list_datasets_local(catalog))
    assert all(not removed for _, _, removed in default)

    with_removed = list(list_datasets_local(catalog, include_removed=True))
    tomb = [row for row in with_removed if row[2]]
    assert tomb, "expected at least one tombstoned version"

    list_datasets(catalog, local=True, all=False, include_removed=True)
    out = capsys.readouterr().out
    assert "(removed)" in out


def test_list_datasets_local_versions_by_name_yields_tombstone(
    test_session, dataset_complete
):
    """Passing a specific dataset name delegates to
    `list_datasets_local_versions`, which yields COMPLETE versions and
    (when include_removed=True) REMOVED tombstones with the correct
    removed flag."""
    from datachain.cli.commands.datasets import list_datasets_local

    catalog = test_session.catalog
    name = dataset_complete.name
    tomb_version = dataset_complete.latest_version
    catalog.remove_dataset_version(dataset_complete, tomb_version, keep_metadata=True)
    live_ds = dc.read_values(value=["v3"], session=test_session).save(name).dataset
    live_version = live_ds.latest_version

    default = list(list_datasets_local(catalog, name))
    assert (name, live_version, False) in default
    assert all(v != tomb_version for _, v, _ in default)

    with_removed = list(list_datasets_local(catalog, name, include_removed=True))
    assert (name, live_version, False) in with_removed
    assert (name, tomb_version, True) in with_removed


def test_remove_keep_metadata_lands_on_removed(test_session, dataset_complete):
    """A normal keep-metadata remove lands on status REMOVED, so the
    finalized tombstone is outside the GC candidate set."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    name = dataset_complete.name

    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=True)

    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    removed = _find_removed(ds, version)
    assert removed is not None
    assert removed.status == DatasetStatus.REMOVED
    assert bool(removed.pending_metadata_drop) is False

    candidates = catalog.metastore.get_dataset_versions_to_clean()
    assert all(d.name != name for d, _ in candidates)


def test_keep_metadata_refused_when_drop_in_progress(test_session, dataset_complete):
    """keep_metadata=True must refuse when the version is already
    REMOVING with pending_metadata_drop=True (a removal that drops
    metadata is already in flight)."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(
        catalog,
        dataset_complete,
        version,
        DatasetStatus.REMOVING,
        pending_metadata_drop=True,
    )

    with pytest.raises(DataChainError, match=r"keep_metadata=False.*in progress"):
        catalog.remove_dataset_version(ds, version, keep_metadata=True)


def test_drop_metadata_refused_when_keep_in_progress(test_session, dataset_complete):
    """keep_metadata=False must refuse when the version is already
    REMOVING with pending_metadata_drop=False (a removal that keeps
    metadata is in flight). The user must wait for the tombstone."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(
        catalog,
        dataset_complete,
        version,
        DatasetStatus.REMOVING,
        pending_metadata_drop=False,
    )

    with pytest.raises(DataChainError, match=r"keep_metadata=True.*in progress"):
        catalog.remove_dataset_version(ds, version, keep_metadata=False)


def test_tombstone_then_full_remove_escalates(test_session, dataset_complete):
    """Calling keep_metadata=False on an already-tombstoned (REMOVED)
    version flips pending_metadata_drop from False to True and drops the
    version row."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    name = dataset_complete.name

    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=True)
    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    removed = _find_removed(ds, version)
    assert removed is not None
    assert not removed.pending_metadata_drop

    catalog.remove_dataset_version(ds, version, keep_metadata=False)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def test_complete_raises_when_version_removed_concurrently(
    test_session, dataset_complete
):
    """If a version is removed before completion finishes, the guarded
    final status flip refuses to stomp it and raises a clean error
    instead of silently corrupting state."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVED)

    with pytest.raises(
        ConcurrentDatasetModificationError, match="Could not update status"
    ):
        catalog.metastore.update_dataset_status(
            ds,
            DatasetStatus.COMPLETE,
            version=version,
            expected_status=DatasetStatus.CREATED,
        )


def test_complete_dataset_version_raises_friendly_when_removed_concurrently(
    test_session, dataset_complete
):
    """End-to-end: when a version is flipped to REMOVED mid-save (e.g. by GC),
    catalog.complete_dataset_version surfaces a specific
    ConcurrentDatasetModificationError with a user-friendly message and the
    original guard error chained as the cause."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVED)

    with pytest.raises(
        ConcurrentDatasetModificationError, match="deleted concurrently"
    ) as exc_info:
        catalog.complete_dataset_version(ds, version)

    assert isinstance(exc_info.value.__cause__, ConcurrentDatasetModificationError)
    assert "Could not update status" in str(exc_info.value.__cause__)
