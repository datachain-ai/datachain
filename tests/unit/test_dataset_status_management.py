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
from datachain.error import DataChainError, DatasetNotFoundError
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
    """A REMOVED tombstone from a soft delete issued inside the failing job
    must survive `mark_job_dataset_versions_as_failed` - otherwise the
    tombstone gets flipped to FAILED and then wiped by GC, breaking the
    soft-delete permanence guarantee."""
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
        assert len(dataset.versions) == 1


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
    """User-facing bulk delete (``keep_metadata=True``) must tombstone COMPLETE
    versions, not wipe them — overrides the GC inference path."""
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
        .values(status=DatasetStatus.REMOVING_TOTAL)
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
    for v in ds.versions:
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
    assert len([v for v in ds.versions if v.status == DatasetStatus.REMOVED]) == 1
    assert _find_removed(ds, version).removed_at == first_removed_at


def test_save_after_remove_skips_removed_version(test_session, dataset_complete):
    """A removed semver is permanently reserved — the next save auto-bumps
    past it instead of reclaiming the slot."""
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


def test_remove_keep_metadata_false_wipes_already_removed_version(
    test_session, dataset_complete
):
    """keep_metadata=False wipes a REMOVED record completely
    (version row gone, dataset row gone if it was the last)."""
    catalog = test_session.catalog
    name = dataset_complete.name
    version = dataset_complete.latest_version

    catalog.remove_dataset_version(dataset_complete, version, keep_metadata=True)
    ds = catalog.get_dataset(name, versions=None, include_incomplete=True)
    assert _find_removed(ds, version) is not None

    catalog.remove_dataset_version(ds, version, keep_metadata=False)

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def test_remove_keep_metadata_false_wipes_live_complete_version(
    test_session, dataset_complete
):
    """keep_metadata=False wipes a fresh COMPLETE version without leaving a record."""
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

    with pytest.raises(RuntimeError, match=f"already has version {version}"):
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


def test_janitor_still_hard_deletes_created_version(test_session, job, dataset_created):
    """The cleanup path must still hard-delete non-COMPLETE versions — we
    don't want REMOVED rows piling up for failed/abandoned saves."""
    catalog = test_session.catalog
    catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    catalog.cleanup_dataset_versions()

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(dataset_created.name, include_incomplete=True)


def test_remove_non_complete_version_is_hard_delete(test_session, dataset_failed):
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


def test_dependency_removed_flag(test_session):
    """A dataset dependency pointing at a soft-deleted version is returned
    with ``removed=True`` so delta-style consumers can filter it without a
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


def test_bulk_wipe_does_not_cascade_dataset_row(test_session, dataset_complete):
    """A GC-shaped wipe of a single version (whose in-memory DatasetRecord
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

    # Soft-delete v1.0.0 to get a tombstone we expect to survive.
    catalog.remove_dataset(dataset_complete.name, version="1.0.0", keep_metadata=True)

    # GC wipes v2.0.0 via the bulk path - this builds a single-version
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
    """`remove_dataset(force=True, keep_metadata=True)` on a mixed-state
    dataset tombstones soft-deletable versions and transparently wipes the
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
    # COMPLETE version -> tombstoned; FAILED version -> wiped.
    assert _find_removed(ds, "1.0.0") is not None
    assert not ds.has_version("2.0.0")


def test_remove_dataset_force_keep_metadata_internal_downgrades_to_wipe(test_session):
    """Internal datasets (`lst__*`, `session_*`) have no semver/lineage to
    preserve, so `keep_metadata=True` transparently downgrades to a full wipe.
    """
    catalog = test_session.catalog
    name = f"{SESSION_DATASET_PREFIX}force_test"
    ds = _make_completed_dataset(catalog, name)

    catalog.remove_dataset(ds.name, force=True, keep_metadata=True)

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(ds.name, include_incomplete=True)


def _force_status(catalog, dataset: DatasetRecord, version: str, status: int):
    """Put a version into a specific status directly. Simulates a mid-flight
    removal that crashed, or another caller having claimed the transition."""
    catalog.metastore.update_dataset_version(dataset, version, status=status)
    return catalog.get_dataset(dataset.name, versions=None, include_incomplete=True)


def test_gc_resumes_stuck_removing(test_session, dataset_complete):
    """A version stuck in REMOVING (previous soft-delete crashed mid-flight)
    is resumed to REMOVED by the GC path — _remove_versions picks the soft
    path from the current status."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVING)
    vid = ds.get_version(version).id

    catalog.remove_dataset_versions(version_ids=[vid])

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    assert _find_removed(ds, version) is not None


def test_gc_skips_removed_tombstones(test_session, dataset_complete):
    """A REMOVED version handed to the GC path must be a no-op - the
    tombstone has to be preserved, not wiped through inferred
    keep_metadata=False."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVED)
    vid = ds.get_version(version).id

    catalog.remove_dataset_versions(version_ids=[vid])

    ds = catalog.get_dataset(
        dataset_complete.name, versions=None, include_incomplete=True
    )
    assert _find_removed(ds, version) is not None


def test_gc_wipes_internal_stuck_in_removing(test_session):
    """An internal `session_*` dataset stuck in REMOVING (crashed mid-flight)
    must be wiped by the GC path - inference must not pick the soft path
    for internal datasets, since they can't be kept as metadata tombstones."""
    catalog = test_session.catalog
    name = f"{SESSION_DATASET_PREFIX}gc_stuck_test"
    ds = _make_completed_dataset(catalog, name)
    version = ds.latest_version
    ds = _force_status(catalog, ds, version, DatasetStatus.REMOVING)
    vid = ds.get_version(version).id

    catalog.remove_dataset_versions(version_ids=[vid])

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def test_gc_resumes_stuck_removing_total(test_session, dataset_complete):
    """A version stuck in REMOVING_TOTAL is resumed to a full wipe by the GC
    path — _remove_versions picks the wipe path."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    name = dataset_complete.name
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVING_TOTAL)
    vid = ds.get_version(version).id

    catalog.remove_dataset_versions(version_ids=[vid])

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(name, include_incomplete=True)


def test_remove_explicit_keep_on_inflight_wipe_raises(test_session, dataset_complete):
    """If a wipe is in flight (REMOVING_TOTAL) and a caller explicitly asks
    to keep metadata, raise — don't silently downgrade the in-flight wipe."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVING_TOTAL)

    with pytest.raises(DataChainError, match="while keeping metadata"):
        catalog.remove_dataset_version(ds, version, keep_metadata=True)


def test_remove_explicit_wipe_on_inflight_keep_raises(test_session, dataset_complete):
    """If a soft delete is in flight (REMOVING) and a caller explicitly asks
    to wipe, raise — don't silently escalate the in-flight soft delete."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVING)

    with pytest.raises(DataChainError, match="entirely"):
        catalog.remove_dataset_version(ds, version, keep_metadata=False)


def test_complete_raises_when_version_removed_concurrently(
    test_session, dataset_complete
):
    """If a version is removed before completion finishes, the guarded
    final status flip refuses to stomp it and raises a clean error
    instead of silently corrupting state."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVED)

    with pytest.raises(DataChainError, match="Could not update status"):
        catalog.metastore.update_dataset_status(
            ds,
            DatasetStatus.COMPLETE,
            version=version,
            expected_status=DatasetStatus.CREATED,
        )


def test_complete_dataset_version_raises_friendly_when_removed_concurrently(
    test_session, dataset_complete
):
    """End-to-end: when a version is flipped to REMOVING mid-save (e.g. by GC),
    catalog.complete_dataset_version surfaces a user-friendly DataChainError
    that explains the concurrent removal and suggests a retry."""
    catalog = test_session.catalog
    version = dataset_complete.latest_version
    ds = _force_status(catalog, dataset_complete, version, DatasetStatus.REMOVING)

    with pytest.raises(DataChainError, match="deleted concurrently") as exc_info:
        catalog.complete_dataset_version(ds, version)

    # original guard error is chained as the cause
    assert isinstance(exc_info.value.__cause__, DataChainError)
    assert "Could not update status" in str(exc_info.value.__cause__)
