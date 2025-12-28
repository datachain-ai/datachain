"""Tests for dataset status management and failed version cleanup."""

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
    dataset = test_session.catalog.get_dataset(dataset_created.name)
    dataset_version = dataset.get_version(dataset.latest_version)
    assert dataset_version.status == DatasetStatus.CREATED
    assert dataset_version.job_id == job.id

    # Mark dataset versions as failed
    test_session.catalog.metastore.mark_job_dataset_versions_as_failed(job.id)

    # Verify status is now FAILED
    dataset = test_session.catalog.get_dataset(dataset_created.name)
    dataset_version = dataset.get_version(dataset.latest_version)
    assert dataset_version.status == DatasetStatus.FAILED
    assert dataset_version.finished_at is not None


def test_mark_job_dataset_versions_as_failed_skips_complete(
    test_session, job, dataset_complete
):
    """Test that mark_job_dataset_versions_as_failed skips COMPLETE versions."""
    # Verify initial status is COMPLETE
    dataset = test_session.catalog.get_dataset(dataset_complete.name)
    dataset_version = dataset.get_version(dataset_complete.latest_version)
    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.job_id == job.id

    # Mark dataset versions as failed
    test_session.catalog.metastore.mark_job_dataset_versions_as_failed(job.id)

    # Verify COMPLETE status is unchanged
    dataset = test_session.catalog.get_dataset(dataset_complete.name)
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


def test_get_incomplete_dataset_versions(
    test_session, job, dataset_created, dataset_failed, dataset_complete
):
    """Test get_incomplete_dataset_versions."""
    # Mark job as failed
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    # Get failed versions to clean
    to_clean = test_session.catalog.metastore.get_incomplete_dataset_versions()

    # Should return CREATED and FAILED datasets, not COMPLETE
    cleaned_names = {dataset.name for dataset, _ in to_clean}
    assert dataset_created.name in cleaned_names
    assert dataset_failed.name in cleaned_names
    assert dataset_complete.name not in cleaned_names

    # Verify each tuple contains dataset and version
    for dataset, version in to_clean:
        assert version is not None
        assert len(dataset.versions) == 1


def test_get_incomplete_dataset_versions_skips_running_jobs(
    test_session, job, dataset_created
):
    """Test that cleanup skips dataset versions from running jobs."""
    # Get failed versions to clean - should be empty since job is RUNNING
    to_clean = test_session.catalog.metastore.get_incomplete_dataset_versions()
    assert dataset_created.name not in {ds.name for ds, _ in to_clean}

    # Mark job as complete
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.COMPLETE)

    # Now should be included
    to_clean = test_session.catalog.metastore.get_incomplete_dataset_versions()
    assert dataset_created.name in {ds.name for ds, _ in to_clean}


def test_cleanup_failed_dataset_versions(test_session, job, dataset_failed):
    """Test cleanup_failed_dataset_versions removes datasets and returns IDs."""
    # Mark job as failed
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    # Cleanup failed versions
    num_removed = test_session.catalog.cleanup_failed_dataset_versions()

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


def test_public_api_read_dataset_rejects_non_complete(
    test_session, dataset_created, dataset_failed
):
    """Test that dc.read_dataset() rejects non-COMPLETE datasets."""
    # Should raise error for CREATED dataset
    with pytest.raises(DatasetNotFoundError):
        read_dataset(dataset_created.name, session=test_session)

    # Should raise error for FAILED dataset
    with pytest.raises(DatasetNotFoundError):
        read_dataset(dataset_failed.name, session=test_session)


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
