"""Tests for dataset status management and failed version cleanup."""

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.data_storage import JobStatus
from datachain.dataset import DatasetStatus
from datachain.sql.types import String


def test_mark_job_dataset_versions_as_failed(test_session):
    """Test that mark_job_dataset_versions_as_failed marks versions as FAILED."""
    # Create a job
    job = test_session.get_or_create_job()
    ds_name = test_session.generate_temp_dataset_name()
    ds_name = "cats"
    version = "1.0.0"

    # Create a dataset version with CREATED status
    # ds = dc.read_values(value=["val1", "val2"], session=test_session)
    dataset = test_session.catalog.create_dataset(ds_name, columns=(sa.Column("name", String),))
    '''
    dataset, version = test_session.catalog.create_dataset(
        dataset_name,
        version="1.0.0",
        query_script="test",
        columns=[sa.Column(name, typ) for name, typ in dogs_dataset.schema.items()],
    )
    '''
    # Verify initial status is CREATED
    dataset = test_session.catalog.get_dataset(ds_name)
    dataset_version = dataset.get_version(version)
    assert dataset_version.status == DatasetStatus.CREATED
    assert dataset_version.job_id == job.id

    # Mark dataset versions as failed
    test_session.catalog.metastore.mark_job_dataset_versions_as_failed(job.id)

    # Verify status is now FAILED
    dataset = test_session.catalog.get_dataset(dataset_name)
    dataset_version = dataset.get_version(version)
    assert dataset_version.status == DatasetStatus.FAILED
    assert dataset_version.finished_at is not None


def test_mark_job_dataset_versions_as_failed_skips_complete(test_session):
    """Test that mark_job_dataset_versions_as_failed skips COMPLETE versions."""
    # Create a job
    job = test_session.get_or_create_job()

    # Create and save a dataset (COMPLETE status)
    ds = dc.read_values(value=["val1", "val2"], session=test_session)
    dataset_name = "test-dataset"
    ds.save(dataset_name)

    # Verify initial status is COMPLETE
    dataset = test_session.catalog.get_dataset(dataset_name)
    dataset_version = dataset.get_version(dataset.latest_version)
    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.job_id == job.id

    # Mark dataset versions as failed
    test_session.catalog.metastore.mark_job_dataset_versions_as_failed(job.id)

    # Verify COMPLETE status is unchanged
    dataset = test_session.catalog.get_dataset(dataset_name)
    dataset_version = dataset.get_version(dataset.latest_version)
    assert dataset_version.status == DatasetStatus.COMPLETE


def test_finalize_job_as_failed_marks_dataset_versions(test_session):
    """Test that _finalize_job_as_failed marks dataset versions as FAILED."""
    # Create a job
    job = test_session.get_or_create_job()

    # Create a dataset version with CREATED status
    ds = dc.read_values(value=["val1", "val2"], session=test_session)
    dataset_name = test_session.generate_temp_dataset_name()
    dataset, version = test_session.catalog.create_dataset(
        dataset_name,
        version="1",
        query_script="test",
        feature_schema=ds.signals_schema,
    )

    # Verify initial status is CREATED
    dataset = test_session.catalog.get_dataset(dataset_name)
    dataset_version = dataset.get_version(version)
    assert dataset_version.status == DatasetStatus.CREATED

    # Simulate job failure
    try:
        raise RuntimeError("test error")
    except RuntimeError as e:
        test_session._finalize_job_as_failed(type(e), e, e.__traceback__)

    # Verify job is marked as FAILED
    db_job = test_session.catalog.metastore.get_job(job.id)
    assert db_job.status == JobStatus.FAILED

    # Verify dataset version is marked as FAILED
    dataset = test_session.catalog.get_dataset(dataset_name)
    dataset_version = dataset.get_version(version)
    assert dataset_version.status == DatasetStatus.FAILED


def test_status_filtering_hides_non_complete_versions(test_session):
    """Test that non-COMPLETE dataset versions are hidden from queries."""
    # Create a job
    job = test_session.get_or_create_job()

    # Create a dataset version with CREATED status (not saved)
    ds = dc.read_values(value=["val1", "val2"], session=test_session)
    dataset_name = "test-dataset-created"
    dataset, version = test_session.catalog.create_dataset(
        dataset_name,
        version="1",
        query_script="test",
        feature_schema=ds.signals_schema,
    )

    # Create a dataset version with FAILED status
    failed_dataset_name = "test-dataset-failed"
    failed_dataset, failed_version = test_session.catalog.create_dataset(
        failed_dataset_name,
        version="1",
        query_script="test",
        feature_schema=ds.signals_schema,
    )
    test_session.catalog.metastore.update_dataset_status(
        failed_dataset, DatasetStatus.FAILED, version=failed_version
    )

    # Create a COMPLETE dataset
    complete_dataset_name = "test-dataset-complete"
    ds.save(complete_dataset_name)

    # List datasets - should only show COMPLETE dataset
    datasets = list(test_session.catalog.ls_datasets_db())
    dataset_names = [d.name for d in datasets]

    # Only COMPLETE dataset should be visible
    assert complete_dataset_name in dataset_names
    assert dataset_name not in dataset_names
    assert failed_dataset_name not in dataset_names

    # Verify we can still get datasets with include_incomplete=True
    all_datasets = list(
        test_session.catalog.metastore.list_datasets(include_incomplete=True)
    )
    all_dataset_names = [d.name for d in all_datasets]
    assert complete_dataset_name in all_dataset_names
    assert dataset_name in all_dataset_names
    assert failed_dataset_name in all_dataset_names


def test_get_failed_dataset_versions_to_clean_no_retention(test_session):
    """Test get_failed_dataset_versions_to_clean without retention period."""
    # Create and finalize a job
    job = test_session.get_or_create_job()

    # Create datasets with different statuses
    ds = dc.read_values(value=["val1", "val2"], session=test_session)

    # CREATED dataset (failed job)
    created_name = "test-created"
    created_dataset, created_version = test_session.catalog.create_dataset(
        created_name,
        version="1",
        query_script="test",
        feature_schema=ds.signals_schema,
    )

    # FAILED dataset (failed job)
    failed_name = "test-failed"
    failed_dataset, failed_version = test_session.catalog.create_dataset(
        failed_name,
        version="1",
        query_script="test",
        feature_schema=ds.signals_schema,
    )
    test_session.catalog.metastore.update_dataset_status(
        failed_dataset, DatasetStatus.FAILED, version=failed_version
    )

    # COMPLETE dataset
    complete_name = "test-complete"
    ds.save(complete_name)

    # Mark job as failed
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    # Get failed versions to clean (no retention)
    to_clean = test_session.catalog.metastore.get_failed_dataset_versions_to_clean()

    # Should return CREATED and FAILED datasets, not COMPLETE
    cleaned_names = {dataset.name for dataset, _ in to_clean}
    assert created_name in cleaned_names
    assert failed_name in cleaned_names
    assert complete_name not in cleaned_names

    # Verify each tuple contains dataset and version
    for dataset, version in to_clean:
        assert version is not None
        assert len(dataset.versions) == 1


def test_get_failed_dataset_versions_to_clean_with_retention(test_session):
    """Test get_failed_dataset_versions_to_clean with retention period."""
    # Create and finalize a job
    job = test_session.get_or_create_job()

    # Create a FAILED dataset
    ds = dc.read_values(value=["val1", "val2"], session=test_session)
    failed_name = "test-failed"
    failed_dataset, failed_version = test_session.catalog.create_dataset(
        failed_name,
        version="1",
        query_script="test",
        feature_schema=ds.signals_schema,
    )
    test_session.catalog.metastore.update_dataset_status(
        failed_dataset, DatasetStatus.FAILED, version=failed_version
    )

    # Mark job as failed
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    # With retention of 30 days, recently created dataset should not be cleaned
    to_clean = test_session.catalog.metastore.get_failed_dataset_versions_to_clean(
        retention_days=30
    )
    cleaned_names = {dataset.name for dataset, _ in to_clean}
    assert failed_name not in cleaned_names

    # With retention of 0 days, all should be cleaned
    to_clean = test_session.catalog.metastore.get_failed_dataset_versions_to_clean(
        retention_days=0
    )
    cleaned_names = {dataset.name for dataset, _ in to_clean}
    assert failed_name in cleaned_names


def test_get_failed_dataset_versions_to_clean_skips_running_jobs(test_session):
    """Test that cleanup skips dataset versions from running jobs."""
    # Create a job (still running)
    job = test_session.get_or_create_job()

    # Create a CREATED dataset
    ds = dc.read_values(value=["val1", "val2"], session=test_session)
    dataset_name = "test-created"
    test_session.catalog.create_dataset(
        dataset_name,
        version="1",
        query_script="test",
        feature_schema=ds.signals_schema,
    )

    # Get failed versions to clean - should be empty since job is RUNNING
    to_clean = test_session.catalog.metastore.get_failed_dataset_versions_to_clean()
    cleaned_names = {dataset.name for dataset, _ in to_clean}
    assert dataset_name not in cleaned_names

    # Mark job as complete
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.COMPLETE)

    # Now should be included
    to_clean = test_session.catalog.metastore.get_failed_dataset_versions_to_clean()
    cleaned_names = {dataset.name for dataset, _ in to_clean}
    assert dataset_name in cleaned_names


def test_cleanup_failed_dataset_versions(test_session):
    """Test cleanup_failed_dataset_versions removes datasets and returns IDs."""
    # Create and finalize a job
    job = test_session.get_or_create_job()

    # Create a FAILED dataset
    ds = dc.read_values(value=["val1", "val2"], session=test_session)
    failed_name = "test-failed"
    failed_dataset, failed_version = test_session.catalog.create_dataset(
        failed_name,
        version="1",
        query_script="test",
        feature_schema=ds.signals_schema,
    )
    test_session.catalog.metastore.update_dataset_status(
        failed_dataset, DatasetStatus.FAILED, version=failed_version
    )

    # Get the version ID before cleanup
    version_obj = failed_dataset.get_version(failed_version)
    version_id = str(version_obj.id)

    # Mark job as failed
    test_session.catalog.metastore.set_job_status(job.id, JobStatus.FAILED)

    # Cleanup failed versions
    cleaned_ids = test_session.catalog.cleanup_failed_dataset_versions()

    # Should return the cleaned version ID
    assert version_id in cleaned_ids
    assert len(cleaned_ids) == 1

    # Verify dataset version is removed
    with pytest.raises(Exception):
        test_session.catalog.get_dataset(failed_name)


def test_save_sets_complete_status_at_end(test_session):
    """Test that save() sets COMPLETE status only after all operations."""
    # This is more of an integration test to verify the behavior
    ds = dc.read_values(value=["val1", "val2"], session=test_session)
    dataset_name = "test-complete-at-end"

    # Save the dataset
    ds.save(dataset_name)

    # Verify status is COMPLETE
    dataset = test_session.catalog.get_dataset(dataset_name)
    dataset_version = dataset.get_version(dataset.latest_version)
    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.finished_at is not None

    # Verify all operations completed (num_objects set, etc.)
    assert dataset_version.num_objects == 2
