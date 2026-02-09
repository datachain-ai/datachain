import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.error import JobAncestryDepthExceededError
from tests.utils import reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


def get_dataset_versions_for_job(metastore, job_id):
    """
    Returns:
        List of tuples (dataset_name, version, is_creator)
    """
    query = (
        sa.select(
            metastore._datasets_versions.c.dataset_id,
            metastore._datasets_versions.c.version,
            metastore._dataset_version_jobs.c.is_creator,
        )
        .select_from(
            metastore._dataset_version_jobs.join(
                metastore._datasets_versions,
                metastore._dataset_version_jobs.c.dataset_version_id
                == metastore._datasets_versions.c.id,
            )
        )
        .where(metastore._dataset_version_jobs.c.job_id == job_id)
    )

    results = list(metastore.db.execute(query))

    dataset_versions = []
    for dataset_id, version, is_creator in results:
        dataset_query = sa.select(metastore._datasets.c.name).where(
            metastore._datasets.c.id == dataset_id
        )
        dataset_name = next(metastore.db.execute(dataset_query))[0]
        dataset_versions.append((dataset_name, version, bool(is_creator)))

    return sorted(dataset_versions)


def test_dataset_job_linking(test_session, monkeypatch, nums_dataset):
    """Test that dataset versions are correctly linked to jobs via many-to-many.

    This test verifies that datasets should appear in ALL jobs that use them in
    the single job "chain", not just the job that created them.
    """
    catalog = test_session.catalog
    metastore = catalog.metastore
    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(False))

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN: Create dataset -------------------
    reset_session_job_state()
    chain.save("nums_linked")
    job1_id = test_session.get_or_create_job().id

    # Verify job1 has the dataset associated (as creator)
    job1_datasets = get_dataset_versions_for_job(metastore, job1_id)
    assert len(job1_datasets) == 1
    assert job1_datasets[0] == ("nums_linked", "1.0.0", True)

    # -------------- SECOND RUN: Reuse dataset via checkpoint -------------------
    reset_session_job_state()
    chain.save("nums_linked")
    job2_id = test_session.get_or_create_job().id

    # Verify job2 also has the dataset associated (not creator)
    job2_datasets = get_dataset_versions_for_job(metastore, job2_id)
    assert len(job2_datasets) == 1
    assert job2_datasets[0] == ("nums_linked", "1.0.0", False)

    # Verify job1 still has it
    job1_datasets = get_dataset_versions_for_job(metastore, job1_id)
    assert len(job1_datasets) == 1
    assert job1_datasets[0][2]  # still creator

    # -------------- THIRD RUN: Another reuse -------------------
    reset_session_job_state()
    chain.save("nums_linked")
    job3_id = test_session.get_or_create_job().id

    # Verify job3 also has the dataset associated (not creator)
    job3_datasets = get_dataset_versions_for_job(metastore, job3_id)
    assert len(job3_datasets) == 1
    assert job3_datasets[0] == ("nums_linked", "1.0.0", False)

    # Verify get_dataset_version_for_job_ancestry works correctly
    dataset = catalog.get_dataset("nums_linked")
    found_version = metastore.get_dataset_version_for_job_ancestry(
        "nums_linked",
        dataset.project.namespace.name,
        dataset.project.name,
        job3_id,
    )
    assert found_version.version == "1.0.0"


def test_dataset_job_linking_with_reset(test_session, monkeypatch, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore
    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(True))

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums_reset")
    job1_id = test_session.get_or_create_job().id

    # Verify job1 created version 1.0.0
    job1_datasets = get_dataset_versions_for_job(metastore, job1_id)
    assert len(job1_datasets) == 1
    assert job1_datasets[0] == ("nums_reset", "1.0.0", True)

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    chain.save("nums_reset")
    job2_id = test_session.get_or_create_job().id

    job2_datasets = get_dataset_versions_for_job(metastore, job2_id)
    assert len(job2_datasets) == 1
    assert job2_datasets[0] == ("nums_reset", "1.0.1", True)

    job1_datasets = get_dataset_versions_for_job(metastore, job1_id)
    assert len(job1_datasets) == 1
    assert job1_datasets[0] == ("nums_reset", "1.0.0", True)


def test_dataset_version_job_id_updates_to_latest(
    test_session, monkeypatch, nums_dataset
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(False))

    chain = dc.read_dataset("nums", session=test_session)
    name = "nums_jobid"

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save(name)
    job1_id = test_session.get_or_create_job().id

    dataset = catalog.get_dataset(name)
    assert dataset.get_version(dataset.latest_version).job_id == job1_id

    # -------------- SECOND RUN: Reuse via checkpoint -------------------
    reset_session_job_state()
    chain.save(name)
    job2_id = test_session.get_or_create_job().id

    # job_id should now point to job2 (latest)
    dataset = catalog.get_dataset(name)
    assert dataset.get_version(dataset.latest_version).job_id == job2_id

    # -------------- THIRD RUN: Another reuse -------------------
    reset_session_job_state()
    chain.save(name)
    job3_id = test_session.get_or_create_job().id

    # job_id should now point to job3 (latest)
    dataset = catalog.get_dataset(name)
    assert dataset.get_version(dataset.latest_version).job_id == job3_id


def test_job_ancestry_depth_exceeded(test_session, monkeypatch, nums_dataset):
    from datachain.data_storage import metastore

    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(False))
    # Mock max depth to a small value (3) for testing
    monkeypatch.setattr(metastore, "JOB_ANCESTRY_MAX_DEPTH", 3)

    chain = dc.read_dataset("nums", session=test_session)

    max_attempts = 10  # Safety limit to prevent infinite loop
    for _ in range(max_attempts):
        reset_session_job_state()
        try:
            chain.save("nums_depth")
        except JobAncestryDepthExceededError as exc_info:
            assert "too deep" in str(exc_info)
            assert "from scratch" in str(exc_info)
            # Test passed - we hit the max depth
            return

    # If we get here, we never hit the max depth error
    pytest.fail(f"Expected JobAncestryDepthExceededError after {max_attempts} saves")
