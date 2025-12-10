import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.error import (
    DatasetNotFoundError,
    JobAncestryDepthExceededError,
    JobNotFoundError,
)
from tests.utils import reset_session_job_state


class CustomMapperError(Exception):
    pass


def mapper_fail(num: int) -> int:
    raise CustomMapperError("Error")


def get_dataset_versions_for_job(metastore, job_id):
    """Helper to get all dataset versions associated with a job.

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

    # Get dataset names
    dataset_versions = []
    for dataset_id, version, is_creator in results:
        dataset_query = sa.select(metastore._datasets.c.name).where(
            metastore._datasets.c.id == dataset_id
        )
        dataset_name = next(metastore.db.execute(dataset_query))[0]
        # Convert is_creator to boolean for consistent assertions across databases
        dataset_versions.append((dataset_name, version, bool(is_creator)))

    return sorted(dataset_versions)


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3], session=test_session).save("nums")


@pytest.mark.skipif(
    "os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Checkpoints test skipped in distributed mode",
)
@pytest.mark.parametrize("reset_checkpoints", [True, False])
@pytest.mark.parametrize("with_delta", [True, False])
@pytest.mark.parametrize("use_datachain_job_id_env", [True, False])
def test_checkpoints(
    test_session,
    monkeypatch,
    nums_dataset,
    reset_checkpoints,
    with_delta,
    use_datachain_job_id_env,
):
    catalog = test_session.catalog
    metastore = catalog.metastore

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(reset_checkpoints))

    if with_delta:
        chain = dc.read_dataset(
            "nums", delta=True, delta_on=["num"], session=test_session
        )
    else:
        chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    if use_datachain_job_id_env:
        monkeypatch.setenv(
            "DATACHAIN_JOB_ID", metastore.create_job("my-job", "echo 1;")
        )

    chain.save("nums1")
    chain.save("nums2")
    with pytest.raises(CustomMapperError):
        chain.map(new=mapper_fail).save("nums3")
    first_job_id = test_session.get_or_create_job().id

    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    if use_datachain_job_id_env:
        monkeypatch.setenv(
            "DATACHAIN_JOB_ID",
            metastore.create_job("my-job", "echo 1;", parent_job_id=first_job_id),
        )

    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    second_job_id = test_session.get_or_create_job().id

    expected_versions = 1 if with_delta or not reset_checkpoints else 2
    assert len(catalog.get_dataset("nums1").versions) == expected_versions
    assert len(catalog.get_dataset("nums2").versions) == expected_versions
    assert len(catalog.get_dataset("nums3").versions) == 1

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 2
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


@pytest.mark.skipif(
    "os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Checkpoints test skipped in distributed mode",
)
@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_modified_chains(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(reset_checkpoints))

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    first_job_id = test_session.get_or_create_job().id

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")  # added change from first run
    chain.save("nums3")
    second_job_id = test_session.get_or_create_job().id

    assert len(catalog.get_dataset("nums1").versions) == 2 if reset_checkpoints else 1
    assert len(catalog.get_dataset("nums2").versions) == 2
    assert len(catalog.get_dataset("nums3").versions) == 2

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


@pytest.mark.skipif(
    "os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Checkpoints test skipped in distributed mode",
)
@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_multiple_runs(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(reset_checkpoints))

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.save("nums2")
    with pytest.raises(CustomMapperError):
        chain.map(new=mapper_fail).save("nums3")
    first_job_id = test_session.get_or_create_job().id

    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    second_job_id = test_session.get_or_create_job().id

    # -------------- THIRD RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")
    with pytest.raises(CustomMapperError):
        chain.map(new=mapper_fail).save("nums3")
    third_job_id = test_session.get_or_create_job().id

    # -------------- FOURTH RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")
    chain.save("nums3")
    fourth_job_id = test_session.get_or_create_job().id

    num1_versions = len(catalog.get_dataset("nums1").versions)
    num2_versions = len(catalog.get_dataset("nums2").versions)
    num3_versions = len(catalog.get_dataset("nums3").versions)

    if reset_checkpoints:
        assert num1_versions == 4
        assert num2_versions == 4
        assert num3_versions == 2

    else:
        assert num1_versions == 1
        assert num2_versions == 2
        assert num3_versions == 2

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 2
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(third_job_id))) == 2
    assert len(list(catalog.metastore.list_checkpoints(fourth_job_id))) == 3


@pytest.mark.skipif(
    "os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Checkpoints test skipped in distributed mode",
)
def test_checkpoints_check_valid_chain_is_returned(
    test_session,
    monkeypatch,
    nums_dataset,
):
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))
    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums1")

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    ds = chain.save("nums1")

    # checking that we return expected DataChain even though we skipped chain creation
    # because of the checkpoints
    assert ds.dataset is not None
    assert ds.dataset.name == "nums1"
    assert len(ds.dataset.versions) == 1
    assert ds.order_by("num").to_list("num") == [(1,), (2,), (3,)]


def test_checkpoints_invalid_parent_job_id(test_session, monkeypatch, nums_dataset):
    # setting wrong job id
    reset_session_job_state()
    monkeypatch.setenv("DATACHAIN_JOB_ID", "caee6c54-6328-4bcd-8ca6-2b31cb4fff94")
    with pytest.raises(JobNotFoundError):
        dc.read_dataset("nums", session=test_session).save("nums1")


def test_dataset_job_linking(test_session, monkeypatch, nums_dataset):
    """Test that dataset versions are correctly linked to jobs via many-to-many.

    This test verifies that datasets should appear in ALL jobs that use them in
    the single job "chain", not just the job that created them.
    """
    catalog = test_session.catalog
    metastore = catalog.metastore
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))

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
    """Test that with CHECKPOINTS_RESET=True, new versions are created each run."""
    catalog = test_session.catalog
    metastore = catalog.metastore
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(True))

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

    # Verify job2 created NEW version 1.0.1 (not reusing 1.0.0)
    job2_datasets = get_dataset_versions_for_job(metastore, job2_id)
    assert len(job2_datasets) == 1
    assert job2_datasets[0] == ("nums_reset", "1.0.1", True)

    # Verify job1 still only has version 1.0.0
    job1_datasets = get_dataset_versions_for_job(metastore, job1_id)
    assert len(job1_datasets) == 1
    assert job1_datasets[0] == ("nums_reset", "1.0.0", True)


def test_dataset_version_job_id_updates_to_latest(
    test_session, monkeypatch, nums_dataset
):
    """Test that dataset_version.job_id is updated to the latest job that used it."""
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))

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

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))
    # Mock max depth to a small value (3) for testing
    monkeypatch.setattr(metastore, "JOB_ANCESTRY_MAX_DEPTH", 3)

    chain = dc.read_dataset("nums", session=test_session)

    # Keep saving until we hit the max depth error
    max_attempts = 10  # Safety limit to prevent infinite loop
    for _ in range(max_attempts):
        reset_session_job_state()
        try:
            chain.save("nums_depth")
        except JobAncestryDepthExceededError as exc_info:
            # Verify the error message
            assert "too deep" in str(exc_info)
            assert "from scratch" in str(exc_info)
            # Test passed - we hit the max depth
            return

    # If we get here, we never hit the max depth error
    pytest.fail(f"Expected JobAncestryDepthExceededError after {max_attempts} saves")


def test_checkpoint_with_deleted_dataset_version(
    test_session, monkeypatch, nums_dataset
):
    """Test checkpoint found but dataset version deleted from ancestry."""
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN: Create dataset -------------------
    reset_session_job_state()
    chain.save("nums_deleted")
    test_session.get_or_create_job()

    dataset = catalog.get_dataset("nums_deleted")
    assert len(dataset.versions) == 1
    assert dataset.latest_version == "1.0.0"

    catalog.remove_dataset("nums_deleted", version="1.0.0", force=True)

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums_deleted")

    # -------------- SECOND RUN: Checkpoint exists but version gone
    reset_session_job_state()
    chain.save("nums_deleted")
    job2_id = test_session.get_or_create_job().id

    # Should create a NEW version since old one was deleted
    dataset = catalog.get_dataset("nums_deleted")
    assert len(dataset.versions) == 1
    assert dataset.latest_version == "1.0.0"

    # Verify the new version was created by job2, not job1
    new_version = dataset.get_version("1.0.0")
    assert new_version.job_id == job2_id
