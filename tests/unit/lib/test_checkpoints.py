import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.error import DatasetNotFoundError, JobNotFoundError
from datachain.lib.utils import DataChainError
from tests.utils import reset_session_job_state


def mapper_fail(num) -> int:
    raise Exception("Error")


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
    with pytest.raises(DataChainError):
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
    with pytest.raises(DataChainError):
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
    with pytest.raises(DataChainError):
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
    """Test that dataset versions are correctly linked to jobs via many-to-many."""
    catalog = test_session.catalog
    metastore = catalog.metastore
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN: Create dataset -------------------
    reset_session_job_state()
    chain.save("nums_linked")
    job1_id = test_session.get_or_create_job().id

    # Get dataset version
    dataset = catalog.get_dataset("nums_linked")
    version = dataset.get_version(dataset.latest_version)

    # Check that job1 is linked with is_creator=True
    query = sa.select(
        metastore._dataset_version_jobs.c.job_id,
        metastore._dataset_version_jobs.c.is_creator,
    ).where(metastore._dataset_version_jobs.c.dataset_version_id == version.id)
    results = list(metastore.db.execute(query))

    assert len(results) == 1
    assert results[0][0] == job1_id
    assert results[0][1]  # is_creator

    # -------------- SECOND RUN: Reuse dataset via checkpoint -------------------
    reset_session_job_state()
    chain.save("nums_linked")
    job2_id = test_session.get_or_create_job().id

    # Check that both jobs are now linked
    results = list(metastore.db.execute(query))
    results_dict = {row[0]: row[1] for row in results}

    assert len(results) == 2
    assert results_dict[job1_id]  # job1 is creator
    assert not results_dict[job2_id]  # job2 is not creator

    # -------------- THIRD RUN: Another reuse -------------------
    reset_session_job_state()
    chain.save("nums_linked")
    job3_id = test_session.get_or_create_job().id

    # Check that all three jobs are linked
    results = list(metastore.db.execute(query))
    results_dict = {row[0]: row[1] for row in results}

    assert len(results) == 3
    assert results_dict[job1_id]  # job1 is creator
    assert not results_dict[job2_id]  # job2 reused
    assert not results_dict[job3_id]  # job3 reused

    # Verify get_dataset_version_for_job_ancestry works correctly
    # Job3's ancestry should find the version created by job1
    ancestor_ids = metastore.get_ancestor_job_ids(job3_id)
    job_ancestry = [job3_id, *ancestor_ids]
    found_version = metastore.get_dataset_version_for_job_ancestry(
        "nums_linked",
        dataset.project.namespace.name,
        dataset.project.name,
        job_ancestry,
    )
    assert found_version is not None
    assert found_version.id == version.id
