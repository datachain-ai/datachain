from uuid import uuid4

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.error import (
    DatasetNotFoundError,
    JobNotFoundError,
)
from tests.utils import reset_session_job_state


def _count_rows(metastore, table) -> int:
    query = sa.select(sa.func.count()).select_from(table)
    return next(iter(metastore.db.execute(query)))[0]


class CustomMapperError(Exception):
    pass


def mapper_fail(num: int) -> int:
    raise CustomMapperError("Error")


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


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

    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(reset_checkpoints))

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
    first_job = test_session.get_or_create_job()
    first_job_id = first_job.id

    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    if use_datachain_job_id_env:
        monkeypatch.setenv(
            "DATACHAIN_JOB_ID",
            metastore.create_job(
                "my-job",
                "echo 1;",
                rerun_from_job_id=first_job_id,
                run_group_id=first_job.run_group_id,
            ),
        )
    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    second_job_id = test_session.get_or_create_job().id

    expected_versions = 1 if with_delta or not reset_checkpoints else 2
    assert len(catalog.get_dataset("nums1").versions) == expected_versions
    assert len(catalog.get_dataset("nums2").versions) == expected_versions
    assert len(catalog.get_dataset("nums3").versions) == 1

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_modified_chains(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(reset_checkpoints))

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


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_multiple_runs(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog

    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(reset_checkpoints))

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

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(third_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(fourth_job_id))) == 3


def test_checkpoints_check_valid_chain_is_returned(
    test_session,
    monkeypatch,
    nums_dataset,
):
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
    assert ds.order_by("num").to_list("num") == [(1,), (2,), (3,), (4,), (5,), (6,)]


def test_checkpoints_invalid_parent_job_id(test_session, monkeypatch, nums_dataset):
    # setting wrong job id
    reset_session_job_state()
    monkeypatch.setenv("DATACHAIN_JOB_ID", "caee6c54-6328-4bcd-8ca6-2b31cb4fff94")
    with pytest.raises(JobNotFoundError):
        dc.read_dataset("nums", session=test_session).save("nums1")


def test_checkpoint_with_deleted_dataset_version(
    test_session, monkeypatch, nums_dataset
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(False))

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

    new_version = dataset.get_version("1.0.0")
    assert new_version.job_id == job2_id


def test_udf_checkpoints_multiple_calls_same_job(
    test_session, monkeypatch, nums_dataset
):
    """
    Test that UDF execution creates checkpoints, but subsequent calls in the same
    job will re-execute because the hash changes (includes previous checkpoint hash).
    Checkpoint reuse is designed for cross-job execution, not within-job execution.
    """
    call_count = {"count": 0}

    def add_ten(num) -> int:
        call_count["count"] += 1
        return num + 10

    chain = dc.read_dataset("nums", session=test_session).map(
        plus_ten=add_ten, output=int
    )

    reset_session_job_state()

    # First count() - should execute UDF
    assert chain.count() == 6
    first_calls = call_count["count"]
    assert first_calls == 6, "Mapper should be called 6 times on first count()"

    # Second count() - will re-execute because hash includes previous checkpoint
    call_count["count"] = 0
    assert chain.count() == 6
    assert call_count["count"] == 6, "Mapper re-executes in same job"

    # Third count() - will also re-execute
    call_count["count"] = 0
    assert chain.count() == 6
    assert call_count["count"] == 6, "Mapper re-executes in same job"

    # Other operations like to_list() will also re-execute
    call_count["count"] = 0
    result = chain.order_by("num").to_list("plus_ten")
    assert result == [(11,), (12,), (13,), (14,), (15,), (16,)]
    assert call_count["count"] == 6, "Mapper re-executes in same job"


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_udf_checkpoints_cross_job_reuse(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(reset_checkpoints))

    call_count = {"count": 0}

    def double_num(num) -> int:
        call_count["count"] += 1
        return num * 2

    chain = dc.read_dataset("nums", session=test_session).map(
        doubled=double_num, output=int
    )

    # -------------- FIRST RUN - count() triggers UDF execution -------------------
    reset_session_job_state()
    assert chain.count() == 6
    first_job_id = test_session.get_or_create_job().id

    assert call_count["count"] == 6

    checkpoints = list(catalog.metastore.list_checkpoints(first_job_id))
    assert len(checkpoints) == 1
    assert checkpoints[0].partial is False

    # -------------- SECOND RUN - should reuse UDF checkpoint -------------------
    reset_session_job_state()
    call_count["count"] = 0  # Reset counter

    assert chain.count() == 6
    second_job_id = test_session.get_or_create_job().id

    if reset_checkpoints:
        assert call_count["count"] == 6, "Mapper should be called again"
    else:
        assert call_count["count"] == 0, "Mapper should NOT be called"

    checkpoints_second = list(catalog.metastore.list_checkpoints(second_job_id))
    # After successful completion, only final checkpoint remains
    # (partial checkpoint is deleted after promotion)
    assert len(checkpoints_second) == 1
    assert checkpoints_second[0].partial is False

    # Verify the data is correct
    result = chain.order_by("num").to_list("doubled")
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_checkpoints_job_without_run_group_id(test_session, monkeypatch, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    call_count = {"count": 0}

    def double_num(num) -> int:
        call_count["count"] += 1
        return num * 2

    chain = dc.read_dataset("nums", session=test_session).map(
        doubled=double_num, output=int
    )

    # -------------- FIRST RUN (from scratch, no run_group_id) -------------------
    reset_session_job_state()

    first_job_id = str(uuid4())
    metastore.create_job(
        "scheduled-task",
        "echo 1;",
        job_id=first_job_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", first_job_id)

    chain.save("doubled_nums")
    first_job = metastore.get_job(first_job_id)
    assert first_job.run_group_id == first_job_id
    assert call_count["count"] == 6

    # -------------- SECOND RUN (skip, no run_group_id) -------------------
    reset_session_job_state()
    call_count["count"] = 0

    # Create rerun job — also without run_group_id (inherits None from parent)
    second_job_id = str(uuid4())
    metastore.create_job(
        "scheduled-task",
        "echo 1;",
        job_id=second_job_id,
        rerun_from_job_id=first_job_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", second_job_id)

    chain.save("doubled_nums")
    second_job = metastore.get_job(second_job_id)
    assert second_job.run_group_id == second_job_id
    assert second_job.rerun_from_job_id == first_job_id

    # UDF should be skipped via checkpoint
    assert call_count["count"] == 0

    result = chain.order_by("num").to_list("doubled")
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_checkpoints_job_without_run_group_id_continue(
    test_session, monkeypatch, nums_dataset
):
    catalog = test_session.catalog
    metastore = catalog.metastore

    processed_count = {"count": 0}
    should_fail = [True]

    def double_num(num) -> int:
        processed_count["count"] += 1
        if should_fail[0] and processed_count["count"] > 3:
            raise RuntimeError("Simulated failure")
        return num * 2

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(batch_size=1)
        .map(doubled=double_num, output=int)
    )

    # -------------- FIRST RUN (fails, no run_group_id) -------------------
    reset_session_job_state()

    first_job_id = str(uuid4())
    metastore.create_job(
        "scheduled-task",
        "echo 1;",
        job_id=first_job_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", first_job_id)

    with pytest.raises(RuntimeError, match="Simulated failure"):
        chain.save("doubled_nums")

    first_count = processed_count["count"]
    assert first_count > 0

    # -------------- SECOND RUN (continue, no run_group_id) -------------------
    reset_session_job_state()
    processed_count["count"] = 0
    should_fail[0] = False

    second_job_id = str(uuid4())
    metastore.create_job(
        "scheduled-task",
        "echo 1;",
        job_id=second_job_id,
        rerun_from_job_id=first_job_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", second_job_id)

    chain.save("doubled_nums")

    # Should only process remaining rows, not all 6
    assert processed_count["count"] < 6

    result = sorted(
        dc.read_dataset("doubled_nums", session=test_session).to_list("doubled")
    )
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_udf_runs_in_ephemeral_mode(test_session, nums_dataset):
    metastore = test_session.catalog.metastore
    jobs_before = _count_rows(metastore, metastore._jobs)
    checkpoints_before = _count_rows(metastore, metastore._checkpoints)

    result = sorted(
        dc.read_dataset("nums", session=test_session)
        .settings(ephemeral=True)
        .map(doubled=lambda num: num * 2, output=int)
        .to_list("doubled")
    )
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]

    # No checkpoints or jobs should have been created
    assert _count_rows(metastore, metastore._checkpoints) == checkpoints_before
    assert _count_rows(metastore, metastore._jobs) == jobs_before


def test_ephemeral_mode_repeated_runs_no_table_collision(test_session, nums_dataset):
    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(ephemeral=True)
        .map(doubled=lambda num: num * 2, output=int)
    )

    for _ in range(3):
        result = sorted(chain.to_list("doubled"))
        assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_ephemeral_mode_no_jobs_on_collect(test_session, nums_dataset):
    metastore = test_session.catalog.metastore
    jobs_before = _count_rows(metastore, metastore._jobs)
    checkpoints_before = _count_rows(metastore, metastore._checkpoints)

    result = sorted(
        dc.read_dataset("nums", session=test_session)
        .settings(ephemeral=True)
        .map(doubled=lambda num: num * 2, output=int)
        .to_values("doubled")
    )
    assert result == [2, 4, 6, 8, 10, 12]

    assert _count_rows(metastore, metastore._jobs) == jobs_before
    assert _count_rows(metastore, metastore._checkpoints) == checkpoints_before
