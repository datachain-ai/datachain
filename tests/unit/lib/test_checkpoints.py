from collections.abc import Iterator

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.error import DatasetNotFoundError, JobNotFoundError
from datachain.lib.utils import DataChainError
from tests.utils import get_partial_tables, reset_session_job_state


def mapper_fail(num) -> int:
    raise Exception("Error")


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


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

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
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

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(third_job_id))) == 3
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


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_udf_checkpoints_cross_job_reuse(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(reset_checkpoints))

    # Track how many times the mapper is called
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

    # Check that second job created checkpoints
    checkpoints_second = list(catalog.metastore.list_checkpoints(second_job_id))
    # After successful completion, only final checkpoint remains
    # (partial checkpoint is deleted after promotion)
    assert len(checkpoints_second) == 1
    assert checkpoints_second[0].partial is False

    # Verify the data is correct
    result = chain.order_by("num").to_list("doubled")
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_udf_checkpoints_multiple_calls_same_job(
    test_session, monkeypatch, nums_dataset
):
    """
    Test that UDF execution creates checkpoints, but subsequent calls in the same
    job will re-execute because the hash changes (includes previous checkpoint hash).
    Checkpoint reuse is designed for cross-job execution, not within-job execution.
    """
    # Track how many times the mapper is called
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


def test_udf_tables_naming(test_session, monkeypatch):
    catalog = test_session.catalog
    warehouse = catalog.warehouse

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("num.num.numbers")

    # Record initial UDF tables (from numbers dataset which uses read_values
    # internally)
    initial_udf_tables = set(warehouse.db.list_tables(prefix="udf_"))

    def get_udf_tables():
        tables = set(warehouse.db.list_tables(prefix="udf_"))
        return sorted(tables - initial_udf_tables)

    def square_num(num) -> int:
        return num * num

    chain = dc.read_dataset("num.num.numbers", session=test_session).map(
        squared=square_num, output=int
    )

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.count()
    first_job_id = test_session.get_or_create_job().id

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 1

    # Construct expected job-specific table names (include job_id in names)
    # After UDF completion, processed table is cleaned up,
    # input and output tables remain
    # Note: Input table uses partial_hash (hash_input + output_schema_hash),
    # not just hash_input, to detect schema changes
    partial_hash = "241cc841b9bd4ba9dca17183ce467b413de6a176e94c14929fd37da94e2445be"
    hash_output = "12a892fbed5f7d557d5fc7f048f3356dda97e7f903a3f998318202a4400e3f16"
    expected_first_run_tables = sorted(
        [
            f"udf_{first_job_id}_{partial_hash}_input",
            f"udf_{first_job_id}_{hash_output}_output",
        ]
    )

    assert get_udf_tables() == expected_first_run_tables

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    chain.count()
    second_job_id = test_session.get_or_create_job().id

    # Second run should:
    # - Reuse first job's input table (found via ancestor search)
    # - Create its own output table (copied from first job)
    expected_all_tables = sorted(
        [
            f"udf_{first_job_id}_{partial_hash}_input",  # Shared input
            f"udf_{first_job_id}_{hash_output}_output",  # First job output
            f"udf_{second_job_id}_{hash_output}_output",  # Second job output
        ]
    )

    assert get_udf_tables() == expected_all_tables


@pytest.mark.parametrize("parallel", [None, 2, 4, 6, 20])
def test_track_processed_items(test_session_tmpfile, parallel):
    """Test that we correctly track processed sys__ids with different parallel
    settings.

    This is a simple test that runs a UDF that fails partway through and verifies
    that the processed sys__ids are properly tracked (no duplicates, no missing values).
    """
    test_session = test_session_tmpfile
    catalog = test_session.catalog
    warehouse = catalog.warehouse

    def gen_numbers(num) -> Iterator[int]:
        """Generator function that fails on a specific input."""
        # Fail on input 7
        if num == 7:
            raise Exception(f"Simulated failure on num={num}")
        yield num * 10

    dc.read_values(num=list(range(1, 100)), session=test_session).save("nums")

    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .order_by("num")
        .settings(batch_size=2)
    )
    if parallel is not None:
        chain = chain.settings(parallel=parallel)

    # Run UDF - should fail on num=7
    with pytest.raises(Exception):  # noqa: B017
        chain.gen(result=gen_numbers, output=int).save("results")

    _, partial_output_table = get_partial_tables(test_session)

    # Get distinct sys__input_id from partial output table to see which inputs were
    # processed
    query = sa.select(sa.distinct(partial_output_table.c.sys__input_id))
    processed_sys_ids = [row[0] for row in warehouse.db.execute(query)]

    # Verify no duplicates
    assert len(processed_sys_ids) == len(set(processed_sys_ids))
    # Verify we processed some but not all inputs (should have failed before completing)
    assert 0 < len(processed_sys_ids) < 100
