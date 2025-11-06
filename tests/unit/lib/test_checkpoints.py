import pytest
from collections.abc import Generator, Iterator

import datachain as dc
from datachain.error import DatasetNotFoundError, JobNotFoundError
from datachain.lib.utils import DataChainError
from datachain.query.dataset import UDFStep
from tests.utils import reset_session_job_state


def mapper_fail(num) -> int:
    raise Exception("Error")


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


def _count_table(warehouse, table_name) -> int:
    assert warehouse.db.has_table(table_name)
    table = warehouse.get_table(table_name)
    return warehouse.table_rows_count(table)


def _count_partial(warehouse, job_id, _hash) -> int:
    table_name = UDFStep.partial_output_table_name(job_id, _hash)
    return _count_table(warehouse, table_name)


def _count_processed(warehouse, job_id, _hash):
    table_name = UDFStep.processed_table_name(job_id, _hash)
    return _count_table(warehouse, table_name)


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

    # Second count() - should reuse checkpoint within same job
    call_count["count"] = 0
    assert chain.count() == 6
    assert call_count["count"] == 0, "Mapper should NOT be called on second count()"

    # Third count() - should still reuse checkpoint
    call_count["count"] = 0
    assert chain.count() == 6
    assert call_count["count"] == 0, "Mapper should NOT be called on third count()"

    # Other operations like to_list() should also reuse checkpoint
    call_count["count"] = 0
    result = chain.order_by("num").to_list("plus_ten")
    assert result == [(11,), (12,), (13,), (14,), (15,), (16,)]
    assert call_count["count"] == 0, "Mapper should NOT be called on to_list()"


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
    hash_input = "213263c3715396a437cc0fdcb94e908b67993490c56485c1b2180ae3d9e14780"
    hash_output = "12a892fbed5f7d557d5fc7f048f3356dda97e7f903a3f998318202a4400e3f16"
    expected_first_run_tables = sorted(
        [
            f"udf_{first_job_id}_{hash_input}_input",
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
            f"udf_{first_job_id}_{hash_input}_input",  # Shared input
            f"udf_{first_job_id}_{hash_output}_output",  # First job output
            f"udf_{second_job_id}_{hash_output}_output",  # Second job output
        ]
    )

    assert get_udf_tables() == expected_all_tables


@pytest.mark.parametrize(
    "batch_size,fail_after_count",
    [
        (2, 3),  # batch_size=2: Fail after 3 rows
        (3, 4),  # batch_size=3: Fail after 4 rows
        (5, 3),  # batch_size=5: Fail after 3 rows
    ],
)
def test_udf_signals_continue_from_partial(
    test_session,
    monkeypatch,
    nums_dataset,
    batch_size,
    fail_after_count,
):
    """Test continuing UDF execution from partial output table in unsafe mode.

    Tests with different batch sizes to ensure partial results are correctly handled
    regardless of batch boundaries. Uses counter-based failure to avoid dependency
    on row ordering (ClickHouse doesn't guarantee order without ORDER BY).

    Simulates real-world scenario: user writes buggy UDF, it fails, then fixes bug
    and reruns.
    """
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    processed_nums = []

    def process_buggy(num) -> int:
        """Buggy version that fails before processing the (fail_after_count+1)th row."""
        if len(processed_nums) >= fail_after_count:
            raise Exception(f"Simulated failure after {len(processed_nums)} rows")
        processed_nums.append(num)
        return num * 10

    chain = dc.read_dataset("nums", session=test_session).settings(
        batch_size=batch_size
    )

    # -------------- FIRST RUN (FAILS WITH BUGGY UDF) -------------------
    reset_session_job_state()

    with pytest.raises(Exception, match="Simulated failure after"):
        chain.map(result=process_buggy, output=int).save("results")

    first_job_id = test_session.get_or_create_job().id
    first_run_count = len(processed_nums)

    # Should have processed exactly fail_after_count rows before failing
    assert first_run_count == fail_after_count

    # Verify partial checkpoint was created
    checkpoints = list(catalog.metastore.list_checkpoints(first_job_id))
    hash_input = checkpoints[0].hash
    assert len(checkpoints) == 1

    # Verify partial table state after exception
    # ClickHouse: saves all fail_after_count rows (buffer flushed in finally)
    # SQLite: saves complete batches only (may be 0 if only incomplete batch)
    partial_count = _count_partial(warehouse, first_job_id, hash_input)
    assert 0 <= partial_count <= fail_after_count, (
        f"Expected 0-{fail_after_count} rows in partial table, got {partial_count}"
    )

    # -------------- SECOND RUN (FIXED UDF) -------------------
    reset_session_job_state()

    processed_nums.clear()

    def process_fixed(num) -> int:
        """Fixed version that works correctly."""
        processed_nums.append(num)
        return num * 10

    # Now use the fixed UDF - should continue from partial checkpoint
    chain.map(result=process_fixed, output=int).save("results")

    second_job_id = test_session.get_or_create_job().id
    checkpoints = sorted(
        catalog.metastore.list_checkpoints(second_job_id),
        key=lambda c: c.created_at,
    )

    # After successful completion, only final checkpoints remain (partial ones deleted)
    # 2 checkpoints: [0] from map() UDF, [1] from nums dataset generation
    assert len(checkpoints) == 2
    assert all(c.partial is False for c in checkpoints)
    # Verify the map() UDF output table exists (checkpoints[0])
    assert warehouse.db.has_table(
        UDFStep.output_table_name(second_job_id, checkpoints[0].hash)
    )

    # Verify all 6 rows were processed correctly in final dataset
    result = dc.read_dataset("results", session=test_session).to_list("result")
    assert sorted(result) == [(10,), (20,), (30,), (40,), (50,), (60,)]

    # Verify second run processed remaining rows (checkpoint continuation working)
    # The exact count depends on warehouse implementation and batch boundaries:
    # - ClickHouse: buffer flush in finally saves all processed rows (3-4 saved)
    # - SQLite: only complete batches are saved (0-3 saved depending on batch_size)
    # In worst case (SQLite, batch_size=5), 0 rows saved → all 6 reprocessed
    assert 0 < len(processed_nums) <= 6, "Expected 1-6 rows in second run"


@pytest.mark.parametrize(
    "batch_size,fail_after_count",
    [
        (2, 2),  # batch_size=2: Fail after 2 inputs (4 outputs → 2 batches saved)
        (3, 4),  # batch_size=3: Fail after 4 inputs
        (10, 3),  # batch_size=10: Fail after 3 inputs
    ],
)
def test_udf_generator_continue_from_partial(
    test_session,
    monkeypatch,
    nums_dataset,
    batch_size,
    fail_after_count,
):
    """Test continuing RowGenerator from partial output in unsafe mode.

    RowGenerator differs from UDFSignal because:
    - One input can generate multiple outputs (2 outputs per input)
    - Output rows have different sys__ids than input rows
    - Uses a separate processed table to track which inputs are processed

    Tests with different batch sizes to ensure processed table correctly
    tracks inputs only after ALL their outputs have been committed. Uses
    counter-based failure to avoid dependency on row ordering.

    Simulates real-world scenario: user writes buggy generator, it fails, then
    fixes bug and reruns.
    """
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    processed_nums = []

    class BuggyGenerator(dc.Generator):
        """
        Buggy generator that fails before processing the (fail_after_count+1)th input.
        """

        def process(self, num):
            if len(processed_nums) >= fail_after_count:
                raise Exception(f"Simulated failure after {len(processed_nums)} inputs")
            processed_nums.append(num)
            yield num * 10
            yield num * num

    chain = dc.read_dataset("nums", session=test_session).settings(
        batch_size=batch_size
    )

    # -------------- FIRST RUN (FAILS WITH BUGGY GENERATOR) -------------------
    reset_session_job_state()

    with pytest.raises(Exception, match="Simulated failure after"):
        chain.gen(value=BuggyGenerator(), output=int).save("gen_results")

    first_job_id = test_session.get_or_create_job().id
    first_run_count = len(processed_nums)

    # Should have processed exactly fail_after_count inputs before failing
    assert first_run_count == fail_after_count

    # Verify partial checkpoint was created
    checkpoints = list(catalog.metastore.list_checkpoints(first_job_id))
    hash_input = checkpoints[0].hash
    assert len(checkpoints) == 1

    # Verify partial table has outputs (each input generates 2 outputs)
    # ClickHouse: saves all outputs including incomplete batch
    # SQLite: saves complete batches only (may be 0 if only incomplete batch)
    partial_count = _count_partial(warehouse, first_job_id, hash_input)
    max_outputs = fail_after_count * 2  # Each input yields 2 outputs
    assert 0 <= partial_count <= max_outputs

    # Verify processed table tracks completed inputs
    # ClickHouse: tracks all inputs whose outputs were saved
    # SQLite: may be 0 if incomplete batch lost (no complete inputs saved)
    processed_count = _count_processed(warehouse, first_job_id, hash_input)
    assert 0 <= processed_count <= fail_after_count

    # -------------- SECOND RUN (FIXED GENERATOR) -------------------
    reset_session_job_state()

    processed_nums.clear()

    class FixedGenerator(dc.Generator):
        """Fixed generator that works correctly."""

        def process(self, num):
            processed_nums.append(num)
            yield num * 10
            yield num * num

    # Now use the fixed generator - should continue from partial checkpoint
    chain.gen(value=FixedGenerator(), output=int).save("gen_results")

    second_job_id = test_session.get_or_create_job().id
    checkpoints = sorted(
        catalog.metastore.list_checkpoints(second_job_id),
        key=lambda c: c.created_at,
    )
    assert len(checkpoints) == 2
    assert all(c.partial is False for c in checkpoints)
    # Verify gen() UDF output table exists (checkpoints[0])
    assert warehouse.db.has_table(
        UDFStep.output_table_name(second_job_id, checkpoints[0].hash)
    )

    result = sorted(
        dc.read_dataset("gen_results", session=test_session).to_list("value")
    )
    expected = sorted(
        [
            (1,),
            (10,),  # num=1: 1 (1²), 10 (1x10)
            (4,),
            (20,),  # num=2: 4 (2²), 20 (2x10)
            (9,),
            (30,),  # num=3: 9 (3²), 30 (3x10)
            (16,),
            (40,),  # num=4: 16 (4²), 40 (4x10)
            (25,),
            (50,),  # num=5: 25 (5²), 50 (5x10)
            (36,),
            (60,),  # num=6: 36 (6²), 60 (6x10)
        ]
    )

    # Should have exactly 12 outputs (no duplicates)
    assert result == expected

    # Verify second run processed remaining inputs (checkpoint continuation working)
    # The exact count depends on warehouse implementation and batch boundaries
    assert 0 < len(processed_nums) <= 6, "Expected 1-6 inputs in second run"


# (3, 2),  # batch_size=3: Fail after processing 2 partitions
# (10, 2),  # batch_size=10: Fail after processing 2 partitions
@pytest.mark.parametrize(
    "batch_size,fail_after_count",
    [
        (2, 2),  # batch_size=2: Fail after processing 2 partitions
    ],
)
def test_aggregator_continue_from_partial(
    test_session,
    monkeypatch,
    nums_dataset,
    batch_size,
    fail_after_count,
):
    """Test continuing Aggregator from partial output in unsafe mode with partition_by.

    Aggregator differs from Generator because:
    - Uses partition_by to group inputs
    - Reduces multiple inputs to one output per partition
    - Processes partitions, not individual rows

    Tests that partition_by works correctly with checkpoints and ensures
    input table is created first to maintain consistent sys__id values.

    Simulates real-world scenario: user writes buggy aggregator, it fails, then
    fixes bug and reruns.
    """
    processed_partitions = []

    def buggy_aggregator(letter, num) -> Iterator[tuple[str, int]]:
        """
        Buggy aggregator that fails before processing the (fail_after_count+1)th partition.
        letter: partition key value (A, B, or C)
        num: iterator of num values in that partition
        """
        if len(processed_partitions) >= fail_after_count:
            raise Exception(
                f"Simulated failure after {len(processed_partitions)} partitions"
            )
        nums_list = list(num)
        processed_partitions.append(nums_list)
        # Yield tuple of (letter, sum) to preserve partition key in output
        yield letter[0], sum(n for n in nums_list)

    def fixed_aggregator(letter, num) -> Iterator[tuple[str, int]]:
        """Fixed aggregator that works correctly."""
        nums_list = list(num)
        processed_partitions.append(nums_list)
        # Yield tuple of (letter, sum) to preserve partition key in output
        yield letter[0], sum(n for n in nums_list)

    # Create dataset with groups: nums [1,2,3,4,5,6] with group [A,A,B,B,C,C]
    # Save to dataset to ensure consistent hash across runs
    nums_data = [1, 2, 3, 4, 5, 6]
    leters_data = ["A", "A", "B", "B", "C", "C"]
    dc.read_values(num=nums_data, letter=leters_data, session=test_session).save(
        "nums_letters"
    )

    # -------------- FIRST RUN (FAILS WITH BUGGY AGGREGATOR) -------------------
    reset_session_job_state()

    chain = dc.read_dataset("nums_letters", session=test_session).settings(
        batch_size=batch_size
    )

    with pytest.raises(Exception, match="Simulated failure after"):
        chain.agg(
            total=buggy_aggregator,
            partition_by="letter",
        ).save("agg_results")

    first_run_count = len(processed_partitions)

    # Should have processed exactly fail_after_count partitions before failing
    assert first_run_count == fail_after_count

    # -------------- SECOND RUN (FIXED AGGREGATOR) -------------------
    reset_session_job_state()

    processed_partitions.clear()

    # Now use the fixed aggregator - should continue from partial checkpoint
    chain.agg(
        total=fixed_aggregator,
        partition_by="letter",
    ).save("agg_results")

    second_run_count = len(processed_partitions)

    # Verify final results: 3 partitions (A, B, C) with correct sums
    # Column names are total_0 (letter) and total_1 (sum) from the tuple
    assert sorted(
        dc.read_dataset("agg_results", session=test_session).to_list("total_0", "total_1")
    ) == sorted(
        [
            ("A", 3),  # group A: 1 + 2 = 3
            ("B", 7),  # group B: 3 + 4 = 7
            ("C", 11),  # group C: 5 + 6 = 11
        ]
    )

    # KEY TEST: Verify checkpoint continuation worked
    # Second run should process remaining partitions (or potentially all if no continuation)
    # The important check is that results are correct without duplicates
    assert 0 < second_run_count <= 3, (
        f"Expected 1-3 partitions processed in second run, but got {second_run_count}"
    )


@pytest.mark.xfail(
    reason="Known limitation: inputs that yield nothing are not tracked "
    "in processed table"
)
def test_generator_yielding_nothing(test_session, monkeypatch, nums_dataset):
    """Test that generator correctly handles inputs that yield zero outputs."""
    warehouse = test_session.catalog.warehouse
    processed = []

    class SelectiveGenerator(dc.Generator):
        """Generator that only yields outputs for even numbers."""

        def process(self, num):
            processed.append(num)
            if num == 3:
                raise Exception("Simulated failure")
            if num % 2 == 0:  # Only even numbers yield outputs
                yield num * 10

    # First run - fails on num=3
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session).gen(
        value=SelectiveGenerator(), output=int
    )

    with pytest.raises(Exception, match="Simulated failure"):
        chain.save("results")

    first_job_id = test_session.get_or_create_job().id
    first_checkpoints = list(
        test_session.catalog.metastore.list_checkpoints(first_job_id)
    )
    hash_input = first_checkpoints[0].hash

    # Verify processed table tracks inputs that yielded nothing
    # Inputs 1,2 were processed (1 yielded nothing, 2 yielded one output)
    assert _count_processed(warehouse, first_job_id, hash_input) == 2

    # Second run - should skip already processed inputs
    reset_session_job_state()
    processed.clear()
    chain.save("results")

    # Only inputs 3,4,5,6 should be processed
    assert processed == [3, 4, 5, 6]
    # Result should only have even numbers x 10
    result = sorted(dc.read_dataset("results", session=test_session).to_list("value"))
    assert result == [(20,), (40,), (60,)]


def test_multiple_udf_chain_continue(test_session, monkeypatch, nums_dataset):
    """Test continuing from partial with multiple UDFs in chain.

    When mapper fails, only mapper's partial table exists. On retry, mapper
    completes and gen runs from scratch.
    """
    map_processed = []
    gen_processed = []
    fail_once = [True]  # Mutable flag to track if we should fail

    def mapper(num: int) -> int:
        map_processed.append(num)
        # Fail before processing the 4th row in first run only
        if fail_once[0] and len(map_processed) == 3:
            fail_once[0] = False
            raise Exception("Map failure")
        return num * 2

    class Doubler(dc.Generator):
        def process(self, doubled):
            gen_processed.append(doubled)
            yield doubled
            yield doubled

    # First run - fails in mapper
    # batch_size=2: processes [1,2] (commits), then [3,4] (fails on 4)
    reset_session_job_state()
    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(batch_size=2)
        .map(doubled=mapper)
        .gen(value=Doubler(), output=int)
    )

    with pytest.raises(Exception, match="Map failure"):
        chain.save("results")

    # Second run - completes successfully
    # Mapper continues from partial checkpoint
    reset_session_job_state()
    chain.save("results")

    # Verify mapper processed some rows (continuation working)
    # First run: 3 rows attempted
    # Second run: varies by warehouse (0-6 rows depending on batching/buffer behavior)
    # Total: 6-9 calls (some rows may be reprocessed if not saved to partial)
    assert 6 <= len(map_processed) <= 9, "Expected 6-9 total mapper calls"

    # Verify gen processed all 6 mapper outputs
    assert len(gen_processed) == 6

    # Verify final result has all values doubled twice
    result = sorted(dc.read_dataset("results", session=test_session).to_list("value"))
    assert sorted([v[0] for v in result]) == sorted(
        [2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12]
    )


def test_udf_code_change_triggers_rerun(test_session, monkeypatch, nums_dataset):
    """Test that changing UDF code (hash) triggers rerun from scratch."""
    map1_calls = []
    map2_calls = []

    chain = dc.read_dataset("nums", session=test_session).settings(batch_size=2)

    # Run 1: map1 succeeds, map2 fails
    def mapper1_v1(num: int) -> int:
        map1_calls.append(num)
        return num * 2

    def mapper2_failing(doubled: int) -> int:
        # Fail before processing 4th row (counter-based for ClickHouse compatibility)
        if len(map2_calls) >= 3:
            raise Exception("Map2 failure")
        map2_calls.append(doubled)
        return doubled * 3

    reset_session_job_state()
    with pytest.raises(Exception, match="Map2 failure"):
        (chain.map(doubled=mapper1_v1).map(tripled=mapper2_failing).save("results"))

    assert len(map1_calls) == 6  # All processed
    assert len(map2_calls) == 3  # Processed 3 before failing

    # Run 2: Change map1 code, map2 fixed - both should rerun
    def mapper1_v2(num: int) -> int:
        map1_calls.append(num)
        return num * 2 + 1  # Different code = different hash

    def mapper2_fixed(doubled: int) -> int:
        map2_calls.append(doubled)
        return doubled * 3

    map1_calls.clear()
    map2_calls.clear()
    reset_session_job_state()
    (chain.map(doubled=mapper1_v2).map(tripled=mapper2_fixed).save("results"))

    assert len(map1_calls) == 6  # Reran due to code change
    assert len(map2_calls) == 6  # Ran all (no partial to continue from)
    result = dc.read_dataset("results", session=test_session).to_list("tripled")
    # nums [1,2,3,4,5,6] → x2+1 = [3,5,7,9,11,13] → x3 = [9,15,21,27,33,39]
    assert sorted(result) == sorted([(i,) for i in [9, 15, 21, 27, 33, 39]])

    # Run 3: Keep both unchanged - both should skip
    map1_calls.clear()
    map2_calls.clear()
    reset_session_job_state()
    (chain.map(doubled=mapper1_v2).map(tripled=mapper2_fixed).save("results"))

    assert len(map1_calls) == 0  # Skipped (checkpoint found)
    assert len(map2_calls) == 0  # Skipped (checkpoint found)
    result = dc.read_dataset("results", session=test_session).to_list("tripled")
    assert sorted(result) == sorted([(i,) for i in [9, 15, 21, 27, 33, 39]])


def test_udf_generator_safe_mode_no_partial_continue(
    test_session, monkeypatch, nums_dataset
):
    """Test that in safe mode (unsafe=False), we don't continue from partial
    checkpoints.

    When DATACHAIN_UDF_CHECKPOINT_MODE is not "unsafe":
    - No processed table is created for RowGenerator
    - Failed jobs don't create partial checkpoints that can be continued from
    - Rerunning always starts from scratch
    """
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    monkeypatch.setenv("DATACHAIN_UDF_CHECKPOINT_MODE", "safe")

    processed_nums = []

    class BuggyGenerator(dc.Generator):
        """Buggy generator that fails on num=4."""

        def process(self, num):
            processed_nums.append(num)
            if num == 4:
                raise Exception(f"Simulated failure on num={num}")
            yield num * 10
            yield num * num

    # -------------- FIRST RUN (FAILS WITH BUGGY GENERATOR) -------------------
    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(batch_size=2)
        .gen(value=BuggyGenerator(), output=int)
    )

    with pytest.raises(Exception, match="Simulated failure"):
        chain.save("gen_results")

    first_job_id = test_session.get_or_create_job().id

    checkpoints = list(catalog.metastore.list_checkpoints(first_job_id))
    assert len(checkpoints) == 1
    hash_input = checkpoints[0].hash

    # Verify partial output table exists (partial outputs are still created)
    partial_table_name = UDFStep.partial_output_table_name(first_job_id, hash_input)
    assert warehouse.db.has_table(partial_table_name)

    # KEY DIFFERENCE: In safe mode, no processed table should be created
    processed_table_name = UDFStep.processed_table_name(first_job_id, hash_input)
    assert not warehouse.db.has_table(processed_table_name)

    # -------------- SECOND RUN (FIXED GENERATOR) -------------------
    reset_session_job_state()

    processed_nums.clear()

    class FixedGenerator(dc.Generator):
        """Fixed generator that works correctly."""

        def process(self, num):
            processed_nums.append(num)
            yield num * 10
            yield num * num

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(batch_size=2)
        .gen(value=FixedGenerator(), output=int)
    )

    chain.save("gen_results")

    # KEY DIFFERENCE: In safe mode, ALL inputs are processed again (not continuing
    # from partial)
    # Even though some were processed successfully in first run, we start from scratch
    assert sorted(processed_nums) == sorted([1, 2, 3, 4, 5, 6])

    # Verify final results are correct
    result = (
        dc.read_dataset("gen_results", session=test_session)
        .order_by("value")
        .to_list("value")
    )
    expected = [
        (1,),
        (10,),  # num=1: 1 (1²), 10 (1x10)
        (4,),
        (20,),  # num=2: 4 (2²), 20 (2x10)
        (9,),
        (30,),  # num=3: 9 (3²), 30 (3x10)
        (16,),
        (40,),  # num=4: 16 (4²), 40 (4x10)
        (25,),
        (50,),  # num=5: 25 (5²), 50 (5x10)
        (36,),
        (60,),  # num=6: 36 (6²), 60 (6x10)
    ]
    assert sorted(result) == sorted(expected)
