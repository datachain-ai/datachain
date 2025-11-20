from collections.abc import Iterator

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.error import DatasetNotFoundError
from datachain.query.dataset import UDFStep
from tests.utils import get_partial_tables, reset_session_job_state


def _count_table(warehouse, table_name) -> int:
    assert warehouse.db.has_table(table_name)
    table = warehouse.get_table(table_name)
    return warehouse.table_rows_count(table)


def _count_partial(warehouse, partial_table) -> int:
    return warehouse.table_rows_count(partial_table)


def _count_processed(warehouse, partial_table, generator=False):
    """Count distinct input sys__ids from partial output table.

    For generators: counts distinct sys__input_id values (non-NULL)
    For mappers: counts all rows (1:1 mapping, sys__input_id is NULL)
    """
    if generator:
        # Generators have sys__input_id populated with actual input sys__ids
        return len(
            list(
                warehouse.db.execute(
                    sa.select(sa.distinct(partial_table.c.sys__input_id)).where(
                        partial_table.c.sys__input_id.isnot(None)
                    )
                )
            )
        )

    # Mapper: count all rows (1:1 mapping)
    return warehouse.table_rows_count(partial_table)


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
def test_checkpoints_parallel(test_session_tmpfile, monkeypatch):
    def mapper_fail(num) -> int:
        raise Exception("Error")

    test_session = test_session_tmpfile
    catalog = test_session.catalog

    dc.read_values(num=list(range(1000)), session=test_session).save("nums")

    chain = dc.read_dataset("nums", session=test_session).settings(parallel=True)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.save("nums2")
    with pytest.raises(RuntimeError):
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

    assert len(catalog.get_dataset("nums1").versions) == 1
    assert len(catalog.get_dataset("nums2").versions) == 1
    assert len(catalog.get_dataset("nums3").versions) == 1

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


def test_udf_generator_continue_parallel(test_session_tmpfile, monkeypatch):
    """Test continuing RowGenerator from partial with parallel=True.

    This tests that processed table is properly passed through parallel
    execution path so that checkpoint recovery works correctly.
    """
    test_session = test_session_tmpfile
    catalog = test_session.catalog
    warehouse = catalog.warehouse

    # Track which numbers have been processed
    processed_nums = []
    run_count = {"count": 0}

    def gen_multiple(num) -> Iterator[int]:
        """Generator that yields multiple outputs per input."""
        processed_nums.append(num)
        # Fail on input 4 in first run only
        if num == 4 and run_count["count"] == 0:
            raise Exception(f"Simulated failure on num={num}")
        # Each input yields 2 outputs
        yield num * 10
        yield num

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN (FAILS) -------------------
    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(parallel=2, batch_size=2)
        .gen(result=gen_multiple, output=int)
    )

    with pytest.raises(RuntimeError):
        chain.save("results")

    _, partial_table = get_partial_tables(test_session)

    # Verify sys__input_id has tracked some inputs
    processed_count_first = len(
        list(
            warehouse.db.execute(sa.select(sa.distinct(partial_table.c.sys__input_id)))
        )
    )
    assert processed_count_first > 0, "Some inputs should be tracked"

    # -------------- SECOND RUN (CONTINUE) -------------------
    reset_session_job_state()

    # Clear processed list and increment run count
    processed_nums.clear()
    run_count["count"] += 1

    # Should complete successfully
    chain.save("results")

    # Verify result
    result = (
        dc.read_dataset("results", session=test_session)
        .order_by("result")
        .to_list("result")
    )
    # Each of 6 inputs yields 2 outputs: [10,1], [20,2], ..., [60,6]
    assert result == [
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (10,),
        (20,),
        (30,),
        (40,),
        (50,),
        (60,),
    ]

    # Verify only unprocessed inputs were processed in second run
    # (should be less than all 6 inputs)
    assert len(processed_nums) < 6


@pytest.mark.parametrize("parallel", [2, 4, 6, 20])
def test_processed_table_data_integrity(test_session_tmpfile, parallel):
    """Test that input table, and output table are consistent after failure.

    Verifies that for a generator that yields n^2 for each input n:
    - Every sys__input_id in  output table has corresponding input in input table
    - Every processed input has correct output (n^2) in partial output table
    - No missing or incorrect outputs
    """
    test_session = test_session_tmpfile
    warehouse = test_session.catalog.warehouse

    def gen_square(num) -> Iterator[int]:
        # Fail on input 7
        if num == 50:
            raise Exception(f"Simulated failure on num={num}")
        yield num * num

    dc.read_values(num=list(range(1, 100)), session=test_session).save("nums")
    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(parallel=parallel, batch_size=2)
        .gen(result=gen_square, output=int)
    )

    # Run UDF - should fail on num=7
    with pytest.raises(RuntimeError):
        chain.save("results")

    input_table, partial_output_table = get_partial_tables(test_session)

    # Get distinct sys__input_id from partial output table to see which inputs were
    # processed
    processed_sys_ids = [
        row[0]
        for row in warehouse.db.execute(
            sa.select(sa.distinct(partial_output_table.c.sys__input_id))
        )
    ]
    # output values in partial output table
    outputs = [
        row[0] for row in warehouse.db.execute(sa.select(partial_output_table.c.result))
    ]
    # Build mapping: sys__id -> input_value from input table
    input_data = {
        row[0]: row[1]
        for row in warehouse.db.execute(
            sa.select(input_table.c.sys__id, input_table.c.num)
        )
    }

    # Verify no duplicates
    assert len(set(outputs)) == len(outputs)

    # Verify each processed sys__id has correct input and output
    for sys_id in processed_sys_ids:
        # Check input exists for this sys__id
        assert sys_id in input_data

        # Verify output value is correct (n^2)
        input_val = input_data[sys_id]
        expected_output = input_val * input_val

        assert expected_output in outputs, (
            f"For sys__id {sys_id}: input={input_val}, "
            f"expected output={expected_output}, "
            f"not found in partial output"
        )

    # Verify we processed some inputs (don't check exact count - varies by warehouse)
    assert len(processed_sys_ids) > 0, "Expected some processing before failure"


def test_udf_code_change_triggers_rerun(test_session, monkeypatch):
    """Test that changing UDF code (hash) triggers rerun from scratch."""
    map1_calls = []
    map2_calls = []

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

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


@pytest.mark.parametrize(
    "batch_size,fail_after_count",
    [
        (2, 3),  # batch_size=2: Fail after 3 rows
        (3, 4),  # batch_size=3: Fail after 4 rows
        (5, 3),  # batch_size=5: Fail after 3 rows
    ],
)
def test_udf_signals_continue_from_partial(
    test_session_tmpfile,
    monkeypatch,
    nums_dataset,
    batch_size,
    fail_after_count,
):
    """Test continuing UDF execution from partial output table.

    Tests with different batch sizes to ensure partial results are correctly handled
    regardless of batch boundaries. Uses counter-based failure to avoid dependency
    on row ordering (ClickHouse doesn't guarantee order without ORDER BY).

    Simulates real-world scenario: user writes buggy UDF, it fails, then fixes bug
    and reruns.
    """
    test_session = test_session_tmpfile
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    processed_nums = []

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

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

    # Should have processed exactly fail_after_count rows before failing
    assert len(processed_nums) == fail_after_count

    _, partial_table = get_partial_tables(test_session)
    assert 0 <= _count_partial(warehouse, partial_table) <= fail_after_count

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
    batch_size,
    fail_after_count,
):
    """Test continuing RowGenerator from partial output.

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

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    def buggy_generator(num) -> Iterator[int]:
        """
        Buggy generator that fails before processing the (fail_after_count+1)th input.
        """
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
        chain.gen(value=buggy_generator, output=int).save("gen_results")

    first_run_count = len(processed_nums)

    # Should have processed exactly fail_after_count inputs before failing
    assert first_run_count == fail_after_count

    _, partial_table = get_partial_tables(test_session)

    # Verify partial table has outputs (each input generates 2 outputs)
    # ClickHouse: saves all outputs including incomplete batch
    # SQLite: saves complete batches only (may be 0 if only incomplete batch)
    partial_count = _count_partial(warehouse, partial_table)
    max_outputs = fail_after_count * 2  # Each input yields 2 outputs
    assert 0 <= partial_count <= max_outputs

    # Verify processed table tracks completed inputs
    # ClickHouse: tracks all inputs whose outputs were saved
    # SQLite: may be 0 if incomplete batch lost (no complete inputs saved)
    processed_count = _count_processed(warehouse, partial_table, generator=True)
    assert 0 <= processed_count <= fail_after_count

    # -------------- SECOND RUN (FIXED GENERATOR) -------------------
    reset_session_job_state()

    processed_nums.clear()

    def fixed_generator(num) -> Iterator[int]:
        """Fixed generator that works correctly."""
        processed_nums.append(num)
        yield num * 10
        yield num * num

    # Now use the fixed generator - should continue from partial checkpoint
    chain.gen(value=fixed_generator, output=int).save("gen_results")

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


@pytest.mark.xfail(
    reason="Known limitation: inputs that yield nothing are not tracked "
    "in processed table"
)
def test_generator_yielding_nothing(test_session, monkeypatch, nums_dataset):
    """Test that generator correctly handles inputs that yield zero outputs."""
    warehouse = test_session.catalog.warehouse
    processed = []

    def selective_generator(num) -> Iterator[int]:
        """Generator that only yields outputs for even numbers."""
        processed.append(num)
        if num == 3:
            raise Exception("Simulated failure")
        if num % 2 == 0:  # Only even numbers yield outputs
            yield num * 10

    # First run - fails on num=3
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session).gen(
        value=selective_generator, output=int
    )

    with pytest.raises(Exception, match="Simulated failure"):
        chain.save("results")

    _, partial_table = get_partial_tables(test_session)

    # Verify processed table tracks inputs that yielded nothing
    # Inputs 1,2 were processed (1 yielded nothing, 2 yielded one output)
    assert _count_processed(warehouse, partial_table) == 2

    # Second run - should skip already processed inputs
    reset_session_job_state()
    processed.clear()
    chain.save("results")

    # Only inputs 3,4,5,6 should be processed
    assert processed == [3, 4, 5, 6]
    # Result should only have even numbers x 10
    result = sorted(dc.read_dataset("results", session=test_session).to_list("value"))
    assert result == [(20,), (40,), (60,)]


@pytest.mark.parametrize(
    "batch_size,fail_after_count",
    [
        (2, 2),  # batch_size=2: Fail after processing 2 partitions
        (3, 2),  # batch_size=3: Fail after processing 2 partitions
        (10, 2),  # batch_size=10: Fail after processing 2 partitions
    ],
)
def test_aggregator_allways_runs_from_scratch(
    test_session,
    monkeypatch,
    nums_dataset,
    batch_size,
    fail_after_count,
):
    """Test running Aggregator always from scratch"""

    processed_partitions = []

    def buggy_aggregator(letter, num) -> Iterator[tuple[str, int]]:
        """
        Buggy aggregator that fails before processing the (fail_after_count+1)th
        partition.
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

    # Now use the fixed aggregator - should run from scratch
    chain.agg(
        total=fixed_aggregator,
        partition_by="letter",
    ).save("agg_results")

    second_run_count = len(processed_partitions)

    # Verify final results: 3 partitions (A, B, C) with correct sums
    assert sorted(
        dc.read_dataset("agg_results", session=test_session).to_list(
            "total_0", "total_1"
        )
    ) == sorted(
        [
            ("A", 3),  # group A: 1 + 2 = 3
            ("B", 7),  # group B: 3 + 4 = 7
            ("C", 11),  # group C: 5 + 6 = 11
        ]
    )

    # should re-process everything
    assert second_run_count == 3


def test_multiple_udf_chain_continue(test_session, monkeypatch):
    """Test continuing from partial with multiple UDFs in chain.

    When mapper fails, only mapper's partial table exists. On retry, mapper
    completes and gen runs from scratch.
    """
    map_processed = []
    gen_processed = []
    fail_once = [True]  # Mutable flag to track if we should fail

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    def mapper(num: int) -> int:
        map_processed.append(num)
        # Fail before processing the 4th row in first run only
        if fail_once[0] and len(map_processed) == 3:
            fail_once[0] = False
            raise Exception("Map failure")
        return num * 2

    def doubler(doubled) -> Iterator[int]:
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
        .gen(value=doubler, output=int)
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


def test_udf_generator_reset_udf(test_session, monkeypatch):
    """Test that when DATACHAIN_UDF_RESET=True, we don't continue from partial
    checkpoints but re-run from scratch.
    """
    monkeypatch.setenv("DATACHAIN_UDF_RESET", "true")
    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    processed_nums = []

    def buggy_generator(num) -> Iterator[int]:
        """Buggy generator that fails on num=4."""
        processed_nums.append(num)
        if num == 4:
            raise Exception(f"Simulated failure on num={num}")
        yield num * 10
        yield num * num

    # -------------- FIRST RUN (FAILS WITH BUGGY GENERATOR) -------------------
    reset_session_job_state()

    chain = dc.read_dataset("nums", session=test_session).settings(batch_size=2)

    with pytest.raises(Exception, match="Simulated failure"):
        chain.gen(value=buggy_generator, output=int).save("gen_results")

    # -------------- SECOND RUN (FIXED GENERATOR) -------------------
    reset_session_job_state()

    processed_nums.clear()

    def fixed_generator(num) -> Iterator[int]:
        """Fixed generator that works correctly."""
        processed_nums.append(num)
        yield num * 10
        yield num * num

    chain.gen(value=fixed_generator, output=int).save("gen_results")

    # KEY DIFFERENCE: In reset mode, ALL inputs are processed again (not continuing
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


def test_generator_output_schema_change_triggers_rerun(test_session, monkeypatch):
    """Test that changing generator output type triggers re-run from scratch.

    When a user changes the output schema of a UDF (e.g., int -> str), the
    system should detect this and re-run from scratch rather than attempting
    to continue from partial results with incompatible schema.
    """
    processed_nums_v1 = []
    processed_nums_v2 = []

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN (INT OUTPUT, FAILS) -------------------
    def generator_v1_int(num) -> Iterator[int]:
        """Generator version 1: yields int, fails on num=4."""
        processed_nums_v1.append(num)
        if num == 4:
            raise Exception(f"Simulated failure on num={num}")
        yield num * 10
        yield num * num

    reset_session_job_state()

    chain = dc.read_dataset("nums", session=test_session).settings(batch_size=2)

    with pytest.raises(Exception, match="Simulated failure"):
        chain.gen(result=generator_v1_int, output=int).save("gen_results")

    # Some inputs were processed before failure
    assert len(processed_nums_v1) > 0

    # -------------- SECOND RUN (STR OUTPUT, DIFFERENT SCHEMA) -------------------
    def generator_v2_str(num) -> Iterator[str]:
        """Generator version 2: yields str instead of int (schema change!)."""
        processed_nums_v2.append(num)
        yield f"value_{num * 10}"
        yield f"square_{num * num}"

    reset_session_job_state()

    # Use generator with different output type - should run from scratch
    chain.gen(result=generator_v2_str, output=str).save("gen_results")

    # Verify ALL inputs were processed in second run (not continuing from partial)
    assert sorted(processed_nums_v2) == sorted([1, 2, 3, 4, 5, 6]), (
        "All inputs should be processed when schema changes"
    )

    # Verify final results are correct with new schema (str)
    result = sorted(
        dc.read_dataset("gen_results", session=test_session).to_list("result")
    )
    expected = sorted(
        [
            ("square_1",),
            ("value_10",),  # num=1
            ("square_4",),
            ("value_20",),  # num=2
            ("square_9",),
            ("value_30",),  # num=3
            ("square_16",),
            ("value_40",),  # num=4
            ("square_25",),
            ("value_50",),  # num=5
            ("square_36",),
            ("value_60",),  # num=6
        ]
    )
    assert result == expected


def test_mapper_output_schema_change_triggers_rerun(test_session, monkeypatch):
    """Test that changing mapper output type triggers re-run from scratch.

    Similar to generator test, but for mappers (1:1 mapping). When output
    schema changes, the system should detect this and re-run from scratch.
    """
    processed_nums_v1 = []
    processed_nums_v2 = []

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN (INT OUTPUT, FAILS) -------------------
    def mapper_v1_int(num) -> int:
        """Mapper version 1: returns int, fails on num=4."""
        processed_nums_v1.append(num)
        if num == 4:
            raise Exception(f"Simulated failure on num={num}")
        return num * 10

    reset_session_job_state()

    chain = dc.read_dataset("nums", session=test_session).settings(batch_size=2)

    with pytest.raises(Exception, match="Simulated failure"):
        chain.map(result=mapper_v1_int, output=int).save("map_results")

    # Some inputs were processed before failure
    assert len(processed_nums_v1) > 0

    # -------------- SECOND RUN (STR OUTPUT, DIFFERENT SCHEMA) -------------------
    def mapper_v2_str(num) -> str:
        """Mapper version 2: returns str instead of int (schema change!)."""
        processed_nums_v2.append(num)
        return f"value_{num * 10}"

    reset_session_job_state()

    # Use mapper with different output type - should run from scratch
    chain.map(result=mapper_v2_str, output=str).save("map_results")

    # Verify ALL inputs were processed in second run (not continuing from partial)
    assert sorted(processed_nums_v2) == sorted([1, 2, 3, 4, 5, 6]), (
        "All inputs should be processed when schema changes"
    )

    # Verify final results are correct with new schema (str)
    result = sorted(
        dc.read_dataset("map_results", session=test_session).to_list("result")
    )
    expected = sorted(
        [
            ("value_10",),  # num=1
            ("value_20",),  # num=2
            ("value_30",),  # num=3
            ("value_40",),  # num=4
            ("value_50",),  # num=5
            ("value_60",),  # num=6
        ]
    )
    assert result == expected


@pytest.mark.parametrize(
    "batch_size,fail_after_count",
    [
        (
            1,
            2,
        ),  # batch_size=1: Commit each output immediately, ensures checkpoint works
    ],
)
def test_aggregator_continue_from_partial(
    test_session,
    monkeypatch,
    nums_dataset,
    batch_size,
    fail_after_count,
):
    """Test continuing Aggregator from partial output with partition_by.

    Aggregator differs from Generator because:
    - Uses partition_by to group inputs
    - Reduces multiple inputs to one output per partition
    - Processes partitions, not individual rows

    Tests that partition_by works correctly with checkpoints and ensures
    input table is created first to maintain consistent sys__id values.

    Simulates real-world scenario: user writes buggy aggregator, it fails, then
    fixes bug and reruns.
    """
    # Reduce INSERT_BATCH_SIZE to 1 so each output is committed immediately
    # This ensures partial outputs are saved before failure
    monkeypatch.setattr("datachain.query.dataset.INSERT_BATCH_SIZE", 1)
    monkeypatch.setattr("datachain.data_storage.warehouse.INSERT_BATCH_SIZE", 1)

    processed_partitions = []

    def buggy_aggregator(letter, num) -> Iterator[tuple[str, int]]:
        """
        Buggy aggregator that fails before processing the (fail_after_count+1)th
        partition.
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

    catalog = test_session.catalog
    warehouse = catalog.warehouse

    _, partial_table = get_partial_tables(test_session)

    # Count processed partitions (via sys__input_id which should track partition_id)
    # For Aggregator with partition_by, sys__input_id tracks which partition produced
    # output
    processed_count_first = _count_processed(warehouse, partial_table, generator=True)

    # Must be > 0 to verify sys__input_id tracking is working
    assert 0 < processed_count_first <= fail_after_count, (
        f"Expected 1-{fail_after_count} processed partitions tracked, "
        f"but got {processed_count_first}. sys__input_id tracking may not be working."
    )

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
    result = sorted(
        dc.read_dataset("agg_results", session=test_session).to_list(
            "total_0", "total_1"
        )
    )
    expected = sorted(
        [
            ("A", 3),  # group A: 1 + 2 = 3
            ("B", 7),  # group B: 3 + 4 = 7
            ("C", 11),  # group C: 5 + 6 = 11
        ]
    )

    # Should have exactly 3 outputs (no duplicates)
    assert result == expected

    # KEY TEST: Verify checkpoint continuation worked
    # Second run should only process remaining partitions
    # Total processed across both runs should equal 3 partitions
    total_processed = processed_count_first + second_run_count
    assert total_processed == 3, (
        f"Expected 3 total partitions processed, but got {total_processed} "
        f"(first run: {processed_count_first}, second run: {second_run_count})"
    )
