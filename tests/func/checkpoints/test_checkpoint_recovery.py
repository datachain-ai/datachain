from collections.abc import Iterator

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.query.dataset import UDFStep
from tests.utils import get_last_udf_partial_table, reset_session_job_state


class CustomMapperError(Exception):
    pass


def mapper_fail(num) -> int:
    raise CustomMapperError("Error")


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

    return warehouse.table_rows_count(partial_table)


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

    assert len(processed_nums) == fail_after_count

    partial_table = get_last_udf_partial_table(test_session)
    assert 0 <= _count_partial(warehouse, partial_table) <= fail_after_count

    # -------------- SECOND RUN (FIXED UDF) -------------------
    reset_session_job_state()

    processed_nums.clear()

    def process_fixed(num) -> int:
        """Fixed version that works correctly."""
        processed_nums.append(num)
        return num * 10

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

    Tests with different batch sizes to ensure processed table correctly
    tracks inputs only after ALL their outputs have been committed.

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

    assert first_run_count == fail_after_count

    partial_table = get_last_udf_partial_table(test_session)

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

    assert result == expected

    assert 0 < len(processed_nums) <= 6, "Expected 1-6 inputs in second run"


def test_generator_incomplete_input_recovery(test_session):
    """Test full recovery flow from incomplete inputs.

    Tests the complete checkpoint recovery mechanism:
    1. First run fails, leaving some inputs incomplete (missing final row)
    2. Second run detects incomplete inputs
    3. Filters out partial results from incomplete inputs
    4. Re-processes incomplete inputs
    5. Final results are correct (no duplicates, no missing values)
    """
    warehouse = test_session.catalog.warehouse
    processed_inputs = []
    run_count = [0]
    numbers = [6, 2, 8, 7]

    def gen_multiple(num) -> Iterator[int]:
        """Generator that yields 5 outputs per input."""
        processed_inputs.append(num)
        for i in range(5):
            if num == 8 and i == 2 and run_count[0] == 0:
                raise Exception("Simulated crash")
            yield num * 100 + i

    dc.read_values(num=numbers, session=test_session).save("nums")

    # -------------- FIRST RUN (FAILS) -------------------
    reset_session_job_state()
    processed_inputs.clear()

    with pytest.raises(Exception, match="Simulated crash"):
        (
            dc.read_dataset("nums", session=test_session)
            .order_by("num")
            .settings(batch_size=2)  # Small batch for partial commits
            .gen(result=gen_multiple, output=int)
            .save("results")
        )

    partial_table = get_last_udf_partial_table(test_session)
    first_run_rows = list(
        warehouse.db.execute(
            sa.select(
                partial_table.c.sys__input_id,
                partial_table.c.result,
                partial_table.c.sys__partial,
            )
        )
    )
    assert len(first_run_rows) > 0, "Should have partial data from first run"

    # With order_by("num") and batch_size=2, sorted order is [2, 6, 7, 8]:
    # - Batch 1: [2, 6] - fully committed before crash
    # - Batch 2: [7, 8] - 7 completes but batch crashes on 8, entire batch uncommitted
    # Both inputs in the crashed batch need re-processing.
    incomplete_batch = [7, 8]
    complete_batch = [2, 6]

    # -------------- SECOND RUN (RECOVERS) -------------------
    reset_session_job_state()
    processed_inputs.clear()
    run_count[0] += 1  # Increment so generator succeeds this time

    # Should complete successfully
    (
        dc.read_dataset("nums", session=test_session)
        .order_by("num")
        .settings(batch_size=2)
        .gen(result=gen_multiple, output=int)
        .save("results")
    )

    # Verify inputs from crashed batch are re-processed
    assert any(inp in processed_inputs for inp in incomplete_batch), (
        f"Inputs from crashed batch {incomplete_batch} should be re-processed, "
        f"but only processed: {processed_inputs}"
    )

    # Verify inputs from committed batch are NOT re-processed
    # (tests sys__partial flag correctness - complete inputs are correctly skipped)
    for inp in complete_batch:
        assert inp not in processed_inputs, (
            f"Input {inp} from committed batch should NOT be re-processed, "
            f"but was found in processed: {processed_inputs}"
        )

    result = (
        dc.read_dataset("results", session=test_session)
        .order_by("result")
        .to_list("result")
    )

    expected = sorted([(num * 100 + i,) for num in numbers for i in range(5)])
    actual = sorted(result)

    assert actual == expected, (
        f"Should have all 20 outputs with no duplicates or missing.\n"
        f"Expected: {expected}\n"
        f"Actual: {actual}"
    )

    # Verify each input has exactly 5 outputs
    result_by_input = {}
    for (val,) in result:
        input_id = val // 100
        result_by_input.setdefault(input_id, []).append(val)

    for input_id in numbers:
        assert len(result_by_input.get(input_id, [])) == 5, (
            f"Input {input_id} should have exactly 5 outputs"
        )

    # Verify no duplicates
    all_results = [val for (val,) in result]
    assert len(all_results) == len(set(all_results)), "Should have no duplicate results"


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

    partial_table = get_last_udf_partial_table(test_session)

    assert _count_processed(warehouse, partial_table) == 2

    reset_session_job_state()
    processed.clear()
    chain.save("results")

    # Only inputs 3,4,5,6 should be processed
    assert processed == [3, 4, 5, 6]
    result = sorted(dc.read_dataset("results", session=test_session).to_list("value"))
    assert result == [(20,), (40,), (60,)]
