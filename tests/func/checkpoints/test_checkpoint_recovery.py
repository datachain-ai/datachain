"""Tests for resuming from partial results after failures.

This module tests partial table continuation and sys__partial tracking.
"""

from collections.abc import Iterator

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.query.dataset import UDFStep
from tests.utils import get_partial_tables, reset_session_job_state


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

    # Mapper: count all rows (1:1 mapping)
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

    def gen_multiple(num) -> Iterator[int]:
        """Generator that yields 5 outputs per input."""
        processed_inputs.append(num)
        for i in range(5):
            # Fail on input 4 after yielding 2 partial outputs (on first run only)
            if num == 4 and i == 2 and run_count[0] == 0:
                raise Exception("Simulated crash")
            yield num * 100 + i

    dc.read_values(num=[1, 2, 3, 4], session=test_session).save("nums")

    # -------------- FIRST RUN (FAILS) -------------------
    reset_session_job_state()
    processed_inputs.clear()

    with pytest.raises(Exception, match="Simulated crash"):
        (
            dc.read_dataset("nums", session=test_session)
            .order_by("num")  # Ensure deterministic ordering
            .settings(batch_size=2)  # Small batch for partial commits
            .gen(result=gen_multiple, output=int)
            .save("results")
        )

    # Verify partial state exists
    _, partial_table = get_partial_tables(test_session)
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

    # Identify incomplete inputs (missing sys__partial=False)
    incomplete_before = [
        row[0]
        for row in warehouse.db.execute(
            sa.select(sa.distinct(partial_table.c.sys__input_id)).where(
                partial_table.c.sys__input_id.not_in(
                    sa.select(partial_table.c.sys__input_id).where(
                        partial_table.c.sys__partial == False  # noqa: E712
                    )
                )
            )
        )
    ]
    assert len(incomplete_before) > 0, "Should have incomplete inputs"

    # -------------- SECOND RUN (RECOVERS) -------------------
    reset_session_job_state()
    processed_inputs.clear()
    run_count[0] += 1  # Increment so generator succeeds this time

    # Should complete successfully
    (
        dc.read_dataset("nums", session=test_session)
        .order_by("num")  # Ensure deterministic ordering
        .settings(batch_size=2)
        .gen(result=gen_multiple, output=int)
        .save("results")
    )

    # Verify incomplete inputs were re-processed
    assert any(inp in processed_inputs for inp in incomplete_before), (
        "Incomplete inputs should be re-processed"
    )

    # Verify final results
    result = (
        dc.read_dataset("results", session=test_session)
        .order_by("result")
        .to_list("result")
    )

    # Should have exactly 20 outputs (4 inputs x 5 outputs each)
    expected = sorted([(num * 100 + i,) for num in [1, 2, 3, 4] for i in range(5)])
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

    for input_id in [1, 2, 3, 4]:
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


def test_generator_sys_partial_flag_correctness(test_session):
    """Test that sys__partial flag is correctly set for generator outputs.

    Verifies that for each input:
    - All outputs except the last have sys__partial=True
    - The last output has sys__partial=False
    - This enables detection of incomplete inputs during checkpoint recovery
    """
    warehouse = test_session.catalog.warehouse

    def gen_multiple(num) -> Iterator[int]:
        """Generator that yields multiple outputs per input."""
        for i in range(5):  # Each input yields 5 outputs
            # Fail on input 4 after yielding 2 partial outputs
            # (after successfully processing inputs 1, 2, 3)
            if num == 4 and i == 2:
                raise Exception("Intentional failure to preserve partial table")
            yield num * 100 + i

    dc.read_values(num=[1, 2, 3, 4], session=test_session).save("nums")

    reset_session_job_state()

    # Run and expect failure - this leaves partial table
    # Use small batch size to force commits between inputs
    with pytest.raises(Exception):  # noqa: B017
        (
            dc.read_dataset("nums", session=test_session)
            .order_by("num")  # Ensure deterministic ordering
            .settings(batch_size=2)  # Very small batch size
            .gen(result=gen_multiple, output=int)
            .save("results")
        )

    # Get the partial table to inspect sys__partial flags
    _, partial_table = get_partial_tables(test_session)

    # Query all rows with their sys__partial flags
    rows = list(
        warehouse.db.execute(
            sa.select(
                partial_table.c.sys__input_id,
                partial_table.c.result,
                partial_table.c.sys__partial,
            ).order_by(partial_table.c.sys__input_id, partial_table.c.result)
        )
    )

    # Group by input
    by_input = {}
    for input_id, result, partial in rows:
        by_input.setdefault(input_id, []).append((result, partial))

    # Verify we have data for some inputs
    assert len(by_input) >= 1, f"Should have at least 1 input, got {len(by_input)}"

    # Check complete inputs (those with 5 outputs)
    complete_inputs = {k: v for k, v in by_input.items() if len(v) == 5}
    incomplete_inputs = {k: v for k, v in by_input.items() if len(v) < 5}

    assert complete_inputs
    assert incomplete_inputs

    # Verify complete inputs have correct sys__partial flags
    for input_id, outputs in complete_inputs.items():
        assert len(outputs) == 5, f"Complete input {input_id} should have 5 outputs"
        # First 4 should be True, last one should be False
        for i, (_, partial) in enumerate(outputs):
            if i < 4:
                assert partial, (
                    f"Output {i} of input {input_id} should have sys__partial=True"
                )
            else:
                assert not partial, (
                    f"Last output of input {input_id} should have sys__partial=False"
                )

    # Verify incomplete inputs have ALL outputs marked as partial=True
    for input_id, outputs in incomplete_inputs.items():
        assert len(outputs) < 5, f"Incomplete input {input_id} should have < 5 outputs"
        # ALL should be True (missing the final False marker)
        for _, (_, partial) in enumerate(outputs):
            assert partial, (
                f"All outputs of incomplete input {input_id} "
                f"should have sys__partial=True"
            )
