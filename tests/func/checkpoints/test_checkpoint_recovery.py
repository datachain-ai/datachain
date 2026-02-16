from collections.abc import Iterator

import pytest

import datachain as dc
from datachain.lib.file import File
from tests.utils import reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


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
    regardless of batch boundaries.
    """
    test_session = test_session_tmpfile
    processed_nums = []

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    def process_buggy(num) -> int:
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

    # -------------- SECOND RUN (FIXED UDF) -------------------
    reset_session_job_state()

    processed_nums.clear()

    def process_fixed(num) -> int:
        processed_nums.append(num)
        return num * 10

    chain.map(result=process_fixed, output=int).save("results")

    result = dc.read_dataset("results", session=test_session).to_list("result")
    assert sorted(result) == [(10,), (20,), (30,), (40,), (50,), (60,)]

    # Second run should process remaining rows (checkpoint continuation working)
    assert 0 < len(processed_nums) <= 6


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
    processed_nums = []

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    def buggy_generator(num) -> Iterator[int]:
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

    assert len(processed_nums) == fail_after_count

    # -------------- SECOND RUN (FIXED GENERATOR) -------------------
    reset_session_job_state()

    processed_nums.clear()

    def fixed_generator(num) -> Iterator[int]:
        processed_nums.append(num)
        yield num * 10
        yield num * num

    chain.gen(value=fixed_generator, output=int).save("gen_results")

    result = sorted(
        dc.read_dataset("gen_results", session=test_session).to_list("value")
    )
    expected = sorted(
        [
            (1,),
            (10,),
            (4,),
            (20,),
            (9,),
            (30,),
            (16,),
            (40,),
            (25,),
            (50,),
            (36,),
            (60,),
        ]
    )

    assert result == expected

    # Second run should process remaining inputs (checkpoint continuation working)
    assert 0 < len(processed_nums) <= 6


def test_generator_incomplete_input_recovery(test_session):
    """Test full recovery flow from incomplete inputs.

    Tests the complete checkpoint recovery mechanism:
    1. First run fails, leaving some inputs incomplete (missing final row)
    2. Second run detects incomplete inputs
    3. Filters out partial results from incomplete inputs
    4. Re-processes incomplete inputs
    5. Final results are correct (no duplicates, no missing values)
    """
    processed_inputs = []
    run_count = [0]
    numbers = [6, 2, 8, 7]

    def gen_multiple(num) -> Iterator[int]:
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
    processed = []

    def selective_generator(num) -> Iterator[int]:
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

    # Second run - should continue from checkpoint
    reset_session_job_state()
    processed.clear()
    chain.save("results")

    # Only inputs 3,4,5,6 should be processed (1,2 were already done)
    assert processed == [3, 4, 5, 6]
    result = sorted(dc.read_dataset("results", session=test_session).to_list("value"))
    assert result == [(20,), (40,), (60,)]


def test_empty_dataset_checkpoint(test_session):
    processed = []

    def mapper(num) -> int:
        processed.append(num)
        return num * 10

    dc.read_values(num=[], session=test_session).save("empty_nums")

    reset_session_job_state()
    chain = dc.read_dataset("empty_nums", session=test_session).map(
        result=mapper, output=int
    )
    chain.save("results")

    assert len(processed) == 0

    # Second run should also work (checkpoint reuse with empty result)
    reset_session_job_state()
    processed.clear()
    chain.save("results")

    assert len(processed) == 0

    result = dc.read_dataset("results", session=test_session).to_list("result")
    assert result == []


def test_single_row_dataset_checkpoint(test_session):
    processed = []
    run_count = {"value": 0}

    def mapper(num) -> int:
        processed.append(num)
        if run_count["value"] == 0:
            raise Exception("First run failure")
        return num * 10

    dc.read_values(num=[42], session=test_session).save("single_num")

    reset_session_job_state()
    chain = (
        dc.read_dataset("single_num", session=test_session)
        .settings(
            batch_size=10  # Batch size larger than dataset
        )
        .map(result=mapper, output=int)
    )

    with pytest.raises(Exception, match="First run failure"):
        chain.save("results")

    assert len(processed) == 1

    reset_session_job_state()
    processed.clear()
    run_count["value"] += 1

    chain.save("results")

    result = dc.read_dataset("results", session=test_session).to_list("result")
    assert result == [(420,)]


def test_multiple_consecutive_failures(test_session):
    """Test checkpoint recovery across multiple consecutive failures.

    Scenario: fail at row 3, then fail at row 5, then succeed.
    Each run should continue from where the previous one left off.
    """
    processed = []
    run_count = {"value": 0}

    def flaky_mapper(num) -> int:
        processed.append(num)
        if run_count["value"] == 0 and len(processed) >= 3:
            raise Exception("First failure at row 3")
        if run_count["value"] == 1 and len(processed) >= 3:
            raise Exception("Second failure at row 3 (of remaining)")
        return num * 10

    dc.read_values(num=[1, 2, 3, 4, 5, 6, 7, 8], session=test_session).save("nums")

    chain = dc.read_dataset("nums", session=test_session).settings(batch_size=2)

    # -------------- FIRST RUN: Fails after processing 3 rows -------------------
    reset_session_job_state()

    with pytest.raises(Exception, match="First failure"):
        chain.map(result=flaky_mapper, output=int).save("results")

    first_run_processed = len(processed)
    assert first_run_processed == 3

    # -------------- SECOND RUN: Continues but fails again -------------------
    reset_session_job_state()
    processed.clear()
    run_count["value"] += 1

    with pytest.raises(Exception, match="Second failure"):
        chain.map(result=flaky_mapper, output=int).save("results")

    second_run_processed = len(processed)
    # Should process some rows (continuing from first run's checkpoint)
    assert second_run_processed > 0

    # -------------- THIRD RUN: Finally succeeds -------------------
    reset_session_job_state()
    processed.clear()
    run_count["value"] += 1

    chain.map(result=flaky_mapper, output=int).save("results")

    # Verify final result is correct
    result = dc.read_dataset("results", session=test_session).to_list("result")
    assert sorted(result) == [(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,)]


def test_generator_multiple_consecutive_failures(test_session):
    processed = []
    run_count = {"value": 0}

    def flaky_generator(num) -> Iterator[int]:
        processed.append(num)
        if run_count["value"] == 0 and num == 3:
            raise Exception("First failure on num=3")
        if run_count["value"] == 1 and num == 5:
            raise Exception("Second failure on num=5")
        yield num * 10
        yield num * 100

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    chain = (
        dc.read_dataset("nums", session=test_session)
        .order_by("num")
        .settings(batch_size=2)
    )

    # -------------- FIRST RUN: Fails on num=3 -------------------
    reset_session_job_state()

    with pytest.raises(Exception, match="First failure"):
        chain.gen(result=flaky_generator, output=int).save("results")

    # -------------- SECOND RUN: Continues but fails on num=5 -------------------
    reset_session_job_state()
    processed.clear()
    run_count["value"] += 1

    with pytest.raises(Exception, match="Second failure"):
        chain.gen(result=flaky_generator, output=int).save("results")

    # -------------- THIRD RUN: Finally succeeds -------------------
    reset_session_job_state()
    processed.clear()
    run_count["value"] += 1

    chain.gen(result=flaky_generator, output=int).save("results")

    # Verify final result is correct (each input produces 2 outputs)
    result = dc.read_dataset("results", session=test_session).to_list("result")
    expected = [(i * 10,) for i in range(1, 7)] + [(i * 100,) for i in range(1, 7)]
    assert sorted(result) == sorted(expected)

    # Verify no duplicates
    values = [r[0] for r in result]
    assert len(values) == len(set(values))


def test_multiple_udf_chain_continue(test_session):
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

    assert len(gen_processed) == 6

    result = sorted(dc.read_dataset("results", session=test_session).to_list("value"))
    assert sorted([v[0] for v in result]) == sorted(
        [2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12]
    )


def test_file_udf_continue_from_partial(test_session, tmp_dir):
    """Test checkpoint continuation with File objects (file downloading UDFs).

    Ensures that File objects are correctly reconstructed from the checkpoint's
    input table on the second run (regression test for bytes vs str path issue).
    """
    # Create test files
    file_names = [f"file_{i}.txt" for i in range(6)]
    for name in file_names:
        (tmp_dir / name).write_text(f"content of {name}", encoding="utf-8")

    processed_files = []

    def process_file(file: File) -> int:
        if len(processed_files) >= 3:
            raise Exception("Simulated failure after 3 files")
        data = file.read()
        processed_files.append(file.path)
        return len(data)

    chain = (
        dc.read_storage(tmp_dir.as_uri(), session=test_session)
        .order_by("file.path")
        .settings(batch_size=2)
    )

    # -------------- FIRST RUN (FAILS AFTER 3 FILES) -------------------
    reset_session_job_state()

    with pytest.raises(Exception, match="Simulated failure after 3 files"):
        chain.map(file_size=process_file).save("file_results")

    assert len(processed_files) == 3

    # -------------- SECOND RUN (CONTINUES FROM CHECKPOINT) -------------------
    reset_session_job_state()
    processed_files.clear()

    def process_file_fixed(file: File) -> int:
        data = file.read()
        processed_files.append(file.path)
        return len(data)

    chain.map(file_size=process_file_fixed).save("file_results")

    result = dc.read_dataset("file_results", session=test_session).to_list("file_size")
    assert len(result) == 6

    # Second run should only process remaining files
    assert 0 < len(processed_files) <= 6


def test_skip_udf_fallback_when_output_table_missing(test_session):
    call_count = {"value": 0}

    def mapper(num) -> int:
        call_count["value"] += 1
        return num * 10

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    chain = dc.read_dataset("nums", session=test_session).map(result=mapper, output=int)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    assert chain.count() == 6
    assert call_count["value"] == 6

    catalog = test_session.catalog

    # Drop all UDF output tables from first run
    for table_name in catalog.warehouse.db.list_tables(prefix="udf_"):
        if "_output" in table_name and "_partial" not in table_name:
            table = catalog.warehouse.db.get_table(table_name)
            catalog.warehouse.db.drop_table(table, if_exists=True)

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    call_count["value"] = 0

    result = chain.order_by("num").to_list("result")

    # UDF should have been re-executed from scratch (fallback from skip)
    assert call_count["value"] == 6
    assert result == [(10,), (20,), (30,), (40,), (50,), (60,)]


def test_continue_udf_fallback_when_partial_table_missing(test_session):
    fail_flag = [True]

    def mapper(num) -> int:
        if fail_flag[0] and num >= 4:
            raise RuntimeError("Simulated failure")
        return num * 10

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    chain = dc.read_dataset("nums", session=test_session).settings(batch_size=2)

    # -------------- FIRST RUN (FAILS) -------------------
    reset_session_job_state()
    with pytest.raises(RuntimeError, match="Simulated failure"):
        chain.map(result=mapper, output=int).save("results")

    catalog = test_session.catalog
    test_session.get_or_create_job()

    # Drop all partial output tables from first run
    for table_name in catalog.warehouse.db.list_tables(prefix="udf_"):
        if "_partial" in table_name:
            table = catalog.warehouse.db.get_table(table_name)
            catalog.warehouse.db.drop_table(table, if_exists=True)

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    fail_flag[0] = False

    chain.map(result=mapper, output=int).save("results")

    # UDF should have been re-executed from scratch (fallback from continue)
    result = dc.read_dataset("results", session=test_session).to_list("result")
    assert sorted(result) == [(10,), (20,), (30,), (40,), (50,), (60,)]


def test_aggregator_checkpoint_no_partial_continuation(test_session):
    call_count = {"value": 0}

    def sum_by_group(num: list[int]) -> Iterator[tuple[int, int]]:
        call_count["value"] += 1
        total = sum(num)
        count = len(num)
        yield total, count

    dc.read_values(
        num=[1, 2, 3, 4, 5, 6],
        category=["a", "a", "a", "b", "b", "b"],
        session=test_session,
    ).save("grouped_nums")

    chain = dc.read_dataset("grouped_nums", session=test_session)

    # -------------- FIRST RUN (FAILS) -------------------
    reset_session_job_state()

    fail_flag = [True]

    def sum_by_group_buggy(num: list[int]) -> Iterator[tuple[int, int]]:
        call_count["value"] += 1
        if fail_flag[0] and call_count["value"] >= 2:
            raise RuntimeError("Simulated aggregator failure")
        total = sum(num)
        count = len(num)
        yield total, count

    with pytest.raises(RuntimeError, match="Simulated aggregator failure"):
        chain.agg(
            sum_by_group_buggy,
            partition_by="category",
            output={"total": int, "count": int},
        ).save("agg_results")

    # -------------- SECOND RUN (FIXED) -------------------
    reset_session_job_state()
    call_count["value"] = 0
    fail_flag[0] = False

    chain.agg(
        sum_by_group,
        partition_by="category",
        output={"total": int, "count": int},
    ).save("agg_results")

    # Aggregator should have run from scratch (not continued from partial)
    # because partition_by prevents partial continuation
    assert call_count["value"] == 2  # Two groups: "a" and "b"

    result = sorted(
        dc.read_dataset("agg_results", session=test_session).to_list("total", "count")
    )
    # Group "a": sum(1,2,3)=6, count=3; Group "b": sum(4,5,6)=15, count=3
    assert result == [(6, 3), (15, 3)]
