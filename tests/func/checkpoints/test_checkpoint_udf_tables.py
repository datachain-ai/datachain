from collections.abc import Iterator

import pytest

import datachain as dc
from tests.utils import reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


def test_track_processed_items(test_session_tmpfile):
    """Test that processed items are correctly tracked.

    Verifies checkpoint recovery works by checking that second run processes
    fewer items than total and final result is correct with no duplicates.
    Note: Parallel checkpoint recovery is tested in test_checkpoint_parallel.py.
    """
    test_session = test_session_tmpfile
    processed_nums = []
    run_count = {"value": 0}

    def gen_numbers(num) -> Iterator[int]:
        processed_nums.append(num)
        if num == 50 and run_count["value"] == 0:
            raise Exception(f"Simulated failure on num={num}")
        yield num * 10

    dc.read_values(num=list(range(1, 100)), session=test_session).save("nums")

    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .order_by("num")
        .settings(batch_size=2)
    )

    # First run - fails partway through
    with pytest.raises(Exception):  # noqa: B017
        chain.gen(result=gen_numbers, output=int).save("results")

    first_run_count = len(processed_nums)
    assert 0 < first_run_count < 99

    # Second run - should continue from checkpoint
    reset_session_job_state()
    processed_nums.clear()
    run_count["value"] += 1

    chain.gen(result=gen_numbers, output=int).save("results")

    # Second run should process remaining items (not all 99)
    assert 0 < len(processed_nums) < 99

    # Verify final result is correct
    result = dc.read_dataset("results", session=test_session).to_list("result")
    assert len(result) == 99

    # Verify no duplicates
    values = [r[0] for r in result]
    assert len(values) == len(set(values))


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

    assert len(gen_processed) == 6

    result = sorted(dc.read_dataset("results", session=test_session).to_list("value"))
    assert sorted([v[0] for v in result]) == sorted(
        [2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12]
    )
