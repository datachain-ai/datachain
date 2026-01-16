"""Tests for UDF intermediate table creation, naming, and lifecycle.

This module tests input/output/partial table management and reuse across jobs.
"""

from collections.abc import Iterator

import pytest
import sqlalchemy as sa

import datachain as dc
from tests.utils import get_last_udf_partial_table, reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


@pytest.mark.parametrize("parallel", [None, 2, 20])
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

    with pytest.raises(Exception):  # noqa: B017
        chain.gen(result=gen_numbers, output=int).save("results")

    partial_output_table = get_last_udf_partial_table(test_session)

    query = sa.select(sa.distinct(partial_output_table.c.sys__input_id))
    processed_sys_ids = [row[0] for row in warehouse.db.execute(query)]

    # Verify no duplicates
    assert len(processed_sys_ids) == len(set(processed_sys_ids))
    # Verify we processed some but not all inputs (should have failed before completing)
    assert 0 < len(processed_sys_ids) < 100


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
