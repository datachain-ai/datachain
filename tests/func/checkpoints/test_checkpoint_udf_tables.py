"""Tests for UDF intermediate table creation, naming, and lifecycle.

This module tests input/output/partial table management and reuse across jobs.
"""

from collections.abc import Iterator

import pytest
import sqlalchemy as sa

import datachain as dc
from tests.utils import get_partial_tables, reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


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

    _, partial_output_table = get_partial_tables(test_session)

    query = sa.select(sa.distinct(partial_output_table.c.sys__input_id))
    processed_sys_ids = [row[0] for row in warehouse.db.execute(query)]

    # Verify no duplicates
    assert len(processed_sys_ids) == len(set(processed_sys_ids))
    # Verify we processed some but not all inputs (should have failed before completing)
    assert 0 < len(processed_sys_ids) < 100


def test_udf_tables_naming(test_session, monkeypatch):
    catalog = test_session.catalog
    warehouse = catalog.warehouse

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("num.num.numbers")

    from tests.utils import list_tables

    initial_udf_tables = set(list_tables(warehouse.db, prefix="udf_"))

    def get_udf_tables():
        tables = set(list_tables(warehouse.db, prefix="udf_"))
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
