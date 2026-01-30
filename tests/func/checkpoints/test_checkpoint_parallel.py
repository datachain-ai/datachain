from collections.abc import Iterator

import pytest

import datachain as dc
from datachain.error import DatasetNotFoundError
from tests.utils import reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


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
    test_session = test_session_tmpfile

    processed_nums = []
    run_count = {"count": 0}

    def gen_multiple(num) -> Iterator[int]:
        """Generator that yields multiple outputs per input."""
        processed_nums.append(num)
        # Fail on input 4 in first run only
        if num == 4 and run_count["count"] == 0:
            raise Exception(f"Simulated failure on num={num}")
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

    # -------------- SECOND RUN (CONTINUE) -------------------
    reset_session_job_state()

    processed_nums.clear()
    run_count["count"] += 1

    # Should complete successfully
    chain.save("results")

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
def test_parallel_checkpoint_recovery_no_duplicates(test_session_tmpfile, parallel):
    """Test that parallel checkpoint recovery processes all inputs exactly once.

    Verifies:
    - No duplicate outputs in final result
    - All inputs produce correct outputs (n^2)
    - Correct total number of outputs (100)
    """
    test_session = test_session_tmpfile

    # Track run count to fail only on first run
    run_count = {"value": 0}

    def gen_square(num) -> Iterator[int]:
        # Fail on input 95 during first run only
        if num == 95 and run_count["value"] == 0:
            raise Exception(f"Simulated failure on num={num}")

        yield num * num

    dc.read_values(num=list(range(1, 101)), session=test_session).save("nums")
    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .order_by("num")
        .settings(parallel=parallel, batch_size=2)
        .gen(result=gen_square, output=int)
    )

    # First run - fails on num=95
    with pytest.raises(RuntimeError):
        chain.save("results")

    # Second run - should recover and complete
    reset_session_job_state()
    run_count["value"] += 1
    chain.save("results")

    # Verify: Final result has correct number of outputs and values
    result = dc.read_dataset("results", session=test_session).to_list("result")
    assert len(result) == 100, f"Expected 100 outputs, got {len(result)}"

    # Verify: No duplicate outputs
    output_values = [row[0] for row in result]
    assert len(output_values) == len(set(output_values)), (
        "Found duplicate outputs in final result"
    )

    # Verify: All expected outputs present (1^2, 2^2, ..., 100^2)
    expected = {i * i for i in range(1, 101)}
    actual = set(output_values)
    assert actual == expected, f"Outputs don't match. Missing: {expected - actual}"
