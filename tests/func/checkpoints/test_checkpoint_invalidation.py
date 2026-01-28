from collections.abc import Iterator

import pytest

import datachain as dc
from tests.utils import reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


def test_udf_code_change_triggers_rerun(test_session, monkeypatch):
    map1_calls = []
    map2_calls = []

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    chain = dc.read_dataset("nums", session=test_session).settings(batch_size=2)

    # Run 1: map1 succeeds, map2 fails
    def mapper1_v1(num: int) -> int:
        map1_calls.append(num)
        return num * 2

    def mapper2_failing(doubled: int) -> int:
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


def test_generator_output_schema_change_triggers_rerun(test_session, monkeypatch):
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
    processed_nums_v1 = []
    processed_nums_v2 = []

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN (INT OUTPUT, FAILS) -------------------
    def mapper_v1_int(num) -> int:
        processed_nums_v1.append(num)
        if num == 4:
            raise Exception(f"Simulated failure on num={num}")
        return num * 10

    reset_session_job_state()

    chain = dc.read_dataset("nums", session=test_session).settings(batch_size=2)

    with pytest.raises(Exception, match="Simulated failure"):
        chain.map(result=mapper_v1_int, output=int).save("map_results")

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
        (2, 2),  # batch_size=2: Fail after processing 2 partitions
        (3, 2),  # batch_size=3: Fail after processing 2 partitions
        (10, 2),  # batch_size=10: Fail after processing 2 partitions
    ],
)
def test_aggregator_always_runs_from_scratch(
    test_session,
    monkeypatch,
    nums_dataset,
    batch_size,
    fail_after_count,
):
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
        nums_list = list(num)
        processed_partitions.append(nums_list)
        # Yield tuple of (letter, sum) to preserve partition key in output
        yield letter[0], sum(n for n in nums_list)

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

    assert first_run_count == fail_after_count

    # -------------- SECOND RUN (FIXED AGGREGATOR) -------------------
    reset_session_job_state()

    processed_partitions.clear()

    chain.agg(
        total=fixed_aggregator,
        partition_by="letter",
    ).save("agg_results")

    second_run_count = len(processed_partitions)

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
