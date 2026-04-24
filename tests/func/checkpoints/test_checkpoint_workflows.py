from collections.abc import Iterator
from uuid import uuid4

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.error import (
    DatasetNotFoundError,
    JobNotFoundError,
)
from tests.utils import reset_session_job_state


def _count_rows(metastore, table) -> int:
    query = sa.select(sa.func.count()).select_from(table)
    return next(iter(metastore.db.execute(query)))[0]


class CustomMapperError(Exception):
    pass


def mapper_fail(num: int) -> int:
    raise CustomMapperError("Error")


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


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

    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(reset_checkpoints))

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
    with pytest.raises(CustomMapperError):
        chain.map(new=mapper_fail).save("nums3")
    first_job = test_session.get_or_create_job()
    first_job_id = first_job.id

    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    if use_datachain_job_id_env:
        monkeypatch.setenv(
            "DATACHAIN_JOB_ID",
            metastore.create_job(
                "my-job",
                "echo 1;",
                rerun_from_job_id=first_job_id,
                run_group_id=first_job.run_group_id,
            ),
        )
    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    second_job_id = test_session.get_or_create_job().id

    expected_versions = 1 if with_delta or not reset_checkpoints else 2
    assert (
        len(catalog.get_dataset("nums1", versions=None).versions) == expected_versions
    )
    assert (
        len(catalog.get_dataset("nums2", versions=None).versions) == expected_versions
    )
    assert len(catalog.get_dataset("nums3", versions=None).versions) == 1

    assert len(list(catalog.metastore.list_checkpoints([first_job_id]))) == 3
    assert len(list(catalog.metastore.list_checkpoints([second_job_id]))) == 3


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_modified_chains(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(reset_checkpoints))

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

    assert (
        len(catalog.get_dataset("nums1", versions=None).versions) == 2
        if reset_checkpoints
        else 1
    )
    assert len(catalog.get_dataset("nums2", versions=None).versions) == 2
    assert (
        len(catalog.get_dataset("nums3", versions=None).versions) == 2
        if reset_checkpoints
        else 1
    )
    assert len(list(catalog.metastore.list_checkpoints([first_job_id]))) == 3
    assert len(list(catalog.metastore.list_checkpoints([second_job_id]))) == 3


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_multiple_runs(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog

    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(reset_checkpoints))

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.save("nums2")
    with pytest.raises(CustomMapperError):
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
    with pytest.raises(CustomMapperError):
        chain.map(new=mapper_fail).save("nums3")
    third_job_id = test_session.get_or_create_job().id

    # -------------- FOURTH RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")
    chain.save("nums3")
    fourth_job_id = test_session.get_or_create_job().id

    num1_versions = len(catalog.get_dataset("nums1", versions=None).versions)
    num2_versions = len(catalog.get_dataset("nums2", versions=None).versions)
    num3_versions = len(catalog.get_dataset("nums3", versions=None).versions)

    if reset_checkpoints:
        assert num1_versions == 4
        assert num2_versions == 4
        assert num3_versions == 2

    else:
        assert num1_versions == 1
        assert num2_versions == 2
        assert num3_versions == 2

    assert len(list(catalog.metastore.list_checkpoints([first_job_id]))) == 3
    assert len(list(catalog.metastore.list_checkpoints([second_job_id]))) == 3
    assert len(list(catalog.metastore.list_checkpoints([third_job_id]))) == 3
    assert len(list(catalog.metastore.list_checkpoints([fourth_job_id]))) == 3


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


def test_checkpoint_with_deleted_dataset_version(test_session, nums_dataset):
    catalog = test_session.catalog

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN: Create dataset -------------------
    reset_session_job_state()
    chain.save("nums_deleted")
    test_session.get_or_create_job()

    dataset = catalog.get_dataset("nums_deleted", versions=None)
    assert len(dataset.versions) == 1
    assert dataset.latest_version == "1.0.0"

    catalog.remove_dataset("nums_deleted", version="1.0.0", force=True)

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums_deleted")

    # -------------- SECOND RUN: Checkpoint exists but version gone
    reset_session_job_state()
    chain.save("nums_deleted")
    job2_id = test_session.get_or_create_job().id

    # Should create a NEW version since old one was deleted
    dataset = catalog.get_dataset("nums_deleted", versions=None)
    assert len(dataset.versions) == 1
    assert dataset.latest_version == "1.0.0"

    new_version = dataset.get_version("1.0.0")
    assert new_version.job_id == job2_id


def test_udf_checkpoints_multiple_calls_same_job(
    test_session, monkeypatch, nums_dataset
):
    """
    Test that UDF execution creates checkpoints and subsequent calls in the
    same job reuse the cached result ("done" checkpoints act as a cache).
    """
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

    # Second count() - skips UDF, reuses cached result
    call_count["count"] = 0
    assert chain.count() == 6
    assert call_count["count"] == 0, "Mapper skipped — cached in same job"

    # Third count() - also skips
    call_count["count"] = 0
    assert chain.count() == 6
    assert call_count["count"] == 0, "Mapper skipped — cached in same job"

    # Other operations like to_list() also reuse cached output
    call_count["count"] = 0
    result = chain.order_by("num").to_list("plus_ten")
    assert result == [(11,), (12,), (13,), (14,), (15,), (16,)]
    assert call_count["count"] == 0, "Mapper skipped — cached in same job"


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_udf_checkpoints_cross_job_reuse(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_IGNORE_CHECKPOINTS", str(reset_checkpoints))

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

    checkpoints = list(catalog.metastore.list_checkpoints([first_job_id]))
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

    checkpoints_second = list(catalog.metastore.list_checkpoints([second_job_id]))
    if reset_checkpoints:
        # Ran from scratch — created its own final checkpoint
        assert len(checkpoints_second) == 1
        assert checkpoints_second[0].partial is False
    else:
        # Skipped — reused first job's checkpoint, no new record created
        assert len(checkpoints_second) == 0

    # Verify the data is correct
    result = chain.order_by("num").to_list("doubled")
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_checkpoints_job_without_run_group_id(test_session, monkeypatch, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    call_count = {"count": 0}

    def double_num(num) -> int:
        call_count["count"] += 1
        return num * 2

    chain = dc.read_dataset("nums", session=test_session).map(
        doubled=double_num, output=int
    )

    # -------------- FIRST RUN (from scratch, no run_group_id) -------------------
    reset_session_job_state()

    first_job_id = str(uuid4())
    metastore.create_job(
        "scheduled-task",
        "echo 1;",
        job_id=first_job_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", first_job_id)

    chain.save("doubled_nums")
    first_job = metastore.get_job(first_job_id)
    assert first_job.run_group_id == first_job_id
    assert call_count["count"] == 6

    # -------------- SECOND RUN (skip, no run_group_id) -------------------
    reset_session_job_state()
    call_count["count"] = 0

    # Create rerun job — also without run_group_id (inherits None from parent)
    second_job_id = str(uuid4())
    metastore.create_job(
        "scheduled-task",
        "echo 1;",
        job_id=second_job_id,
        rerun_from_job_id=first_job_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", second_job_id)

    chain.save("doubled_nums")
    second_job = metastore.get_job(second_job_id)
    assert second_job.run_group_id == second_job_id
    assert second_job.rerun_from_job_id == first_job_id

    # UDF should be skipped via checkpoint
    assert call_count["count"] == 0

    result = chain.order_by("num").to_list("doubled")
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_checkpoints_job_without_run_group_id_continue(
    test_session, monkeypatch, nums_dataset
):
    catalog = test_session.catalog
    metastore = catalog.metastore

    processed_count = {"count": 0}
    should_fail = [True]

    def double_num(num) -> int:
        processed_count["count"] += 1
        if should_fail[0] and processed_count["count"] > 3:
            raise RuntimeError("Simulated failure")
        return num * 2

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(batch_size=1)
        .map(doubled=double_num, output=int)
    )

    # -------------- FIRST RUN (fails, no run_group_id) -------------------
    reset_session_job_state()

    first_job_id = str(uuid4())
    metastore.create_job(
        "scheduled-task",
        "echo 1;",
        job_id=first_job_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", first_job_id)

    with pytest.raises(RuntimeError, match="Simulated failure"):
        chain.save("doubled_nums")

    first_count = processed_count["count"]
    assert first_count > 0

    # -------------- SECOND RUN (continue, no run_group_id) -------------------
    reset_session_job_state()
    processed_count["count"] = 0
    should_fail[0] = False

    second_job_id = str(uuid4())
    metastore.create_job(
        "scheduled-task",
        "echo 1;",
        job_id=second_job_id,
        rerun_from_job_id=first_job_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", second_job_id)

    chain.save("doubled_nums")

    # Should only process remaining rows, not all 6
    assert processed_count["count"] < 6

    result = sorted(
        dc.read_dataset("doubled_nums", session=test_session).to_list("doubled")
    )
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_udf_runs_in_ephemeral_mode(test_session, nums_dataset):
    metastore = test_session.catalog.metastore
    jobs_before = _count_rows(metastore, metastore._jobs)
    checkpoints_before = _count_rows(metastore, metastore._checkpoints)

    result = sorted(
        dc.read_dataset("nums", session=test_session)
        .settings(ephemeral=True)
        .map(doubled=lambda num: num * 2, output=int)
        .to_list("doubled")
    )
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]

    # No checkpoints or jobs should have been created
    assert _count_rows(metastore, metastore._checkpoints) == checkpoints_before
    assert _count_rows(metastore, metastore._jobs) == jobs_before


def test_ephemeral_mode_repeated_runs_no_table_collision(test_session, nums_dataset):
    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(ephemeral=True)
        .map(doubled=lambda num: num * 2, output=int)
    )

    for _ in range(3):
        result = sorted(chain.to_list("doubled"))
        assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_ephemeral_mode_no_jobs_on_collect(test_session, nums_dataset):
    metastore = test_session.catalog.metastore
    jobs_before = _count_rows(metastore, metastore._jobs)
    checkpoints_before = _count_rows(metastore, metastore._checkpoints)

    result = sorted(
        dc.read_dataset("nums", session=test_session)
        .settings(ephemeral=True)
        .map(doubled=lambda num: num * 2, output=int)
        .to_values("doubled")
    )
    assert result == [2, 4, 6, 8, 10, 12]

    assert _count_rows(metastore, metastore._jobs) == jobs_before
    assert _count_rows(metastore, metastore._checkpoints) == checkpoints_before


def test_ephemeral_mode_aggregator_with_partition_by(test_session):
    metastore = test_session.catalog.metastore

    dc.read_values(
        num=[1, 2, 3, 4, 5, 6],
        letter=["A", "A", "B", "B", "C", "C"],
        session=test_session,
    ).save("nums_letters_eph")

    jobs_before = _count_rows(metastore, metastore._jobs)
    checkpoints_before = _count_rows(metastore, metastore._checkpoints)

    result = sorted(
        dc.read_dataset("nums_letters_eph", session=test_session)
        .settings(ephemeral=True)
        .agg(
            total=lambda num: [sum(num)],
            output=int,
            partition_by="letter",
        )
        .to_values("total")
    )
    assert result == [3, 7, 11]

    assert _count_rows(metastore, metastore._jobs) == jobs_before
    assert _count_rows(metastore, metastore._checkpoints) == checkpoints_before

    with pytest.raises(RuntimeError, match="Cannot save datasets in ephemeral mode"):
        dc.read_dataset("nums_letters_eph", session=test_session).settings(
            ephemeral=True
        ).agg(
            total=lambda num: [sum(num)],
            output=int,
            partition_by="letter",
        ).save("should_fail")


def test_independent_chains_no_transient_invalidation(test_session, nums_dataset):
    call_count = {"count": 0}

    def add_ten(num) -> int:
        call_count["count"] += 1
        return num + 10

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).filter(dc.C("num") > 3).save("big")
    dc.read_dataset("nums", session=test_session).map(
        plus_ten=add_ten, output=int
    ).save("mapped")

    assert call_count["count"] == 6

    # -------------- SECOND RUN - modify the filter chain -------------------
    reset_session_job_state()
    call_count["count"] = 0

    # Change the filter — this should NOT affect the map chain
    dc.read_dataset("nums", session=test_session).filter(dc.C("num") > 1).save("big")
    dc.read_dataset("nums", session=test_session).map(
        plus_ten=add_ten, output=int
    ).save("mapped")

    assert call_count["count"] == 0, "UDF should be skipped — unrelated chain changed"


def test_reordering_chains_no_invalidation(test_session, nums_dataset):
    call_count_a = {"count": 0}
    call_count_b = {"count": 0}

    def double(num) -> int:
        call_count_a["count"] += 1
        return num * 2

    def triple(num) -> int:
        call_count_b["count"] += 1
        return num * 3

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN: double then triple -------------------
    reset_session_job_state()
    chain.map(result=double, output=int).save("doubled")
    chain.map(result=triple, output=int).save("tripled")

    assert call_count_a["count"] == 6
    assert call_count_b["count"] == 6

    # -------------- SECOND RUN: triple then double (swapped) -------------------
    reset_session_job_state()
    call_count_a["count"] = 0
    call_count_b["count"] = 0

    chain.map(result=triple, output=int).save("tripled")
    chain.map(result=double, output=int).save("doubled")

    assert call_count_a["count"] == 0, "double UDF should be skipped"
    assert call_count_b["count"] == 0, "triple UDF should be skipped"


def test_try_except_identical_chains(test_session, nums_dataset):

    def compute(num) -> int:
        if num > 3:
            raise CustomMapperError("Simulated failure")
        return num * 10

    reset_session_job_state()

    # Chain A crashes mid-UDF, leaves partial output table with 3 rows
    try:
        dc.read_dataset("nums", session=test_session).settings(batch_size=1).map(
            result=compute, output=int
        ).save("ds1")
    except CustomMapperError:
        pass

    # Chain B: same function name, same partial_hash, no crash
    def compute(num) -> int:
        return num * 10

    dc.read_dataset("nums", session=test_session).settings(batch_size=1).map(
        result=compute, output=int
    ).save("ds2")

    # ds2 should have exactly 6 rows, not 9 (6 + chain A's 3 leftover)
    result = sorted(dc.read_dataset("ds2", session=test_session).to_list("result"))
    assert result == [(10,), (20,), (30,), (40,), (50,), (60,)]


def test_loop_same_chain_creates_one_version(test_session, nums_dataset):
    """Running the same chain in a loop within one job runs the UDF once and
    creates exactly one dataset version — "done" checkpoints act as a cache."""
    catalog = test_session.catalog
    call_count = {"count": 0}

    def plus_ten(num) -> int:
        call_count["count"] += 1
        return num + 10

    reset_session_job_state()

    for _ in range(3):
        dc.read_dataset("nums", session=test_session).map(
            plus_ten=plus_ten, output=int
        ).save("looped")

    # UDF ran exactly once; subsequent iterations hit the cache
    assert call_count["count"] == 6

    # Only one dataset version was created
    assert len(catalog.get_dataset("looped", versions=None).versions) == 1

    # Data is correct
    result = sorted(dc.read_dataset("looped", session=test_session).to_list("plus_ten"))
    assert result == [(11,), (12,), (13,), (14,), (15,), (16,)]


def test_chain_multiple_operations_udf_runs_once(test_session, nums_dataset):
    """Running multiple operations (e.g. .to_list() then .save()) on the same
    chain within one job triggers the UDF only once — the second operation
    reuses the cached output."""
    call_count = {"count": 0}

    def plus_ten(num) -> int:
        call_count["count"] += 1
        return num + 10

    reset_session_job_state()

    chain = dc.read_dataset("nums", session=test_session).map(
        plus_ten=plus_ten, output=int
    )

    # First operation: triggers UDF execution
    first_result = sorted(chain.to_list("plus_ten"))
    assert call_count["count"] == 6
    assert first_result == [(11,), (12,), (13,), (14,), (15,), (16,)]

    # Second operation: reuses cached output from first run
    chain.save("saved")
    assert call_count["count"] == 6, "UDF should be skipped on second operation"

    # Saved data is correct
    saved = sorted(dc.read_dataset("saved", session=test_session).to_list("plus_ten"))
    assert saved == [(11,), (12,), (13,), (14,), (15,), (16,)]


def test_retry_loop_accumulates_partial_progress(test_session, nums_dataset):
    """Retry pattern within one job: each attempt continues from the partial
    checkpoint left by the previous attempt, accumulating progress until the
    chain completes."""
    processed = []
    fail_after = {"count": 2}  # each attempt fails after this many new rows

    def compute(num) -> int:
        # Fail after processing `fail_after` NEW rows in current attempt
        if fail_after["count"] <= 0:
            raise CustomMapperError("Simulated failure")
        fail_after["count"] -= 1
        processed.append(num)
        return num * 10

    reset_session_job_state()

    attempts = 0
    max_attempts = 10
    while attempts < max_attempts:
        attempts += 1
        try:
            dc.read_dataset("nums", session=test_session).settings(batch_size=1).map(
                result=compute, output=int
            ).save("retried")
            break
        except CustomMapperError:
            # Reset the fail counter for the next attempt; partial will carry over
            fail_after["count"] = 2

    # Must have retried several times, then succeeded
    assert attempts > 1
    assert attempts < max_attempts, "Retry loop did not converge"

    # Each attempt only processed rows that weren't already in the partial table
    # Total UDF invocations must equal total rows (6), not rows * attempts
    assert len(processed) == 6, (
        f"UDF ran {len(processed)} times — partial progress was not accumulated"
    )

    # Final data is correct
    result = sorted(dc.read_dataset("retried", session=test_session).to_list("result"))
    assert result == [(10,), (20,), (30,), (40,), (50,), (60,)]


def test_independent_chains_same_udf_within_job(test_session, nums_dataset):
    """Two independent chains with the same UDF hash within one job: the UDF
    runs once (second chain finds the first chain's final checkpoint and skips)
    and both saves produce their own datasets with the correct data."""
    catalog = test_session.catalog
    call_count = {"count": 0}

    def plus_ten(num) -> int:
        call_count["count"] += 1
        return num + 10

    reset_session_job_state()

    # Chain A: runs the UDF from scratch, saves to ds_a
    dc.read_dataset("nums", session=test_session).map(
        plus_ten=plus_ten, output=int
    ).save("ds_a")
    assert call_count["count"] == 6

    # Chain B: same UDF chain, different save target. UDF hash matches → skip.
    dc.read_dataset("nums", session=test_session).map(
        plus_ten=plus_ten, output=int
    ).save("ds_b")
    assert call_count["count"] == 6, "UDF should be skipped — same hash in same job"

    # Both datasets exist and have the correct data
    assert len(catalog.get_dataset("ds_a", versions=None).versions) == 1
    assert len(catalog.get_dataset("ds_b", versions=None).versions) == 1

    expected = [(11,), (12,), (13,), (14,), (15,), (16,)]
    assert (
        sorted(dc.read_dataset("ds_a", session=test_session).to_list("plus_ten"))
        == expected
    )
    assert (
        sorted(dc.read_dataset("ds_b", session=test_session).to_list("plus_ten"))
        == expected
    )


def test_loop_same_chain_generator_one_version(test_session, nums_dataset):
    """Generator equivalent of test_loop_same_chain_creates_one_version:
    a `for` loop running a generator chain creates exactly one dataset version."""
    catalog = test_session.catalog
    call_count = {"count": 0}

    def expand(num) -> Iterator[int]:
        call_count["count"] += 1
        yield num
        yield num * 10

    reset_session_job_state()

    for _ in range(3):
        dc.read_dataset("nums", session=test_session).gen(
            value=expand, output=int
        ).save("expanded")

    # Generator ran exactly once per input; subsequent iterations hit the cache
    assert call_count["count"] == 6
    assert len(catalog.get_dataset("expanded", versions=None).versions) == 1

    result = sorted(dc.read_dataset("expanded", session=test_session).to_list("value"))
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


def test_retry_loop_generator_accumulates_partial_progress(test_session, nums_dataset):
    """Generator retry pattern within one job: each attempt continues from the
    partial checkpoint left by the previous attempt. Exercises same-job partial
    continuation for generators — the in-place deletion of incomplete rows and
    the 1:N input→output tracking via sys__input_id."""
    processed = []
    fail_after = {"count": 2}

    def expand(num) -> Iterator[int]:
        # Fail after producing outputs for `fail_after` NEW inputs
        if fail_after["count"] <= 0:
            raise CustomMapperError("Simulated failure")
        fail_after["count"] -= 1
        processed.append(num)
        yield num
        yield num * 10

    reset_session_job_state()

    attempts = 0
    max_attempts = 10
    while attempts < max_attempts:
        attempts += 1
        try:
            dc.read_dataset("nums", session=test_session).settings(batch_size=1).gen(
                value=expand, output=int
            ).save("expanded")
            break
        except CustomMapperError:
            fail_after["count"] = 2

    assert attempts > 1
    assert attempts < max_attempts, "Retry loop did not converge"

    # Each attempt only processed inputs not already in the partial table
    # Total generator invocations must equal total inputs (6), not 6 * attempts
    assert len(processed) == 6, (
        f"Generator ran {len(processed)} times — partial progress was not accumulated"
    )

    # All 12 output rows are present (2 yields per input x 6 inputs), no duplicates
    result = sorted(dc.read_dataset("expanded", session=test_session).to_list("value"))
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
