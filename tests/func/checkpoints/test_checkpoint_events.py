from collections.abc import Iterator

import pytest

import datachain as dc
from datachain.checkpoint_event import CheckpointEventType, CheckpointStepType
from datachain.dataset import create_dataset_full_name
from tests.utils import reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


def get_events(metastore, job_id):
    return list(metastore.get_checkpoint_events(job_id=job_id))


def get_udf_events(metastore, job_id):
    return [
        e
        for e in get_events(metastore, job_id)
        if e.event_type
        in (
            CheckpointEventType.UDF_SKIPPED,
            CheckpointEventType.UDF_CONTINUED,
            CheckpointEventType.UDF_FROM_SCRATCH,
        )
    ]


def get_dataset_events(metastore, job_id):
    return [
        e
        for e in get_events(metastore, job_id)
        if e.event_type
        in (
            CheckpointEventType.DATASET_SAVE_SKIPPED,
            CheckpointEventType.DATASET_SAVE_COMPLETED,
        )
    ]


def test_map_from_scratch_event(test_session, nums_dataset):
    metastore = test_session.catalog.metastore

    def double(num) -> int:
        return num * 2

    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).map(doubled=double, output=int).save(
        "doubled"
    )
    job_id = test_session.get_or_create_job().id

    events = get_udf_events(metastore, job_id)
    assert len(events) == 1

    map_event = events[0]
    assert map_event.event_type == CheckpointEventType.UDF_FROM_SCRATCH
    assert map_event.step_type == CheckpointStepType.UDF_MAP
    assert map_event.udf_name == "double"
    assert map_event.rows_input == 6
    assert map_event.rows_processed == 6
    assert map_event.rows_output == 6
    assert map_event.rows_input_reused == 0
    assert map_event.rows_output_reused == 0
    assert map_event.rerun_from_job_id is None
    assert map_event.hash_partial is None


def test_gen_from_scratch_event(test_session, nums_dataset):
    metastore = test_session.catalog.metastore

    def duplicate(num) -> Iterator[int]:
        yield num
        yield num

    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).gen(dup=duplicate, output=int).save(
        "duplicated"
    )
    job_id = test_session.get_or_create_job().id

    events = get_udf_events(metastore, job_id)
    gen_event = next(e for e in events if e.udf_name == "duplicate")

    assert gen_event.event_type == CheckpointEventType.UDF_FROM_SCRATCH
    assert gen_event.step_type == CheckpointStepType.UDF_GEN
    assert gen_event.rows_input == 6
    assert gen_event.rows_processed == 6
    assert gen_event.rows_output == 12
    assert gen_event.rows_input_reused == 0
    assert gen_event.rows_output_reused == 0


def test_map_skipped_event(test_session, nums_dataset):
    metastore = test_session.catalog.metastore

    def double(num) -> int:
        return num * 2

    chain = dc.read_dataset("nums", session=test_session).map(
        doubled=double, output=int
    )

    reset_session_job_state()
    chain.save("doubled")
    first_job_id = test_session.get_or_create_job().id

    reset_session_job_state()
    chain.save("doubled2")
    second_job_id = test_session.get_or_create_job().id

    events = get_udf_events(metastore, second_job_id)
    assert len(events) == 1

    map_event = events[0]
    assert map_event.event_type == CheckpointEventType.UDF_SKIPPED
    assert map_event.udf_name == "double"
    assert map_event.rows_input == 6
    assert map_event.rows_processed == 0
    assert map_event.rows_output == 0
    assert map_event.rows_input_reused == 6
    assert map_event.rows_output_reused == 6
    assert map_event.rerun_from_job_id == first_job_id
    assert map_event.hash_partial is None


def test_gen_skipped_event(test_session, nums_dataset):
    metastore = test_session.catalog.metastore

    def duplicate(num) -> Iterator[int]:
        yield num
        yield num

    chain = dc.read_dataset("nums", session=test_session).gen(dup=duplicate, output=int)

    reset_session_job_state()
    chain.save("duplicated")
    first_job_id = test_session.get_or_create_job().id

    reset_session_job_state()
    chain.save("duplicated2")
    second_job_id = test_session.get_or_create_job().id

    events = get_udf_events(metastore, second_job_id)
    gen_event = next(e for e in events if e.udf_name == "duplicate")

    assert gen_event.event_type == CheckpointEventType.UDF_SKIPPED
    assert gen_event.rows_input == 6
    assert gen_event.rows_processed == 0
    assert gen_event.rows_output == 0
    assert gen_event.rows_input_reused == 6
    assert gen_event.rows_output_reused == 12
    assert gen_event.rerun_from_job_id == first_job_id


def test_map_continued_event(test_session, nums_dataset):
    metastore = test_session.catalog.metastore
    processed = []

    def buggy_double(num) -> int:
        if len(processed) >= 3:
            raise Exception("Simulated failure")
        processed.append(num)
        return num * 2

    chain = dc.read_dataset("nums", session=test_session).map(
        doubled=buggy_double, output=int
    )

    reset_session_job_state()
    with pytest.raises(Exception, match="Simulated failure"):
        chain.save("doubled")
    first_job_id = test_session.get_or_create_job().id

    reset_session_job_state()
    processed.clear()

    def fixed_double(num) -> int:
        processed.append(num)
        return num * 2

    dc.read_dataset("nums", session=test_session).map(
        doubled=fixed_double, output=int
    ).save("doubled")
    second_job_id = test_session.get_or_create_job().id

    events = get_udf_events(metastore, second_job_id)
    map_event = next(e for e in events if e.udf_name == "fixed_double")

    assert map_event.event_type == CheckpointEventType.UDF_CONTINUED
    assert map_event.rows_input == 6
    assert map_event.rows_input_reused == 3
    assert map_event.rows_output_reused == 3
    assert map_event.rows_processed == 3
    assert map_event.rows_output == 3
    assert map_event.rerun_from_job_id == first_job_id
    assert map_event.hash_partial is not None


def test_gen_continued_event(test_session, nums_dataset):
    metastore = test_session.catalog.metastore
    processed = []

    def buggy_gen(num) -> Iterator[int]:
        if len(processed) >= 2:
            raise Exception("Simulated failure")
        processed.append(num)
        yield num
        yield num * 10

    chain = dc.read_dataset("nums", session=test_session).gen(
        result=buggy_gen, output=int
    )

    reset_session_job_state()
    with pytest.raises(Exception, match="Simulated failure"):
        chain.save("results")
    first_job_id = test_session.get_or_create_job().id

    reset_session_job_state()
    processed.clear()

    def fixed_gen(num) -> Iterator[int]:
        processed.append(num)
        yield num
        yield num * 10

    dc.read_dataset("nums", session=test_session).gen(
        result=fixed_gen, output=int
    ).save("results")
    second_job_id = test_session.get_or_create_job().id

    events = get_udf_events(metastore, second_job_id)
    gen_event = next(e for e in events if e.udf_name == "fixed_gen")

    assert gen_event.event_type == CheckpointEventType.UDF_CONTINUED
    assert gen_event.rows_input == 6
    assert gen_event.rows_input_reused == 2
    assert gen_event.rows_output_reused == 4
    assert gen_event.rows_processed == 4
    assert gen_event.rows_output == 8
    assert gen_event.rerun_from_job_id == first_job_id
    assert gen_event.hash_partial is not None


def test_dataset_save_completed_event(test_session, nums_dataset):
    metastore = test_session.catalog.metastore

    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).save("nums_copy")
    job_id = test_session.get_or_create_job().id

    events = get_dataset_events(metastore, job_id)

    assert len(events) == 1
    event = events[0]

    expected_name = create_dataset_full_name(
        metastore.default_namespace_name,
        metastore.default_project_name,
        "nums_copy",
        "1.0.0",
    )
    assert event.event_type == CheckpointEventType.DATASET_SAVE_COMPLETED
    assert event.step_type == CheckpointStepType.DATASET_SAVE
    assert event.dataset_name == expected_name
    assert event.checkpoint_hash is not None


def test_dataset_save_skipped_event(test_session, nums_dataset):
    metastore = test_session.catalog.metastore
    chain = dc.read_dataset("nums", session=test_session)

    reset_session_job_state()
    chain.save("nums_copy")
    first_job_id = test_session.get_or_create_job().id

    first_events = get_dataset_events(metastore, first_job_id)
    assert len(first_events) == 1
    assert first_events[0].event_type == CheckpointEventType.DATASET_SAVE_COMPLETED

    reset_session_job_state()
    chain.save("nums_copy")
    second_job_id = test_session.get_or_create_job().id

    second_events = get_dataset_events(metastore, second_job_id)

    assert len(second_events) == 1
    event = second_events[0]

    expected_name = create_dataset_full_name(
        metastore.default_namespace_name,
        metastore.default_project_name,
        "nums_copy",
        "1.0.0",
    )
    assert event.event_type == CheckpointEventType.DATASET_SAVE_SKIPPED
    assert event.step_type == CheckpointStepType.DATASET_SAVE
    assert event.dataset_name == expected_name
    assert event.rerun_from_job_id is not None


def test_events_by_run_group(test_session, monkeypatch, nums_dataset):
    metastore = test_session.catalog.metastore

    def double(num) -> int:
        return num * 2

    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).map(doubled=double, output=int).save(
        "doubled"
    )
    first_job = test_session.get_or_create_job()

    reset_session_job_state()
    second_job_id = metastore.create_job(
        "test-job",
        "echo 1",
        rerun_from_job_id=first_job.id,
        run_group_id=first_job.run_group_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", second_job_id)

    dc.read_dataset("nums", session=test_session).map(doubled=double, output=int).save(
        "doubled2"
    )

    run_group_events = list(
        metastore.get_checkpoint_events(run_group_id=first_job.run_group_id)
    )

    job_ids = {e.job_id for e in run_group_events}
    assert first_job.id in job_ids
    assert second_job_id in job_ids


def test_hash_fields_populated(test_session, nums_dataset):
    metastore = test_session.catalog.metastore

    def double(num) -> int:
        return num * 2

    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).map(doubled=double, output=int).save(
        "doubled"
    )
    job_id = test_session.get_or_create_job().id

    events = get_udf_events(metastore, job_id)

    for event in events:
        assert event.checkpoint_hash is not None
        assert event.hash_input is not None
        assert event.hash_output is not None
        if event.event_type == CheckpointEventType.UDF_FROM_SCRATCH:
            assert event.hash_partial is None
