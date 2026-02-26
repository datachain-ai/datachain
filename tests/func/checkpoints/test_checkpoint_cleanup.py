from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.data_storage import JobStatus
from tests.utils import reset_session_job_state


def finish_job(metastore, job_id):
    """Mark a job as complete so GC considers it eligible for cleanup."""
    metastore.set_job_status(job_id, JobStatus.COMPLETE)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3], session=test_session).save("nums")


def test_cleanup_checkpoints_with_ttl(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore
    warehouse = catalog.warehouse

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    job_id = test_session.get_or_create_job().id
    finish_job(metastore, job_id)

    assert len(list(metastore.list_checkpoints([job_id]))) == 4
    assert len(warehouse.db.list_tables(pattern="udf_%")) > 0

    # Make all checkpoints older than the default TTL (4h)
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints()

    assert len(list(metastore.list_checkpoints([job_id]))) == 0
    assert len(warehouse.db.list_tables(pattern=f"udf_{job_id}_%")) == 0


def test_cleanup_checkpoints_with_custom_ttl(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id
    finish_job(metastore, job_id)

    assert len(list(metastore.list_checkpoints([job_id]))) == 2

    # Make checkpoints 2 hours old, then clean with 1h TTL
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints(ttl_seconds=3600)

    assert len(list(metastore.list_checkpoints([job_id]))) == 0


def test_cleanup_checkpoints_no_old_checkpoints(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    checkpoints_before = list(metastore.list_checkpoints([job_id]))
    assert len(checkpoints_before) == 2

    catalog.cleanup_checkpoints()

    checkpoints_after = list(metastore.list_checkpoints([job_id]))
    assert len(checkpoints_after) == 2
    assert {cp.id for cp in checkpoints_before} == {cp.id for cp in checkpoints_after}


def test_cleanup_does_not_remove_unrelated_tables(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore
    warehouse = catalog.warehouse

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id
    finish_job(metastore, job_id)

    assert len(list(metastore.list_checkpoints([job_id]))) == 2

    fake_table_name = "udf_unrelated_job_fakehash_output"
    warehouse.create_dataset_rows_table(
        fake_table_name,
        columns=[sa.Column("val", sa.Integer)],
    )
    assert warehouse.db.has_table(fake_table_name)

    # Make checkpoints outdated and clean up
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints()

    assert len(list(metastore.list_checkpoints([job_id]))) == 0
    assert len(warehouse.db.list_tables(pattern=f"udf_{job_id}_%")) == 0
    assert warehouse.db.has_table(fake_table_name)

    warehouse.cleanup_tables([fake_table_name])


def test_cleanup_checkpoints_multiple_jobs(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job1_id = test_session.get_or_create_job().id
    finish_job(metastore, job1_id)

    reset_session_job_state()
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    job2_id = test_session.get_or_create_job().id
    finish_job(metastore, job2_id)

    reset_session_job_state()
    chain.map(quadrupled=lambda num: num * 4, output=int).save("nums_quadrupled")
    job3_id = test_session.get_or_create_job().id
    finish_job(metastore, job3_id)

    # Make ALL checkpoints outdated
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints()

    for job_id in [job1_id, job2_id, job3_id]:
        assert len(list(metastore.list_checkpoints([job_id]))) == 0


def test_cleanup_skips_job_with_active_checkpoints(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    # Job 1: will have all outdated checkpoints
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    old_job_id = test_session.get_or_create_job().id
    finish_job(metastore, old_job_id)

    # Job 2: will have a mix of outdated and active checkpoints
    reset_session_job_state()
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    chain.map(quadrupled=lambda num: num * 4, output=int).save("nums_quadrupled")
    mixed_job_id = test_session.get_or_create_job().id
    finish_job(metastore, mixed_job_id)

    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    for ch in metastore.list_checkpoints([old_job_id]):
        metastore.db.execute(
            metastore._checkpoints.update()
            .where(metastore._checkpoints.c.id == ch.id)
            .values(created_at=old_time)
        )

    # Make only SOME of job 2's checkpoints outdated
    mixed_checkpoints = list(metastore.list_checkpoints([mixed_job_id]))
    assert len(mixed_checkpoints) >= 2
    metastore.db.execute(
        metastore._checkpoints.update()
        .where(metastore._checkpoints.c.id == mixed_checkpoints[0].id)
        .values(created_at=old_time)
    )

    catalog.cleanup_checkpoints()

    # Job 1: fully cleaned
    assert len(list(metastore.list_checkpoints([old_job_id]))) == 0
    # Job 2: untouched (has active checkpoints)
    assert len(list(metastore.list_checkpoints([mixed_job_id]))) == len(
        mixed_checkpoints
    )


def test_cleanup_skips_running_jobs(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    # Job stays in RUNNING status (no finish_job call)
    assert metastore.get_job(job_id).status == JobStatus.RUNNING

    checkpoints_before = list(metastore.list_checkpoints([job_id]))
    assert len(checkpoints_before) == 2

    # Make checkpoints older than TTL
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints()

    # Checkpoints preserved — job is still running
    assert len(list(metastore.list_checkpoints([job_id]))) == 2


def test_cleanup_preserves_input_tables_when_run_group_active(
    test_session, monkeypatch, nums_dataset
):
    catalog = test_session.catalog
    metastore = catalog.metastore
    warehouse = catalog.warehouse

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job1_id = test_session.get_or_create_job().id
    finish_job(metastore, job1_id)
    job1 = metastore.get_job(job1_id)
    run_group_id = job1.run_group_id

    # Create a shared input table (named with run_group_id)
    input_table = f"udf_{run_group_id}_somehash_input"
    warehouse.create_udf_table(name=input_table)
    assert warehouse.db.has_table(input_table)

    # Create a second job in the same run group with active checkpoints
    reset_session_job_state()
    job2_id = metastore.create_job(
        "test-job",
        "echo 1",
        rerun_from_job_id=job1_id,
        run_group_id=run_group_id,
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", job2_id)
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    finish_job(metastore, job2_id)

    # Record job1's UDF output tables before cleanup
    job1_output_tables = [
        t
        for t in warehouse.db.list_tables(pattern=f"udf_{job1_id}_%")
        if t.endswith("_output")
    ]
    assert len(job1_output_tables) > 0

    # Make only job 1's checkpoints outdated
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    for ch in metastore.list_checkpoints([job1_id]):
        metastore.db.execute(
            metastore._checkpoints.update()
            .where(metastore._checkpoints.c.id == ch.id)
            .values(created_at=old_time)
        )

    catalog.cleanup_checkpoints()

    # Job 1's output tables removed
    assert all(not warehouse.db.has_table(t) for t in job1_output_tables)
    assert len(list(metastore.list_checkpoints([job1_id]))) == 0
    assert len(list(metastore.list_checkpoints([job2_id]))) > 0
    # Shared input table preserved (run group still active)
    assert warehouse.db.has_table(input_table)

    warehouse.cleanup_tables([input_table])
