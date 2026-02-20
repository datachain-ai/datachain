from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

import datachain as dc
from tests.utils import reset_session_job_state


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

    assert len(list(metastore.list_checkpoints(job_id))) == 4
    assert len(warehouse.db.list_tables(prefix="udf_")) > 0

    # Make all checkpoints older than the default TTL (4h)
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints()

    assert len(list(metastore.list_checkpoints(job_id))) == 0
    assert len(warehouse.db.list_tables(prefix=f"udf_{job_id}_")) == 0


def test_cleanup_checkpoints_with_custom_ttl(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    assert len(list(metastore.list_checkpoints(job_id))) == 2

    # Make checkpoints 2 hours old, then clean with 1h TTL
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints(ttl_seconds=3600)

    assert len(list(metastore.list_checkpoints(job_id))) == 0


def test_cleanup_checkpoints_no_old_checkpoints(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    checkpoints_before = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints_before) == 2

    catalog.cleanup_checkpoints()

    checkpoints_after = list(metastore.list_checkpoints(job_id))
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

    assert len(list(metastore.list_checkpoints(job_id))) == 2

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

    assert len(list(metastore.list_checkpoints(job_id))) == 0
    assert len(warehouse.db.list_tables(prefix=f"udf_{job_id}_")) == 0
    assert warehouse.db.has_table(fake_table_name)

    warehouse.cleanup_tables([fake_table_name])


def test_cleanup_checkpoints_multiple_jobs(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job1_id = test_session.get_or_create_job().id

    reset_session_job_state()
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    job2_id = test_session.get_or_create_job().id

    reset_session_job_state()
    chain.map(quadrupled=lambda num: num * 4, output=int).save("nums_quadrupled")
    job3_id = test_session.get_or_create_job().id

    # Make ALL checkpoints outdated
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints()

    for job_id in [job1_id, job2_id, job3_id]:
        assert len(list(metastore.list_checkpoints(job_id))) == 0
