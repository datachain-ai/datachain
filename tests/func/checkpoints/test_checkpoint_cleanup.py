from datetime import datetime, timedelta, timezone

import pytest

import datachain as dc
from tests.utils import reset_session_job_state


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3], session=test_session).save("nums")


def test_cleanup_checkpoints_with_ttl(test_session, monkeypatch, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore
    warehouse = catalog.warehouse

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    test_session.get_or_create_job()
    job_id = test_session.get_or_create_job().id

    checkpoints_before = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints_before) == 4
    assert all(c.partial is False for c in checkpoints_before)

    # Verify UDF tables exist by checking all tables with udf_ prefix
    # Note: Due to checkpoint skipping, some jobs may reuse parent tables
    all_udf_tables_before = warehouse.db.list_tables(prefix="udf_")

    assert len(all_udf_tables_before) > 0

    # Modify ALL checkpoints to be older than TTL (4 hours by default)
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints()

    checkpoints_after = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints_after) == 0

    udf_tables_after = warehouse.db.list_tables(prefix=f"udf_{job_id}_")
    assert len(udf_tables_after) == 0


def test_cleanup_checkpoints_with_custom_ttl(test_session, monkeypatch, nums_dataset):
    from datetime import datetime, timedelta, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    checkpoints = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints) == 2
    assert all(c.partial is False for c in checkpoints)

    # Modify ALL checkpoints to be 2 hours old
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    # Run cleanup with custom TTL of 1 hour (3600 seconds)
    # Checkpoints are 2 hours old, so they should be removed
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
    checkpoint_ids_before = {cp.id for cp in checkpoints_before}
    checkpoint_ids_after = {cp.id for cp in checkpoints_after}
    assert checkpoint_ids_before == checkpoint_ids_after


def test_cleanup_does_not_remove_unrelated_tables(test_session, nums_dataset):
    catalog = test_session.catalog
    metastore = catalog.metastore
    warehouse = catalog.warehouse

    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    assert len(list(metastore.list_checkpoints(job_id))) == 2

    # Create an unrelated table that looks like a UDF table from another job
    import sqlalchemy as sa

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

    # Expired job's checkpoints and tables are gone
    assert len(list(metastore.list_checkpoints(job_id))) == 0
    assert len(warehouse.db.list_tables(prefix=f"udf_{job_id}_")) == 0

    # Unrelated table survives
    assert warehouse.db.has_table(fake_table_name)

    # Clean up
    warehouse.cleanup_tables([fake_table_name])


def test_cleanup_checkpoints_branch_pruning(test_session, nums_dataset):
    from datetime import datetime, timedelta, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create a lineage: root -> child -> grandchild
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    root_job_id = test_session.get_or_create_job().id

    reset_session_job_state()
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    child_job_id = test_session.get_or_create_job().id

    reset_session_job_state()
    chain.map(quadrupled=lambda num: num * 4, output=int).save("nums_quadrupled")
    grandchild_job_id = test_session.get_or_create_job().id

    # Make ALL checkpoints outdated (older than TTL)
    all_job_ids = [root_job_id, child_job_id, grandchild_job_id]
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    metastore.db.execute(metastore._checkpoints.update().values(created_at=old_time))

    catalog.cleanup_checkpoints()

    for job_id in all_job_ids:
        remaining = list(metastore.list_checkpoints(job_id))
        assert len(remaining) == 0, f"Job {job_id} should have been cleaned"
