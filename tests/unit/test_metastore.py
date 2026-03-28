import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

import datachain as dc
from datachain.checkpoint import CheckpointStatus
from datachain.data_storage.job import JobQueryType, JobStatus
from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import SCHEMA_VERSION, SQLiteMetastore
from datachain.error import DatasetStateNotLoadedError, OutdatedDatabaseSchemaError
from tests.conftest import cleanup_sqlite_db


def test_sqlite_metastore(sqlite_db):
    obj = SQLiteMetastore(db=sqlite_db)
    assert obj.db == sqlite_db

    # Test clone
    obj2 = obj.clone()
    try:
        assert isinstance(obj2, SQLiteMetastore)
        assert obj2.db.db_file == sqlite_db.db_file
        assert obj2.clone_params() == obj.clone_params()

        # Test serialization JSON format
        serialized = obj.serialize()
        assert serialized
        raw = base64.b64decode(serialized.encode())
        data = json.loads(raw.decode())
        assert data["callable"] == "sqlite.metastore.init_after_clone"
        assert data["args"] == []
        nested = data["kwargs"]["db_clone_params"]
        assert nested["callable"] == "sqlite.from_db_file"
        assert nested["args"] == [":memory:"]
        assert nested["kwargs"] == {}

        obj3 = deserialize(serialized)
        try:
            assert isinstance(obj3, SQLiteMetastore)
            assert obj3.db.db_file == sqlite_db.db_file
            assert obj3.clone_params() == obj.clone_params()
        finally:
            obj3.close_on_exit()
    finally:
        obj2.close_on_exit()


def test_outdated_schema_meta_not_present():
    metastore = SQLiteMetastore(db_file=":memory:")
    try:
        metastore.db.drop_table(metastore._meta)

        with pytest.raises(OutdatedDatabaseSchemaError):
            SQLiteMetastore(db_file=":memory:")

        cleanup_sqlite_db(metastore.db.clone(), metastore.default_table_names)
    finally:
        metastore.close_on_exit()


def test_outdated_schema():
    metastore = SQLiteMetastore(db_file=":memory:")
    try:
        # update schema version to be lower than current one
        stmt = (
            metastore._meta.update()
            .where(metastore._meta.c.id == 1)
            .values(schema_version=SCHEMA_VERSION - 1)
        )
        metastore.db.execute(stmt)

        with pytest.raises(OutdatedDatabaseSchemaError):
            SQLiteMetastore(db_file=":memory:")

        cleanup_sqlite_db(metastore.db.clone(), metastore.default_table_names)
    finally:
        metastore.close_on_exit()


def test_expire_checkpoints():
    metastore = SQLiteMetastore(db_file=":memory:")
    try:
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=2)
        ttl_threshold = now - timedelta(hours=1)

        # Create two jobs
        job1_id = metastore.create_job(
            "job1", "q", query_type=JobQueryType.PYTHON, status=JobStatus.COMPLETE
        )
        job2_id = metastore.create_job(
            "job2", "q", query_type=JobQueryType.PYTHON, status=JobStatus.COMPLETE
        )

        # Job1: all checkpoints older than TTL
        metastore.get_or_create_checkpoint(job1_id, "hash_a")
        metastore.get_or_create_checkpoint(job1_id, "hash_b")

        # Job2: has one active (recent) checkpoint
        metastore.get_or_create_checkpoint(job2_id, "hash_c")
        metastore.get_or_create_checkpoint(job2_id, "hash_d")

        # Backdate job1 checkpoints and one of job2's checkpoints
        ch = metastore._checkpoints
        metastore.db.execute(
            ch.update().where(ch.c.job_id == job1_id).values(created_at=old)
        )
        metastore.db.execute(
            ch.update()
            .where(ch.c.job_id == job2_id)
            .where(ch.c.hash == "hash_c")
            .values(created_at=old)
        )

        assert list(metastore.list_checkpoints(status=CheckpointStatus.EXPIRED)) == []

        # Expire checkpoints — only job1's should be marked EXPIRED
        checkpoints, inactive_group_ids = metastore.expire_checkpoints(ttl_threshold)

        assert len(checkpoints) == 2
        assert all(cp.job_id == job1_id for cp in checkpoints)
        assert inactive_group_ids == [job1_id]

        # Job2's checkpoints are untouched (has one recent checkpoint)
        job2_checkpoints = list(metastore.list_checkpoints(job_ids=[job2_id]))
        assert len(job2_checkpoints) == 2
        assert all(cp.status == CheckpointStatus.ACTIVE for cp in job2_checkpoints)
    finally:
        metastore.close_on_exit()


def test_get_dataset_can_skip_preview_loading(test_session):
    ds = dc.read_values(value=["a", "b"], session=test_session).save("preview-ds")
    metastore = test_session.catalog.metastore

    with_preview = metastore.get_dataset(
        ds.name,
        versions=None,
        include_preview=True,
    )
    without_preview = metastore.get_dataset(
        ds.name,
        versions=None,
        include_preview=False,
    )

    assert with_preview.get_version("1.0.0").preview is not None
    with pytest.raises(DatasetStateNotLoadedError):
        _ = without_preview.get_version("1.0.0").preview


def test_update_dataset_version_marks_preview_loaded_after_explicit_preview_update(
    test_session,
):
    ds = dc.read_values(value=["a", "b"], session=test_session).save(
        "preview-update-ds"
    )
    metastore = test_session.catalog.metastore

    without_preview = metastore.get_dataset(
        ds.name,
        versions=None,
        include_preview=False,
    )
    version = without_preview.get_version("1.0.0")

    with pytest.raises(DatasetStateNotLoadedError):
        _ = version.preview

    updated = metastore.update_dataset_version(
        without_preview,
        "1.0.0",
        preview=[{"sys__id": 1, "value": "updated"}],
    )

    assert updated._preview_loaded is True
    assert updated.preview == [{"sys__id": 1, "value": "updated"}]
    assert "_preview_loaded" not in updated.to_dict()


def test_dataset_record_versions_setter_marks_loaded(test_session):
    ds = dc.read_values(value=["a", "b"], session=test_session).save("setter-ds")
    metastore = test_session.catalog.metastore

    record = metastore.get_dataset(ds.name, versions=())
    with pytest.raises(DatasetStateNotLoadedError):
        _ = record.versions

    loaded = metastore.get_dataset(ds.name, versions=None)
    record.versions = loaded.versions

    assert record.versions == loaded.versions
