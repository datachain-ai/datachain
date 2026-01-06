import base64
import json

import pytest

from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import SCHEMA_VERSION, SQLiteMetastore
from datachain.dataset import StorageURI
from datachain.error import OutdatedDatabaseSchemaError
from tests.conftest import cleanup_sqlite_db


def test_sqlite_metastore(sqlite_db):
    uri = StorageURI("s3://bucket")

    obj = SQLiteMetastore(uri, sqlite_db)
    assert obj.uri == uri
    assert obj.db == sqlite_db

    # Test clone
    obj2 = obj.clone()
    try:
        assert isinstance(obj2, SQLiteMetastore)
        assert obj2.uri == uri
        assert obj2.db.db_file == sqlite_db.db_file
        assert obj2.clone_params() == obj.clone_params()

        # Test serialization JSON format
        serialized = obj.serialize()
        assert serialized
        raw = base64.b64decode(serialized.encode())
        data = json.loads(raw.decode())
        assert data["callable"] == "sqlite.metastore.init_after_clone"
        assert data["args"] == []
        assert data["kwargs"]["uri"] == uri
        nested = data["kwargs"]["db_clone_params"]
        assert nested["callable"] == "sqlite.from_db_file"
        assert nested["args"] == [":memory:"]
        assert nested["kwargs"] == {}

        obj3 = deserialize(serialized)
        try:
            assert isinstance(obj3, SQLiteMetastore)
            assert obj3.uri == uri
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
