import base64
import json
import os

import pytest
from sqlalchemy import Column, Integer, Table

from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import (
    SQLiteDatabaseEngine,
    get_db_file_in_memory,
)
from tests.utils import skip_if_not_sqlite


@pytest.mark.parametrize(
    "db_file,expected_db_file",
    [
        (":memory:", ":memory:"),
        ("file.db", "file.db"),
        (None, os.path.join(".datachain", "db")),
    ],
)
@skip_if_not_sqlite
def test_init_clone(tmp_dir, db_file, expected_db_file):
    if db_file is None:
        expected_db_file = os.fspath(tmp_dir / expected_db_file)

    with SQLiteDatabaseEngine.from_db_file(db_file) as db:
        assert isinstance(db, SQLiteDatabaseEngine)
        assert db.db_file == expected_db_file

        # Test clone
        with db.clone() as db2:
            assert isinstance(db2, SQLiteDatabaseEngine)
            assert db2.db_file == expected_db_file


@pytest.mark.parametrize(
    "db_file,in_memory,expected",
    [
        (None, False, None),
        (None, True, ":memory:"),
        (":memory:", False, ":memory:"),
        (":memory:", True, ":memory:"),
        ("file.db", False, "file.db"),
        ("file.db", True, RuntimeError),
    ],
)
@skip_if_not_sqlite
def test_get_db_file_in_memory(db_file, in_memory, expected):
    if expected is RuntimeError:
        with pytest.raises(RuntimeError):
            get_db_file_in_memory(db_file, in_memory)
    else:
        assert get_db_file_in_memory(db_file, in_memory) == expected


def test_serialize(sqlite_db):
    # JSON serialization format
    serialized = sqlite_db.serialize()
    assert serialized
    raw = base64.b64decode(serialized.encode())
    data = json.loads(raw.decode())
    assert data["callable"] == "sqlite.from_db_file"
    assert data["args"] == [":memory:"]
    assert data["kwargs"] == {}

    obj3 = deserialize(serialized)
    assert isinstance(obj3, SQLiteDatabaseEngine)
    assert obj3.db_file == ":memory:"
    assert obj3.clone_params() == sqlite_db.clone_params()
    obj3.close()


def test_table(sqlite_db):
    table = Table(
        "test_table", sqlite_db.metadata, Column("id", Integer, primary_key=True)
    )
    assert not sqlite_db.has_table("test_table")
    assert not sqlite_db.has_table("test_table_2")

    table.create(sqlite_db.engine)
    assert sqlite_db.has_table("test_table")
    assert not sqlite_db.has_table("test_table_2")

    sqlite_db.rename_table("test_table", "test_table_2")
    assert sqlite_db.has_table("test_table_2")
    assert not sqlite_db.has_table("test_table")

    sqlite_db.drop_table(Table("test_table_2", sqlite_db.metadata))
    assert not sqlite_db.has_table("test_table")
    assert not sqlite_db.has_table("test_table_2")


def test_table_in_transaction(sqlite_db):
    table = Table(
        "test_table", sqlite_db.metadata, Column("id", Integer, primary_key=True)
    )
    assert not sqlite_db.has_table("test_table")
    assert not sqlite_db.has_table("test_table_2")

    with sqlite_db.transaction():
        table.create(sqlite_db.engine)
        assert sqlite_db.has_table("test_table")
        assert not sqlite_db.has_table("test_table_2")

        sqlite_db.rename_table("test_table", "test_table_2")
        assert sqlite_db.has_table("test_table_2")
        assert not sqlite_db.has_table("test_table")

        sqlite_db.drop_table(Table("test_table_2", sqlite_db.metadata))
        assert not sqlite_db.has_table("test_table")
        assert not sqlite_db.has_table("test_table_2")
