import base64
import json
import os

import pytest
import sqlalchemy as sa
from sqlalchemy import Column, Integer, Table

from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import (
    SQLiteDatabaseEngine,
    get_db_file_in_memory,
)
from datachain.sql.types import Int64
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


def test_reflect_table_temp_table_and_view(sqlite_db):
    sqlite_db.execute_str("CREATE TEMP TABLE tmp_rel (x INTEGER)")
    assert "x" in sqlite_db.reflect_table("tmp_rel").c

    sqlite_db.execute_str("CREATE VIEW view_rel AS SELECT 1 AS x")
    assert "x" in sqlite_db.reflect_table("view_rel").c
    assert sqlite_db.has_table("view_rel") is False


def test_get_table_reconnects_and_reflects_added_column(tmp_dir):
    db_path = os.fspath(tmp_dir / "reconnect_reflect.db")
    with SQLiteDatabaseEngine.from_db_file(db_path) as db:
        db.execute_str("CREATE TABLE MixedCaseRel (x INTEGER)")
        assert "x" in db.get_table("mixedcaserel").c

        db.add_column("MixedCaseRel", sa.Column("y", Int64))
        db.close()
        assert "x" in db.get_table("mixedcaserel").c
        assert db.is_closed is False
        refreshed = db.reflect_table("MixedCaseRel")
        assert "y" in refreshed.c


def test_reflect_table_column_metadata(sqlite_db):
    sqlite_db.execute_str(
        "CREATE TABLE col_meta"
        " (id INTEGER PRIMARY KEY, x INTEGER NOT NULL DEFAULT 7, opt TEXT)"
    )
    sqlite_db.execute_str("INSERT INTO col_meta (id) VALUES (1)")
    tbl = sqlite_db.reflect_table("col_meta")
    assert tbl.c.id.primary_key is True
    assert tbl.c.x.primary_key is False
    assert tbl.c.x.nullable is False
    assert tbl.c.opt.nullable is True
    assert tbl.c.x.server_default is not None
    assert sqlite_db.execute_str("SELECT id, x FROM col_meta").fetchall() == [(1, 7)]


def test_reflect_table_updates_recreated_schema(sqlite_db):
    sqlite_db.execute_str(
        "CREATE TABLE recreated_rel (id INTEGER PRIMARY KEY, x INTEGER)"
    )
    first = sqlite_db.reflect_table("recreated_rel")
    assert first.c.id.primary_key is True
    assert isinstance(first.c.x.type, sa.Integer)

    sqlite_db.execute_str("DROP TABLE recreated_rel")
    sqlite_db.execute_str("CREATE TABLE recreated_rel (id INTEGER, x TEXT PRIMARY KEY)")
    second = sqlite_db.reflect_table("recreated_rel")
    assert second.c.id.primary_key is False
    assert second.c.x.primary_key is True
    assert isinstance(second.c.x.type, sa.Text)

    sqlite_db.execute_str("DROP TABLE recreated_rel")
    sqlite_db.execute_str("CREATE TABLE recreated_rel (id INTEGER PRIMARY KEY)")
    third = sqlite_db.reflect_table("recreated_rel")
    assert list(third.c.keys()) == ["id"]


def test_reflect_table_includes_generated_columns_and_skips_hidden_columns(sqlite_db):
    sqlite_db.execute_str(
        "CREATE TABLE measure"
        " (x INTEGER, doubled INTEGER GENERATED ALWAYS AS (2*x) STORED)"
    )
    measure = sqlite_db.reflect_table("measure")
    assert list(measure.c.keys()) == ["x", "doubled"]
    assert measure.c.doubled.computed is not None

    sqlite_db.execute_str("CREATE VIRTUAL TABLE vt_rel USING fts5(body)")
    assert list(sqlite_db.reflect_table("vt_rel").c.keys()) == ["body"]
