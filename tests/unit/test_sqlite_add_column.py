# ruff: noqa: S608

from uuid import uuid4

import pytest
from sqlalchemy import Boolean, Column, String, Text

from tests.utils import skip_if_not_sqlite


@skip_if_not_sqlite
@pytest.mark.parametrize(
    "column_def,expected_value",
    [
        (Column("is_active", Boolean, default=True, nullable=False), 1),
        (Column("is_deleted", Boolean, default=False, nullable=False), 0),
        (
            Column("status", String(50), default="it's working", nullable=False),
            "it's working",
        ),
    ],
)
def test_add_column_with_defaults(catalog, column_def, expected_value):
    db = catalog.metastore.db
    table_name = f"test_{column_def.name}"

    db.execute_str(f"DROP TABLE IF EXISTS {table_name}")
    db.execute_str(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY)")

    db.add_column(table_name, column_def)

    db.execute_str(f"INSERT INTO {table_name} (id) VALUES (1)")
    result = db.execute_str(
        f"SELECT {column_def.name} FROM {table_name} WHERE id=1"
    ).fetchone()
    assert result[0] == expected_value

    db.execute_str(f"DROP TABLE {table_name}")


@skip_if_not_sqlite
def test_add_column_callable_default(catalog):
    db = catalog.metastore.db
    table_name = "test_callable_default"

    db.execute_str(f"DROP TABLE IF EXISTS {table_name}")
    db.execute_str(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY)")

    callable_column = Column(
        "uuid", String(36), default=lambda: str(uuid4()), nullable=False
    )
    db.add_column(table_name, callable_column)

    columns_query = f"PRAGMA table_info({table_name})"
    columns = db.execute_str(columns_query).fetchall()
    column_names = {col[1] for col in columns}
    assert "uuid" in column_names

    db.execute_str(f"DROP TABLE {table_name}")


@skip_if_not_sqlite
def test_add_column_unsupported_default_type(catalog):
    db = catalog.metastore.db
    table_name = "test_unsupported_default"

    db.execute_str(f"DROP TABLE IF EXISTS {table_name}")
    db.execute_str(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY)")

    unsupported_column = Column("data", Text, default=[1, 2, 3], nullable=False)

    with pytest.raises(
        RuntimeError,
        match=r"unsupported default for test_unsupported_default\.data: list",
    ):
        db.add_column(table_name, unsupported_column)

    db.execute_str(f"DROP TABLE {table_name}")


@skip_if_not_sqlite
def test_add_column_not_null_with_existing_rows(catalog):
    db = catalog.metastore.db
    table_name = "test_migration_with_data"

    db.execute_str(f"DROP TABLE IF EXISTS {table_name}")
    db.execute_str(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, name TEXT)")

    db.execute_str(f"INSERT INTO {table_name} (id, name) VALUES (1, 'existing')")
    db.execute_str(f"INSERT INTO {table_name} (id, name) VALUES (2, 'data')")

    uuid_column = Column(
        "uuid", String(36), default=lambda: str(uuid4()), nullable=False
    )
    db.add_column(table_name, uuid_column)

    result = db.execute_str(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    assert result[0] == 2

    columns_query = f"PRAGMA table_info({table_name})"
    columns = db.execute_str(columns_query).fetchall()
    column_names = {col[1] for col in columns}
    assert "uuid" in column_names

    db.execute_str(f"DROP TABLE {table_name}")
