"""Tests for automatic local database schema migration.

These tests verify that the MongoDB-style lazy schema evolution works correctly
for SQLite databases. ClickHouse databases used in SaaS have proper migrations,
so these tests are skipped when running against ClickHouse.
"""

# ruff: noqa: S608

import time

from sqlalchemy import Column, Index, Integer, Table, Text

from tests.utils import skip_if_not_sqlite


@skip_if_not_sqlite
def test_automatic_schema_migration(catalog):
    """Test that missing columns are automatically added during initialization.

    This test simulates upgrading from an old database schema by:
    1. Creating a table with a subset of columns (old schema)
    2. Calling migration logic
    3. Verifying missing columns were added
    4. Verifying default values were applied
    5. Verifying indexes were created
    6. Verifying the table is functional
    """
    metastore = catalog.metastore
    db = metastore.db

    old_table_name = "test_migration_table"

    # Clean up if exists from previous test run
    db.execute_str(f"DROP TABLE IF EXISTS {old_table_name}")

    # Create table with old schema (only 2 columns)
    db.execute_str(
        f"""
        CREATE TABLE {old_table_name} (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    """
    )

    db.execute_str(
        f"INSERT INTO {old_table_name} (id, name) VALUES (?, ?)",
        ("test-id", "test-name"),
    )

    # Define the "new" schema with additional columns
    new_table = Table(
        old_table_name,
        db.metadata,
        Column("id", Text, primary_key=True),
        Column("name", Text, nullable=False),
        Column("description", Text, nullable=True),
        Column("count", Integer, nullable=True, default=0),
        Column("status", Text, nullable=False, default="active"),
        Index("idx_test_name", "name"),
    )

    metastore._migrate_table_schema(new_table)

    # Verify: Check that new columns were added
    columns_query = f"PRAGMA table_info({old_table_name})"
    columns = db.execute_str(columns_query).fetchall()
    column_names = {col[1] for col in columns}

    assert "id" in column_names
    assert "name" in column_names
    assert "description" in column_names, (
        "Missing column 'description' should have been added"
    )
    assert "count" in column_names, "Missing column 'count' should have been added"
    assert "status" in column_names, "Missing column 'status' should have been added"

    # Verify: Check that index was created
    indexes_query = f"PRAGMA index_list({old_table_name})"
    indexes = db.execute_str(indexes_query).fetchall()
    index_names = {idx[1] for idx in indexes}

    assert "idx_test_name" in index_names, "Index should have been created"

    # Verify: Old data still exists
    result = db.execute_str(
        f"SELECT id, name FROM {old_table_name} WHERE id = ?",
        ("test-id",),
    ).fetchone()
    assert result[0] == "test-id"
    assert result[1] == "test-name"

    # Verify: Can insert data with new columns
    db.execute_str(
        f"INSERT INTO {old_table_name} (id, name, description, count, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("test-id-2", "test-name-2", "test description", 42, "pending"),
    )

    # Verify: Can query new columns
    result = db.execute_str(
        f"SELECT description, count, status FROM {old_table_name} WHERE id = ?",
        ("test-id-2",),
    ).fetchone()
    assert result[0] == "test description"
    assert result[1] == 42
    assert result[2] == "pending"

    # Verify: Old rows have NULL for nullable columns without defaults,
    # but get default values for columns with defaults
    result = db.execute_str(
        f"SELECT description, count, status FROM {old_table_name} WHERE id = ?",
        ("test-id",),
    ).fetchone()
    assert result[0] is None, "Nullable column without default should be NULL"
    assert result[1] == 0, (
        "Column with default=0 should have default applied to existing rows"
    )
    assert result[2] == "active", (
        "Column with default='active' should have default applied to existing rows"
    )

    # Verify: New rows get default values when not specified
    db.execute_str(
        f"INSERT INTO {old_table_name} (id, name) VALUES (?, ?)",
        ("test-id-3", "test-name-3"),
    )
    result = db.execute_str(
        f"SELECT count, status FROM {old_table_name} WHERE id = ?",
        ("test-id-3",),
    ).fetchone()
    # SQLite applies defaults on INSERT
    assert result[0] == 0, "Default value for count should be 0"
    assert result[1] == "active", "Default value for status should be 'active'"

    db.execute_str(f"DROP TABLE {old_table_name}")


@skip_if_not_sqlite
def test_migration_is_idempotent(catalog):
    """Test that running migration multiple times doesn't cause errors."""
    metastore = catalog.metastore
    db = metastore.db

    old_table_name = "test_idempotent_migration"

    db.execute_str(f"DROP TABLE IF EXISTS {old_table_name}")

    db.execute_str(
        f"""
        CREATE TABLE {old_table_name} (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """
    )

    new_table = Table(
        old_table_name,
        db.metadata,
        Column("id", Text, primary_key=True),
        Column("name", Text),
        Column("extra", Text, nullable=True),
        Index("idx_test_idempotent", "name"),
    )

    # Run migration multiple times - should not fail
    metastore._migrate_table_schema(new_table)
    metastore._migrate_table_schema(new_table)
    metastore._migrate_table_schema(new_table)

    # Verify column exists
    columns_query = f"PRAGMA table_info({old_table_name})"
    columns = db.execute_str(columns_query).fetchall()
    column_names = {col[1] for col in columns}
    assert "extra" in column_names

    db.execute_str(f"DROP TABLE {old_table_name}")


@skip_if_not_sqlite
def test_migration_with_data_preservation(catalog):
    """Test that migration preserves existing data correctly."""
    metastore = catalog.metastore
    db = metastore.db

    table_name = "test_data_preservation"

    db.execute_str(f"DROP TABLE IF EXISTS {table_name}")

    db.execute_str(
        f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY,
            value TEXT NOT NULL
        )
    """
    )

    for i in range(100):
        db.execute_str(
            f"INSERT INTO {table_name} (id, value) VALUES (?, ?)",
            (i, f"value-{i}"),
        )

    new_table = Table(
        table_name,
        db.metadata,
        Column("id", Integer, primary_key=True),
        Column("value", Text, nullable=False),
        Column("new_field", Text, nullable=True),
    )

    metastore._migrate_table_schema(new_table)

    # Verify all data is preserved
    result = db.execute_str(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    assert result[0] == 100, "All rows should be preserved"

    # Verify data integrity
    for i in range(100):
        result = db.execute_str(
            f"SELECT value, new_field FROM {table_name} WHERE id = ?",
            (i,),
        ).fetchone()
        assert result[0] == f"value-{i}", f"Value for row {i} should be preserved"
        assert result[1] is None, f"New field for row {i} should be NULL"

    db.execute_str(f"DROP TABLE {table_name}")


@skip_if_not_sqlite
def test_backward_compatibility_with_extra_columns(catalog):
    """Test that old code works with DB that has extra columns (downgrade scenario).

    Simulates the case where:
    1. User runs new version (DB has new columns)
    2. User downgrades to old version (schema doesn't define new columns)

    Verifies that SQLAlchemy-based queries handle extra columns gracefully.
    """
    metastore = catalog.metastore
    db = metastore.db

    table_name = "test_backward_compat"

    db.execute_str(f"DROP TABLE IF EXISTS {table_name}")

    # Simulate NEW version DB: create table with extra columns
    db.execute_str(
        f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            extra_column_1 TEXT,
            extra_column_2 INTEGER
        )
    """
    )

    db.execute_str(
        f"INSERT INTO {table_name} VALUES (1, 'Alice', 'extra_data', 42)",
    )

    # Simulate OLD version code: schema WITHOUT extra columns
    old_schema_table = Table(
        table_name,
        db.metadata,
        Column("id", Integer, primary_key=True),
        Column("name", Text, nullable=False),
        # Note: extra_column_1 and extra_column_2 are NOT defined
        extend_existing=True,
    )

    result = db.execute_str(f"SELECT id, name FROM {table_name} WHERE id=1").fetchone()
    assert result[0] == 1
    assert result[1] == "Alice"

    from sqlalchemy import insert

    db.execute(insert(old_schema_table).values(id=2, name="Bob"))
    result = db.execute_str(
        f"SELECT id, name, extra_column_1, extra_column_2 FROM {table_name} WHERE id=2"
    ).fetchone()
    assert result[0] == 2
    assert result[1] == "Bob"
    assert result[2] is None  # extra_column_1 is NULL
    assert result[3] is None  # extra_column_2 is NULL

    from sqlalchemy import update

    db.execute(
        update(old_schema_table).where(old_schema_table.c.id == 1).values(name="Alice2")
    )
    result = db.execute_str(
        f"SELECT name, extra_column_1, extra_column_2 FROM {table_name} WHERE id=1"
    ).fetchone()
    assert result[0] == "Alice2"  # name updated
    assert result[1] == "extra_data"  # extra_column_1 preserved
    assert result[2] == 42  # extra_column_2 preserved

    db.execute_str(f"DROP TABLE {table_name}")


@skip_if_not_sqlite
def test_migration_performance_overhead(catalog):
    """Measure the overhead of migration checks when schema is already up-to-date.

    This simulates the common case where users run commands and the schema
    hasn't changed - we want to ensure the migration check is fast.
    """
    metastore = catalog.metastore

    for table in metastore._metastore_tables:
        metastore._migrate_table_schema(table)

    # Measure overhead by running multiple times
    num_runs = 100
    start = time.perf_counter()
    for _ in range(num_runs):
        for table in metastore._metastore_tables:
            metastore._migrate_table_schema(table)
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / num_runs) * 1000
    num_tables = len(metastore._metastore_tables)

    print(f"\nMigration check overhead (average of {num_runs} runs):")
    print(f"  Total: {avg_time_ms:.3f}ms for {num_tables} tables")
    print(f"  Per table: {avg_time_ms / num_tables:.3f}ms")

    # Assert reasonable performance: should be under 5ms for all tables
    assert avg_time_ms < 5.0, (
        f"Migration check overhead is {avg_time_ms:.2f}ms, should be under 5ms. "
        "This might indicate a performance regression."
    )
