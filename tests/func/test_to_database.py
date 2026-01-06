import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

import datachain as dc


def get_postgres_uri():
    return os.environ.get("TEST_POSTGRES_URI", "postgresql://test:test@localhost:5432")


def is_postgres_available():
    try:
        from sqlalchemy import create_engine

        base_postgres_uri = get_postgres_uri()
        engine = create_engine(f"{base_postgres_uri}/postgres")
        with engine.connect():
            pass
        return True
    except sqlalchemy.exc.SQLAlchemyError:
        return False


def _get_engine_from_connection(connection):
    """Extract the SQLAlchemy engine from a connection object."""
    if isinstance(connection, str):
        return sqlalchemy.create_engine(connection)
    if isinstance(connection, sqlalchemy.Engine):
        return connection
    if isinstance(connection, sqlalchemy.Connection):
        return connection.engine
    if isinstance(connection, Session):
        if connection.bind is None:
            raise ValueError("Session has no bound engine")
        # Session.bind can be Engine or Connection, we need Engine
        if isinstance(connection.bind, sqlalchemy.Engine):
            return connection.bind
        if isinstance(connection.bind, sqlalchemy.Connection):
            return connection.bind.engine
        raise TypeError(f"Unexpected bind type: {type(connection.bind)}")
    if hasattr(connection, "execute"):  # sqlite3.Connection
        return sqlalchemy.create_engine("sqlite://", creator=lambda: connection)

    raise TypeError(f"Unsupported connection type: {type(connection)}")


def _query_table(connection, query, params=None):
    """Execute a query against a connection and return the result."""
    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        return conn.execute(text(query), params or {})


def _fetch_all_rows(connection, table_name, order_by="id"):
    """Fetch all rows from a table, ordered by the specified column."""
    query = f"SELECT * FROM {table_name} ORDER BY {order_by}"  # noqa: S608
    return _query_table(connection, query).fetchall()


def _count_table_rows(connection, table_name):
    """Count the number of rows in a table."""
    query = f"SELECT COUNT(*) FROM {table_name}"  # noqa: S608
    return _query_table(connection, query).scalar()


def _query_scalar(connection, query, params=None):
    """Execute a query and return a scalar result."""
    return _query_table(connection, query, params).scalar()


def _create_conflict_test_table(engine, table_name):
    """Create a test table with initial data for conflict testing."""
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(
                text(f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value INTEGER
                )
            """)
            )
            conn.execute(
                text(f"""
                INSERT INTO {table_name} VALUES
                (1, 'Alice', 100), (2, 'Bob', 200), (3, 'Charlie', 300)
            """)  # noqa: S608
            )


def _get_table_columns(connection, table_name):
    engine = _get_engine_from_connection(connection)
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return [col["name"] for col in columns]


def _parse_json_if_needed(connection, value):
    """Parse JSON string if needed based on database type."""
    if value is None:
        return None
    engine = _get_engine_from_connection(connection)
    if engine.name == "sqlite" and isinstance(value, str):
        return json.loads(value)
    return value


def _parse_datetime_if_needed(connection, value, expected_timezone=None):
    """Parse datetime string if needed based on database type."""
    if value is None:
        return None
    engine = _get_engine_from_connection(connection)

    if engine.name == "sqlite" and isinstance(value, str):
        # SQLite stores naive datetime strings that represent UTC time
        parsed = datetime.fromisoformat(value)
        parsed_utc = parsed.replace(tzinfo=timezone.utc)
        if expected_timezone is not None:
            return parsed_utc.astimezone(expected_timezone)
        return parsed_utc

    return value


@pytest.fixture
def ensure_sqlite_adapter():
    """
    Ensure SQLite adapters are setup for DataChain.

    This fixture forcefully registers DataChain's SQLite adapters to handle
    potential conflicts with other libraries, particularly pandas.io.sql.

    pandas.io.sql registers its own datetime adapter when performing to_sql()
    operations, which can overwrite DataChain's custom datetime adapter that
    handles timezone-aware datetime serialization. This fixture ensures
    DataChain's adapters are active for all SQLite-based tests.
    """
    from datachain.sql.sqlite.base import adapt_datetime

    sqlite3.register_adapter(datetime, adapt_datetime)

    yield


@pytest.fixture
def sqlite_uri():
    return "sqlite:///:memory:"


@pytest.fixture
def sqlite_engine(sqlite_uri, ensure_sqlite_adapter):
    engine = sqlalchemy.create_engine(sqlite_uri)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def sqlite_connection(sqlite_engine, ensure_sqlite_adapter):
    with closing(sqlite_engine.connect()) as conn:
        yield conn


@pytest.fixture
def sqlite_session(sqlite_engine, ensure_sqlite_adapter):
    with Session(bind=sqlite_engine) as session:
        yield session


@pytest.fixture
def sqlite3_connection(ensure_sqlite_adapter):
    with closing(sqlite3.connect(":memory:")) as conn:
        yield conn


@pytest.fixture(scope="session")
def postgres_session_database():
    if not is_postgres_available():
        pytest.skip("PostgreSQL not available")

    unique_db_name = f"datachain_test_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    base_postgres_uri = get_postgres_uri()
    admin_engine = sqlalchemy.create_engine(f"{base_postgres_uri}/postgres")

    try:
        with admin_engine.connect() as admin_conn:
            # SQLAlchemy automatically starts transactions, PG needs to be
            # outside a transaction to run admin commands
            admin_conn.execute(text("COMMIT"))
            admin_conn.execute(text(f"CREATE DATABASE {unique_db_name}"))

        test_uri = f"{base_postgres_uri}/{unique_db_name}"
        test_engine = sqlalchemy.create_engine(test_uri)

        yield {
            "engine": test_engine,
            "db_name": unique_db_name,
            "admin_engine": admin_engine,
        }

    finally:
        try:
            with admin_engine.connect() as admin_conn:
                admin_conn.execute(text("COMMIT"))
                admin_conn.execute(
                    text("""
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = :db_name AND pid <> pg_backend_pid()
                """),
                    {"db_name": unique_db_name},
                )
                admin_conn.execute(text(f"DROP DATABASE IF EXISTS {unique_db_name}"))
        except sqlalchemy.exc.SQLAlchemyError:
            pass


@pytest.fixture
def postgres_connection(postgres_session_database):
    """PostgreSQL connection fixture that uses the session database."""
    engine = postgres_session_database["engine"]
    with closing(engine.connect()) as conn:
        yield conn


@pytest.fixture(
    params=(
        "sqlite_connection",
        "sqlite_engine",
        "sqlite_session",
        "sqlite3_connection",
        "postgres_connection",
    )
)
def connection(request):
    return request.getfixturevalue(request.param)


def test_basic_to_database(tmp_dir, connection):
    """Test basic functionality with actual DataChain data."""
    chain = dc.read_values(
        id=[1, 2, 3], name=["Alice", "Bob", "Charlie"], age=[25, 30, 35]
    )

    rows_affected = chain.to_database("basic_test_table", connection)

    result = _fetch_all_rows(connection, "basic_test_table")
    assert len(result) == 3
    assert rows_affected == 3  # Check that we get the correct row count
    assert result[0] == (1, "Alice", 25)
    assert result[1] == (2, "Bob", 30)
    assert result[2] == (3, "Charlie", 35)


def test_to_database_with_uri(sqlite_uri, ensure_sqlite_adapter):
    """Test basic functionality with URI connection string."""
    chain = dc.read_values(
        id=[1, 2, 3], name=["Alice", "Bob", "Charlie"], age=[25, 30, 35]
    )

    chain.to_database("uri_test_table", sqlite_uri)

    # URI connections create isolated databases, so we can't verify externally
    # The absence of exceptions indicates success


def test_to_database_large_dataset(connection, test_session):
    """Test to_database with large amount of data and custom batch size."""
    large_size = 10000
    chain = dc.read_values(
        id=list(range(large_size)),
        value=[f"value_{i}" for i in range(large_size)],
        number=[i * 2.5 for i in range(large_size)],
        session=test_session,
    )

    rows_affected = chain.to_database("large_table", connection, batch_size=1000)

    count = _count_table_rows(connection, "large_table")
    assert count == large_size
    assert rows_affected == large_size  # Check correct count for large datasets

    rows = _query_table(
        connection,
        "SELECT * FROM large_table WHERE id IN (0, 999, 5000, 9999) ORDER BY id",
    ).fetchall()

    assert len(rows) == 4
    assert rows[0] == (0, "value_0", 0.0)
    assert rows[1] == (999, "value_999", 2497.5)
    assert rows[2] == (5000, "value_5000", 12500.0)
    assert rows[3] == (9999, "value_9999", 24997.5)


def test_to_database_on_conflict_ignore(connection, test_session):
    """Test to_database with on_conflict='ignore' for duplicate handling."""
    table_name = "conflict_ignore_table"
    engine = _get_engine_from_connection(connection)
    _create_conflict_test_table(engine, table_name)

    conflict_chain = dc.read_values(
        id=[2, 3, 4, 5],  # 2 and 3 will conflict
        name=["Bob_Updated", "Charlie_Updated", "Diana", "Eve"],
        value=[999, 888, 400, 500],
        session=test_session,
    )

    rows_affected = conflict_chain.to_database(
        table_name, connection, on_conflict="ignore"
    )

    rows = _fetch_all_rows(connection, table_name)
    assert len(rows) == 5
    assert rows_affected == 2  # Only 2 new inserts (conflicts ignored)
    assert rows[0] == (1, "Alice", 100)  # Original
    assert rows[1] == (2, "Bob", 200)  # Original (conflict ignored)
    assert rows[2] == (3, "Charlie", 300)  # Original (conflict ignored)
    assert rows[3] == (4, "Diana", 400)  # New
    assert rows[4] == (5, "Eve", 500)  # New


def test_to_database_on_conflict_update(connection, test_session):
    """Test to_database with on_conflict='update' for duplicate handling."""
    table_name = "conflict_update_table"
    engine = _get_engine_from_connection(connection)
    _create_conflict_test_table(engine, table_name)

    update_chain = dc.read_values(
        id=[2, 3, 4, 5],  # 2 and 3 will update existing records
        name=["Bob_Updated", "Charlie_Updated", "Diana", "Eve"],
        value=[999, 888, 400, 500],
        session=test_session,
    )

    rows_affected = update_chain.to_database(
        table_name, connection, on_conflict="update", conflict_columns=["id"]
    )

    rows = _fetch_all_rows(connection, table_name)
    assert len(rows) == 5
    assert rows_affected == 4  # 2 updates + 2 inserts
    assert rows[0] == (1, "Alice", 100)  # Original (unchanged)
    assert rows[1] == (2, "Bob_Updated", 999)  # Updated
    assert rows[2] == (3, "Charlie_Updated", 888)  # Updated
    assert rows[3] == (4, "Diana", 400)  # New
    assert rows[4] == (5, "Eve", 500)  # New


def test_to_database_on_conflict_update_postgres_missing_conflict_columns(
    postgres_connection, test_session
):
    table_name = "conflict_missing_columns_table"
    engine = _get_engine_from_connection(postgres_connection)
    _create_conflict_test_table(engine, table_name)

    update_chain = dc.read_values(
        id=[2, 3, 4],
        name=["Bob_Updated", "Charlie_Updated", "Diana"],
        value=[999, 888, 400],
        session=test_session,
    )

    with pytest.raises(
        ValueError,
        match=(
            "conflict_columns parameter is required when "
            "on_conflict='update' with PostgreSQL"
        ),
    ):
        update_chain.to_database(
            table_name, postgres_connection, on_conflict="update", conflict_columns=None
        )

    with pytest.raises(
        ValueError,
        match=(
            "conflict_columns parameter is required when "
            "on_conflict='update' with PostgreSQL"
        ),
    ):
        update_chain.to_database(table_name, postgres_connection, on_conflict="update")

    # Verify the table remains unchanged after failed operations
    rows = _fetch_all_rows(postgres_connection, table_name)
    assert len(rows) == 3
    assert rows[0] == (1, "Alice", 100)
    assert rows[1] == (2, "Bob", 200)
    assert rows[2] == (3, "Charlie", 300)


def test_to_database_conflict_columns_normalization(postgres_connection, test_session):
    """Test that conflict_columns supports DataChain format and column mapping."""
    table_name = "conflict_normalization_table"
    engine = _get_engine_from_connection(postgres_connection)
    _create_conflict_test_table(engine, table_name)

    class User(dc.DataModel):
        id: int
        name: str

    class Metadata(dc.DataModel):
        value: int

    update_chain = dc.read_values(
        user=[
            User(id=2, name="Bob_Updated"),
            User(id=3, name="Charlie_Updated"),
            User(id=4, name="Dave"),
        ],
        metadata=[
            Metadata(value=250),
            Metadata(value=350),
            Metadata(value=400),
        ],
        session=test_session,
    )

    column_mapping = {
        "user.id": "id",  # DataChain format with dots
        "user.name": "name",
        "metadata__value": "value",  # DataChain format with underscores
    }

    rows_affected = update_chain.to_database(
        table_name,
        postgres_connection,
        on_conflict="update",
        conflict_columns=["user.id"],  # DataChain format, will be mapped to "id"
        column_mapping=column_mapping,
    )

    rows = _fetch_all_rows(postgres_connection, table_name, order_by="id")
    assert len(rows) == 4
    assert rows_affected == 3  # 2 updates + 1 insert
    assert rows[0] == (1, "Alice", 100)  # Original unchanged
    assert rows[1] == (2, "Bob_Updated", 250)  # Updated
    assert rows[2] == (3, "Charlie_Updated", 350)  # Updated
    assert rows[3] == (4, "Dave", 400)  # New


def test_to_database_table_exists_different_schema(connection, test_session):
    """Test to_database when table exists but has different schema."""
    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(
                sqlalchemy.text("""
                CREATE TABLE schema_mismatch (
                    id INTEGER,
                    old_column TEXT,
                    extra_column REAL
                )
            """)
            )
            conn.execute(
                sqlalchemy.text(
                    "INSERT INTO schema_mismatch VALUES (1, 'existing', 1.5)"
                )
            )

    chain = dc.read_values(
        id=[2, 3],
        name=["Alice", "Bob"],  # Different column name
        session=test_session,
    )

    with pytest.raises(
        (sqlalchemy.exc.OperationalError, sqlalchemy.exc.ProgrammingError)
    ):
        chain.to_database("schema_mismatch", connection)


def test_to_database_transaction_rollback_on_error(connection, test_session):
    """Test that transaction is rolled back when an exception occurs."""
    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(
                sqlalchemy.text("""
                CREATE TABLE strict_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL
                )
            """)
            )
            conn.execute(
                sqlalchemy.text(
                    "INSERT INTO strict_table VALUES (1, 'Initial', 'initial@test.com')"
                )
            )

    initial_count = _count_table_rows(connection, "strict_table")

    assert initial_count == 1

    chain = dc.read_values(
        id=[2, 3, 4],
        name=["Alice", "Bob", "Charlie"],
        # Second email is duplicate and will cause violation
        email=["alice@test.com", "initial@test.com", "charlie@test.com"],
        session=test_session,
    )

    with pytest.raises(sqlalchemy.exc.IntegrityError):
        chain.to_database("strict_table", connection)

    final_count = _count_table_rows(connection, "strict_table")
    assert final_count == 1


def test_to_database_empty_chain(connection, test_session):
    """Test to_database with an empty DataChain."""
    chain = dc.read_values(
        id=[],
        name=[],
        session=test_session,
    )

    rows_affected = chain.to_database("empty_table", connection)

    count = _count_table_rows(connection, "empty_table")
    assert count == 0
    assert rows_affected == 0  # Check that empty chain returns 0


def test_to_database_column_mapping(connection, test_session):
    """Test to_database with column mapping functionality."""
    chain = dc.read_values(
        internal_id=[1, 2, 3],
        full_name=["Alice Smith", "Bob Jones", "Charlie Brown"],
        session=test_session,
    )

    column_mapping = {"internal_id": "id", "full_name": "name"}

    chain.to_database("mapped_table", connection, column_mapping=column_mapping)

    rows = _query_table(
        connection, "SELECT id, name FROM mapped_table ORDER BY id"
    ).fetchall()

    assert len(rows) == 3
    assert rows[0] == (1, "Alice Smith")
    assert rows[1] == (2, "Bob Jones")
    assert rows[2] == (3, "Charlie Brown")


def test_to_database_column_mapping_skip_columns(connection, test_session):
    """Test column mapping with explicit column skipping using None."""
    chain = dc.read_values(
        id=[1, 2, 3],
        name=["Alice", "Bob", "Charlie"],
        age=[25, 30, 35],
        internal_notes=["note1", "note2", "note3"],
        session=test_session,
    )

    column_mapping = {
        "id": "user_id",
        "name": "full_name",
        "age": None,  # Skip this column
        "internal_notes": None,  # Skip this column too
    }

    chain.to_database(
        "skipped_columns_table", connection, column_mapping=column_mapping
    )

    columns = _get_table_columns(connection, "skipped_columns_table")

    assert "user_id" in columns
    assert "full_name" in columns
    assert "age" not in columns
    assert "internal_notes" not in columns

    rows = _query_table(
        connection,
        "SELECT user_id, full_name FROM skipped_columns_table ORDER BY user_id",
    ).fetchall()

    assert len(rows) == 3
    assert rows[0] == (1, "Alice")
    assert rows[1] == (2, "Bob")
    assert rows[2] == (3, "Charlie")


def test_to_database_column_mapping_defaultdict(connection, test_session):
    """Test column mapping using defaultdict to skip columns by default."""
    from collections import defaultdict

    chain = dc.read_values(
        id=[1, 2, 3],
        name=["Alice", "Bob", "Charlie"],
        age=[25, 30, 35],
        internal_id=["i1", "i2", "i3"],
        secret_key=["sk1", "sk2", "sk3"],
        session=test_session,
    )

    column_mapping = defaultdict(lambda: None)
    column_mapping.update(
        {
            "id": "user_id",
            "name": "user_name",
        }
    )

    chain.to_database("defaultdict_table", connection, column_mapping=column_mapping)

    columns = _get_table_columns(connection, "defaultdict_table")

    assert "user_id" in columns
    assert "user_name" in columns
    assert "age" not in columns
    assert "internal_id" not in columns
    assert "secret_key" not in columns

    rows = _query_table(
        connection, "SELECT user_id, user_name FROM defaultdict_table ORDER BY user_id"
    ).fetchall()

    assert len(rows) == 3
    assert rows[0] == (1, "Alice")
    assert rows[1] == (2, "Bob")
    assert rows[2] == (3, "Charlie")


def test_to_database_column_mapping_defaultdict_with_datachain_format(
    connection, test_session
):
    """Test defaultdict column mapping with DataChain format (dots) in nested fields."""
    from collections import defaultdict

    class NestedInfo(dc.DataModel):
        value: str
        priority: int

    chain = dc.read_values(
        id=[1, 2],
        name=["Alice", "Bob"],
        age=[25, 30],
        nested_info=[
            NestedInfo(value="important", priority=1),
            NestedInfo(value="normal", priority=2),
        ],
        secret_field=["secret1", "secret2"],
        session=test_session,
    )

    column_mapping = defaultdict(lambda: None)
    column_mapping.update(
        {
            "id": "user_id",
            "name": "user_name",
            "nested_info.value": "info_value",  # DataChain format with dots
            # Skip age, nested_info.priority, and secret_field
        }
    )

    chain.to_database(
        "defaultdict_datachain_table",
        connection,
        column_mapping=column_mapping,
    )

    columns = _get_table_columns(connection, "defaultdict_datachain_table")

    assert "user_id" in columns
    assert "user_name" in columns
    assert "info_value" in columns
    # These should be skipped
    assert "age" not in columns
    assert "nested_info__priority" not in columns
    assert "secret_field" not in columns

    rows = _query_table(
        connection,
        "SELECT user_id, user_name, info_value "
        "FROM defaultdict_datachain_table ORDER BY user_id",
    ).fetchall()

    assert len(rows) == 2
    assert rows[0] == (1, "Alice", "important")
    assert rows[1] == (2, "Bob", "normal")


def test_to_database_column_mapping_complex_nested_names(connection, test_session):
    """Test column mapping with complex/nested column names."""

    class NestedData(dc.DataModel):
        value: str
        metadata: dict

    chain = dc.read_values(
        id=[1, 2, 3],
        user_profile_name=["Alice", "Bob", "Charlie"],
        system_config_theme=["dark", "light", "auto"],
        nested_data=[
            NestedData(value="data1", metadata={"type": "A"}),
            NestedData(value="data2", metadata={"type": "B"}),
            NestedData(value="data3", metadata={"type": "C"}),
        ],
        session=test_session,
    )

    column_mapping = {
        "id": "id",
        "user_profile_name": "name",
        "system_config_theme": "theme",
        "nested_data.value": "data_value",  # Use DataChain format with dots
        "nested_data.metadata": "data_meta",  # Use DataChain format with dots
    }

    chain.to_database(
        "complex_mapping_table", connection, column_mapping=column_mapping
    )

    columns = _get_table_columns(connection, "complex_mapping_table")

    assert "id" in columns
    assert "name" in columns
    assert "theme" in columns
    assert "data_value" in columns
    assert "data_meta" in columns
    # Original complex names should not be present
    assert "user_profile_name" not in columns
    assert "system_config_theme" not in columns

    rows = _query_table(
        connection,
        "SELECT id, name, theme, data_value FROM complex_mapping_table ORDER BY id",
    ).fetchall()

    assert len(rows) == 3
    assert rows[0][:4] == (1, "Alice", "dark", "data1")
    assert rows[1][:4] == (2, "Bob", "light", "data2")
    assert rows[2][:4] == (3, "Charlie", "auto", "data3")


def test_to_database_column_mapping_datachain_format_backward_compatibility(
    connection, test_session
):
    """Test that both DataChain format (dots) and database format (underscores)
    work in column mapping."""

    class NestedData(dc.DataModel):
        value: str
        config: dict

    chain = dc.read_values(
        id=[1, 2],
        simple_field=["test1", "test2"],
        nested_data=[
            NestedData(value="val1", config={"setting": 1}),
            NestedData(value="val2", config={"setting": 2}),
        ],
        session=test_session,
    )

    # Test DataChain format (dots) - this should work
    column_mapping_dots = {
        "id": "record_id",
        "simple_field": "simple",
        "nested_data.value": "nested_val",
        "nested_data.config": "nested_cfg",
    }

    chain.to_database(
        "dots_format_table",
        connection,
        column_mapping=column_mapping_dots,
    )

    # Test database format (double underscores) - this should also work
    column_mapping_underscores = {
        "id": "record_id",
        "simple_field": "simple",
        "nested_data__value": "nested_val",
        "nested_data__config": "nested_cfg",
    }

    chain.to_database(
        "underscores_format_table",
        connection,
        column_mapping=column_mapping_underscores,
    )

    rows_dots = _query_table(
        connection,
        "SELECT record_id, simple, nested_val FROM dots_format_table "
        "ORDER BY record_id",
    ).fetchall()

    rows_underscores = _query_table(
        connection,
        "SELECT record_id, simple, nested_val FROM underscores_format_table "
        "ORDER BY record_id",
    ).fetchall()

    # Both should have identical data
    assert len(rows_dots) == 2
    assert len(rows_underscores) == 2
    assert rows_dots == rows_underscores
    assert rows_dots[0][:3] == (1, "test1", "val1")
    assert rows_dots[1][:3] == (2, "test2", "val2")


def test_to_database_invalid_on_conflict_value(connection, test_session):
    """Test that invalid on_conflict values raise ValueError."""
    chain = dc.read_values(
        id=[1, 2, 3],
        name=["Alice", "Bob", "Charlie"],
        session=test_session,
    )

    with pytest.raises(ValueError, match="on_conflict must be 'ignore' or 'update'"):
        chain.to_database("test_table", connection, on_conflict="invalid")


def test_to_database_with_null_values(connection, test_session, warehouse):
    """Test to_database handles NULL values correctly."""
    from datachain.data_storage.sqlite import SQLiteWarehouse

    chain = dc.read_values(
        id=[1, 2, 3, 4],
        name=["Alice", None, "Charlie", "Diana"],
        age=[25, 30, None, 28],
        session=test_session,
    )

    chain.to_database("null_table", connection)

    rows = _fetch_all_rows(connection, "null_table")

    # Different warehouse implementations handle NULL values differently
    default_str_value = None if isinstance(warehouse, SQLiteWarehouse) else ""
    default_int_value = None if isinstance(warehouse, SQLiteWarehouse) else 0

    assert len(rows) == 4
    assert rows[0] == (1, "Alice", 25)
    assert rows[1] == (2, default_str_value, 30)
    assert rows[2] == (3, "Charlie", default_int_value)
    assert rows[3] == (4, "Diana", 28)


def test_to_database_column_mapping_collision(connection, test_session):
    """Test that providing mapping keys which normalize to the same DB name raises."""

    class Nested(dc.DataModel):
        data: int

    chain = dc.read_values(
        id=[1, 2],
        nested=[Nested(data=10), Nested(data=20)],
        session=test_session,
    )

    # Both keys normalize to 'nested__data'
    column_mapping = {"nested.data": "col1", "nested__data": "col2"}

    with pytest.raises(ValueError, match="Column mapping collision"):
        chain.to_database("collision_table", connection, column_mapping=column_mapping)


def test_to_database_table_cleanup_on_map_exception(connection, test_session):
    """Test that newly created tables are cleaned up when an exception occurs
    during DataChain iteration using map function."""
    table_name = "cleanup_test_table"

    engine = _get_engine_from_connection(connection)
    inspector = sqlalchemy.inspect(engine)
    assert table_name not in inspector.get_table_names()

    def process_item(id_val):
        # Process first few rows normally, then fail on the third item
        # This ensures table creation happens first, but exception occurs during
        # processing
        if id_val >= 3:
            raise ValueError("Processing failed on third item")
        return f"item_{id_val}"

    chain = dc.read_values(
        id=[1, 2, 3, 4, 5],
        session=test_session,
    ).map(process_item, params=["id"], output={"data": str})

    # The export should fail during processing, after table creation
    with pytest.raises(ValueError, match="Processing failed on third item"):
        chain.to_database(table_name, connection)

    # Verify the table was cleaned up (removed) due to the exception
    inspector = sqlalchemy.inspect(engine)
    assert table_name not in inspector.get_table_names(), (
        f"Table {table_name} should have been cleaned up after processing exception"
    )


def test_to_database_comprehensive_data_types(connection, test_session):
    table_name = "comprehensive_data_types_table"

    # Use Eastern Time (UTC-5) for testing timezone handling
    eastern_tz = timezone(timedelta(hours=-5))

    class UserProfile(dc.DataModel):
        name: str
        settings: dict
        tags: list[str]

    chain = dc.read_values(
        id=[1, 2, 3],
        name=["Alice", "Bob", None],  # Include null values
        profile=[
            UserProfile(
                name="Alice",
                settings={"theme": "dark", "notifications": True, "score": 95.5},
                tags=["admin", "vip"],
            ),
            UserProfile(
                name="Bob",
                settings={"theme": "light", "notifications": False, "score": 87.2},
                tags=["user"],
            ),
            UserProfile(
                name="Charlie",
                settings={"theme": "auto", "notifications": True, "score": None},
                tags=["user", "beta"],
            ),
        ],
        scores=[
            [0.95, 0.87, 0.92],  # Test arrays
            [0.88, 0.91],
            [0.76, 0.89, 0.93, 0.85],
        ],
        created_at=[
            datetime(2024, 1, 15, 5, 30, 0, tzinfo=eastern_tz),  # Eastern Time
            datetime(2024, 2, 20, 9, 45, 30, tzinfo=eastern_tz),  # Eastern Time
            datetime(2024, 3, 10, 4, 15, 45, tzinfo=eastern_tz),  # Eastern Time
        ],
        metadata=[
            {"preferences": {"theme": "dark", "lang": "en"}},  # Test nested JSON
            {"preferences": {"theme": "light", "lang": "es"}},
            {"age": 30, "active": True},
        ],
        session=test_session,
    )

    chain.to_database(table_name, connection)

    result = _fetch_all_rows(connection, table_name)
    assert len(result) == 3

    columns = _get_table_columns(connection, table_name)
    expected_columns = [
        "id",
        "name",
        "profile__name",
        "profile__settings",
        "profile__tags",
        "scores",
        "created_at",
        "metadata",
    ]
    for col in expected_columns:
        assert col in columns, f"Expected column '{col}' not found in {columns}"

    # Verify first row has all expected data
    # Handle differences between database types:
    # SQLite returns certain data as strings, PostgreSQL returns as native objects
    first_row = result[0]
    assert first_row[0] == 1  # id
    assert first_row[1] == "Alice"  # name

    profile_name = first_row[2]  # profile.name
    assert profile_name == "Alice"

    # JSONs and dicts
    profile_settings = _parse_json_if_needed(connection, first_row[3])
    assert profile_settings == {"theme": "dark", "notifications": True, "score": 95.5}

    profile_tags = _parse_json_if_needed(connection, first_row[4])  # profile.tags
    assert profile_tags == ["admin", "vip"]

    # Verify arrays
    scores = _parse_json_if_needed(connection, first_row[5])
    assert scores == [0.95, 0.87, 0.92]

    # Verify datetime
    created_at = _parse_datetime_if_needed(connection, first_row[6], eastern_tz)
    original_dt = datetime(2024, 1, 15, 5, 30, 0, tzinfo=eastern_tz)
    assert created_at == original_dt
    assert created_at.tzinfo is not None

    metadata = _parse_json_if_needed(connection, first_row[7])  # metadata
    assert metadata == {"preferences": {"theme": "dark", "lang": "en"}}

    # Verify we can query the data back
    count_result = _count_table_rows(connection, table_name)
    assert count_result == 3
