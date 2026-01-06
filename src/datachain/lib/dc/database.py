import contextlib
import itertools
import os
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import sqlalchemy

from datachain.query.schema import ColumnMeta
from datachain.utils import batched

DEFAULT_DATABASE_BATCH_SIZE = 10_000

if TYPE_CHECKING:
    import sqlalchemy.orm  # noqa: TC004

    from datachain.lib.data_model import DataType
    from datachain.query import Session

    from .datachain import DataChain

    ConnectionType = (
        str
        | sqlalchemy.engine.URL
        | sqlalchemy.engine.interfaces.Connectable
        | sqlalchemy.engine.Engine
        | sqlalchemy.engine.Connection
        | sqlalchemy.orm.Session
        | sqlite3.Connection
    )


@contextlib.contextmanager
def _connect(
    connection: "ConnectionType",
) -> Iterator[sqlalchemy.engine.Connection]:
    import sqlalchemy.orm

    with contextlib.ExitStack() as stack:
        engine_kwargs = {"echo": bool(os.environ.get("DEBUG_SHOW_SQL_QUERIES"))}
        if isinstance(connection, (str, sqlalchemy.URL)):
            engine = sqlalchemy.create_engine(connection, **engine_kwargs)
            stack.callback(engine.dispose)
            yield stack.enter_context(engine.connect())
        elif isinstance(connection, sqlite3.Connection):
            engine = sqlalchemy.create_engine(
                "sqlite://",
                creator=lambda: connection,
                poolclass=sqlalchemy.pool.StaticPool,
                **engine_kwargs,
            )
            # Close only the SQLAlchemy connection wrapper; the underlying
            # sqlite3 connection remains managed by the caller via StaticPool.
            yield stack.enter_context(engine.connect())
        elif isinstance(connection, sqlalchemy.Engine):
            yield stack.enter_context(connection.connect())
        elif isinstance(connection, sqlalchemy.Connection):
            # do not close the connection, as it is managed by the caller
            yield connection
        elif isinstance(connection, sqlalchemy.orm.Session):
            # For Session objects, get the underlying bind (Engine or Connection)
            # Sessions don't support DDL operations directly
            bind = connection.get_bind()
            if isinstance(bind, sqlalchemy.Engine):
                yield stack.enter_context(bind.connect())
            else:
                # bind is already a Connection
                yield bind
        else:
            raise TypeError(f"Unsupported connection type: {type(connection).__name__}")


def to_database(
    chain: "DataChain",
    table_name: str,
    connection: "ConnectionType",
    *,
    batch_size: int = DEFAULT_DATABASE_BATCH_SIZE,
    on_conflict: str | None = None,
    conflict_columns: list[str] | None = None,
    column_mapping: dict[str, str | None] | None = None,
) -> int:
    """
    Implementation function for exporting DataChain to database tables.

    This is the core implementation that handles the actual database operations.
    For user-facing documentation, see DataChain.to_database() method.

    Returns:
        int: Number of rows affected (inserted/updated).
    """
    if on_conflict and on_conflict not in ("ignore", "update"):
        raise ValueError(
            f"on_conflict must be 'ignore' or 'update', got: {on_conflict}"
        )

    signals_schema = chain.signals_schema.clone_without_sys_signals()
    all_columns = [
        sqlalchemy.Column(c.name, c.type)  # type: ignore[union-attr]
        for c in signals_schema.db_signals(as_columns=True)
    ]

    column_mapping = column_mapping or {}
    normalized_column_mapping = _normalize_column_mapping(column_mapping)
    column_indices_and_names, columns = _prepare_columns(
        all_columns, normalized_column_mapping
    )

    normalized_conflict_columns = _normalize_conflict_columns(
        conflict_columns, normalized_column_mapping
    )

    with _connect(connection) as conn:
        metadata = sqlalchemy.MetaData()
        table = sqlalchemy.Table(table_name, metadata, *columns)

        table_existed_before = False
        total_rows_affected = 0
        try:
            with conn.begin():
                # Check if table exists to determine if we should clean up on error.
                inspector = sqlalchemy.inspect(conn)
                assert inspector  # to satisfy mypy
                table_existed_before = table_name in inspector.get_table_names()

                table.create(conn, checkfirst=True)

                rows_iter = chain._leaf_values()
                for batch in batched(rows_iter, batch_size):
                    rows_affected = _process_batch(
                        conn,
                        table,
                        batch,
                        on_conflict,
                        normalized_conflict_columns,
                        column_indices_and_names,
                    )
                    if rows_affected < 0 or total_rows_affected < 0:
                        total_rows_affected = -1
                    else:
                        total_rows_affected += rows_affected
        except Exception:
            if not table_existed_before:
                try:
                    table.drop(conn, checkfirst=True)
                    conn.commit()
                except sqlalchemy.exc.SQLAlchemyError:
                    pass
            raise

    return total_rows_affected


def _normalize_column_mapping(
    column_mapping: dict[str, str | None],
) -> dict[str, str | None]:
    """
    Convert column mapping keys from DataChain format (dots) to database format
    (double underscores).

    This allows users to specify column mappings using the intuitive DataChain
    format like: {"nested_data.value": "data_value"} instead of
    {"nested_data__value": "data_value"}
    """
    if not column_mapping:
        return {}

    normalized_mapping: dict[str, str | None] = {}
    original_keys: dict[str, str] = {}
    for key, value in column_mapping.items():
        db_key = ColumnMeta.to_db_name(key)
        if db_key in normalized_mapping:
            prev = original_keys[db_key]
            raise ValueError(
                "Column mapping collision: multiple keys map to the same "
                f"database column name '{db_key}': '{prev}' and '{key}'. "
            )
        normalized_mapping[db_key] = value
        original_keys[db_key] = key

    # If it's a defaultdict, preserve the default factory
    if hasattr(column_mapping, "default_factory"):
        from collections import defaultdict

        default_factory = column_mapping.default_factory
        result: dict[str, str | None] = defaultdict(default_factory)
        result.update(normalized_mapping)
        return result

    return normalized_mapping


def _normalize_conflict_columns(
    conflict_columns: list[str] | None, column_mapping: dict[str, str | None]
) -> list[str] | None:
    """
    Normalize conflict_columns by converting DataChain format to database format
    and applying column mapping.
    """
    if not conflict_columns:
        return None

    normalized_columns = []
    for col in conflict_columns:
        db_col = ColumnMeta.to_db_name(col)

        if db_col in column_mapping or hasattr(column_mapping, "default_factory"):
            mapped_name = column_mapping[db_col]
            if mapped_name:
                normalized_columns.append(mapped_name)
        else:
            normalized_columns.append(db_col)

    return normalized_columns


def _prepare_columns(all_columns, column_mapping):
    """Prepare column mapping and column definitions."""
    column_indices_and_names = []  # List of (index, target_name) tuples
    columns = []
    for idx, col in enumerate(all_columns):
        if col.name in column_mapping or hasattr(column_mapping, "default_factory"):
            mapped_name = column_mapping[col.name]
            if mapped_name:
                columns.append(sqlalchemy.Column(mapped_name, col.type))
                column_indices_and_names.append((idx, mapped_name))
        else:
            columns.append(col)
            column_indices_and_names.append((idx, col.name))
    return column_indices_and_names, columns


def _process_batch(
    conn, table, batch, on_conflict, conflict_columns, column_indices_and_names
) -> int:
    """Process a batch of rows with conflict resolution.

    Returns:
        int: Number of rows affected by the insert operation.
    """

    def prepare_row(row_values):
        """Convert a row tuple to a dictionary with proper DB column names."""
        return {
            target_name: row_values[idx]
            for idx, target_name in column_indices_and_names
        }

    rows_to_insert = [prepare_row(row) for row in batch]

    supports_conflict = on_conflict and conn.engine.name in ("postgresql", "sqlite")

    insert_stmt: Any  # Can be PostgreSQL, SQLite, or regular insert statement
    if supports_conflict:
        # Use dialect-specific insert for conflict resolution
        if conn.engine.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            insert_stmt = pg_insert(table)
        elif conn.engine.name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            insert_stmt = sqlite_insert(table)
    else:
        insert_stmt = table.insert()

    if supports_conflict:
        if on_conflict == "ignore":
            insert_stmt = insert_stmt.on_conflict_do_nothing()
        elif on_conflict == "update":
            update_values = {
                col.name: insert_stmt.excluded[col.name] for col in table.columns
            }
            if conn.engine.name == "postgresql":
                if not conflict_columns:
                    raise ValueError(
                        "conflict_columns parameter is required when "
                        "on_conflict='update' with PostgreSQL. Specify the column "
                        "names that form a unique constraint."
                    )

                insert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=conflict_columns, set_=update_values
                )
            else:
                insert_stmt = insert_stmt.on_conflict_do_update(set_=update_values)
    elif on_conflict:
        import warnings

        warnings.warn(
            f"Database does not support conflict resolution. "
            f"Ignoring on_conflict='{on_conflict}' parameter.",
            UserWarning,
            stacklevel=2,
        )

    result = conn.execute(insert_stmt, rows_to_insert)
    return result.rowcount


def read_database(
    query: "str | sqlalchemy.sql.expression.Executable",
    connection: "ConnectionType",
    params: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    output: dict[str, "DataType"] | None = None,
    session: "Session | None" = None,
    settings: dict | None = None,
    in_memory: bool = False,
    infer_schema_length: int | None = 100,
) -> "DataChain":
    """
    Read the results of a SQL query into a DataChain, using a given database connection.

    Args:
        query:
            The SQL query to execute. Can be a raw SQL string or a SQLAlchemy
            `Executable` object.
        connection: SQLAlchemy connectable, str, or a sqlite3 connection
            Using SQLAlchemy makes it possible to use any DB supported by that
            library. If a DBAPI2 object, only sqlite3 is supported. The user is
            responsible for engine disposal and connection closure for the
            SQLAlchemy connectable; str connections are closed automatically.
        params: Parameters to pass to execute method.
        output: A dictionary mapping column names to types, used to override the
            schema inferred from the query results.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        in_memory: If True, creates an in-memory session. Defaults to False.
        infer_schema_length:
            The maximum number of rows to scan for inferring schema.
            If set to `None`, the full data may be scanned.
            The rows used for schema inference are stored in memory,
            so large values can lead to high memory usage.
            Only applies if the `output` parameter is not set for the given column.

    Examples:
        Reading from a SQL query against a user-supplied connection:
        ```python
        query = "SELECT key, value FROM tbl"
        chain = dc.read_database(query, connection, output={"value": float})
        ```

        Load data from a SQLAlchemy driver/engine:
        ```python
        from sqlalchemy import create_engine
        engine = create_engine("postgresql+psycopg://myuser:mypassword@localhost:5432/mydb")
        chain = dc.read_database("select * from tbl", engine)
        ```

        Load data from a parameterized SQLAlchemy query:
        ```python
        query = "SELECT key, value FROM tbl WHERE value > :value"
        dc.read_database(query, engine, params={"value": 50})
        ```

    Notes:
        - This function works with a variety of databases — including,
        but not limited to, SQLite, DuckDB, PostgreSQL, and Snowflake,
        provided the appropriate driver is installed.
        - This call is blocking, and will execute the query and return once the
          results are saved.
    """
    from datachain.lib.dc.records import read_records

    output = output or {}
    if isinstance(query, str):
        query = sqlalchemy.text(query)
    kw = {"execution_options": {"stream_results": True}}  # use server-side cursors
    with _connect(connection) as conn, conn.execute(query, params, **kw) as result:
        cols = result.keys()
        to_infer = [k for k in cols if k not in output]  # preserve the order
        rows, inferred_schema = _infer_schema(result, to_infer, infer_schema_length)
        records = (row._asdict() for row in itertools.chain(rows, result))
        return read_records(
            records,
            session=session,
            settings=settings,
            in_memory=in_memory,
            schema=inferred_schema | output,
        )


def _infer_schema(
    result: "sqlalchemy.engine.Result",
    to_infer: list[str],
    infer_schema_length: int | None = 100,
) -> tuple[list["sqlalchemy.Row"], dict[str, "DataType"]]:
    from datachain.lib.convert.values_to_tuples import values_to_tuples

    if not to_infer:
        return [], {}

    rows = list(itertools.islice(result, infer_schema_length))
    values = {col: [row._mapping[col] for row in rows] for col in to_infer}
    _, output_schema, _ = values_to_tuples("", **values)
    return rows, output_schema
