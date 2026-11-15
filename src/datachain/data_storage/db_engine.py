import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Self

import sqlalchemy as sa
from sqlalchemy.sql import FROM_LINTING
from sqlalchemy.sql.roles import DDLRole

from datachain.data_storage.serializer import Serializable
from datachain.error import TableMissingError

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Table
    from sqlalchemy.engine.base import Engine
    from sqlalchemy.engine.interfaces import Dialect
    from sqlalchemy.sql.compiler import Compiled
    from sqlalchemy.sql.elements import ClauseElement


logger = logging.getLogger("datachain")

SELECT_BATCH_SIZE = 100_000  # number of rows to fetch at a time


class DatabaseEngine(ABC, Serializable):
    dialect: ClassVar["Dialect"]

    engine: "Engine"
    metadata: "MetaData"
    # Compiled metastore statements by shape (SQL text and bind names, never
    # data, so no other process can make an entry stale); None turns it off.
    compiled_cache: dict | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @abstractmethod
    def clone(self) -> "DatabaseEngine":
        """Clones DatabaseEngine implementation."""

    @classmethod
    def compile(cls, statement: "ClauseElement", **kwargs) -> "Compiled":
        """
        Compile a sqlalchemy query or ddl object to a Compiled object.

        Use the `string` and `params` properties of this object to get
        the resulting sql string and parameters.
        """
        if not isinstance(statement, DDLRole):
            # render_postcompile is needed for in_ queries to work
            kwargs["compile_kwargs"] = {
                **kwargs.pop("compile_kwargs", {}),
                "render_postcompile": True,
            }
            kwargs = {"linting": FROM_LINTING} | kwargs
        return statement.compile(dialect=cls.dialect, **kwargs)

    @classmethod
    def compile_to_args(
        cls, statement: "ClauseElement", **kwargs
    ) -> tuple[str] | tuple[str, dict[str, Any]]:
        """
        Compile a sqlalchemy query or ddl object to an args tuple.

        This tuple is formatted specifically for calling
        `cursor.execute(*args)` according to the python DB-API.
        """
        result = cls.compile(statement, **kwargs)
        params = result.params
        if params is None:
            return (result.string,)
        return result.string, params

    def to_args(
        self, statement: "ClauseElement"
    ) -> tuple[str] | tuple[str, dict[str, Any]]:
        """`compile_to_args`, compiling each shape once if `compiled_cache` is
        set. An IN list renders into the text, so it compiles every time."""
        cache = self.compiled_cache
        if cache is None or not (key := statement._generate_cache_key()):
            return self.compile_to_args(statement)
        if (compiled := cache.get(key.key)) is None:
            compiled = self.compile(statement, cache_key=key)
            if getattr(compiled, "post_compile_params", None):
                return compiled.string, compiled.params or {}
            cache[key.key] = compiled
        params: Any = compiled.construct_params(extracted_parameters=key[1])
        return compiled.string, params or {}

    @abstractmethod
    def execute(self, query) -> Iterator[tuple[Any, ...]]: ...

    def get_table(self, name: str) -> "Table":
        """Get a table by name, raising TableMissingError if not found. Only its
        columns are reflected: one query, where indexes and keys take ten more."""
        if (table := self.metadata.tables.get(name)) is not None:
            return table
        try:
            columns = [
                sa.Column(c["name"], c["type"], primary_key=bool(c.get("primary_key")))
                for c in sa.inspect(self.engine).get_columns(name)
            ]
        except sa.exc.NoSuchTableError as e:
            raise TableMissingError(f"Table '{name}' not found") from e
        return sa.Table(name, self.metadata, *columns)

    @abstractmethod
    def executemany(self, sql: str, params) -> None: ...

    @abstractmethod
    def execute_str(self, sql: str, parameters=None) -> Iterator[tuple[Any, ...]]: ...

    @abstractmethod
    def insert_dataframe(self, table_name: str, df) -> int: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def transaction(self): ...

    def has_table(self, name: str) -> bool:
        """
        Return True if a table exists with the given name
        """
        return sa.inspect(self.engine).has_table(name)

    @abstractmethod
    def list_tables(self, pattern: str = "") -> list[str]:
        """
        List all table names, optionally filtered by a SQL LIKE pattern.

        Args:
            pattern: SQL LIKE pattern to filter table names (e.g. 'udf_%')

        Returns:
            List of table names matching the pattern
        """

    @abstractmethod
    def create_table(
        self,
        table: "Table",
        if_not_exists: bool = True,
        *,
        kind: str | None = None,
    ) -> None:
        """
        Create table. Does nothing if table already exists when if_not_exists=True.
        """

    @abstractmethod
    def drop_table(self, table: "Table", if_exists: bool = False) -> None: ...

    @abstractmethod
    def rename_table(self, old_name: str, new_name: str): ...
