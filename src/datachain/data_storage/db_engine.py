import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.sql import FROM_LINTING
from sqlalchemy.sql.roles import DDLRole

from datachain.data_storage.serializer import Serializable

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

    def __enter__(self) -> "DatabaseEngine":
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

    @abstractmethod
    def execute(
        self,
        query,
        cursor: Any | None = None,
        conn: Any | None = None,
    ) -> Iterator[tuple[Any, ...]]: ...

    def get_table(self, name: str) -> "Table":
        from datachain.error import TableMissingError

        table = self.metadata.tables.get(name)
        if table is None:
            try:
                sa.Table(name, self.metadata, autoload_with=self.engine)
                # ^^^ This table may not be correctly initialised on some dialects
                # Grab it from metadata instead.
                table = self.metadata.tables.get(name)
                if table is None:
                    raise TableMissingError(f"Table '{name}' not found")
            except sa.exc.NoSuchTableError as e:
                raise TableMissingError(f"Table '{name}' not found") from e
        return table

    @abstractmethod
    def executemany(
        self, query, params, cursor: Any | None = None
    ) -> Iterator[tuple[Any, ...]]: ...

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

    def list_tables(self, prefix: str = "") -> list[str]:
        """
        Return a list of table names that start with the given prefix.
        If no prefix is provided, returns all table names.
        """
        all_tables = sa.inspect(self.engine).get_table_names()
        if not prefix:
            return all_tables
        return [table for table in all_tables if table.startswith(prefix)]

    @abstractmethod
    def create_table(
        self,
        table: "Table",
        if_not_exists: bool = True,
        *,
        kind: str | None = None,
    ) -> bool:
        """Create table and return True if created, False if already existed."""
        ...

    @abstractmethod
    def drop_table(self, table: "Table", if_exists: bool = False) -> None: ...

    @abstractmethod
    def rename_table(self, old_name: str, new_name: str): ...
