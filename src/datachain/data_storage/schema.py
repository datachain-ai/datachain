import inspect
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import sqlalchemy as sa
from sqlalchemy.sql import func as f
from sqlalchemy.sql.expression import false, null, true

from datachain.sql.functions import path as pathfunc
from datachain.sql.types import (
    JSON,
    Boolean,
    DateTime,
    Int64,
    SQLType,
    String,
    UInt64,
)

if TYPE_CHECKING:
    from sqlalchemy.engine.interfaces import Dialect
    from sqlalchemy.sql.base import (
        ColumnCollection,
        Executable,
        ReadOnlyColumnCollection,
    )
    from sqlalchemy.sql.elements import ColumnElement

    from datachain.data_storage.db_engine import DatabaseEngine


DEFAULT_DELIMITER = "__"


def col_name(name: str, column: str = "file") -> str:
    return f"{column}{DEFAULT_DELIMITER}{name}"


def dedup_columns(columns: Iterable[sa.Column]) -> list[sa.Column]:
    """
    Removes duplicate columns from a list of columns.
    If column with the same name and different type is found, exception is
    raised
    """
    c_set: dict[str, sa.Column] = {}
    for c in columns:
        if (ec := c_set.get(c.name)) is not None:
            if str(ec.type) != str(c.type):
                raise ValueError(
                    f"conflicting types for column {c.name}:{c.type!s} and {ec.type!s}"
                )
            continue
        c_set[c.name] = c

    return list(c_set.values())


def convert_rows_custom_column_types(
    columns: "ColumnCollection[str, ColumnElement[Any]]",
    rows: Iterator[tuple[Any, ...]],
    dialect: "Dialect",
) -> Iterator[tuple[Any, ...]]:
    """
    This function converts values of rows columns based on their types which are
    defined in columns. We are only converting column values for which types are
    subclasses of our SQLType, as only for those we have converters registered.
    """
    # indexes of SQLType column in a list of columns so that we can skip the rest
    custom_columns_types: list[tuple[int, SQLType]] = [
        (idx, c.type) for idx, c in enumerate(columns) if isinstance(c.type, SQLType)
    ]

    if not custom_columns_types:
        yield from rows

    for row in rows:
        row_list = list(row)
        for idx, t in custom_columns_types:
            row_list[idx] = (
                t.default_value(dialect)
                if row_list[idx] is None
                else t.on_read_convert(row_list[idx], dialect)
            )

        yield tuple(row_list)


class DirExpansion:
    def __init__(self, column: str):
        self.column = column

    def col_name(self, name: str, column: str | None = None) -> str:
        column = column or self.column
        return col_name(name, column)

    def c(self, query, name: str, column: str | None = None) -> str:
        return getattr(query.c, self.col_name(name, column=column))

    def base_select(self, q):
        return sa.select(
            self.c(q, "id", column="sys"),
            false().label(self.col_name("is_dir")),
            self.c(q, "source"),
            self.c(q, "path"),
            self.c(q, "version"),
            self.c(q, "location"),
        )

    def apply_group_by(self, q):
        return (
            sa.select(
                f.min(q.c.sys__id).label("sys__id"),
                self.c(q, "is_dir"),
                self.c(q, "source"),
                self.c(q, "path"),
                self.c(q, "version"),
                f.max(self.c(q, "location")).label(self.col_name("location")),
            )
            .select_from(q)
            .group_by(
                self.c(q, "source"),
                self.c(q, "path"),
                self.c(q, "is_dir"),
                self.c(q, "version"),
            )
            .order_by(
                self.c(q, "source"),
                self.c(q, "path"),
                self.c(q, "is_dir"),
                self.c(q, "version"),
            )
        )

    def query(self, q):
        q = self.base_select(q).cte(recursive=True)
        parent = pathfunc.parent(self.c(q, "path"))
        q = q.union_all(
            sa.select(
                sa.literal(-1).label("sys__id"),
                true().label(self.col_name("is_dir")),
                self.c(q, "source"),
                parent.label(self.col_name("path")),
                sa.literal("").label(self.col_name("version")),
                null().label(self.col_name("location")),
            ).where(parent != "")
        )
        return self.apply_group_by(q)


class DataTable:
    MAX_RANDOM = 2**63 - 1

    def __init__(
        self,
        name: str,
        engine: "DatabaseEngine",
        column_types: dict[str, SQLType] | None = None,
        column: str = "file",
    ):
        self.name: str = name
        self.engine = engine
        self.column_types: dict[str, SQLType] = column_types or {}
        self.column = column

    @staticmethod
    def copy_column(
        column: sa.Column,
        primary_key: bool | None = None,
        index: bool | None = None,
        nullable: bool | None = None,
        default: Any | None = None,
        server_default: Any | None = None,
        unique: bool | None = None,
    ) -> sa.Column:
        """
        Copy a sqlalchemy Column object intended for use as a signal column.

        This does not copy all attributes as certain attributes such as
        table are too context-dependent and the purpose of this function is
        adding a signal column from one table to another table.

        We can't use Column.copy() as it only works in certain contexts.
        See https://github.com/sqlalchemy/sqlalchemy/issues/5953
        """
        return sa.Column(
            column.name,
            column.type,
            primary_key=primary_key if primary_key is not None else column.primary_key,
            index=index if index is not None else column.index,
            nullable=nullable if nullable is not None else column.nullable,
            default=default if default is not None else column.default,
            server_default=(
                server_default if server_default is not None else column.server_default
            ),
            unique=unique if unique is not None else column.unique,
        )

    @classmethod
    def new_table(
        cls,
        name: str,
        columns: Sequence[sa.Column] = (),
        metadata: sa.MetaData | None = None,
    ):
        # copy columns, since reusing the same objects from another table
        # may raise an error
        columns = cls.sys_columns() + [cls.copy_column(c) for c in columns]
        columns = dedup_columns(columns)

        if metadata is None:
            metadata = sa.MetaData()
        return sa.Table(name, metadata, *columns)

    def get_table(self) -> sa.Table:
        table = self.engine.get_table(self.name)

        column_types = self.column_types | {c.name: c.type for c in self.sys_columns()}
        # adjusting types for custom columns to be instances of SQLType if possible
        for c in table.columns:
            if c.name in column_types:
                t = column_types[c.name]
                c.type = t() if inspect.isclass(t) else t
        return table

    @property
    def columns(self) -> "ReadOnlyColumnCollection[str, sa.Column[Any]]":
        return self.table.columns

    def col_name(self, name: str, column: str | None = None) -> str:
        column = column or self.column
        return col_name(name, column)

    def without_object(self, column_name: str, column: str | None = None) -> str:
        column = column or self.column
        return column_name.removeprefix(f"{column}{DEFAULT_DELIMITER}")

    def c(self, name: str, column: str | None = None):
        return getattr(self.columns, self.col_name(name, column=column))

    @property
    def table(self) -> sa.Table:
        return self.get_table()

    def apply_conditions(self, query: "Executable") -> "Executable":
        """
        Apply any conditions that belong on all selecting queries.

        This could be used to filter tables that use access control.
        """
        return query

    def select(self, *columns):
        if not columns:
            query = self.table.select()
        else:
            query = sa.select(*columns)
        return self.apply_conditions(query)

    def insert(self):
        return self.table.insert()

    def update(self):
        return self.apply_conditions(self.table.update())

    def delete(self):
        return self.apply_conditions(self.table.delete())

    @classmethod
    def sys_columns(cls):
        return [
            sa.Column("sys__id", UInt64, primary_key=True),
            sa.Column(
                "sys__rand", UInt64, nullable=False, server_default=f.abs(f.random())
            ),
        ]

    @classmethod
    def listing_columns(cls):
        return [
            sa.Column("file__source", String()),
            sa.Column("file__path", String()),
            sa.Column("file__size", Int64()),
            sa.Column("file__version", String()),
            sa.Column("file__etag", String()),
            sa.Column("file__is_latest", Boolean()),
            sa.Column("file__last_modified", DateTime()),
            sa.Column("file__location", JSON()),
        ]

    def dir_expansion(self):
        return DirExpansion(self.column)


PARTITION_COLUMN_ID = "partition_id"

partition_col_names = [PARTITION_COLUMN_ID]


def partition_columns() -> Sequence[sa.Column]:
    return [
        sa.Column(PARTITION_COLUMN_ID, sa.Integer),
    ]


DataTableT = TypeVar("DataTableT", bound=DataTable)


class Schema(Generic[DataTableT]):
    dataset_row_cls: type[DataTableT]


class DefaultSchema(Schema[DataTable]):
    def __init__(self):
        self.dataset_row_cls = DataTable
