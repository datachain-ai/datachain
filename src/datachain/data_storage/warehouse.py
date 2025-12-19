import glob
import logging
import posixpath
import secrets
import string
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Union, cast
from urllib.parse import urlparse

import attrs
import sqlalchemy as sa
from sqlalchemy.sql.expression import true

from datachain import json
from datachain.client import Client
from datachain.data_storage.schema import convert_rows_custom_column_types
from datachain.data_storage.serializer import Serializable
from datachain.dataset import DatasetRecord, StorageURI
from datachain.lib.file import File
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema
from datachain.node import DirType, DirTypeGroup, Node, NodeWithPath, get_path
from datachain.query.batch import RowsOutput
from datachain.query.schema import ColumnMeta
from datachain.sql.functions import path as pathfunc
from datachain.sql.types import SQLType
from datachain.utils import sql_escape_like

if TYPE_CHECKING:
    from sqlalchemy.sql._typing import (
        _ColumnsClauseArgument,
        _FromClauseArgument,
        _OnClauseArgument,
    )
    from sqlalchemy.sql.selectable import FromClause
    from sqlalchemy.types import TypeEngine

    from datachain.data_storage import schema
    from datachain.data_storage.db_engine import DatabaseEngine
    from datachain.data_storage.schema import DataTable


logger = logging.getLogger("datachain")

SELECT_BATCH_SIZE = 100_000  # number of rows to fetch at a time
INSERT_BATCH_SIZE = 10_000  # number of rows to insert at a time


class AbstractWarehouse(ABC, Serializable):
    """
    Abstract Warehouse class, to be implemented by any Database Adapters
    for a specific database system. This manages the storing, searching, and
    retrieval of datasets data, and has shared logic for all database
    systems currently in use.
    """

    #
    # Constants, Initialization, and Tables
    #

    DATASET_TABLE_PREFIX = "ds_"
    DATASET_SOURCE_TABLE_PREFIX = "src_"
    UDF_TABLE_NAME_PREFIX = "udf_"
    TMP_TABLE_NAME_PREFIX = "tmp_"

    schema: "schema.Schema"
    db: "DatabaseEngine"

    def __enter__(self) -> "AbstractWarehouse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Default behavior is to do nothing, as connections may be shared."""

    def cleanup_for_tests(self):
        """Cleanup for tests."""

    def _to_jsonable(self, obj: Any) -> Any:
        """Recursively convert Python/Pydantic structures into JSON-serializable
        objects.
        """

        if ModelStore.is_pydantic(type(obj)):
            # Use Pydantic's JSON mode to ensure datetime and other non-JSON
            # native types are serialized in a compatible way.
            return obj.model_dump(mode="json")

        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                if not isinstance(k, str):
                    key_str = json.dumps(self._to_jsonable(k), ensure_ascii=False)
                else:
                    key_str = k
                out[key_str] = self._to_jsonable(v)
            return out

        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(i) for i in obj]

        return obj

    def convert_type(  # noqa: PLR0911
        self,
        val: Any,
        col_type: SQLType,
        col_python_type: Any,
        col_type_name: str,
        col_name: str,
    ) -> Any:
        """
        Tries to convert value to specific type if needed and if compatible,
        otherwise throws an ValueError.
        If value is a list or some other iterable, it tries to convert sub elements
        as well
        """
        import numpy as np

        if isinstance(val, (np.ndarray, np.generic)):
            val = val.tolist()

        # Optimization: Precompute all the column type variables.
        value_type = type(val)

        exc = None
        try:
            if col_python_type is list and value_type in (list, tuple, set):
                if len(val) == 0:
                    return []

                item_python_type = self.python_type(col_type.item_type)

                if item_python_type is not list:
                    if isinstance(val[0], item_python_type):
                        # SQLite ARRAY storage expects a list; tuples/sets must be
                        # converted to lists even when element types already match.
                        return list(val)
                    if item_python_type is float and isinstance(val[0], int):
                        return [float(i) for i in val]

                # Optimization: Reuse these values for each function call within the
                # list comprehension.
                item_type_info = (
                    col_type.item_type,
                    item_python_type,
                    type(col_type.item_type).__name__,
                    col_name,
                )
                return [self.convert_type(i, *item_type_info) for i in val]
            # Special use case with JSON type as we save it as string
            if col_python_type is dict or col_type_name == "JSON":
                if value_type is str:
                    return val
                try:
                    json_ready = self._to_jsonable(val)
                    return json.dumps(json_ready, ensure_ascii=False)
                except Exception as e:
                    raise ValueError(
                        f"Cannot convert value {val!r} with type {value_type} to JSON"
                    ) from e

            if isinstance(val, col_python_type):
                return val
            if col_python_type is float and isinstance(val, int):
                return float(val)
        except Exception as e:  # noqa: BLE001
            exc = e
        ve = ValueError(
            f"Value {val!r} with type {value_type} incompatible for "
            f"column type {col_type_name}"
        )
        # This is the same as "from exc" when not raising the exception.
        if exc:
            ve.__cause__ = exc
        # Optimization: Log here, so only one try/except is needed, since the ValueError
        # above is raised after logging.
        logger.exception(
            "Error while validating/converting type for column "
            "%s with value %s, original error %s",
            col_name,
            val,
            ve,
        )
        raise ve

    @abstractmethod
    def clone(self, use_new_connection: bool = False) -> "AbstractWarehouse":
        """Clones Warehouse implementation for some Storage input.
        Setting use_new_connection will always use a new database connection.
        New connections should only be used if needed due to errors with
        closed connections."""

    def close(self) -> None:
        """Closes any active database connections."""
        self.db.close()

    def close_on_exit(self) -> None:
        """Closes any active database or HTTP connections, called on Session exit or
        for test cleanup only, as some Warehouse implementations may handle this
        differently."""
        self.close()

    #
    # Query Tables
    #

    @abstractmethod
    def is_ready(self, timeout: int | None = None) -> bool: ...

    def dataset_rows(
        self,
        dataset: DatasetRecord,
        version: str | None = None,
        column: str = "file",
    ):
        version = version or dataset.latest_version

        table_name = self.dataset_table_name(dataset, version)
        return self.schema.dataset_row_cls(
            table_name,
            self.db,
            dataset.get_schema(version),
            column=column,
        )

    @property
    def dataset_row_cls(self) -> type["DataTable"]:
        return self.schema.dataset_row_cls

    #
    # Query Execution
    #

    def query_count(self, query: sa.Select) -> int:
        """Count the number of rows in a query."""
        count_query = sa.select(sa.func.count(1)).select_from(query.subquery())
        return next(self.db.execute(count_query))[0]

    def table_rows_count(self, table) -> int:
        count_query = sa.select(sa.func.count(1)).select_from(table)
        return next(self.db.execute(count_query))[0]

    def dataset_select_paginated(
        self,
        query,
        page_size: int = SELECT_BATCH_SIZE,
    ) -> Generator[Sequence, None, None]:
        """
        This is equivalent to `db.execute`, but for selecting rows in batches
        """
        limit = query._limit
        paginated_query = query.limit(page_size)

        offset = query._offset or 0
        num_yielded = 0

        # Ensure we're using a thread-local connection
        with self.clone() as wh:
            while True:
                if limit is not None:
                    limit -= num_yielded
                    num_yielded = 0
                    if limit <= 0:
                        break
                    if limit < page_size:
                        paginated_query = paginated_query.limit(None).limit(limit)

                # Cursor results are not thread-safe, so we convert them to a list
                results = list(wh.dataset_rows_select(paginated_query.offset(offset)))

                processed = 0
                for row in results:
                    processed += 1
                    yield row
                    num_yielded += 1

                if processed < page_size:
                    break  # no more results
                offset += page_size

    def _regenerate_system_columns(
        self,
        selectable: sa.Select,
        keep_existing_columns: bool = False,
        regenerate_columns: Iterable[str] | None = None,
    ) -> sa.Select:
        """
        Return a SELECT that regenerates system columns deterministically.

        If keep_existing_columns is True, existing system columns will be kept as-is
        even when they are listed in ``regenerate_columns``.

        Args:
            selectable: Base SELECT
            keep_existing_columns: When True, reuse existing system columns even if
                they are part of the regeneration set.
            regenerate_columns: Names of system columns to regenerate. Defaults to
                {"sys__id", "sys__rand"}. Columns not listed are left untouched.
        """
        system_columns = {
            sys_col.name: sys_col.type
            for sys_col in self.schema.dataset_row_cls.sys_columns()
        }
        regenerate = set(regenerate_columns or system_columns)
        generators = {
            "sys__id": self._system_row_number_expr,
            "sys__rand": self._system_random_expr,
        }

        base = cast("FromClause", selectable.subquery())

        def build(name: str) -> sa.ColumnElement:
            expr = generators[name]()
            return sa.cast(expr, system_columns[name]).label(name)

        columns: list[sa.ColumnElement] = []
        present: set[str] = set()
        changed = False

        for col in base.c:
            present.add(col.name)
            regen = col.name in regenerate and not keep_existing_columns
            columns.append(build(col.name) if regen else col)
            changed |= regen

        for name in regenerate - present:
            columns.append(build(name))
            changed = True

        if not changed:
            return selectable

        inner = sa.select(*columns).select_from(base).subquery()
        return sa.select(*inner.c).select_from(inner)

    def _system_row_number_expr(self):
        """Return an expression that produces deterministic row numbers."""

        raise NotImplementedError

    def _system_random_expr(self):
        """Return an expression that produces deterministic random values."""

        raise NotImplementedError

    #
    # Table Name Internal Functions
    #

    @staticmethod
    def uri_to_storage_info(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        name = parsed.path if parsed.scheme == "file" else parsed.netloc
        return parsed.scheme, name

    def dataset_table_name(self, dataset: DatasetRecord, version: str) -> str:
        return self._construct_dataset_table_name(
            dataset.project.namespace.name,
            dataset.project.name,
            dataset.name,
            version,
        )

    def _construct_dataset_table_name(
        self, namespace: str, project: str, dataset_name: str, version: str
    ) -> str:
        prefix = self.DATASET_TABLE_PREFIX
        if Client.is_data_source_uri(dataset_name):
            # for datasets that are created for bucket listing we use different prefix
            prefix = self.DATASET_SOURCE_TABLE_PREFIX
        return (
            f"{prefix}{namespace}_{project}_{dataset_name}_{version.replace('.', '_')}"
        )

    def temp_table_name(self) -> str:
        return self.TMP_TABLE_NAME_PREFIX + _random_string(6)

    def udf_table_name(self) -> str:
        return self.UDF_TABLE_NAME_PREFIX + _random_string(6)

    #
    # Datasets
    #

    @abstractmethod
    def create_dataset_rows_table(
        self,
        name: str,
        columns: Sequence["sa.Column"] = (),
        if_not_exists: bool = True,
    ) -> sa.Table:
        """Creates a dataset rows table for the given dataset name and columns"""

    def drop_dataset_rows_table(
        self,
        dataset: DatasetRecord,
        version: str,
        if_exists: bool = True,
    ) -> None:
        """Drops a dataset rows table for the given dataset name."""
        table_name = self.dataset_table_name(dataset, version)
        table = sa.Table(table_name, self.db.metadata)
        self.db.drop_table(table, if_exists=if_exists)
        # Remove from metadata cache to allow recreation
        if table_name in self.db.metadata.tables:
            self.db.metadata.remove(self.db.metadata.tables[table_name])

    def dataset_rows_select(
        self,
        query: sa.Select,
        **kwargs,
    ) -> Iterator[tuple[Any, ...]]:
        """
        Fetch dataset rows from database.
        """
        rows = self.db.execute(query, **kwargs)
        yield from convert_rows_custom_column_types(
            query.selected_columns, rows, self.db.dialect
        )

    def dataset_rows_select_from_ids(
        self,
        query: sa.Select,
        ids: Iterable[RowsOutput],
        is_batched: bool,
    ) -> Iterator[RowsOutput]:
        """
        Fetch dataset rows from database using a list of IDs.
        """
        if (id_col := query.selected_columns.get("sys__id")) is None:
            raise RuntimeError("sys__id column not found in query")

        query = query._clone().offset(None).limit(None).order_by(None)

        if is_batched:
            for batch in ids:
                yield list(self.dataset_rows_select(query.where(id_col.in_(batch))))
        else:
            yield from self.dataset_rows_select(query.where(id_col.in_(ids)))

    @abstractmethod
    def get_dataset_sources(
        self, dataset: DatasetRecord, version: str
    ) -> list[StorageURI]: ...

    def rename_dataset_tables(
        self, dataset: DatasetRecord, dataset_updated: DatasetRecord
    ) -> None:
        """
        Renames all dataset version tables when parts of the dataset that
        are used in constructing table name are updated.
        If nothing important is changed, nothing will be renamed (no DB calls
        will be made at all).
        """
        for version in [v.version for v in dataset_updated.versions]:
            if not dataset.has_version(version):
                continue
            src = self.dataset_table_name(dataset, version)
            dest = self.dataset_table_name(dataset_updated, version)
            if src == dest:
                continue
            self.db.rename_table(src, dest)

    def dataset_rows_count(self, dataset: DatasetRecord, version=None) -> int:
        """Returns total number of rows in a dataset"""
        dr = self.dataset_rows(dataset, version)
        table = dr.get_table()
        query = sa.select(sa.func.count(table.c.sys__id))
        (res,) = self.db.execute(query)
        return res[0]

    def dataset_stats(
        self, dataset: DatasetRecord, version: str
    ) -> tuple[int | None, int | None]:
        """
        Returns tuple with dataset stats: total number of rows and total dataset size.
        """
        if not (self.db.has_table(self.dataset_table_name(dataset, version))):
            return None, None

        file_signals = list(
            SignalSchema.deserialize(dataset.feature_schema).get_signals(File)
        )

        dr = self.dataset_rows(dataset, version)
        table = dr.get_table()
        expressions: tuple[_ColumnsClauseArgument[Any], ...] = (
            sa.func.count(table.c.sys__id),
        )
        size_column_names = [ColumnMeta.to_db_name(s) + "__size" for s in file_signals]
        size_columns = [c for c in table.columns if c.name in size_column_names]

        if size_columns:
            expressions = (*expressions, sa.func.sum(sum(size_columns)))
        query = sa.select(*expressions)
        ((nrows, *rest),) = self.db.execute(query)
        return nrows, rest[0] if rest else 0

    @abstractmethod
    def prepare_entries(self, entries: "Iterable[File]") -> Iterable[dict[str, Any]]:
        """Convert File entries so they can be passed on to `insert_rows()`"""

    @abstractmethod
    def insert_rows(
        self,
        table: sa.Table,
        rows: Iterable[dict[str, Any]],
        batch_size: int = INSERT_BATCH_SIZE,
    ) -> None:
        """Does batch inserts of any kind of rows into table"""

    def insert_rows_done(self, table: sa.Table) -> None:
        """
        Only needed for certain implementations
        to signal when rows inserts are complete.
        """

    @abstractmethod
    def insert_dataset_rows(self, df, dataset: DatasetRecord, version: str) -> int:
        """Inserts dataset rows directly into dataset table"""

    @abstractmethod
    def instr(self, source, target) -> sa.ColumnElement:
        """
        Return SQLAlchemy Boolean determining if a target substring is present in
        source string column
        """

    @abstractmethod
    def get_table(self, name: str) -> sa.Table:
        """
        Returns a SQLAlchemy Table object by name. If table doesn't exist, it should
        create it
        """

    @abstractmethod
    def export_dataset_table(
        self,
        bucket: str,
        dataset: DatasetRecord,
        version: str,
        *,
        file_format: str | None = None,
        base_file_name: str,
        client_config=None,
    ) -> None:
        """Exports dataset table to the cloud, e.g to some s3 bucket"""

    def python_type(self, col_type: Union["TypeEngine", "SQLType"]) -> Any:
        """Returns python type representation of some Sqlalchemy column type"""
        return col_type.python_type

    def add_node_type_where(
        self,
        query: sa.Select,
        type: str,
        dataset_rows: "DataTable",
        include_subobjects: bool = True,
    ) -> sa.Select:
        dr = dataset_rows

        def col(name: str):
            return getattr(query.selected_columns, dr.col_name(name))

        file_group: Sequence[int]
        if type in {"f", "file", "files"}:
            if include_subobjects:
                file_group = DirTypeGroup.SUBOBJ_FILE
            else:
                file_group = DirTypeGroup.FILE
        elif type in {"d", "dir", "directory", "directories"}:
            if include_subobjects:
                file_group = DirTypeGroup.SUBOBJ_DIR
            else:
                file_group = DirTypeGroup.DIR
        else:
            raise ValueError(f"invalid file type: {type!r}")

        q = query.where(col("dir_type").in_(file_group))
        if not include_subobjects:
            q = q.where((col("location") == "") | (col("location").is_(None)))
        return q

    def get_nodes(self, query, dataset_rows: "DataTable") -> Iterator[Node]:
        """
        This gets nodes based on the provided query, and should be used sparingly,
        as it will be slow on any OLAP database systems.
        """
        dr = dataset_rows
        columns = [c.name for c in query.selected_columns]
        for row in self.db.execute(query):
            d = dict(zip(columns, row, strict=False))
            yield Node(**{dr.without_object(k): v for k, v in d.items()})

    def get_dirs_by_parent_path(
        self,
        dataset_rows: "DataTable",
        parent_path: str,
    ) -> Iterator[Node]:
        """Gets nodes from database by parent path, with optional filtering"""
        dr = dataset_rows
        query = self._find_query(
            dr,
            parent_path,
            type="dir",
            conds=[pathfunc.parent(sa.Column(dr.col_name("path"))) == parent_path],
            order_by=[dr.col_name("source"), dr.col_name("path")],
        )
        return self.get_nodes(query, dr)

    def _get_nodes_by_glob_path_pattern(
        self,
        dataset_rows: "DataTable",
        path_list: list[str],
        glob_name: str,
        column="file",
    ) -> Iterator[Node]:
        """Finds all Nodes that correspond to GLOB like path pattern."""
        dr = dataset_rows
        de = dr.dir_expansion()
        q = de.query(
            dr.select().where(dr.c("is_latest") == true()).subquery()
        ).subquery()
        path_glob = "/".join([*path_list, glob_name])
        dirpath = path_glob[: -len(glob_name)]
        relpath = sa.func.substr(de.c(q, "path"), len(dirpath) + 1)

        return self.get_nodes(
            self.expand_query(de, q, dr)
            .where(
                (de.c(q, "path").op("GLOB")(path_glob))
                & ~self.instr(relpath, "/")
                & (de.c(q, "path") != dirpath)
            )
            .order_by(de.c(q, "source"), de.c(q, "path"), de.c(q, "version")),
            dr,
        )

    def _get_node_by_path_list(
        self, dataset_rows: "DataTable", path_list: list[str], name: str
    ) -> "Node":
        """
        Gets node that correspond some path list, e.g ["data-lakes", "dogs-and-cats"]
        """
        parent = "/".join(path_list)
        dr = dataset_rows
        de = dr.dir_expansion()
        q = de.query(
            dr.select().where(dr.c("is_latest") == true()).subquery(),
            column=dr.column,
        ).subquery()
        q = self.expand_query(de, q, dr)

        q = q.where(de.c(q, "path") == get_path(parent, name)).order_by(
            de.c(q, "source"), de.c(q, "path"), de.c(q, "version")
        )
        row = next(self.dataset_rows_select(q), None)
        if not row:
            path = f"{parent}/{name}"
            raise FileNotFoundError(f"Unable to resolve path {path!r}")
        return Node(*row)

    def _populate_nodes_by_path(
        self, dataset_rows: "DataTable", path_list: list[str]
    ) -> list[Node]:
        """
        Puts all nodes found by path_list into the res input variable.
        Note that path can have GLOB like pattern matching which means that
        res can have multiple nodes as result.
        If there is no GLOB pattern, res should have one node as result that
        match exact path by path_list
        """
        if not path_list:
            return [self._get_node_by_path_list(dataset_rows, [], "")]
        matched_paths: list[list[str]] = [[]]
        for curr_name in path_list[:-1]:
            if glob.has_magic(curr_name):
                new_paths: list[list[str]] = []
                for path in matched_paths:
                    nodes = self._get_nodes_by_glob_path_pattern(
                        dataset_rows, path, curr_name
                    )
                    new_paths.extend([*path, n.name] for n in nodes if n.is_container)
                matched_paths = new_paths
            else:
                for path in matched_paths:
                    path.append(curr_name)
        curr_name = path_list[-1]
        if glob.has_magic(curr_name):
            result: list[Node] = []
            for path in matched_paths:
                nodes = self._get_nodes_by_glob_path_pattern(
                    dataset_rows, path, curr_name
                )
                result.extend(nodes)
        else:
            result = [
                self._get_node_by_path_list(dataset_rows, path, curr_name)
                for path in matched_paths
            ]
        return result

    @staticmethod
    def expand_query(dir_expansion, dir_expanded_query, dataset_rows: "DataTable"):
        dr = dataset_rows
        de = dir_expansion
        q = dir_expanded_query

        def with_default(column):
            default = getattr(
                attrs.fields(Node), dr.without_object(column.name)
            ).default
            return sa.func.coalesce(column, default).label(column.name)

        return sa.select(
            q.c.sys__id,
            sa.case(
                (de.c(q, "is_dir") == true(), DirType.DIR), else_=DirType.FILE
            ).label(dr.col_name("dir_type")),
            de.c(q, "path"),
            with_default(dr.c("etag")),
            de.c(q, "version"),
            with_default(dr.c("is_latest")),
            dr.c("last_modified"),
            with_default(dr.c("size")),
            with_default(dr.c("rand", column="sys")),
            dr.c("location"),
            de.c(q, "source"),
        ).select_from(q.outerjoin(dr.table, q.c.sys__id == dr.c("id", column="sys")))

    def get_node_by_path(self, dataset_rows: "DataTable", path: str) -> Node:
        """Gets node that corresponds to some path"""
        if path == "":
            return Node.root()
        dr = dataset_rows
        if not path.endswith("/"):
            query = dr.select().where(
                dr.c("path") == path,
                dr.c("is_latest") == true(),
            )
            node = next(self.get_nodes(query, dr), None)
            if node:
                return node
            path += "/"
        query = sa.select(1).where(
            dr.select()
            .where(
                dr.c("is_latest") == true(),
                dr.c("path").startswith(path),
            )
            .exists()
        )
        row = next(self.db.execute(query), None)
        if not row:
            raise FileNotFoundError(f"Unable to resolve path {path}")
        path = path.removesuffix("/")
        return Node.from_dir(path)

    def expand_path(self, dataset_rows: "DataTable", path: str) -> list[Node]:
        """Simulates Unix-like shell expansion"""
        clean_path = path.strip("/")
        path_list = clean_path.split("/") if clean_path != "" else []
        res = self._populate_nodes_by_path(dataset_rows, path_list)
        if path.endswith("/"):
            res = [node for node in res if node.dir_type in DirTypeGroup.SUBOBJ_DIR]
        return res

    def select_node_fields_by_parent_path(
        self,
        dataset_rows: "DataTable",
        parent_path: str,
        fields: Iterable[str],
    ) -> Iterator[tuple[Any, ...]]:
        """
        Gets latest-version file nodes from the provided parent path
        """
        dr = dataset_rows
        de = dr.dir_expansion()
        q = de.query(
            dr.select().where(dr.c("is_latest") == true()).subquery()
        ).subquery()
        where_cond = pathfunc.parent(de.c(q, "path")) == parent_path
        if parent_path == "":
            # Exclude the root dir
            where_cond = where_cond & (de.c(q, "path") != "")
        inner_query = self.expand_query(de, q, dr).where(where_cond).subquery()

        def field_to_expr(f):
            if f == "name":
                return pathfunc.name(de.c(inner_query, "path"))
            return de.c(inner_query, f)

        return self.db.execute(
            sa.select(*(field_to_expr(f) for f in fields)).order_by(
                de.c(inner_query, "source"),
                de.c(inner_query, "path"),
                de.c(inner_query, "version"),
            )
        )

    def select_node_fields_by_parent_path_tar(
        self, dataset_rows: "DataTable", parent_path: str, fields: Iterable[str]
    ) -> Iterator[tuple[Any, ...]]:
        """
        Gets latest-version file nodes from the provided parent path
        """
        dr = dataset_rows
        dirpath = f"{parent_path}/"

        def field_to_expr(f):
            if f == "name":
                return pathfunc.name(dr.c("path"))
            return dr.c(f)

        q = (
            sa.select(*(field_to_expr(f) for f in fields))
            .where(
                dr.c("path").like(f"{sql_escape_like(dirpath)}%"),
                ~self.instr(pathfunc.name(dr.c("path")), "/"),
                dr.c("is_latest") == true(),
            )
            .order_by(dr.c("source"), dr.c("path"), dr.c("version"), dr.c("etag"))
        )
        return self.db.execute(q)

    def size(
        self,
        dataset_rows: "DataTable",
        node: Node | dict[str, Any],
        count_files: bool = False,
    ) -> tuple[int, int]:
        """
        Calculates size of some node (and subtree below node).
        Returns size in bytes as int and total files as int
        """
        if isinstance(node, dict):
            is_dir = node.get("is_dir", node["dir_type"] in DirTypeGroup.SUBOBJ_DIR)
            node_size = node["size"]
            path = node["path"]
        else:
            is_dir = node.is_container
            node_size = node.size
            path = node.path
        if not is_dir:
            # Return node size if this is not a directory
            return node_size, 1

        sub_glob = posixpath.join(path, "*")
        dr = dataset_rows
        selections: list[sa.ColumnElement] = [
            sa.func.sum(dr.c("size")),
        ]
        if count_files:
            selections.append(sa.func.count())
        results = next(
            self.db.execute(
                dr.select(*selections).where(
                    (dr.c("path").op("GLOB")(sub_glob)) & (dr.c("is_latest") == true())
                )
            ),
            (0, 0),
        )
        if count_files:
            return results[0] or 0, results[1] or 0
        return results[0] or 0, 0

    def _find_query(
        self,
        dataset_rows: "DataTable",
        parent_path: str,
        fields: Sequence[str] | None = None,
        type: str | None = None,
        conds=None,
        order_by: str | list[str] | None = None,
        include_subobjects: bool = True,
    ) -> sa.Select:
        if not conds:
            conds = []

        dr = dataset_rows
        de = dr.dir_expansion()
        q = de.query(
            dr.select().where(dr.c("is_latest") == true()).subquery()
        ).subquery()
        q = self.expand_query(de, q, dr).subquery()
        path = de.c(q, "path")

        if parent_path:
            sub_glob = posixpath.join(parent_path, "*")
            conds.append(path.op("GLOB")(sub_glob))
        else:
            conds.append(path != "")

        columns = q.c
        if fields:
            columns = [getattr(columns, f) for f in fields]

        query = sa.select(*columns)
        query = query.where(*conds)
        if type is not None:
            query = self.add_node_type_where(query, type, dr, include_subobjects)
        if order_by is not None:
            if isinstance(order_by, str):
                order_by = [order_by]
            query = query.order_by(*order_by)
        return query

    def get_subtree_files(
        self,
        dataset_rows: "DataTable",
        node: Node,
        sort: list[str] | str | None = None,
        include_subobjects: bool = True,
    ) -> Iterator[NodeWithPath]:
        """
        Returns all file nodes that are "below" some node.
        Nodes can be sorted as well.
        """
        dr = dataset_rows
        query = self._find_query(
            dr,
            node.path,
            type="f",
            include_subobjects=include_subobjects,
        )
        if sort is not None:
            if not isinstance(sort, list):
                sort = [sort]
            query = query.order_by(*(sa.text(dr.col_name(s)) for s in sort))  # type: ignore [attr-defined]

        prefix_len = len(node.path)

        def make_node_with_path(node: Node) -> NodeWithPath:
            return NodeWithPath(node, node.path[prefix_len:].lstrip("/").split("/"))

        return map(make_node_with_path, self.get_nodes(query, dr))

    def find(
        self,
        dataset_rows: "DataTable",
        node: Node,
        fields: Sequence[str],
        type=None,
        conds=None,
        order_by=None,
    ) -> Iterator[tuple[Any, ...]]:
        """
        Finds nodes that match certain criteria and only looks for latest nodes
        under the passed node.
        """
        dr = dataset_rows
        fields = [dr.col_name(f) for f in fields]
        query = self._find_query(
            dr,
            node.path,
            fields=fields,
            type=type,
            conds=conds,
            order_by=order_by,
        )
        return self.db.execute(query)

    def update_node(self, node_id: int, values: dict[str, Any]) -> None:
        # TODO used only in formats which will be deleted
        """Update entry of a specific node in the database."""

    def create_udf_table(
        self,
        columns: Sequence["sa.Column"] = (),
        name: str | None = None,
    ) -> sa.Table:
        """
        Create a temporary table for storing custom signals generated by a UDF.
        SQLite TEMPORARY tables cannot be directly used as they are process-specific,
        and UDFs are run in other processes when run in parallel.
        """
        columns = [
            c
            for c in columns
            if c.name not in [col.name for col in self.dataset_row_cls.sys_columns()]
        ]
        tbl = sa.Table(
            name or self.udf_table_name(),
            sa.MetaData(),
            *self.dataset_row_cls.sys_columns(),
            *columns,
        )
        self.db.create_table(tbl, if_not_exists=True, kind="udf")
        return tbl

    @abstractmethod
    def copy_table(
        self,
        table: sa.Table,
        query: sa.Select,
        progress_cb: Callable[[int], None] | None = None,
    ) -> None:
        """
        Copy the results of a query into a table.
        """

    @abstractmethod
    def join(
        self,
        left: "_FromClauseArgument",
        right: "_FromClauseArgument",
        onclause: "_OnClauseArgument",
        inner: bool = True,
        full: bool = False,
        columns=None,
    ) -> sa.Select:
        """
        Join two tables together.
        """

    @abstractmethod
    def create_pre_udf_table(self, query: sa.Select) -> sa.Table:
        """
        Create a temporary table from a query for use in a UDF.
        """

    def is_temp_table_name(self, name: str) -> bool:
        """Returns if the given table name refers to a temporary
        or no longer needed table."""
        return name.startswith((self.TMP_TABLE_NAME_PREFIX, self.UDF_TABLE_NAME_PREFIX))

    def get_temp_table_names(self) -> list[str]:
        return [
            t
            for t in sa.inspect(self.db.engine).get_table_names()
            if self.is_temp_table_name(t)
        ]

    def cleanup_tables(self, names: Iterable[str]) -> None:
        """
        Drop tables passed.

        This should be implemented to ensure that the provided tables
        are cleaned up as soon as they are no longer needed.
        """
        to_drop = set(names)
        for name in to_drop:
            self.db.drop_table(sa.Table(name, self.db.metadata), if_exists=True)


def _random_string(length: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))
