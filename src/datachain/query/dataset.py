import contextlib
import hashlib
import inspect
import logging
import os
import secrets
import string
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from copy import copy
from functools import wraps
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import attrs
import sqlalchemy
import sqlalchemy as sa
from attrs import frozen
from fsspec.callbacks import DEFAULT_CALLBACK, Callback, TqdmCallback
from sqlalchemy import Column
from sqlalchemy.sql import func as f
from sqlalchemy.sql.elements import ColumnClause, ColumnElement, Label
from sqlalchemy.sql.expression import label
from sqlalchemy.sql.schema import TableClause
from sqlalchemy.sql.selectable import Select
from tqdm.auto import tqdm

from datachain.asyn import ASYNC_WORKERS, AsyncMapper, OrderedMapper
from datachain.catalog.catalog import clone_catalog_with_cache
from datachain.data_storage.schema import (
    PARTITION_COLUMN_ID,
    partition_col_names,
    partition_columns,
)
from datachain.dataset import DatasetDependency, DatasetStatus, RowDict
from datachain.error import DatasetNotFoundError, QueryScriptCancelError
from datachain.func.base import Function
from datachain.hash_utils import hash_column_elements
from datachain.lib.listing import is_listing_dataset, listing_dataset_expired
from datachain.lib.signal_schema import SignalSchema, generate_merge_root_mapping
from datachain.lib.udf import UDFAdapter, _get_cache
from datachain.progress import CombinedDownloadCallback, TqdmCombinedDownloadCallback
from datachain.project import Project
from datachain.query.schema import DEFAULT_DELIMITER, C, UDFParamSpec, normalize_param
from datachain.query.session import Session
from datachain.query.udf import UdfInfo
from datachain.sql.functions.random import rand
from datachain.sql.types import SQLType
from datachain.utils import (
    determine_processes,
    determine_workers,
    ensure_sequence,
    filtered_cloudpickle_dumps,
    get_datachain_executable,
    safe_closing,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Concatenate

    from sqlalchemy.sql.elements import ClauseElement, KeyedColumnElement
    from sqlalchemy.sql.schema import Table
    from sqlalchemy.sql.selectable import GenerativeSelect
    from typing_extensions import ParamSpec, Self

    from datachain.catalog import Catalog
    from datachain.data_storage import AbstractWarehouse
    from datachain.dataset import DatasetRecord
    from datachain.lib.udf import UDFAdapter, UDFResult

    P = ParamSpec("P")


INSERT_BATCH_SIZE = 10000

PartitionByType = (
    str | Function | ColumnElement | Sequence[str | Function | ColumnElement]
)
JoinPredicateType = str | ColumnClause | ColumnElement
DatasetDependencyType = tuple["DatasetRecord", str]

logger = logging.getLogger("datachain")


T = TypeVar("T", bound="DatasetQuery")


def detach(
    method: "Callable[Concatenate[T, P], T]",
) -> "Callable[Concatenate[T, P], T]":
    """
    Decorator that needs to be put on a method that modifies existing DatasetQuery
    which was 100% representing one particular dataset and had name and version of
    that dataset set, and which returns new instance of it.
    This kind of DatasetQuery, which represent one whole dataset, we return from
    .save() method.
    Example of modifying method is .filter() as that one filters out part
    of a dataset which means DatasetQuery no longer 100% represents it (in this case
    it can represents only a part of it)
    """

    @wraps(method)
    def _inner(self: T, *args: "P.args", **kwargs: "P.kwargs") -> T:
        cloned = method(self, *args, **kwargs)
        cloned.name = None
        cloned.version = None
        return cloned

    return _inner


class QueryGeneratorFunc(Protocol):
    def __call__(self, *columns: ColumnElement) -> Select: ...


@frozen
class QueryGenerator:
    func: QueryGeneratorFunc
    columns: tuple[ColumnElement, ...]

    def only(self, column_names: Sequence[str]) -> Select:
        return self.func(*(c for c in self.columns if c.name in column_names))

    def exclude(self, column_names: Sequence[str]) -> Select:
        return self.func(*(c for c in self.columns if c.name not in column_names))

    def select(self, column_names=None) -> Select:
        if column_names is None:
            return self.func(*self.columns)
        return self.func(*(c for c in self.columns if c.name in column_names))


@frozen
class StepResult:
    query_generator: QueryGenerator
    dependencies: tuple[DatasetDependencyType, ...]


def step_result(
    func: QueryGeneratorFunc,
    columns: Iterable[ColumnElement],
    dependencies: Iterable[DatasetDependencyType] = (),
) -> "StepResult":
    return StepResult(
        query_generator=QueryGenerator(func=func, columns=tuple(columns)),
        dependencies=tuple(dependencies),
    )


@frozen
class Step(ABC):
    """A query processing step (filtering, mutation, etc.)"""

    @abstractmethod
    def apply(
        self, query_generator: "QueryGenerator", temp_tables: list[str]
    ) -> "StepResult":
        """Apply the processing step."""

    @abstractmethod
    def hash_inputs(self) -> str:
        """Calculates hash of step inputs"""

    def hash(self) -> str:
        """
        Calculates hash for step which includes step name and hash of it's inputs
        """
        return hashlib.sha256(
            f"{self.__class__.__name__}|{self.hash_inputs()}".encode()
        ).hexdigest()


@frozen
class QueryStep:
    """A query that returns all rows from specific dataset version"""

    catalog: "Catalog"
    dataset: "DatasetRecord"
    dataset_version: str

    def apply(self) -> "StepResult":
        def q(*columns):
            return sqlalchemy.select(*columns)

        dr = self.catalog.warehouse.dataset_rows(self.dataset, self.dataset_version)

        return step_result(
            q, dr.columns, dependencies=[(self.dataset, self.dataset_version)]
        )

    def hash(self) -> str:
        return hashlib.sha256(
            self.dataset.uri(self.dataset_version).encode()
        ).hexdigest()


def generator_then_call(generator, func: Callable):
    """
    Yield items from generator then execute a function and yield
    its result.
    """
    yield from generator
    yield func() or []


@frozen
class DatasetDiffOperation(Step):
    """
    Abstract class for operations that are calculation some kind of diff between
    datasets queries like subtract etc.
    """

    dq: "DatasetQuery"
    catalog: "Catalog"

    def clone(self) -> "Self":
        return self.__class__(self.dq, self.catalog)

    @abstractmethod
    def query(
        self,
        source_query: Select,
        target_query: Select,
    ) -> sa.Selectable:
        """
        Should return select query that calculates desired diff between dataset queries
        """

    def apply(self, query_generator, temp_tables: list[str]) -> "StepResult":
        source_query = query_generator.exclude(("sys__id",))
        right_before = len(self.dq.temp_table_names)
        target_query = self.dq.apply_steps().select()
        temp_tables.extend(self.dq.temp_table_names[right_before:])

        # creating temp table that will hold subtract results
        temp_table_name = self.catalog.warehouse.temp_table_name()
        temp_tables.append(temp_table_name)

        columns = [
            c if isinstance(c, Column) else Column(c.name, c.type)
            for c in source_query.selected_columns
        ]
        temp_table = self.catalog.warehouse.create_dataset_rows_table(
            temp_table_name,
            columns=columns,
            if_not_exists=False,
        )

        diff_q = self.query(source_query, target_query)

        insert_q = temp_table.insert().from_select(
            source_query.selected_columns, diff_q
        )

        self.catalog.warehouse.db.execute(insert_q)

        def q(*columns):
            return sqlalchemy.select(*columns)

        return step_result(q, temp_table.c)


@frozen
class Subtract(DatasetDiffOperation):
    on: Sequence[tuple[str, str]]

    def hash_inputs(self) -> str:
        on_bytes = b"".join(
            f"{a}:{b}".encode() for a, b in sorted(self.on, key=lambda t: (t[0], t[1]))
        )

        return hashlib.sha256(bytes.fromhex(self.dq.hash()) + on_bytes).hexdigest()

    def query(self, source_query: Select, target_query: Select) -> sa.Selectable:
        sq = source_query.alias("source_query")
        tq = target_query.alias("target_query")
        where_clause = sa.and_(
            *[
                getattr(
                    sq.c, col_name[0] if isinstance(col_name, tuple) else col_name
                ).is_not_distinct_from(
                    getattr(
                        tq.c, col_name[1] if isinstance(col_name, tuple) else col_name
                    )
                )
                for col_name in self.on
            ]
        )
        return sq.select().except_(sq.select().where(where_clause))


def adjust_outputs(
    warehouse: "AbstractWarehouse",
    row: dict[str, Any],
    col_types: list[tuple[str, SQLType, type, str, Any]],
) -> dict[str, Any]:
    """
    This function does a couple of things to prepare a row for inserting into the db:
    1. Fill default values for columns that have None and add missing columns
    2. Validate values with its corresponding DB column types and convert types
       if needed and possible
    """
    # Optimization: Use precomputed column type values as these do not change for each
    # row in the same UDF.
    for (
        col_name,
        col_type,
        col_python_type,
        col_type_name,
        default_value,
    ) in col_types:
        row_val = row.get(col_name)

        # Fill None or missing values with defaults (get returns None if not in the row)
        if row_val is None:
            row[col_name] = default_value
            continue

        # Validate and convert type if needed and possible
        row[col_name] = warehouse.convert_type(
            row_val, col_type, col_python_type, col_type_name, col_name
        )
    return row


def get_col_types(
    warehouse: "AbstractWarehouse", output: "Mapping[str, Any]"
) -> list[tuple]:
    """Optimization: Precompute column types so these don't have to be computed
    in the convert_type function for each row in a loop."""
    dialect = warehouse.db.dialect
    return [
        (
            col_name,
            # Check if type is already instantiated or not
            col_type_inst := col_type() if inspect.isclass(col_type) else col_type,
            warehouse.python_type(col_type_inst),
            type(col_type_inst).__name__,
            col_type.default_value(dialect),
        )
        for col_name, col_type in output.items()
    ]


def process_udf_outputs(
    warehouse: "AbstractWarehouse",
    udf_table: "Table",
    udf_results: Iterator[Iterable["UDFResult"]],
    udf: "UDFAdapter",
    cb: Callback = DEFAULT_CALLBACK,
    batch_size: int = INSERT_BATCH_SIZE,
) -> None:
    # Optimization: Compute row types once, rather than for every row.
    udf_col_types = get_col_types(warehouse, udf.output)

    def _insert_rows():
        for udf_output in udf_results:
            if not udf_output:
                continue

            with safe_closing(udf_output):
                for row in udf_output:
                    cb.relative_update()
                    yield adjust_outputs(warehouse, row, udf_col_types)

    warehouse.insert_rows(udf_table, _insert_rows(), batch_size=batch_size)
    warehouse.insert_rows_done(udf_table)


def get_download_callback(suffix: str = "", **kwargs) -> CombinedDownloadCallback:
    return TqdmCombinedDownloadCallback(
        tqdm_kwargs={
            "desc": "Download" + suffix,
            "unit": "B",
            "unit_scale": True,
            "unit_divisor": 1024,
            "leave": False,
            **kwargs,
        },
        tqdm_cls=tqdm,
    )


def get_processed_callback() -> Callback:
    return TqdmCallback(
        {"desc": "Processed", "unit": " rows", "leave": False}, tqdm_cls=tqdm
    )


def get_generated_callback(is_generator: bool = False) -> Callback:
    if is_generator:
        return TqdmCallback(
            {"desc": "Generated", "unit": " rows", "leave": False}, tqdm_cls=tqdm
        )
    return DEFAULT_CALLBACK


@frozen
class UDFStep(Step, ABC):
    udf: "UDFAdapter"
    catalog: "Catalog"
    partition_by: PartitionByType | None = None
    is_generator = False
    # Parameters from Settings
    cache: bool = False
    parallel: int | None = None
    workers: bool | int = False
    min_task_size: int | None = None
    batch_size: int | None = None

    def hash_inputs(self) -> str:
        partition_by = ensure_sequence(self.partition_by or [])
        parts = [
            bytes.fromhex(self.udf.hash()),
            bytes.fromhex(hash_column_elements(partition_by)),
            str(self.is_generator).encode(),
        ]

        return hashlib.sha256(b"".join(parts)).hexdigest()

    @abstractmethod
    def create_udf_table(self, query: Select) -> "Table":
        """Method that creates a table where temp udf results will be saved"""

    def process_input_query(self, query: Select) -> tuple[Select, list["Table"]]:
        """Materialize inputs, ensure sys columns are available, needed for checkpoints,
        needed for map to work (merge results)"""
        table = self.catalog.warehouse.create_pre_udf_table(query)
        return sqlalchemy.select(*table.c), [table]

    @abstractmethod
    def create_result_query(
        self, udf_table: "Table", query: Select
    ) -> tuple[QueryGeneratorFunc, list["sqlalchemy.Column"]]:
        """
        Method that should return query to fetch results from udf and columns
        to select
        """

    def populate_udf_table(self, udf_table: "Table", query: Select) -> None:
        if (rows_total := self.catalog.warehouse.query_count(query)) == 0:
            return

        from datachain.catalog import QUERY_SCRIPT_CANCELED_EXIT_CODE
        from datachain.catalog.loader import (
            DISTRIBUTED_IMPORT_PATH,
            get_udf_distributor_class,
        )

        workers = determine_workers(self.workers, rows_total=rows_total)
        processes = determine_processes(self.parallel, rows_total=rows_total)

        use_partitioning = self.partition_by is not None
        batching = self.udf.get_batching(use_partitioning)
        udf_fields = [str(c.name) for c in query.selected_columns]
        udf_distributor_class = get_udf_distributor_class()

        prefetch = self.udf.prefetch
        with _get_cache(self.catalog.cache, prefetch, use_cache=self.cache) as _cache:
            catalog = clone_catalog_with_cache(self.catalog, _cache)

            try:
                if udf_distributor_class and not catalog.in_memory:
                    # Use the UDF distributor if available (running in SaaS)
                    udf_distributor = udf_distributor_class(
                        catalog=catalog,
                        table=udf_table,
                        query=query,
                        udf_data=filtered_cloudpickle_dumps(self.udf),
                        batching=batching,
                        workers=workers,
                        processes=processes,
                        udf_fields=udf_fields,
                        rows_total=rows_total,
                        use_cache=self.cache,
                        is_generator=self.is_generator,
                        min_task_size=self.min_task_size,
                        batch_size=self.batch_size,
                    )
                    udf_distributor()
                    return

                if workers:
                    if catalog.in_memory:
                        raise RuntimeError(
                            "In-memory databases cannot be used with "
                            "distributed processing."
                        )

                    raise RuntimeError(
                        f"{DISTRIBUTED_IMPORT_PATH} import path is required "
                        "for distributed UDF processing."
                    )
                if processes:
                    # Parallel processing (faster for more CPU-heavy UDFs)
                    if catalog.in_memory:
                        raise RuntimeError(
                            "In-memory databases cannot be used "
                            "with parallel processing."
                        )

                    udf_info = UdfInfo(
                        udf_data=filtered_cloudpickle_dumps(self.udf),
                        catalog_init=catalog.get_init_params(),
                        metastore_clone_params=catalog.metastore.clone_params(),
                        warehouse_clone_params=catalog.warehouse.clone_params(),
                        table=udf_table,
                        query=query,
                        udf_fields=udf_fields,
                        batching=batching,
                        processes=processes,
                        is_generator=self.is_generator,
                        cache=self.cache,
                        rows_total=rows_total,
                        batch_size=self.batch_size or INSERT_BATCH_SIZE,
                    )

                    # Run the UDFDispatcher in another process to avoid needing
                    # if __name__ == '__main__': in user scripts
                    exec_cmd = get_datachain_executable()
                    cmd = [*exec_cmd, "internal-run-udf"]
                    envs = dict(os.environ)
                    envs.update({"PYTHONPATH": os.getcwd()})
                    process_data = filtered_cloudpickle_dumps(udf_info)

                    with subprocess.Popen(  # noqa: S603
                        cmd, env=envs, stdin=subprocess.PIPE
                    ) as process:
                        try:
                            process.communicate(process_data)
                        except KeyboardInterrupt:
                            raise QueryScriptCancelError(
                                "UDF execution was canceled by the user."
                            ) from None
                        if retval := process.poll():
                            raise RuntimeError(
                                f"UDF Execution Failed! Exit code: {retval}"
                            )
                else:
                    # Otherwise process single-threaded (faster for smaller UDFs)
                    warehouse = catalog.warehouse

                    udf_inputs = batching(warehouse.dataset_select_paginated, query)
                    download_cb = get_download_callback()
                    processed_cb = get_processed_callback()
                    generated_cb = get_generated_callback(self.is_generator)

                    try:
                        udf_results = self.udf.run(
                            udf_fields,
                            udf_inputs,
                            catalog,
                            self.cache,
                            download_cb,
                            processed_cb,
                        )
                        with safe_closing(udf_results):
                            process_udf_outputs(
                                warehouse,
                                udf_table,
                                udf_results,
                                self.udf,
                                cb=generated_cb,
                                batch_size=self.batch_size or INSERT_BATCH_SIZE,
                            )
                    finally:
                        download_cb.close()
                        processed_cb.close()
                        generated_cb.close()

            except QueryScriptCancelError:
                self.catalog.warehouse.close()
                sys.exit(QUERY_SCRIPT_CANCELED_EXIT_CODE)
            except (Exception, KeyboardInterrupt):
                # Close any open database connections if an error is encountered
                self.catalog.warehouse.close()
                raise

    def create_partitions_table(self, query: Select) -> "Table":
        """
        Create temporary table with group by partitions.
        """
        if self.partition_by is None:
            raise RuntimeError("Query must have partition_by set to use partitioning")
        if (id_col := query.selected_columns.get("sys__id")) is None:
            raise RuntimeError("Query must have sys__id column to use partitioning")

        if isinstance(self.partition_by, (list, tuple, GeneratorType)):
            list_partition_by = list(self.partition_by)
        else:
            list_partition_by = [self.partition_by]

        partition_by = [
            p.get_column() if isinstance(p, Function) else p for p in list_partition_by
        ]

        # create table with partitions
        tbl = self.catalog.warehouse.create_udf_table(partition_columns())

        # fill table with partitions
        cols = [
            id_col,
            f.dense_rank().over(order_by=partition_by).label(PARTITION_COLUMN_ID),
        ]
        self.catalog.warehouse.db.execute(
            tbl.insert().from_select(
                cols,
                query.offset(None).limit(None).with_only_columns(*cols),
            )
        )

        return tbl

    def clone(self, partition_by: PartitionByType | None = None) -> "Self":
        if partition_by is not None:
            return self.__class__(
                self.udf,
                self.catalog,
                partition_by=partition_by,
                parallel=self.parallel,
                workers=self.workers,
                min_task_size=self.min_task_size,
                batch_size=self.batch_size,
            )
        return self.__class__(self.udf, self.catalog)

    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> "StepResult":
        query, tables = self.process_input_query(query_generator.select())
        _query = query

        # Apply partitioning if needed.
        if self.partition_by is not None:
            partition_tbl = self.create_partitions_table(query)
            query = query.outerjoin(
                partition_tbl,
                partition_tbl.c.sys__id == query.selected_columns.sys__id,
            ).add_columns(*partition_columns())
            tables = [*tables, partition_tbl]

        temp_tables.extend(t.name for t in tables)
        udf_table = self.create_udf_table(_query)
        temp_tables.append(udf_table.name)
        self.populate_udf_table(udf_table, query)
        q, cols = self.create_result_query(udf_table, query)

        return step_result(q, cols)


@frozen
class UDFSignal(UDFStep):
    udf: "UDFAdapter"
    catalog: "Catalog"
    partition_by: PartitionByType | None = None
    is_generator = False
    # Parameters from Settings
    cache: bool = False
    parallel: int | None = None
    workers: bool | int = False
    min_task_size: int | None = None
    batch_size: int | None = None

    def create_udf_table(self, query: Select) -> "Table":
        udf_output_columns: list[sqlalchemy.Column[Any]] = [
            sqlalchemy.Column(col_name, col_type)
            for (col_name, col_type) in self.udf.output.items()
        ]

        return self.catalog.warehouse.create_udf_table(udf_output_columns)

    def create_result_query(
        self, udf_table, query
    ) -> tuple[QueryGeneratorFunc, list["sqlalchemy.Column"]]:
        subq = query.subquery()
        original_cols = [c for c in subq.c if c.name not in partition_col_names]

        # new signal columns that are added to udf_table
        signal_cols = [c for c in udf_table.c if not c.name.startswith("sys__")]
        signal_name_cols = {c.name: c for c in signal_cols}
        cols = signal_cols

        original_names = {c.name for c in original_cols}
        new_names = {c.name for c in cols}

        overlap = original_names & new_names
        if overlap:
            raise ValueError(
                "Column already exists or added in the previous steps: "
                + ", ".join(sorted(overlap))
            )

        def _root(name: str) -> str:
            return name.split(DEFAULT_DELIMITER, 1)[0]

        existing_roots = {_root(name) for name in original_names}
        new_roots = {_root(name) for name in new_names}
        root_conflicts = existing_roots & new_roots
        if root_conflicts:
            raise ValueError(
                "Signals already exist in the previous steps: "
                + ", ".join(sorted(root_conflicts))
            )

        def q(*columns):
            cols1 = []
            cols2 = []
            for c in columns:
                if c.name in partition_col_names:
                    continue
                cols.append(signal_name_cols.get(c.name, c))
                if c.name in signal_name_cols:
                    cols2.append(c)
                else:
                    cols1.append(c)

            if cols2:
                res = (
                    sqlalchemy.select(*cols1)
                    .select_from(subq)
                    .outerjoin(udf_table, udf_table.c.sys__id == subq.c.sys__id)
                    .add_columns(*cols2)
                )
            else:
                res = sqlalchemy.select(*cols1).select_from(subq)

            if self.partition_by is not None:
                subquery = res.subquery()
                res = sqlalchemy.select(*subquery.c).select_from(subquery)

            return res

        return q, [*original_cols, *cols]


@frozen
class RowGenerator(UDFStep):
    """Extend dataset with new rows."""

    udf: "UDFAdapter"
    catalog: "Catalog"
    partition_by: PartitionByType | None = None
    is_generator = True
    # Parameters from Settings
    cache: bool = False
    parallel: int | None = None
    workers: bool | int = False
    min_task_size: int | None = None
    batch_size: int | None = None

    def create_udf_table(self, query: Select) -> "Table":
        warehouse = self.catalog.warehouse

        table_name = self.catalog.warehouse.udf_table_name()
        columns: tuple[Column, ...] = tuple(
            Column(name, typ) for name, typ in self.udf.output.items()
        )
        return warehouse.create_dataset_rows_table(
            table_name,
            columns=columns,
            if_not_exists=False,
        )

    def create_result_query(
        self, udf_table, query: Select
    ) -> tuple[QueryGeneratorFunc, list["sqlalchemy.Column"]]:
        udf_table_query = udf_table.select().subquery()
        udf_table_cols: list[sqlalchemy.Label[Any]] = [
            label(c.name, c) for c in udf_table_query.columns
        ]

        def q(*columns):
            names = {c.name for c in columns}
            # Columns for the generated table.
            cols = [c for c in udf_table_cols if c.name in names]
            return sqlalchemy.select(*cols).select_from(udf_table_query)

        return q, udf_table_query.columns


@frozen
class SQLClause(Step, ABC):
    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> StepResult:
        query = query_generator.select()
        new_query = self.apply_sql_clause(query)

        def q(*columns):
            return new_query.with_only_columns(*columns)

        return step_result(q, new_query.selected_columns)

    def parse_cols(
        self,
        cols: Sequence[Function | ColumnElement],
    ) -> tuple[ColumnElement, ...]:
        return tuple(c.get_column() if isinstance(c, Function) else c for c in cols)

    @abstractmethod
    def apply_sql_clause(self, query: Any) -> Any:
        pass


@frozen
class RegenerateSystemColumns(Step):
    catalog: "Catalog"

    def hash_inputs(self) -> str:
        return hashlib.sha256(b"regenerate_system_columns").hexdigest()

    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> StepResult:
        query = query_generator.select()
        new_query = self.catalog.warehouse._regenerate_system_columns(
            query, keep_existing_columns=True
        )

        def q(*columns):
            return new_query.with_only_columns(*columns)

        return step_result(q, new_query.selected_columns)


@frozen
class SQLSelect(SQLClause):
    args: tuple[Function | ColumnElement, ...]

    def hash_inputs(self) -> str:
        return hash_column_elements(self.args)

    def apply_sql_clause(self, query) -> Select:
        subquery = query.subquery()
        args = [
            subquery.c[str(c)] if isinstance(c, (str, C)) else c
            for c in self.parse_cols(self.args)
        ]
        if not args:
            args = subquery.c

        return sqlalchemy.select(*args).select_from(subquery)


@frozen
class SQLSelectExcept(SQLClause):
    args: tuple[Function | ColumnElement, ...]

    def hash_inputs(self) -> str:
        return hash_column_elements(self.args)

    def apply_sql_clause(self, query: Select) -> Select:
        subquery = query.subquery()
        args = [c for c in subquery.c if c.name not in set(self.parse_cols(self.args))]
        return sqlalchemy.select(*args).select_from(subquery)


@frozen
class SQLMutate(SQLClause):
    args: tuple[Label, ...]
    new_schema: SignalSchema

    def hash_inputs(self) -> str:
        return hash_column_elements(self.args)

    def apply_sql_clause(self, query: Select) -> Select:
        original_subquery = query.subquery()
        to_mutate = {c.name for c in self.args}

        # Drop the original versions to avoid name collisions, exclude renamed
        # columns. Always keep system columns (sys__*) if they exist in original query
        new_schema_columns = set(self.new_schema.db_signals())
        base_cols = [
            c
            for c in original_subquery.c
            if c.name not in to_mutate
            and (c.name in new_schema_columns or c.name.startswith("sys__"))
        ]

        # Create intermediate subquery to properly handle window functions
        intermediate_query = sqlalchemy.select(*base_cols, *self.args).select_from(
            original_subquery
        )
        intermediate_subquery = intermediate_query.subquery()

        return sqlalchemy.select(*intermediate_subquery.c).select_from(
            intermediate_subquery
        )


@frozen
class SQLFilter(SQLClause):
    expressions: tuple[Function | ColumnElement, ...]

    def hash_inputs(self) -> str:
        return hash_column_elements(self.expressions)

    def __and__(self, other):
        expressions = self.parse_cols(self.expressions)
        return self.__class__(expressions + other)

    def apply_sql_clause(self, query: Select) -> Select:
        expressions = self.parse_cols(self.expressions)
        return query.filter(*expressions)


@frozen
class SQLOrderBy(SQLClause):
    args: tuple[Function | ColumnElement, ...]

    def hash_inputs(self) -> str:
        return hash_column_elements(self.args)

    def apply_sql_clause(self, query: Select) -> Select:
        args = self.parse_cols(self.args)
        return query.order_by(*args)


@frozen
class SQLLimit(SQLClause):
    n: int

    def hash_inputs(self) -> str:
        return hashlib.sha256(str(self.n).encode()).hexdigest()

    def apply_sql_clause(self, query: Select) -> Select:
        return query.limit(self.n)


@frozen
class SQLOffset(SQLClause):
    offset: int

    def hash_inputs(self) -> str:
        return hashlib.sha256(str(self.offset).encode()).hexdigest()

    def apply_sql_clause(self, query: "GenerativeSelect"):
        return query.offset(self.offset)


@frozen
class SQLCount(SQLClause):
    def hash_inputs(self) -> str:
        return ""

    def apply_sql_clause(self, query):
        return sqlalchemy.select(f.count(1)).select_from(query.subquery())


@frozen
class SQLDistinct(SQLClause):
    args: tuple[ColumnElement, ...]
    dialect: str

    def hash_inputs(self) -> str:
        return hash_column_elements(self.args)

    def apply_sql_clause(self, query):
        if self.dialect == "sqlite":
            return query.group_by(*self.args)

        return query.distinct(*self.args)


@frozen
class SQLUnion(Step):
    query1: "DatasetQuery"
    query2: "DatasetQuery"

    def hash_inputs(self) -> str:
        return hashlib.sha256(
            bytes.fromhex(self.query1.hash()) + bytes.fromhex(self.query2.hash())
        ).hexdigest()

    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> StepResult:
        left_before = len(self.query1.temp_table_names)
        q1 = self.query1.apply_steps().select().subquery()
        temp_tables.extend(self.query1.temp_table_names[left_before:])
        right_before = len(self.query2.temp_table_names)
        q2 = self.query2.apply_steps().select().subquery()
        temp_tables.extend(self.query2.temp_table_names[right_before:])

        columns1 = _drop_system_columns(q1.columns)
        columns2 = _drop_system_columns(q2.columns)
        columns1, columns2 = _order_columns(columns1, columns2)

        def q(*columns):
            selected_names = [c.name for c in columns]
            col1 = [c for c in columns1 if c.name in selected_names]
            col2 = [c for c in columns2 if c.name in selected_names]
            union_query = sqlalchemy.select(*col1).union_all(sqlalchemy.select(*col2))

            union_cte = union_query.cte()
            select_cols = [union_cte.c[name] for name in selected_names]
            return sqlalchemy.select(*select_cols)

        return step_result(
            q,
            columns1,
            dependencies=self.query1.dependencies | self.query2.dependencies,
        )


@frozen
class SQLJoin(Step):
    catalog: "Catalog"
    query1: "DatasetQuery"
    query2: "DatasetQuery"
    predicates: JoinPredicateType | tuple[JoinPredicateType, ...]
    inner: bool
    full: bool
    rname: str

    @staticmethod
    def _split_db_name(name: str) -> tuple[str, str]:
        if DEFAULT_DELIMITER in name:
            head, tail = name.split(DEFAULT_DELIMITER, 1)
            return head, tail
        return name, ""

    @classmethod
    def _root_name(cls, name: str) -> str:
        return cls._split_db_name(name)[0]

    def hash_inputs(self) -> str:
        predicates = (
            ensure_sequence(self.predicates) if self.predicates is not None else []
        )

        parts = [
            bytes.fromhex(self.query1.hash()),
            bytes.fromhex(self.query2.hash()),
            bytes.fromhex(hash_column_elements(predicates)),
            str(self.inner).encode(),
            str(self.full).encode(),
            self.rname.encode("utf-8"),
        ]

        return hashlib.sha256(b"".join(parts)).hexdigest()

    def get_query(self, dq: "DatasetQuery", temp_tables: list[str]) -> sa.Subquery:
        temp_tables_before = len(dq.temp_table_names)
        query = dq.apply_steps().select()
        temp_tables.extend(dq.temp_table_names[temp_tables_before:])

        if not any(isinstance(step, (SQLJoin, SQLUnion)) for step in dq.steps):
            return query.subquery(dq.table.name)

        warehouse = self.catalog.warehouse

        columns = [
            c if isinstance(c, Column) else Column(c.name, c.type)
            for c in query.subquery().columns
        ]
        temp_table = warehouse.create_dataset_rows_table(
            warehouse.temp_table_name(),
            columns=columns,
        )
        temp_tables.append(temp_table.name)

        warehouse.copy_table(temp_table, query)

        return temp_table.select().subquery(dq.table.name)

    def validate_expression(self, exp: "ClauseElement", q1, q2):
        """
        Checking if columns used in expression actually exist in left / right
        part of the join.
        """
        for c in exp.get_children():
            if isinstance(c, ColumnClause):
                assert isinstance(c.table, TableClause)

                q1_c = q1.c.get(c.name)
                q2_c = q2.c.get(c.name)

                if c.table.name == q1.name and q1_c is None:
                    raise ValueError(
                        f"Column {c.name} was not found in left part of the join"
                    )

                if c.table.name == q2.name and q2_c is None:
                    raise ValueError(
                        f"Column {c.name} was not found in right part of the join"
                    )
                if c.table.name not in [q1.name, q2.name]:
                    raise ValueError(
                        f"Column {c.name} was not found in left or right"
                        " part of the join"
                    )
                continue
            self.validate_expression(c, q1, q2)

    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> StepResult:
        q1 = self.get_query(self.query1, temp_tables)
        q2 = self.get_query(self.query2, temp_tables)

        q1_columns = _drop_system_columns(q1.c)
        existing_column_names = {c.name for c in q1_columns}
        right_columns: list[KeyedColumnElement[Any]] = []
        right_column_names: list[str] = []
        for column in q2.c:
            if column.name.startswith("sys__"):
                continue
            right_columns.append(column)
            right_column_names.append(column.name)

        root_mapping = generate_merge_root_mapping(
            existing_column_names,
            right_column_names,
            extract_root=self._root_name,
            prefix=self.rname,
        )

        q2_columns: list[KeyedColumnElement[Any]] = []
        for column in right_columns:
            original_name = column.name
            column_root, column_tail = self._split_db_name(original_name)
            mapped_root = root_mapping[column_root]

            new_name = (
                mapped_root
                if not column_tail
                else DEFAULT_DELIMITER.join([mapped_root, column_tail])
            )

            if new_name != original_name:
                column = column.label(new_name)

            q2_columns.append(column)

        res_columns = q1_columns + q2_columns
        predicates = (
            (self.predicates,)
            if not isinstance(self.predicates, tuple)
            else self.predicates
        )

        expressions = []
        for p in predicates:
            if isinstance(p, ColumnClause):
                expressions.append(self.query1.c(p.name) == self.query2.c(p.name))
            elif isinstance(p, str):
                expressions.append(self.query1.c(p) == self.query2.c(p))
            elif isinstance(p, ColumnElement):
                expressions.append(p)
            else:
                raise TypeError(f"Unsupported predicate {p} for join expression")

        if not expressions:
            raise ValueError("Missing predicates")

        join_expression = sqlalchemy.and_(*expressions)
        self.validate_expression(join_expression, q1, q2)

        def q(*columns):
            return self.catalog.warehouse.join(
                q1,
                q2,
                join_expression,
                inner=self.inner,
                full=self.full,
                columns=columns,
            )

        return step_result(
            q,
            res_columns,
            dependencies=self.query1.dependencies | self.query2.dependencies,
        )


@frozen
class SQLGroupBy(SQLClause):
    cols: Sequence[str | Function | ColumnElement]
    group_by: Sequence[str | Function | ColumnElement]

    def hash_inputs(self) -> str:
        return hashlib.sha256(
            bytes.fromhex(
                hash_column_elements(self.cols) + hash_column_elements(self.group_by)
            )
        ).hexdigest()

    def apply_sql_clause(self, query) -> Select:
        if not self.cols:
            raise ValueError("No columns to select")

        subquery = query.subquery()

        group_by = [
            c.get_column() if isinstance(c, Function) else c for c in self.group_by
        ]

        cols_dict: dict[str, Any] = {}
        for c in (*group_by, *self.cols):
            if isinstance(c, Function):
                key = c.name
                value = c.get_column()
            elif isinstance(c, (str, C)):
                key = str(c)
                value = subquery.c[str(c)]
            else:
                key = c.name
                value = c  # type: ignore[assignment]
            cols_dict[key] = value

        unique_cols = cols_dict.values()

        return sqlalchemy.select(*unique_cols).select_from(subquery).group_by(*group_by)


class UnionSchemaMismatchError(ValueError):
    """Union input columns mismatch."""

    @classmethod
    def from_column_sets(
        cls,
        missing_left: set[str],
        missing_right: set[str],
    ) -> "UnionSchemaMismatchError":
        def _describe(cols: set[str], side: str) -> str:
            return f"{', '.join(sorted(cols))} only present in {side}"

        parts = []
        if missing_left:
            parts.append(_describe(missing_left, "left"))
        if missing_right:
            parts.append(_describe(missing_right, "right"))

        return cls(f"Cannot perform union. {'. '.join(parts)}")


def _order_columns(
    left_columns: Iterable[ColumnElement], right_columns: Iterable[ColumnElement]
) -> list[list[ColumnElement]]:
    left_names = [c.name for c in left_columns]
    right_names = [c.name for c in right_columns]

    # validate
    if sorted(left_names) != sorted(right_names):
        left_names_set = set(left_names)
        right_names_set = set(right_names)
        raise UnionSchemaMismatchError.from_column_sets(
            left_names_set - right_names_set,
            right_names_set - left_names_set,
        )

    # Order columns to match left_names order
    column_dicts = [
        {c.name: c for c in columns} for columns in [left_columns, right_columns]
    ]

    return [[d[n] for n in left_names] for d in column_dicts]


def _drop_system_columns(columns: Iterable[ColumnElement]) -> list[ColumnElement]:
    return [c for c in columns if not c.name.startswith("sys__")]


@attrs.define
class ResultIter:
    _row_iter: Iterable[Any]
    columns: list[str]

    def __iter__(self):
        yield from self._row_iter


class DatasetQuery:
    def __init__(
        self,
        name: str,
        version: str | None = None,
        project_name: str | None = None,
        namespace_name: str | None = None,
        catalog: "Catalog | None" = None,
        session: Session | None = None,
        in_memory: bool = False,
        update: bool = False,
        include_incomplete: bool = False,
    ) -> None:
        self.session = Session.get(session, catalog=catalog, in_memory=in_memory)
        self.catalog = catalog or self.session.catalog
        self.steps: list[Step] = []
        self._chunk_index: int | None = None
        self._chunk_total: int | None = None
        self.temp_table_names: list[str] = []
        self.dependencies: set[DatasetDependencyType] = set()
        self.table = self.get_table()
        self.starting_step: QueryStep | None = None
        self.name: str | None = None
        self.version: str | None = None
        self.feature_schema: dict | None = None
        self.column_types: dict[str, Any] | None = None
        self.before_steps: list[Callable] = []
        self.listing_fn: Callable | None = None
        self.update = update

        self.list_ds_name: str | None = None

        self.name = name
        self.dialect = self.catalog.warehouse.db.dialect
        if version:
            self.version = version

        if namespace_name is None:
            namespace_name = self.catalog.metastore.default_namespace_name
        if project_name is None:
            project_name = self.catalog.metastore.default_project_name

        if is_listing_dataset(name) and not version:
            # not setting query step yet as listing dataset might not exist at
            # this point
            self.list_ds_name = name
        else:
            self._set_starting_step(
                self.catalog.get_dataset_with_remote_fallback(
                    name,
                    namespace_name=namespace_name,
                    project_name=project_name,
                    version=version,
                    pull_dataset=True,
                    update=update,
                    include_incomplete=include_incomplete,
                )
            )

    def _set_starting_step(self, ds: "DatasetRecord") -> None:
        if not self.version:
            self.version = ds.latest_version

        self.starting_step = QueryStep(self.catalog, ds, self.version)

        # at this point we know our starting dataset so setting up schemas
        self.feature_schema = ds.get_version(self.version).feature_schema
        self.column_types = copy(ds.schema)
        if "sys__id" in self.column_types:
            self.column_types.pop("sys__id")
        self.project = ds.project

    def __iter__(self):
        return iter(self.db_results())

    def __or__(self, other):
        return self.union(other)

    def hash(self) -> str:
        """
        Calculates hash of this class taking into account hash of starting step
        and hashes of each following steps. Ordering is important.
        """
        hasher = hashlib.sha256()
        if self.starting_step:
            hasher.update(self.starting_step.hash().encode("utf-8"))
        else:
            assert self.list_ds_name
            hasher.update(self.list_ds_name.encode("utf-8"))

        for step in self.steps:
            hasher.update(step.hash().encode("utf-8"))

        return hasher.hexdigest()

    @staticmethod
    def get_table() -> "TableClause":
        table_name = "".join(secrets.choice(string.ascii_letters) for _ in range(16))
        return sqlalchemy.table(table_name)

    @property
    def attached(self) -> bool:
        """
        DatasetQuery is considered "attached" to underlying dataset if it represents
        it completely. If this is the case, name and version of underlying dataset
        will be defined.
        DatasetQuery instance can become attached in two scenarios:
            1. ds = DatasetQuery(name="dogs", version="1.0.0") -> ds is attached to dogs
            2. ds = ds.save("dogs", version="1.0.0") -> ds is attached to dogs dataset
        It can move to detached state if filter or similar methods are called on it,
        as then it no longer 100% represents underlying datasets.
        """
        return self.name is not None and self.version is not None

    def c(self, column: C | str) -> "ColumnClause[Any]":
        col: sqlalchemy.ColumnClause = (
            sqlalchemy.column(column)
            if isinstance(column, str)
            else sqlalchemy.column(column.name, column.type)
        )
        col.table = self.table
        return col

    def set_listing_fn(self, fn: Callable) -> None:
        """Setting listing function to be run if needed"""
        self.listing_fn = fn

    def apply_listing_pre_step(self) -> None:
        """Runs listing pre-step if needed"""
        if self.list_ds_name and not self.starting_step:
            listing_ds = None
            try:
                listing_ds = self.catalog.get_dataset(
                    self.list_ds_name, include_incomplete=False
                )
            except DatasetNotFoundError:
                pass

            if not listing_ds or self.update or listing_dataset_expired(listing_ds):
                assert self.listing_fn
                self.listing_fn()
                listing_ds = self.catalog.get_dataset(
                    self.list_ds_name, include_incomplete=False
                )

            # at this point we know what is our starting listing dataset name
            self._set_starting_step(listing_ds)  # type: ignore [arg-type]

    def apply_steps(self) -> QueryGenerator:
        """
        Apply the steps in the query and return the resulting
        sqlalchemy.SelectBase.
        """
        self.apply_listing_pre_step()

        query = self.clone()

        index = os.getenv("DATACHAIN_QUERY_CHUNK_INDEX", self._chunk_index)
        total = os.getenv("DATACHAIN_QUERY_CHUNK_TOTAL", self._chunk_total)

        if index is not None and total is not None:
            index, total = int(index), int(total)  # os.getenv returns str

            if not (0 <= index < total):
                raise ValueError("chunk index must be between 0 and total")

            # Respect limit in chunks
            query.steps = self._chunk_limit(query.steps, index, total)

            # Prepend the chunk filter to the step chain.
            query = query.filter(C.sys__rand % total == index)
            query.steps = query.steps[-1:] + query.steps[:-1]

        assert query.starting_step
        result = query.starting_step.apply()
        self.dependencies.update(result.dependencies)

        for step in query.steps:
            result = step.apply(
                result.query_generator, self.temp_table_names
            )  # a chain of steps linked by results
            self.dependencies.update(result.dependencies)

        return result.query_generator

    @staticmethod
    def _chunk_limit(steps: list["Step"], index: int, total: int) -> list["Step"]:
        no_limit_steps = []
        limit = None
        for step in steps:
            # Remember last limit
            if isinstance(step, SQLLimit):
                limit = step.n
            # Only keep non-limit steps
            else:
                no_limit_steps.append(step)
        # Chunk the limit
        if limit:
            limit_modulo = limit % total
            limit = limit // total
            if index < limit_modulo:
                limit += 1
            return [*no_limit_steps, SQLLimit(limit)]
        return steps

    def cleanup(self) -> None:
        """Cleanup any temporary tables."""
        if not self.temp_table_names:
            # Nothing to clean up.
            return
        # This is needed to always use a new connection with all metastore and warehouse
        # implementations, as errors may close or render unusable the existing
        # connections.
        assert len(self.temp_table_names) == len(set(self.temp_table_names))
        with self.catalog.metastore.clone(use_new_connection=True) as metastore:
            metastore.cleanup_tables(self.temp_table_names)
        with self.catalog.warehouse.clone(use_new_connection=True) as warehouse:
            warehouse.cleanup_tables(self.temp_table_names)
        self.temp_table_names = []

    def db_results(self, row_factory=None, **kwargs):
        with self.as_iterable(**kwargs) as result:
            if row_factory:
                cols = result.columns
                return [row_factory(cols, r) for r in result]
            return list(result)

    def to_db_records(self) -> list[dict[str, Any]]:
        return self.db_results(lambda cols, row: dict(zip(cols, row, strict=False)))

    @contextlib.contextmanager
    def as_iterable(self, **kwargs) -> Iterator[ResultIter]:
        try:
            query = self.apply_steps().select()
            selected_columns = [c.name for c in query.selected_columns]
            yield ResultIter(
                self.catalog.warehouse.dataset_rows_select(query, **kwargs),
                selected_columns,
            )
        finally:
            self.cleanup()

    def extract(
        self, *params: UDFParamSpec, workers=ASYNC_WORKERS, **kwargs
    ) -> Iterable[tuple]:
        """
        Extract columns from each row in the query.

        Returns an iterable of tuples matching the given params.

        To ensure prompt resource cleanup, it is recommended to wrap this
        with contextlib.closing().
        """
        actual_params = [normalize_param(p) for p in params]
        try:
            query = self.apply_steps().select()
            query_fields = [str(c.name) for c in query.selected_columns]

            def row_iter() -> Generator[Sequence, None, None]:
                # warehouse isn't threadsafe, we need to clone() it
                # in the thread that uses the results
                with self.catalog.warehouse.clone() as warehouse:
                    gen = warehouse.dataset_select_paginated(query)
                    with contextlib.closing(gen) as rows:
                        yield from rows

            async def get_params(row: Sequence) -> tuple:
                row_dict = RowDict(zip(query_fields, row, strict=False))
                return tuple(  # noqa: C409
                    [
                        await p.get_value_async(
                            self.catalog, row_dict, mapper, **kwargs
                        )
                        for p in actual_params
                    ]
                )

            MapperCls = OrderedMapper if query._order_by_clauses else AsyncMapper  # noqa: N806
            with contextlib.closing(row_iter()) as rows:
                mapper = MapperCls(get_params, rows, workers=workers)
                yield from mapper.iterate()
        finally:
            self.cleanup()

    def sample(self, n) -> "Self":
        """
        Return a random sample from the dataset.

        Args:
            n (int): Number of samples to draw.

        NOTE: Sampled are not deterministic, and streamed/paginated queries or
        multiple workers will draw samples with replacement.
        """
        sampled = self.order_by(rand())

        return sampled.limit(n)

    def clone(self, new_table=True) -> "Self":
        obj = copy(self)
        obj.steps = obj.steps.copy()
        if new_table:
            obj.table = self.get_table()
        obj.temp_table_names = []
        return obj

    @detach
    def select(self, *args, **kwargs) -> "Self":
        """
        Select the given columns or expressions using a subquery.

        If used with no arguments, this simply creates a subquery and
        select all columns from it.

        Note that the `save` function expects default dataset columns to
        be present. This function is meant to be followed by a call to
        `results` if used to exclude any default columns.

        Example:
            >>> ds.select(C.name, C.size * 10).results()
            >>> ds.select(C.name, size10x=C.size * 10).order_by(C.size10x).results()
        """
        named_args = [v.label(k) for k, v in kwargs.items()]
        query = self.clone()
        query.steps.append(SQLSelect((*args, *named_args)))
        return query

    @detach
    def ordered_select(self, *args, **kwargs) -> "Self":
        """
        Select the given columns or expressions using a subquery whilst
        maintaining query ordering (only applicable if last step was order_by).

        If used with no arguments, this simply creates a subquery and
        select all columns from it.

        Example:
            >>> ds.ordered_select(C.name, C.size * 10)
            >>> ds.ordered_select(C.name, size10x=C.size * 10)
        """
        named_args = [v.label(k) for k, v in kwargs.items()]
        query = self.clone()
        order_by = query.last_step if query.is_ordered else None
        query.steps.append(SQLSelect((*args, *named_args)))
        if order_by:
            query.steps.append(order_by)
        return query

    @detach
    def select_except(self, *args) -> "Self":
        """
        Exclude certain columns from this query using a subquery.

        Note that the `save` function expects default dataset columns to
        be present. This function is meant to be followed by a call to
        `results` if used to exclude any default columns.

        Example:
            >>> (
            ...     ds.mutate(size10x=C.size * 10)
            ...     .order_by(C.size10x)
            ...     .select_except(C.size10x)
            ...     .results()
            ... )
        """

        if not args:
            raise TypeError("select_except expected at least 1 argument, got 0")
        query_args = [c if isinstance(c, str) else c.name for c in args]
        query = self.clone()
        query.steps.append(SQLSelectExcept(query_args))  # type: ignore [arg-type]
        return query

    @detach
    def mutate(self, *args, new_schema, **kwargs) -> "Self":
        """
        Add new columns to this query.

        This function selects all existing columns from this query and
        adds in the new columns specified.

        Example:
            >>> ds.mutate(size10x=C.size * 10).order_by(C.size10x).results()
        """
        query_args = [v.label(k) for k, v in dict(args, **kwargs).items()]
        query = self.clone()
        query.steps.append(SQLMutate((*query_args,), new_schema))
        return query

    @detach
    def filter(self, *args) -> "Self":
        query = self.clone(new_table=False)
        steps = query.steps
        if steps and isinstance(steps[-1], SQLFilter):
            steps[-1] = steps[-1] & args
        else:
            steps.append(SQLFilter(args))
        return query

    @detach
    def order_by(self, *args) -> "Self":
        query = self.clone(new_table=False)
        query.steps.append(SQLOrderBy(args))
        return query

    @detach
    def limit(self, n: int) -> "Self":
        query = self.clone(new_table=False)
        if (
            query.steps
            and (last_step := query.last_step)
            and isinstance(last_step, SQLLimit)
        ):
            query.steps[-1] = SQLLimit(min(n, last_step.n))
        else:
            query.steps.append(SQLLimit(n))
        return query

    @detach
    def offset(self, offset: int) -> "Self":
        query = self.clone(new_table=False)
        query.steps.append(SQLOffset(offset))
        return query

    @detach
    def distinct(self, *args) -> "Self":
        query = self.clone()
        query.steps.append(
            SQLDistinct(args, dialect=self.catalog.warehouse.db.dialect.name)
        )
        return query

    def as_scalar(self) -> Any:
        with self.as_iterable() as rows:
            row = next(iter(rows))
        return row[0]

    def count(self) -> int:
        query = self.clone()
        query.steps.append(SQLCount())
        return query.as_scalar()

    def sum(self, col: ColumnElement) -> int:
        query = self.clone()
        query.steps.append(SQLSelect((f.sum(col),)))
        return query.as_scalar()

    def avg(self, col: ColumnElement) -> int:
        query = self.clone()
        query.steps.append(SQLSelect((f.avg(col),)))
        return query.as_scalar()

    def min(self, col: ColumnElement) -> int:
        query = self.clone()
        query.steps.append(SQLSelect((f.min(col),)))
        return query.as_scalar()

    def max(self, col: ColumnElement) -> int:
        query = self.clone()
        query.steps.append(SQLSelect((f.max(col),)))
        return query.as_scalar()

    @detach
    def group_by(
        self,
        cols: Sequence[ColumnElement],
        group_by: Sequence[ColumnElement],
    ) -> "Self":
        query = self.clone()
        query.steps.append(SQLGroupBy(cols, group_by))
        return query

    @detach
    def union(self, dataset_query: "DatasetQuery") -> "Self":
        left = self.clone()
        right = dataset_query.clone()
        new_query = self.clone()
        new_query.steps = [SQLUnion(left, right)]
        return new_query

    @detach
    def join(
        self,
        dataset_query: "DatasetQuery",
        predicates: JoinPredicateType | Sequence[JoinPredicateType],
        inner=False,
        full=False,
        rname="right_",
    ) -> "Self":
        left = self.clone(new_table=False)
        if self.table.name == dataset_query.table.name:
            # for use case where we join with itself, e.g dogs.join(dogs, "name")
            right = dataset_query.clone(new_table=True)
        else:
            right = dataset_query.clone(new_table=False)

        new_query = self.clone()
        predicates = (
            predicates
            if isinstance(predicates, (str, ColumnClause, ColumnElement))
            else tuple(predicates)
        )
        new_query.steps = [
            SQLJoin(self.catalog, left, right, predicates, inner, full, rname)
        ]
        return new_query

    @detach
    def chunk(self, index: int, total: int) -> "Self":
        """Split a query into smaller chunks for e.g. parallelization.
        Example:
            >>> query = DatasetQuery(...)
            >>> chunk_1 = query._chunk(0, 2)
            >>> chunk_2 = query._chunk(1, 2)
        Note:
            Bear in mind that `index` is 0-indexed but `total` isn't.
            Use 0/3, 1/3 and 2/3, not 1/3, 2/3 and 3/3.
        """
        query = self.clone()
        query._chunk_index, query._chunk_total = index, total
        return query

    @detach
    def add_signals(
        self,
        udf: "UDFAdapter",
        partition_by: PartitionByType | None = None,
        # Parameters from Settings
        cache: bool = False,
        parallel: int | None = None,
        workers: bool | int = False,
        min_task_size: int | None = None,
        batch_size: int | None = None,
        # Parameters are unused, kept only to match the signature of Settings.to_dict
        prefetch: int | None = None,
        namespace: str | None = None,
        project: str | None = None,
    ) -> "Self":
        """
        Adds one or more signals based on the results from the provided UDF.

        Parallel can optionally be specified as >= 1 for parallel processing with a
        specific number of processes, or set to -1 for the default of
        the number of CPUs (cores) on the current machine.

        For distributed processing with the appropriate distributed module installed,
        workers can optionally be specified as >= 1 for a specific number of workers,
        or set to True for the default of all nodes in the cluster.
        As well, a custom minimum task size (min_task_size) can be provided to send
        at least that minimum number of rows to each distributed worker, mostly useful
        if there are a very large number of small tasks to process.
        """
        query = self.clone()
        query.steps.append(
            UDFSignal(
                udf,
                self.catalog,
                partition_by=partition_by,
                parallel=parallel,
                workers=workers,
                min_task_size=min_task_size,
                cache=cache,
                batch_size=batch_size,
            )
        )
        return query

    @detach
    def subtract(self, dq: "DatasetQuery", on: Sequence[tuple[str, str]]) -> "Self":
        query = self.clone()
        query.steps.append(Subtract(dq, self.catalog, on=on))
        return query

    @detach
    def generate(
        self,
        udf: "UDFAdapter",
        partition_by: PartitionByType | None = None,
        # Parameters from Settings
        cache: bool = False,
        parallel: int | None = None,
        workers: bool | int = False,
        min_task_size: int | None = None,
        batch_size: int | None = None,
        # Parameters are unused, kept only to match the signature of Settings.to_dict:
        prefetch: int | None = None,
        namespace: str | None = None,
        project: str | None = None,
    ) -> "Self":
        query = self.clone()
        steps = query.steps
        steps.append(
            RowGenerator(
                udf,
                self.catalog,
                partition_by=partition_by,
                parallel=parallel,
                workers=workers,
                min_task_size=min_task_size,
                cache=cache,
                batch_size=batch_size,
            )
        )
        return query

    def _add_dependencies(self, dataset: "DatasetRecord", version: str):
        dependencies: set[DatasetDependencyType] = set()
        for dep_dataset, dep_dataset_version in self.dependencies:
            if Session.is_temp_dataset(dep_dataset.name):
                # temp dataset are created for optimization and they will be removed
                # afterwards. Therefore, we should not put them as dependencies, but
                # their own direct dependencies
                for dep in self.catalog.get_dataset_dependencies(
                    dep_dataset.name,
                    dep_dataset_version,
                    namespace_name=dep_dataset.project.namespace.name,
                    project_name=dep_dataset.project.name,
                    indirect=False,
                ):
                    if dep:
                        dependencies.add(
                            (
                                self.catalog.get_dataset(
                                    dep.name,
                                    namespace_name=dep.namespace,
                                    project_name=dep.project,
                                    include_incomplete=False,
                                ),
                                dep.version,
                            )
                        )
            else:
                dependencies.add((dep_dataset, dep_dataset_version))

        for dep_dataset, dep_dataset_version in dependencies:
            self.catalog.metastore.add_dataset_dependency(
                dataset,
                version,
                dep_dataset,
                dep_dataset_version,
            )

    def exec(self) -> "Self":
        """Execute the query."""
        query = self.clone()
        try:
            query.apply_steps()
        finally:
            query.cleanup()
        return query

    def save(
        self,
        name: str | None = None,
        version: str | None = None,
        project: Project | None = None,
        feature_schema: dict | None = None,
        dependencies: list[DatasetDependency] | None = None,
        description: str | None = None,
        attrs: list[str] | None = None,
        update_version: str | None = "patch",
        **kwargs,
    ) -> "Self":
        """Save the query as a dataset."""
        # Get job from session to link dataset version to job
        job = self.session.get_or_create_job()
        job_id = job.id

        project = project or self.catalog.metastore.default_project
        try:
            if (
                name
                and version
                and self.catalog.get_dataset(
                    name,
                    namespace_name=project.namespace.name,
                    project_name=project.name,
                    include_incomplete=True,
                ).has_version(version)
            ):
                raise RuntimeError(f"Dataset {name} already has version {version}")
        except DatasetNotFoundError:
            pass
        if not name and version:
            raise RuntimeError("Cannot set version for temporary datasets")

        if not name:
            name = self.session.generate_temp_dataset_name()

        try:
            query = self.apply_steps()

            columns = [
                c if isinstance(c, Column) else Column(c.name, c.type)
                for c in query.columns
            ]
            if not [c for c in columns if c.name != "sys__id"]:
                raise RuntimeError(
                    "No columns to save in the query. "
                    "Ensure at least one column (other than 'id') is selected."
                )

            dataset = self.catalog.create_dataset(
                name,
                project,
                version=version,
                feature_schema=feature_schema,
                columns=columns,
                description=description,
                attrs=attrs,
                update_version=update_version,
                job_id=job_id,
                **kwargs,
            )
            version = version or dataset.latest_version

            dr = self.catalog.warehouse.dataset_rows(dataset)

            self.catalog.warehouse.copy_table(dr.get_table(), query.select())

            self.catalog.update_dataset_version_with_warehouse_info(dataset, version)

            # Link this dataset version to the job that created it
            self.catalog.metastore.link_dataset_version_to_job(
                dataset.get_version(version).id, job_id, is_creator=True
            )

            if dependencies:
                # overriding dependencies
                self.dependencies = set()
                for dep in dependencies:
                    self.dependencies.add(
                        (
                            self.catalog.get_dataset(
                                dep.name,
                                namespace_name=dep.namespace,
                                project_name=dep.project,
                                include_incomplete=False,
                            ),
                            dep.version,
                        )
                    )

            self._add_dependencies(dataset, version)  # type: ignore [arg-type]

            # Mark as COMPLETE only after all operations succeed
            self.catalog.metastore.update_dataset_status(
                dataset, DatasetStatus.COMPLETE, version=version
            )
        finally:
            self.cleanup()
        return self.__class__(
            name=name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            version=version,
            catalog=self.catalog,
        )

    @property
    def is_ordered(self) -> bool:
        return isinstance(self.last_step, SQLOrderBy)

    @property
    def last_step(self) -> Step | None:
        return self.steps[-1] if self.steps else None
