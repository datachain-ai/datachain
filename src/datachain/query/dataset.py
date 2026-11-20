import contextlib
import hashlib
import logging
import os
import secrets
import string
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import copy
from functools import wraps
from types import GeneratorType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, cast
from uuid import uuid4

import attrs
import sqlalchemy
import sqlalchemy as sa
from attrs import frozen
from fsspec import Callback
from fsspec.callbacks import DEFAULT_CALLBACK, TqdmCallback
from sqlalchemy import Column
from sqlalchemy.sql import func as f
from sqlalchemy.sql.elements import ColumnClause, ColumnElement, Label, TextClause
from sqlalchemy.sql.expression import label
from sqlalchemy.sql.functions import count as sa_count
from sqlalchemy.sql.selectable import (
    GenerativeSelect,
    Select,
    SelectBase,
    TableClause,
)
from sqlalchemy.sql.visitors import iterate, replacement_traverse

# Resolve parent packages before their children to keep import-lock ordering consistent.
from datachain.catalog import catalog as _catalog
from datachain.checkpoint import Checkpoint, CheckpointStatus
from datachain.checkpoint_event import (
    CheckpointEventType,
    CheckpointStepType,
)
from datachain.data_storage.schema import (
    PARTITION_COLUMN_ID,
    coerce_table_types,
    committed_part_keys,
    partition_col_names,
    partition_columns,
    parts_table_name,
)
from datachain.dataset import DatasetDependency
from datachain.error import (
    DataChainError,
    DatasetInvalidVersionError,
    DatasetNotFoundError,
    QueryScriptAbortError,
    QueryScriptCancelError,
    TableMissingError,
)
from datachain.func import base
from datachain.hash_utils import hash_column_elements
from datachain.job import Job
from datachain.lib.data_model import NULLABLE_SCALARS
from datachain.lib.listing import ensure_listing, is_listing_dataset
from datachain.lib.signal_schema import (
    KeyPresence,
    SignalSchema,
    generate_merge_root_mapping,
)
from datachain.lib.udf import _get_cache, _Progress
from datachain.progress import (
    CombinedDownloadCallback,
    TqdmCombinedDownloadCallback,
    tqdm,
)
from datachain.project import Project
from datachain.query.schema import DEFAULT_DELIMITER, C, ColumnExpr
from datachain.query.session import Session
from datachain.query.udf import UdfInfo
from datachain.sql.functions.random import rand
from datachain.sql.types import SQLType
from datachain.utils import (
    determine_processes,
    determine_workers,
    ensure_sequence,
    env2bool,
    filtered_cloudpickle_dumps,
    get_datachain_executable,
    safe_closing,
)

if TYPE_CHECKING:
    from typing import Concatenate, Self

    from sqlalchemy.sql.elements import KeyedColumnElement
    from sqlalchemy.sql.schema import Table
    from typing_extensions import ParamSpec

    from datachain.catalog import Catalog
    from datachain.data_storage import AbstractWarehouse
    from datachain.dataset import DatasetRecord
    from datachain.lib.udf import UDFAdapter, UDFPart, UDFResult

    P = ParamSpec("P")


Function = base.Function
PartitionByType = str | Function | ColumnExpr | Sequence[str | Function | ColumnExpr]
JoinPredicateType = str | ColumnExpr
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


def _bind_shared_names(cols: Iterable[Any], query: Select) -> tuple[Any, ...]:
    """Bind bare names that several of the query's sources have, like both
    sides of a join, to the columns the query selects under those names."""
    seen: set[str] = set()
    shared: set[str] = set()
    for source in query.columns_clause_froms:
        shared |= seen & set(source.c.keys())
        seen |= set(source.c.keys())
    names = {
        name: c.element if isinstance(c, Label) else c
        for name, c in query.selected_columns.items()
        if name in shared
    }
    if not names:
        return tuple(cols)

    def bind(element, **_kwargs):
        # A nested SELECT resolves names against its own FROM. Artificial for
        # now: only hand-written SQLAlchemy puts one in these clauses.
        if isinstance(element, SelectBase):
            return element
        if isinstance(element, C) and element.table is None:
            return names.get(element.name)
        return None

    return tuple(
        names.get(C(c).name, c)
        if isinstance(c, str)
        else replacement_traverse(c, {}, bind)
        for c in cols
    )


class QueryState:
    """Tracks which terminal SQL clauses have been applied to the in-flight
    query during ``DatasetQuery.apply_steps``.

    A subsequent clause inspects these flags to decide whether the prior
    output must be wrapped in a subquery (a *state boundary*) before the
    new clause is applied — otherwise constructs like ``WHERE`` after
    ``GROUP BY`` or ``ORDER BY`` after ``LIMIT`` would affect the
    underlying rows instead of the materialized state output.
    """

    def __init__(self, warehouse: "AbstractWarehouse") -> None:
        self.warehouse = warehouse
        self.reset()

    def reset(self) -> None:
        self.grouped = False
        self.limited = False
        self.offset_applied = False
        self.distinct = False

    @property
    def has_finalizing_clause(self) -> bool:
        """True if a prior clause finalized the current row set."""
        return self.grouped or self.limited or self.offset_applied or self.distinct

    def emit_boundary(self, query_generator: "QueryGenerator") -> "QueryGenerator":
        """Materialize the in-flight query as a subquery and reset flags.

        After this call, the returned generator represents a fresh state:
        any prior ``LIMIT``/``OFFSET``/``GROUP BY``/``DISTINCT`` is folded
        into the inner subquery, so the caller can apply its clause to the
        outer level without affecting underlying rows.
        """
        query: GenerativeSelect = query_generator.select()
        # Some warehouses need dialect-specific LIMIT/OFFSET normalization
        # before the query is wrapped.
        query = self.warehouse.normalize_limit_offset(query)
        sub = query.subquery()
        new_query = sqlalchemy.select(*sub.c).select_from(sub)

        def q(*columns):
            return new_query.with_only_columns(*columns)

        self.reset()
        return QueryGenerator(func=q, columns=tuple(new_query.selected_columns))


@frozen
class Step(ABC):
    """A query processing step (filtering, mutation, etc.)"""

    resets_query_state_after_apply: ClassVar[bool] = True
    bounds_columns: ClassVar[bool] = False

    def used_columns(self) -> set[str] | None:
        """Names of the input columns this step reads, or None if unknown."""
        return None

    @abstractmethod
    def apply(
        self,
        query_generator: QueryGenerator,
        temp_tables: list[str],
        state: "QueryState",
        **kwargs,
    ) -> "StepResult":
        """Apply the processing step.

        ``state`` carries the in-flight query-state flags. SQL clauses
        that finalize output (``filter`` after ``GROUP BY``/``LIMIT``, etc.)
        inspect and update it via ``state.emit_boundary`` to materialize
        a subquery boundary when needed. Steps that don't care may
        ignore it.
        """

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

    def collect_delta_sources(
        self,
    ) -> list["DatasetQuery"]:
        return []

    @property
    def sub_queries(self) -> list["DatasetQuery"]:
        """Immediate sub-queries contained in this step."""
        return []

    def replace_sources(
        self,
        replacements: dict[int, "DatasetQuery"],
    ) -> "Step":
        return self


@frozen
class DeltaSpec:
    on: str | Sequence[str]
    right_on: str | Sequence[str] | None = None
    compare: str | Sequence[str] | None = None
    delta_retry: bool | str | None = None
    delta_unsafe: bool = False

    @staticmethod
    def _normalize_hash_value(
        value: str | Sequence[str] | bool | None,
    ) -> str | tuple[str, ...] | bool | None:
        if isinstance(value, str) or value is None or isinstance(value, bool):
            return value
        return tuple(value)

    def hash(self) -> str:
        normalized = (
            self._normalize_hash_value(self.on),
            self._normalize_hash_value(self.right_on),
            self._normalize_hash_value(self.compare),
            self.delta_retry,
            self.delta_unsafe,
        )
        return hashlib.sha256(repr(normalized).encode("utf-8")).hexdigest()


@frozen
class QueryStep:
    """A query that returns all rows from specific dataset version"""

    catalog: "Catalog"
    dataset: "DatasetRecord"
    dataset_version: str

    def _record_access(self) -> None:
        self.catalog.metastore.record_dataset_version_access(
            self.dataset,
            self.dataset_version,
        )

    def apply(self) -> "StepResult":
        def q(*columns):
            return sqlalchemy.select(*columns)

        self._record_access()
        dr = self.catalog.warehouse.dataset_rows(self.dataset, self.dataset_version)
        # Use a short alias with dataset ID suffix for uniqueness and SQL brevity
        ds_id = dr.table.name.rsplit("_", 1)[-1]
        aliased_table = dr.table.alias(f"__ds_t_{ds_id}")

        return step_result(
            q,
            aliased_table.columns,
            dependencies=[(self.dataset, self.dataset_version)],
        )

    def hash(self) -> str:
        version = self.dataset.get_version(self.dataset_version)
        anchor = (
            version.content_hash
            if version.content_hash is not None
            and not is_listing_dataset(self.dataset.name)
            else version.uuid
        )
        return hashlib.sha256(anchor.encode()).hexdigest()


def _key_present(columns: Any, presence: KeyPresence) -> ColumnElement:
    """Whether a key node holds a value, by ``SignalSchema.key_presence``."""
    if presence is None:
        return sa.true()
    tag, arm = presence
    is_set = columns[tag].is_not(None)
    return is_set if arm is None else sa.and_(is_set, columns[tag] == arm)


@frozen
class Subtract(Step):
    """Rows of the query whose keys ``on`` are not in ``dq``."""

    dq: "DatasetQuery"
    catalog: "Catalog"
    on: Sequence[str | tuple[str, str]]
    # Key nodes compared by whether they hold a value on each side.
    presence: Sequence[tuple[KeyPresence, KeyPresence]] = ()

    def clone(self) -> "Self":
        return attrs.evolve(self)

    def collect_delta_sources(self) -> list["DatasetQuery"]:
        return self.dq.delta_sources()

    @property
    def sub_queries(self) -> list["DatasetQuery"]:
        return [self.dq]

    def replace_sources(self, replacements: dict[int, "DatasetQuery"]) -> "Step":
        return attrs.evolve(self, dq=self.dq.replace_sources(replacements))

    def apply(
        self, query_generator, temp_tables: list[str], state: "QueryState", **kwargs
    ) -> "StepResult":
        source_query = query_generator.select()
        right_before = len(self.dq.temp_table_names)
        target_full = self.dq.apply_steps().select()
        temp_tables.extend(self.dq.temp_table_names[right_before:])
        # Exclude sys columns from target - only key columns are used for matching
        target_query = target_full.with_only_columns(
            *(c for c in target_full.selected_columns if not c.name.startswith("sys__"))
        )

        diff_q = cast("SelectBase", self.query(source_query, target_query)).subquery()

        def q(*columns):
            return sqlalchemy.select(*columns).select_from(diff_q)

        columns = [diff_q.c[c.name] for c in source_query.selected_columns]
        return step_result(q, columns, dependencies=self.dq.dependencies)

    def _normalize_on(self) -> list[tuple[str, str]]:
        return [(col, col) if isinstance(col, str) else col for col in self.on]

    def hash_inputs(self) -> str:
        normalized = self._normalize_on()
        on_bytes = b"".join(
            f"{a}:{b}".encode()
            for a, b in sorted(normalized, key=lambda t: (t[0], t[1]))
        )
        on_bytes += "".join(sorted(f"{a}:{b}" for a, b in self.presence)).encode()

        return hashlib.sha256(bytes.fromhex(self.dq.hash()) + on_bytes).hexdigest()

    def query(self, source_query: Select, target_query: Select) -> sa.Selectable:
        on = self._normalize_on()
        if not self.presence:
            return self.catalog.warehouse.subtract_query(source_query, target_query, on)
        # Compare presence as extra key columns, dropped again from the result.
        used = {
            c.name.casefold()
            for query in (source_query, target_query)
            for c in query.selected_columns
        }
        names = []
        for i in range(len(self.presence)):
            name = f"__dc_present_{i}"
            while name.casefold() in used:
                name = "_" + name
            names.append(name)
            used.add(name.casefold())

        def with_presence(query: Select, side: int) -> Select:
            columns = query.selected_columns
            return query.add_columns(
                *(
                    sa.case((_key_present(columns, pair[side]), 1), else_=0).label(name)
                    for name, pair in zip(names, self.presence, strict=True)
                ),
            )

        return self.catalog.warehouse.subtract_query(
            with_presence(source_query, 0),
            with_presence(target_query, 1),
            [*on, *zip(names, names, strict=True)],
        )


# Layout of UDF output tables and their ledgers, part of every UDF checkpoint
# hash: a change makes checkpoints of the old layout recompute once.
UDF_OUTPUT_FORMAT = b"writer-ledger-1"


def process_udf_outputs(
    warehouse: "AbstractWarehouse",
    udf_table: "Table",
    udf_results: "Iterator[Iterable[UDFResult] | UDFPart]",
    udf: "UDFAdapter",
    cb: Callback = DEFAULT_CALLBACK,
    batch_size: int | None = None,
    writer_no: int | None = None,
    parts: "Table | None" = None,
) -> None:
    """Write UDF results into udf_table through one Writer.

    A map result is a batch of rows keyed by their input ``sys__id``. A gen or
    agg result is ``(part_key, rows)``: the writer assigns the ids under
    ``writer_no`` and records the finished part in the ``parts`` ledger. With a
    ledger and no writer_no given, the writer_no is claimed from the ledger.
    """
    if writer_no is None:
        writer_no = 0 if parts is None else warehouse.claim_writer_no(parts)
    # The UDF yields its outputs already in stored form; a map row then its sys__id.
    is_map = udf.inner.udf_kind == "map"
    columns = [*udf.output, "sys__id"] if is_map else list(udf.output)

    def _rows(udf_output, progress):
        for row in udf_output:
            progress.add(1 if type(row) is tuple else row.num_rows)
            yield row

    with (
        _Progress(cb) as progress,
        warehouse.writer(udf_table, columns, writer_no, parts, batch_size) as writer,
    ):
        for result in udf_results:
            part_key, udf_output = (None, result) if is_map else result
            # an error mid-part closes the UDF's outputs now, not at GC
            with (
                safe_closing(udf_output),
                contextlib.closing(_rows(udf_output, progress)) as rows,
            ):
                writer.extend(rows)
            if not is_map:
                writer.commit(part_key)


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
    session: "Session"
    partition_by: PartitionByType | None = None
    is_generator = False
    is_aggregator = False
    # Parameters from Settings
    cache: bool = False
    parallel: int | None = None
    workers: bool | int = False
    min_task_size: int | None = None
    batch_size: int | None = None

    @property
    def is_single_partition_aggregator(self) -> bool:
        """Aggregator without partition_by — processes all rows as a single batch."""
        return self.is_aggregator and self.partition_by is None

    def hash_inputs(self) -> str:
        partition_by = ensure_sequence(self.partition_by or [])
        parts = [
            bytes.fromhex(self.udf.hash()),
            bytes.fromhex(hash_column_elements(partition_by)),
            str(self.is_generator).encode(),
            UDF_OUTPUT_FORMAT,
        ]

        return hashlib.sha256(b"".join(parts)).hexdigest()

    def used_columns(self) -> set[str] | None:
        assert self.udf.inner.params is not None
        return _expr_columns(
            (
                *self.udf.inner.params.db_signals(),
                *ensure_sequence(self.partition_by or ()),
            )
        )

    @abstractmethod
    def create_output_table(self, name: str) -> "Table":
        """Method that creates a table where temp udf results will be saved"""

    def get_input_query(self, input_table_name: str, original_query: Select) -> Select:
        """Get a select query for UDF input."""
        # Table was created from original_query by create_pre_udf_table,
        # so they should have the same columns. However, get_table() reflects
        # the table with database-specific types (e.g ClickHouse types) instead of
        # SQLTypes.
        # To preserve SQLTypes for proper type conversion while keeping columns bound
        # to the table (to avoid ambiguous column names), we use type_coerce.
        table = self.warehouse.db.get_table(input_table_name)

        # Create a mapping of column names to SQLTypes from original query
        orig_col_types = {col.name: col.type for col in original_query.selected_columns}

        # Sys columns are added by create_udf_table and may not be in original query
        sys_col_types = {
            col.name: col.type for col in self.warehouse.dataset_row_cls.sys_columns()
        }

        # Build select using bound columns from table, with type coercion for SQLTypes
        select_columns = []
        for table_col in table.c:
            if table_col.name in orig_col_types:
                # Use type_coerce to preserve SQLType while keeping column bound
                # to table. Use label() to preserve the column name
                select_columns.append(
                    sqlalchemy.type_coerce(
                        table_col, orig_col_types[table_col.name]
                    ).label(table_col.name)
                )
            elif table_col.name in sys_col_types:
                # Sys column added by create_udf_table - use known type
                select_columns.append(
                    sqlalchemy.type_coerce(
                        table_col, sys_col_types[table_col.name]
                    ).label(table_col.name)
                )
            else:
                raise RuntimeError(
                    f"Unexpected column '{table_col.name}' in input table"
                )

        return sqlalchemy.select(*select_columns).select_from(table)

    @abstractmethod
    def create_result_query(
        self, udf_table: "Table", query: Select
    ) -> tuple[QueryGeneratorFunc, list["sqlalchemy.Column"]]:
        """
        Method that should return query to fetch results from udf and columns
        to select
        """

    def populate_udf_output_table(  # noqa: PLR0915
        self,
        udf_table: "Table",
        query: Select,
        continued: bool = False,
        rows_reused: int = 0,
        output_rows_reused: int = 0,
        rows_total: int | None = None,
        parts: "Table | None" = None,
    ) -> None:
        catalog = self.session.catalog
        # read only the columns the UDF reads: a wide input costs nothing extra
        read = {"sys__id", PARTITION_COLUMN_ID, *self.udf.inner.read_columns()}
        query = query.with_only_columns(
            *(c for c in query.selected_columns if c.name in read)
        )
        rows_to_process = catalog.warehouse.query_count(query)
        if rows_to_process == 0:
            logger.debug(
                "UDF(%s): No rows to process, skipping",
                self._udf_name,
            )
            return

        from datachain.catalog import (
            QUERY_SCRIPT_ABORTED_EXIT_CODE,
            QUERY_SCRIPT_CANCELED_EXIT_CODE,
        )
        from datachain.catalog.loader import (
            DISTRIBUTED_IMPORT_PATH,
            get_udf_distributor_class,
        )

        workers = determine_workers(self.workers, rows_total=rows_to_process)
        processes = determine_processes(self.parallel, rows_total=rows_to_process)
        logger.debug(
            "UDF(%s): Processing %d rows (workers=%s, processes=%s, batch_size=%s)",
            self._udf_name,
            rows_to_process,
            workers,
            processes,
            self.batch_size,
        )

        use_partitioning = self.partition_by is not None
        batching = self.udf.get_batching(use_partitioning)
        udf_fields = [str(c.name) for c in query.selected_columns]
        udf_distributor_class = get_udf_distributor_class()

        prefetch = self.udf.prefetch
        with _get_cache(catalog.cache, prefetch, use_cache=self.cache) as _cache:
            catalog = _catalog.clone_catalog_with_cache(catalog, _cache)

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
                        rows_to_process=rows_to_process,
                        rows_total=rows_total,
                        use_cache=self.cache,
                        is_generator=self.is_generator,
                        min_task_size=self.min_task_size,
                        batch_size=self.batch_size,
                        continued=continued,
                        rows_reused=rows_reused,
                        output_rows_reused=output_rows_reused,
                        # only with a ledger: a distributor without it still runs
                        **({"parts": parts} if parts is not None else {}),
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
                        rows_total=rows_to_process,
                        batch_size=self.batch_size,
                        parts=parts,
                    )

                    # Run the UDFDispatcher in another process to avoid needing
                    # if __name__ == '__main__': in user scripts
                    exec_cmd = get_datachain_executable()
                    cmd = [*exec_cmd, "internal-run-udf"]
                    envs = dict(os.environ)
                    if "PYTHONSAFEPATH" not in envs:
                        envs["PYTHONSAFEPATH"] = "1"
                        envs["_DATACHAIN_UDF_SAFE_PATH"] = "1"
                    paths = [
                        sys.path[0] or os.getcwd(),
                        *os.environ.get("PYTHONPATH", "").split(os.pathsep),
                        os.getcwd(),
                    ]
                    envs["PYTHONPATH"] = os.pathsep.join(
                        dict.fromkeys(path or os.getcwd() for path in paths)
                    )
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
                            if retval == QUERY_SCRIPT_CANCELED_EXIT_CODE:
                                raise QueryScriptCancelError(
                                    "UDF execution was canceled by the user."
                                )
                            if retval == QUERY_SCRIPT_ABORTED_EXIT_CODE:
                                raise QueryScriptAbortError(
                                    "UDF execution aborted: job already terminated."
                                )
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
                                batch_size=self.batch_size,
                                parts=parts,
                            )
                    finally:
                        download_cb.close()
                        processed_cb.close()
                        generated_cb.close()

            except QueryScriptAbortError:
                catalog.warehouse.close()
                sys.exit(QUERY_SCRIPT_ABORTED_EXIT_CODE)
            except QueryScriptCancelError:
                catalog.warehouse.close()
                sys.exit(QUERY_SCRIPT_CANCELED_EXIT_CODE)
            except (Exception, KeyboardInterrupt):
                # Close any open database connections if an error is encountered
                catalog.warehouse.close()
                raise

    def create_partitions_table(
        self, query: Select, table_name: str | None = None
    ) -> "Table":
        """
        Create table with partition mappings (sys__id -> partition_id).

        Args:
            query: Input query with sys__id column
            table_name: Name for the partition table. If None, auto-generated.
        """
        catalog = self.session.catalog

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

        # Drop stale partition table from earlier chain with same hash in this job
        if table_name:
            catalog.warehouse.db.drop_table(
                sa.Table(table_name, sa.MetaData()), if_exists=True
            )

        # create table with partitions
        tbl = catalog.warehouse.create_udf_table(partition_columns(), name=table_name)

        # fill table with partitions
        cols = [
            id_col,
            f.dense_rank().over(order_by=partition_by).label(PARTITION_COLUMN_ID),
        ]
        catalog.warehouse.db.execute(
            tbl.insert().from_select(
                cols,
                query.offset(None).limit(None).with_only_columns(*cols),
            )
        )

        return tbl

    @property
    def _udf_name(self) -> str:
        """Get UDF name for logging."""
        return self.udf.inner.verbose_name

    @staticmethod
    def _job_id_short(job: "Job | None") -> str:
        """Get short job_id for logging."""
        return job.id[:8] if job and job.id else "none"

    @staticmethod
    def _run_group_id_short(job: "Job | None") -> str:
        """Get short run_group_id for logging."""
        return job.run_group_id[:8] if job and job.run_group_id else "none"

    @property
    @abstractmethod
    def _step_type(self) -> CheckpointStepType:
        """Get the step type for checkpoint events."""

    def _log_event(
        self,
        job: Job,
        event_type: CheckpointEventType,
        checkpoint_hash: str | None = None,
        hash_partial: str | None = None,
        hash_input: str | None = None,
        hash_output: str | None = None,
        rows_input: int | None = None,
        rows_processed: int | None = None,
        rows_output: int | None = None,
        rows_input_reused: int | None = None,
        rows_output_reused: int | None = None,
        rerun_from_job_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Log a checkpoint event and emit a log message."""
        self.metastore.log_checkpoint_event(
            job_id=job.id,
            event_type=event_type,
            step_type=self._step_type,
            run_group_id=job.run_group_id,
            udf_name=self._udf_name,
            checkpoint_hash=checkpoint_hash,
            hash_partial=hash_partial,
            hash_input=hash_input,
            hash_output=hash_output,
            rows_input=rows_input,
            rows_processed=rows_processed,
            rows_output=rows_output,
            rows_input_reused=rows_input_reused,
            rows_output_reused=rows_output_reused,
            rerun_from_job_id=rerun_from_job_id,
            details=details,
        )
        logger.debug(
            "UDF(%s) [job=%s run_group=%s]: %s - "
            "input=%s, processed=%s, output=%s, input_reused=%s, output_reused=%s",
            self._udf_name,
            self._job_id_short(job),
            self._run_group_id_short(job),
            event_type.value,
            rows_input,
            rows_processed,
            rows_output,
            rows_input_reused,
            rows_output_reused,
        )

    def _find_udf_checkpoint(self, job: "Job", _hash: str) -> Checkpoint | None:
        """
        Find a reusable UDF checkpoint for the given hash.
        Searches current job first, then previous job.
        Returns the Checkpoint object if found, None otherwise.
        """
        ignore_checkpoints = env2bool("DATACHAIN_IGNORE_CHECKPOINTS", undefined=False)
        if ignore_checkpoints:
            return None

        # Search current job first, then previous job
        for job_id in [job.id, job.rerun_from_job_id]:
            if job_id and (checkpoint := self.metastore.find_checkpoint(job_id, _hash)):
                logger.debug(
                    "UDF(%s) [job=%s run_group=%s]: Found %scheckpoint "
                    "hash=%s from job_id=%s",
                    self._udf_name,
                    self._job_id_short(job),
                    self._run_group_id_short(job),
                    "partial " if checkpoint.partial else "",
                    _hash[:8],
                    checkpoint.job_id,
                )
                return checkpoint

        return None

    @property
    def metastore(self):
        return self.session.catalog.metastore

    @property
    def warehouse(self):
        return self.session.catalog.warehouse

    def get_or_create_input_table(
        self, query: Select, _hash: str, job: "Job | None"
    ) -> "Table":
        """
        Get or create input table for the given hash.

        Uses run_group_id for table naming so all jobs in the same run group
        share the same input table.

        Returns the input table.
        """
        group_id = (job.run_group_id or job.id) if job else str(uuid4())
        input_table_name = Checkpoint.input_table_name(group_id, _hash)

        # Check if input table already exists (created by ancestor job)
        if self.warehouse.db.has_table(input_table_name):
            return self.warehouse.get_table(input_table_name)

        # Create input table from original query
        return self.warehouse.create_pre_udf_table(query, input_table_name)

    def _get_partition_table_for_continue(
        self, checkpoint: Checkpoint, _hash: str, job: Job
    ) -> "Table":
        """
        Get partition table for continue flow.
        Same job: reuses existing partition table directly.
        Different job: copies partition table to current job.
        Raises TableMissingError if source partition table not found.
        """
        source_table = self.warehouse.get_table(
            Checkpoint.partition_table_name(checkpoint.job_id, _hash)
        )

        if checkpoint.job_id == job.id:
            return source_table

        # Different job: copy to current job
        return self.warehouse.create_table_from_query(
            Checkpoint.partition_table_name(job.id, _hash),
            sa.select(source_table),
            create_fn=lambda name: self.warehouse.create_udf_table(
                partition_columns(), name=name
            ),
            preserve_sys_ids=True,
        )

    def _prepare_partition_query(
        self,
        query: Select,
        hash_input: str,
        temp_tables: list[str],
        job: "Job | None" = None,
        ch_partial: "Checkpoint | None" = None,
        _continue: bool = False,
    ) -> tuple[Select, bool]:
        """Create input and partition tables for partition_by.

        Returns (updated_query, continue_flag). The continue flag may be set
        to False if the parent partition table is missing and we fall back
        to creating a fresh one.
        """
        input_table = self.get_or_create_input_table(query, hash_input, job)
        query = self.get_input_query(input_table.name, query)

        if _continue:
            assert ch_partial
            assert job
            try:
                partition_tbl = self._get_partition_table_for_continue(
                    ch_partial, hash_input, job
                )
            except TableMissingError:
                logger.warning(
                    "UDF(%s) [job=%s run_group=%s]: Parent partition table "
                    "not found. Running from scratch.",
                    self._udf_name,
                    self._job_id_short(job),
                    self._run_group_id_short(job),
                )
                _continue = False
                partition_tbl = self.create_partitions_table(
                    query, Checkpoint.partition_table_name(job.id, hash_input)
                )
        elif job:
            partition_tbl = self.create_partitions_table(
                query, Checkpoint.partition_table_name(job.id, hash_input)
            )
        else:
            partition_tbl = self.create_partitions_table(query)
            temp_tables.append(partition_tbl.name)

        # On SQLite, the partition table's index on (sys__partition_id, rowid)
        # lets each input page seek to its first row instead of sorting. The
        # alias is required on ClickHouse when continuing from a reflected
        # partition table (test_aggregator_continue_from_partial_same_job).
        part = partition_tbl.alias()
        result_query = query.join(
            part,
            part.c.sys__id == query.selected_columns.sys__id,
        ).with_only_columns(
            *(
                part.c.sys__id.label("sys__id") if c.name == "sys__id" else c
                for c in query.selected_columns
            ),
            part.c[PARTITION_COLUMN_ID],
        )
        return result_query, _continue

    def apply(
        self,
        query_generator: QueryGenerator,
        temp_tables: list[str],
        state: "QueryState",
        *,
        hash_input: str = "",
        hash_output: str = "",
        checkpoints_enabled: bool = True,
        **kwargs,
    ) -> "StepResult":
        query = query_generator.select()
        if PARTITION_COLUMN_ID in query.selected_columns:
            raise DataChainError(
                f"Column '{PARTITION_COLUMN_ID}' is reserved for internal partitioning"
            )

        # Calculate partial hash using UDF identity (name + params + output schema)
        # and partition_by columns. This allows continuing from partial when only
        # code changes (bug fix), but distinguishes different UDFs and forces
        # re-run when schema or partitioning changes.
        partition_by = ensure_sequence(self.partition_by or [])
        partial_hash = hashlib.sha256(
            (
                hash_input
                + self.udf.hash(include_body=False)
                + hash_column_elements(partition_by)
                + (PARTITION_COLUMN_ID if partition_by else "")
            ).encode()
            + UDF_OUTPUT_FORMAT
        ).hexdigest()

        if not checkpoints_enabled:
            if self.partition_by is not None:
                query, _ = self._prepare_partition_query(
                    query,
                    hash_input,
                    temp_tables,
                )
            output_table, input_table = self._run_from_scratch(
                partial_hash,
                hash_output,
                hash_input,
                query,
                None,
                checkpoints_enabled=False,
            )
            # No checkpoints — tables won't be reused, register for cleanup
            temp_tables.append(output_table.name)
            temp_tables.append(input_table.name)
        else:
            job = self.session.get_or_create_job()

            ch = self._find_udf_checkpoint(job, hash_output)
            ch_partial = (
                self._find_udf_checkpoint(job, partial_hash) if not ch else None
            )

            _skip = bool(ch)
            _continue = bool(
                not _skip and ch_partial and not self.is_single_partition_aggregator
            )

            if self.partition_by is not None and not _skip:
                query, _continue = self._prepare_partition_query(
                    query,
                    hash_input,
                    temp_tables,
                    job=job,
                    ch_partial=ch_partial,
                    _continue=_continue,
                )

            if _skip:
                assert ch
                try:
                    output_table, input_table = self._skip_udf(
                        ch, hash_input, query, job
                    )
                except TableMissingError:
                    logger.warning(
                        "UDF(%s) [job=%s run_group=%s]: Output table not found for "
                        "checkpoint %s. Running UDF from scratch.",
                        self._udf_name,
                        self._job_id_short(job),
                        self._run_group_id_short(job),
                        ch,
                    )
                    output_table, input_table = self._run_from_scratch(
                        partial_hash, ch.hash, hash_input, query, job
                    )
            elif _continue:
                assert ch_partial
                output_table, input_table = self._continue_udf(
                    ch_partial, hash_output, hash_input, query, job
                )
            else:
                output_table, input_table = self._run_from_scratch(
                    partial_hash, hash_output, hash_input, query, job
                )

        # Create result query from output table
        input_query = self.get_input_query(input_table.name, query)
        q, cols = self.create_result_query(output_table, input_query)
        return step_result(q, cols)

    def _skip_udf(
        self, checkpoint: Checkpoint, hash_input: str, query, job: Job
    ) -> tuple["Table", "Table"]:
        """
        Skip UDF by reusing existing output table from checkpoint.
        The checkpoint's table is used directly — no copy, no new checkpoint
        record. "Done" checkpoints act as a cache keyed by hash.
        Returns (output_table, input_table).
        """
        logger.debug(
            "UDF(%s) [job=%s run_group=%s]: Skipping execution, "
            "reusing output from job_id=%s",
            self._udf_name,
            self._job_id_short(job),
            self._run_group_id_short(job),
            checkpoint.job_id,
        )
        output_table = self._output_table(
            Checkpoint.output_table_name(checkpoint.job_id, checkpoint.hash)
        )

        input_table = self.get_or_create_input_table(query, hash_input, job)

        # Log checkpoint event with row counts
        rows_input = self.warehouse.table_rows_count(input_table)
        output_rows_reused = self.warehouse.table_rows_count(output_table)
        self._log_event(
            job,
            CheckpointEventType.UDF_SKIPPED,
            checkpoint_hash=checkpoint.hash,
            hash_input=hash_input,
            hash_output=checkpoint.hash,
            rerun_from_job_id=checkpoint.job_id,
            rows_input=rows_input,
            rows_processed=0,
            rows_output=0,
            rows_input_reused=rows_input,
            rows_output_reused=output_rows_reused,
        )

        # Register skipped UDF in the registry (no-op for local metastores)
        self.metastore.add_udf(
            udf_id=str(uuid4()),
            name=self._udf_name,
            status="DONE",
            rows_total=rows_input,
            job_id=job.id,
            tasks_created=0,
            skipped=True,
            rows_reused=rows_input,
            output_rows_reused=output_rows_reused,
        )

        return output_table, input_table

    def _output_table(self, name: str) -> "Table":
        """An existing output table with our SQLTypes over the reflected ones."""
        # Reflection returns generic backend types (e.g. ClickHouse String gives
        # bytes instead of str, DateTime defaults break CREATE TABLE downstream).
        # Restore SQLTypes from our known schema so downstream query generation
        # decodes values correctly.
        known_types = {
            **dict(self.udf.output.items()),
            **{c.name: c.type for c in self.warehouse.dataset_row_cls.sys_columns()},
        }
        return coerce_table_types(self.warehouse.get_table(name), known_types)

    def _resumable_in_parts(self) -> bool:
        """Whether the output gets a ledger: gen or agg with partition_by.
        A map output row is keyed by its input row, so the table is its own
        record; an agg without partition_by is one part and reruns whole."""
        return self.is_generator and not self.is_single_partition_aggregator

    def _promote_partial(
        self, partial_table: "Table", parts: "Table | None", final_name: str
    ) -> "Table":
        if parts is not None:
            self.warehouse.trim_to_ledger(partial_table, parts)
        # Same hash may already have a final table from earlier in this job
        self.warehouse.db.drop_table(
            sa.Table(final_name, sa.MetaData()), if_exists=True
        )
        output_table = self.warehouse.rename_table(partial_table, final_name)
        if parts is not None:
            self.warehouse.db.drop_table(parts, if_exists=True)
        return output_table

    def _run_from_scratch(
        self,
        partial_hash: str,
        hash_output: str,
        hash_input: str,
        query,
        job: "Job | None",
        checkpoints_enabled: bool = True,
    ) -> tuple["Table", "Table"]:
        """Execute UDF from scratch. Returns (output_table, input_table)."""
        run_id = job.id if job else str(uuid4())  # unique ID for table naming

        logger.debug(
            "UDF(%s) [job=%s run_group=%s]: Running from scratch",
            self._udf_name,
            self._job_id_short(job),
            self._run_group_id_short(job),
        )

        partial_checkpoint = None
        if checkpoints_enabled and job:
            partial_checkpoint = self.metastore.get_or_create_checkpoint(
                job.id, partial_hash, partial=True
            )
            logger.debug(
                "UDF(%s) [job=%s run_group=%s]: Created partial checkpoint hash=%s",
                self._udf_name,
                self._job_id_short(job),
                self._run_group_id_short(job),
                partial_hash[:8],
            )

        input_table = self.get_or_create_input_table(query, hash_input, job)

        partial_table_name = Checkpoint.partial_output_table_name(run_id, partial_hash)
        parts_name = parts_table_name(partial_table_name)
        # Drop stale partial tables from earlier chain with same hash in this job
        for name in (partial_table_name, parts_name):
            self.warehouse.db.drop_table(sa.Table(name, sa.MetaData()), if_exists=True)
        partial_output_table = self.create_output_table(partial_table_name)
        parts = None
        if partial_checkpoint and self._resumable_in_parts():
            parts = self.warehouse.create_parts_table(parts_name)

        if self.partition_by is not None:
            input_query = query
        else:
            input_query = self.get_input_query(input_table.name, query)

        self.populate_udf_output_table(partial_output_table, input_query, parts=parts)

        output_table = self._promote_partial(
            partial_output_table,
            parts,
            Checkpoint.output_table_name(run_id, hash_output),
        )

        if checkpoints_enabled and job:
            rows_input = self.warehouse.table_rows_count(input_table)
            self._finish_checkpoint(
                job,
                partial_checkpoint,
                output_table,
                CheckpointEventType.UDF_FROM_SCRATCH,
                hash_input=hash_input,
                hash_output=hash_output,
                rows_input=rows_input,
                rows_processed=rows_input,
                rows_input_reused=0,
                rows_output_reused=0,
            )

        return output_table, input_table

    def _continue_udf(
        self, checkpoint: Checkpoint, hash_output: str, hash_input: str, query, job: Job
    ) -> tuple["Table", "Table"]:
        """
        Continue UDF from partial checkpoint. The checkpoint may be from the
        current job (e.g. try/except retry) or a previous job (rerun).

        Same job: appends directly to existing partial table, no copy.
        Different job: copies partial table to current job first.

        gen and partitioned agg resume from the ledger: rows outside its
        committed ranges are deleted, inputs whose part is committed are
        skipped, and the rest are written under a fresh writer_no.

        Returns (output_table, input_table).
        """
        is_same_job = checkpoint.job_id == job.id

        logger.debug(
            "UDF(%s) [job=%s run_group=%s]: Continuing from partial checkpoint, "
            "source_job_id=%s%s",
            self._udf_name,
            self._job_id_short(job),
            self._run_group_id_short(job),
            checkpoint.job_id,
            " (same job)" if is_same_job else "",
        )

        if is_same_job:
            # Partial checkpoint already exists for this job — reuse it
            partial_checkpoint = checkpoint
        else:
            partial_checkpoint = self.metastore.get_or_create_checkpoint(
                job.id, checkpoint.hash, partial=True
            )

        input_table = self.get_or_create_input_table(query, hash_input, job)

        # Locate the source partial table using the checkpoint's job_id
        source_name = Checkpoint.partial_output_table_name(
            checkpoint.job_id, checkpoint.hash
        )
        partial_name = (
            f"{source_name}_{uuid4().hex[:8]}"
            if is_same_job
            else Checkpoint.partial_output_table_name(job.id, checkpoint.hash)
        )
        parts = None
        try:
            source_table = self._output_table(source_name)
            # a gen/agg partial table without its ledger is never trusted:
            # TableMissingError makes it run again from scratch
            source_parts = (
                self.warehouse.get_table(parts_table_name(source_name))
                if self.is_generator
                else None
            )
            # Copy to a fresh table even in the same job so any leftover worker
            # from the failed attempt writes only into the old table.
            # The source may still be running: copy its ledger first. Rows of
            # every part in that copy are already durable, so the data copy taken
            # after it holds them; anything newer is trimmed as uncommitted.
            if source_parts is not None:
                parts = self.warehouse.create_table_from_query(
                    parts_table_name(partial_name),
                    sa.select(source_parts),
                    create_fn=self.warehouse.create_parts_table,
                    preserve_sys_ids=True,
                )
            partial_table = self.warehouse.create_table_from_query(
                partial_name,
                sa.select(source_table),
                create_fn=self.create_output_table,
                preserve_sys_ids=True,
            )
        except Exception:
            if parts is not None:
                self.warehouse.db.drop_table(parts, if_exists=True)
            if self.warehouse.db.has_table(source_name) and (
                not self.is_generator
                or self.warehouse.db.has_table(parts_table_name(source_name))
            ):
                raise
            if ch := self._find_udf_checkpoint(job, hash_output):
                try:
                    return self._skip_udf(ch, hash_input, query, job)
                except TableMissingError:
                    pass
            logger.warning(
                "UDF(%s) [job=%s run_group=%s]: Partial table not found for "
                "checkpoint %s, falling back to run from scratch",
                self._udf_name,
                self._job_id_short(job),
                self._run_group_id_short(job),
                checkpoint,
            )
            return self._run_from_scratch(
                checkpoint.hash, hash_output, hash_input, query, job
            )

        if parts is not None:
            # Drop any ledger commit whose rows were removed by source duplicate
            # arbitration between the ledger copy and the data copy.
            ids = sa.select(partial_table.c.sys__id)
            self.warehouse.db.execute(
                parts.delete().where(
                    parts.c.part_key.is_not(None),
                    parts.c.lo <= parts.c.hi,
                    ~(parts.c.lo.in_(ids) & parts.c.hi.in_(ids)),
                )
            )
            self.warehouse.trim_to_ledger(partial_table, parts)
            processed = committed_part_keys(parts)
        else:
            processed = sa.select(partial_table.c.sys__id)

        input_query = self.get_input_query(input_table.name, query)

        # For aggregators with partition_by, use `query` which already has
        # partition columns joined (from _prepare_partition_query).
        # `input_query` is rebuilt from the input table and lacks partition_id.
        unprocessed_input = query if self.partition_by is not None else input_query

        try:
            # The anti-join must stay fixed while resumed outputs are appended.
            with self._snapshot_processed_input_ids(processed) as processed_ids:
                unprocessed_query = self.calculate_unprocessed_rows(
                    unprocessed_input, processed_ids
                )

                # Count rows before populating with new rows
                output_rows_reused = self.warehouse.table_rows_count(partial_table)
                rows_input = self.warehouse.table_rows_count(input_table)
                rows_to_process = self.warehouse.query_count(unprocessed_query)
                rows_reused = rows_input - rows_to_process  # input rows reused

                self.populate_udf_output_table(
                    partial_table,
                    unprocessed_query,
                    continued=True,
                    rows_reused=rows_reused,
                    output_rows_reused=output_rows_reused,
                    rows_total=rows_input,
                    parts=parts,
                )
        except BaseException:
            if is_same_job:
                self.warehouse.db.drop_table(source_table, if_exists=True)
                self.warehouse.rename_table(partial_table, source_name)
                if parts is not None and source_parts is not None:
                    self.warehouse.db.drop_table(source_parts, if_exists=True)
                    self.warehouse.rename_table(parts, parts_table_name(source_name))
            raise

        if is_same_job:
            self.warehouse.db.drop_table(source_table, if_exists=True)
            if source_parts is not None:
                self.warehouse.db.drop_table(source_parts, if_exists=True)

        output_table = self._promote_partial(
            partial_table, parts, Checkpoint.output_table_name(job.id, hash_output)
        )

        self._finish_checkpoint(
            job,
            partial_checkpoint,
            output_table,
            CheckpointEventType.UDF_CONTINUED,
            hash_partial=checkpoint.hash,
            hash_input=hash_input,
            hash_output=hash_output,
            rerun_from_job_id=checkpoint.job_id,
            rows_input=rows_input,
            rows_processed=rows_to_process,
            rows_input_reused=rows_reused,
            rows_output_reused=output_rows_reused,
        )

        return output_table, input_table

    def _finish_checkpoint(
        self,
        job: Job,
        partial_checkpoint: Checkpoint | None,
        output_table: "Table",
        event_type: CheckpointEventType,
        *,
        hash_output: str,
        rows_output_reused: int,
        **event,
    ) -> None:
        """Promote the step's partial checkpoint to final and log its event."""
        if partial_checkpoint:
            self.metastore.update_checkpoints(
                [partial_checkpoint.id], status=CheckpointStatus.DELETED
            )
        self.metastore.get_or_create_checkpoint(job.id, hash_output)
        logger.debug(
            "UDF(%s) [job=%s run_group=%s]: Promoted partial to final, hash=%s",
            self._udf_name,
            self._job_id_short(job),
            self._run_group_id_short(job),
            hash_output[:8],
        )
        rows_output = self.warehouse.table_rows_count(output_table)
        self._log_event(
            job,
            event_type,
            checkpoint_hash=hash_output,
            hash_output=hash_output,
            rows_output=rows_output - rows_output_reused,
            rows_output_reused=rows_output_reused,
            **event,
        )

    @contextlib.contextmanager
    def _snapshot_processed_input_ids(self, processed: Select):
        snapshot = self.warehouse.create_pre_udf_table(
            processed.with_only_columns(
                processed.selected_columns[0].label("sys__processed_id")
            ),
            self.warehouse.temp_table_name(),
        )
        try:
            yield snapshot
        finally:
            self.warehouse.db.drop_table(snapshot, if_exists=True)

    def calculate_unprocessed_rows(
        self, input_query: Select, processed_ids: "Table"
    ) -> Select:
        """The input rows, or for partition_by the partitions, not processed yet."""
        key = "sys__id" if self.partition_by is None else PARTITION_COLUMN_ID
        return input_query.where(
            input_query.selected_columns[key].notin_(
                sa.select(processed_ids.c.sys__processed_id)
            )
        )


@frozen
class UDFSignal(UDFStep):
    udf: "UDFAdapter"
    session: "Session"
    partition_by: PartitionByType | None = None
    is_generator = False
    # Parameters from Settings
    cache: bool = False
    parallel: int | None = None
    workers: bool | int = False
    min_task_size: int | None = None
    batch_size: int | None = None

    @property
    def _step_type(self) -> CheckpointStepType:
        return CheckpointStepType.UDF_MAP

    def create_output_table(self, name: str) -> "Table":
        columns: list[sqlalchemy.Column[Any]] = [
            sqlalchemy.Column(col_name, col_type)
            for (col_name, col_type) in self.udf.output.items()
        ]
        return self.warehouse.create_udf_table(columns, name=name)

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

    bounds_columns: ClassVar[bool] = True
    udf: "UDFAdapter"
    session: "Session"
    partition_by: PartitionByType | None = None
    is_generator = True
    is_aggregator: bool = False
    # Parameters from Settings
    cache: bool = False
    parallel: int | None = None
    workers: bool | int = False
    min_task_size: int | None = None
    batch_size: int | None = None

    @property
    def _step_type(self) -> CheckpointStepType:
        return CheckpointStepType.UDF_GEN

    def create_output_table(self, name: str) -> "Table":
        columns: list[Column] = [
            Column(name, typ) for name, typ in self.udf.output.items()
        ]
        return self.warehouse.create_dataset_rows_table(
            name,
            columns=tuple(columns),
            if_not_exists=True,
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

        return q, list(udf_table_query.columns)


@frozen
class SQLClause(Step, ABC):
    resets_query_state_after_apply: ClassVar[bool] = False

    def apply(
        self,
        query_generator: QueryGenerator,
        temp_tables: list[str],
        state: "QueryState",
        **kwargs,
    ) -> StepResult:
        if self.requires_boundary(state):
            query_generator = state.emit_boundary(query_generator)

        query = query_generator.select()
        new_query = self.apply_sql_clause(query)
        new_query = state.warehouse.normalize_limit_offset(new_query)
        self.update_query_state(state)

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

    def requires_boundary(self, state: "QueryState") -> bool:
        return False

    def update_query_state(self, state: "QueryState") -> None:
        # SELECT / MUTATE / SELECT_EXCEPT / COUNT etc. all wrap the prior
        # result in a fresh subquery, so any prior state flags no longer apply.
        state.reset()


@frozen
class RegenerateSystemColumns(Step):
    # This step resets state conditionally inside apply(): preserve pending flags
    # for true no-ops, but reset when system-column regeneration wraps the query.
    resets_query_state_after_apply: ClassVar[bool] = False

    catalog: "Catalog"

    def hash_inputs(self) -> str:
        return hashlib.sha256(b"regenerate_system_columns").hexdigest()

    def apply(
        self,
        query_generator: QueryGenerator,
        temp_tables: list[str],
        state: "QueryState",
        **kwargs,
    ) -> StepResult:
        query = query_generator.select()
        new_query = self.catalog.warehouse._regenerate_system_columns(
            query, keep_existing_columns=True
        )
        if new_query is not query:
            state.reset()

        def q(*columns):
            return new_query.with_only_columns(*columns)

        return step_result(q, new_query.selected_columns)


_SYS = frozenset({"sys__id", "sys__rand"})


@frozen
class SQLSelect(SQLClause):
    bounds_columns: ClassVar[bool] = True
    args: tuple[Function | ColumnElement, ...]
    implicit_sys: bool = False

    def hash_inputs(self) -> str:
        return hash_column_elements(self.args)

    def used_columns(self) -> set[str] | None:
        if not self.args:
            return None
        cols = _expr_columns(self.args)
        return cols - _SYS if cols is not None and self.implicit_sys else cols

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

    def used_columns(self) -> set[str] | None:
        return _expr_columns(self.args)

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

    def used_columns(self) -> set[str] | None:
        return _expr_columns(self.expressions)

    def __and__(self, other):
        expressions = self.parse_cols(self.expressions)
        return self.__class__(expressions + other)

    def apply_sql_clause(self, query: Select) -> Select:
        expressions = self.parse_cols(self.expressions)
        return query.filter(*_bind_shared_names(expressions, query))

    def requires_boundary(self, state: "QueryState") -> bool:
        # WHERE applied after GROUP BY / LIMIT / OFFSET / DISTINCT would filter
        # the underlying rows, not the materialized state output.
        return state.has_finalizing_clause

    def update_query_state(self, state: "QueryState") -> None:
        # filter does not finalize the output; state flags carry over unless a
        # boundary was emitted first, in which case QueryState is already reset.
        pass


@frozen
class SQLOrderBy(SQLClause):
    args: tuple[Function | ColumnElement, ...]

    def hash_inputs(self) -> str:
        return hash_column_elements(self.args)

    def used_columns(self) -> set[str] | None:
        return _expr_columns(self.args)

    def apply_sql_clause(self, query: Select) -> Select:
        args = self.parse_cols(self.args)
        return query.order_by(*_bind_shared_names(args, query))

    def requires_boundary(self, state: "QueryState") -> bool:
        # Re-ordering after LIMIT/OFFSET would change which rows are kept.
        return state.limited or state.offset_applied

    def update_query_state(self, state: "QueryState") -> None:
        # order_by does not finalize the output; state flags carry over unless
        # a boundary was emitted first.
        pass


@frozen
class SQLLimit(SQLClause):
    n: int

    def hash_inputs(self) -> str:
        return hashlib.sha256(str(self.n).encode()).hexdigest()

    def used_columns(self) -> set[str]:
        return set()

    def apply_sql_clause(self, query: Select) -> Select:
        return query.limit(self.n)

    def requires_boundary(self, state: "QueryState") -> bool:
        # Stacked LIMITs must materialize first. (LIMIT after OFFSET combines
        # into a single ``LIMIT N OFFSET M`` clause, so no boundary needed.)
        return state.limited

    def update_query_state(self, state: "QueryState") -> None:
        state.limited = True


@frozen
class SQLOffset(SQLClause):
    offset: int

    def hash_inputs(self) -> str:
        return hashlib.sha256(str(self.offset).encode()).hexdigest()

    def used_columns(self) -> set[str]:
        return set()

    def apply_sql_clause(self, query: GenerativeSelect) -> GenerativeSelect:
        return query.offset(self.offset)

    def requires_boundary(self, state: "QueryState") -> bool:
        # Re-applying OFFSET requires materializing the previous offset.
        # (OFFSET after LIMIT combines into ``LIMIT N OFFSET M``, no boundary.)
        return state.offset_applied

    def update_query_state(self, state: "QueryState") -> None:
        state.offset_applied = True


@frozen
class SQLCount(SQLClause):
    bounds_columns: ClassVar[bool] = True

    def hash_inputs(self) -> str:
        return ""

    def used_columns(self) -> set[str]:
        return set()

    def apply_sql_clause(self, query):
        return sqlalchemy.select(f.count(1)).select_from(query.subquery())


@frozen
class SQLDistinct(SQLClause):
    args: tuple[ColumnElement, ...]
    dialect: str

    def hash_inputs(self) -> str:
        return hash_column_elements(self.args)

    def used_columns(self) -> set[str] | None:
        return _expr_columns(self.args) if self.args else None

    def apply_sql_clause(self, query):
        args = _bind_shared_names(self.args, query)
        if self.dialect == "sqlite":
            return query.group_by(*args)

        return query.distinct(*args)

    def requires_boundary(self, state: "QueryState") -> bool:
        # DISTINCT after GROUP BY / LIMIT / OFFSET should de-duplicate the
        # materialized rows, not the underlying ones. Re-distinct also needs
        # a boundary.
        return state.grouped or state.limited or state.offset_applied or state.distinct

    def update_query_state(self, state: "QueryState") -> None:
        state.distinct = True


@frozen
class SQLUnion(Step):
    query1: "DatasetQuery"
    query2: "DatasetQuery"

    def hash_inputs(self) -> str:
        return hashlib.sha256(
            bytes.fromhex(self.query1.hash()) + bytes.fromhex(self.query2.hash())
        ).hexdigest()

    def collect_delta_sources(
        self,
    ) -> list["DatasetQuery"]:
        return [*self.query1.delta_sources(), *self.query2.delta_sources()]

    @property
    def sub_queries(self) -> list["DatasetQuery"]:
        return [self.query1, self.query2]

    def replace_sources(
        self,
        replacements: dict[int, "DatasetQuery"],
    ) -> "Step":
        return attrs.evolve(
            self,
            query1=self.query1.replace_sources(replacements),
            query2=self.query2.replace_sources(replacements),
        )

    def apply(
        self,
        query_generator: QueryGenerator,
        temp_tables: list[str],
        state: "QueryState",
        **kwargs,
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


def _as_nullable_scalar(column: Any) -> Any:
    """Re-type a join column nullable so an outer-join NULL survives instead of
    coercing to the type default. This applies to a model's leaves too: an
    unmatched row's leaves then read back NULL, so the model hydrates as None
    (via the all-values-None fallback) rather than a default-filled instance."""
    col_type = getattr(column, "type", None)
    if not isinstance(col_type, SQLType) or getattr(col_type, "dc_nullable", False):
        return column
    try:
        py_type = col_type.python_type
    except NotImplementedError:
        return column
    if py_type not in NULLABLE_SCALARS:
        return column
    nullable_type = SQLType.as_nullable(col_type)
    return sqlalchemy.type_coerce(column, nullable_type).label(column.name)


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

    def collect_delta_sources(
        self,
    ) -> list["DatasetQuery"]:
        return [*self.query1.delta_sources(), *self.query2.delta_sources()]

    @property
    def sub_queries(self) -> list["DatasetQuery"]:
        return [self.query1, self.query2]

    def replace_sources(
        self,
        replacements: dict[int, "DatasetQuery"],
    ) -> "Step":
        return attrs.evolve(
            self,
            query1=self.query1.replace_sources(replacements),
            query2=self.query2.replace_sources(replacements),
        )

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
        table = warehouse.create_dataset_rows_table(
            warehouse.temp_table_name(), columns=columns
        )
        temp_tables.append(table.name)
        warehouse.insert_into(table, query)
        return table.select().subquery(dq.table.name)

    def bind_and_validate_expression(self, exp: ColumnElement, q1, q2) -> ColumnElement:
        left_name = self.query1.table.name
        right_name = self.query2.table.name

        def replace(element, **_kwargs):
            if not isinstance(element, ColumnClause):
                return None

            table = element.table
            if not isinstance(table, TableClause):
                raise TypeError(
                    "Join predicate contains a column without a supported table "
                    f"binding: {element.name}"
                )

            table_name = table.name

            if table_name == left_name:
                bound_col = q1.c.get(element.name)
                if bound_col is None:
                    raise ValueError(
                        f"Column {element.name} was not found in left part of the join"
                    )
                return bound_col

            if table_name == right_name:
                bound_col = q2.c.get(element.name)
                if bound_col is None:
                    raise ValueError(
                        f"Column {element.name} was not found in right part of the join"
                    )
                return bound_col

            raise ValueError(
                f"Column {element.name} was not found in left or right part of the join"
            )

        bound = replacement_traverse(exp, {}, replace)
        assert isinstance(bound, ColumnElement)
        return bound

    def _map_right_columns(
        self,
        right_columns: "list[KeyedColumnElement[Any]]",
        root_mapping: dict[str, str],
        nullable: bool,
    ) -> "list[KeyedColumnElement[Any]]":
        mapped: list[KeyedColumnElement[Any]] = []
        for column in right_columns:
            original_name = column.name
            column_root, column_tail = self._split_db_name(original_name)
            mapped_root = root_mapping[column_root]
            new_name = (
                mapped_root
                if not column_tail
                else DEFAULT_DELIMITER.join([mapped_root, column_tail])
            )
            if nullable:
                column = _as_nullable_scalar(column)
            if column.name != new_name:
                column = column.label(new_name)
            mapped.append(column)
        return mapped

    def apply(
        self,
        query_generator: QueryGenerator,
        temp_tables: list[str],
        state: "QueryState",
        **kwargs,
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

        # right side nullable for a left join, both sides for a full join
        q2_columns = self._map_right_columns(
            right_columns, root_mapping, nullable=not self.inner
        )
        if self.full:
            q1_columns = [_as_nullable_scalar(c) for c in q1_columns]

        res_columns = q1_columns + q2_columns
        predicates = (
            (self.predicates,)
            if not isinstance(self.predicates, tuple)
            else self.predicates
        )

        def bind_same_name_predicate(column_name: str) -> ColumnElement:
            left_col = q1.c.get(column_name)
            right_col = q2.c.get(column_name)

            if left_col is None and right_col is None:
                raise ValueError(
                    f"Column {column_name} was not found in left or right part "
                    "of the join"
                )
            if left_col is None:
                raise ValueError(
                    f"Column {column_name} was not found in left part of the join"
                )
            if right_col is None:
                raise ValueError(
                    f"Column {column_name} was not found in right part of the join"
                )

            return left_col == right_col

        expressions = []
        for p in predicates:
            if isinstance(p, ColumnClause):
                expressions.append(bind_same_name_predicate(p.name))
            elif isinstance(p, str):
                expressions.append(bind_same_name_predicate(p))
            elif isinstance(p, ColumnElement):
                expressions.append(self.bind_and_validate_expression(p, q1, q2))
            else:
                raise TypeError(f"Unsupported predicate {p} for join expression")

        if not expressions:
            raise ValueError("Missing predicates")

        join_expression = sqlalchemy.and_(*expressions)

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
    bounds_columns: ClassVar[bool] = True
    cols: Sequence[str | Function | ColumnElement]
    group_by: Sequence[str | Function | ColumnElement]

    def hash_inputs(self) -> str:
        return hashlib.sha256(
            bytes.fromhex(
                hash_column_elements(self.cols) + hash_column_elements(self.group_by)
            )
        ).hexdigest()

    def used_columns(self) -> set[str] | None:
        return _expr_columns((*self.cols, *self.group_by))

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
                key = c.name if isinstance(c, C) else str(c)  # str() quotes names
                value = subquery.c[key]
            else:
                key = c.name
                value = c  # type: ignore[assignment]
            cols_dict[key] = value

        unique_cols = cols_dict.values()

        return sqlalchemy.select(*unique_cols).select_from(subquery).group_by(*group_by)

    def update_query_state(self, state: "QueryState") -> None:
        # SQLGroupBy.apply_sql_clause() selects from query.subquery(), so any
        # prior finalized output is already folded into that inner selectable.
        state.reset()
        state.grouped = True


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

    def close(self) -> None:
        if hasattr(self._row_iter, "close"):
            self._row_iter.close()  # type: ignore[attr-defined]


def _expr_columns(exprs: Iterable[Any]) -> set[str] | None:
    names: set[str] = set()
    for expr in exprs:
        if isinstance(expr, str):
            names.add(expr)
            continue
        if isinstance(expr, Function):
            expr = expr.get_column()
        count_star: set[int] = set()
        for elem in iterate(expr):
            if isinstance(elem, sa_count):
                # count(*) names no column.
                count_star.update(
                    id(c)
                    for c in elem.clauses
                    if isinstance(c, ColumnClause) and c.is_literal and c.name == "*"
                )
            # Safety guard: raw SQL text or literal clauses may reference any column.
            if isinstance(elem, TextClause):
                return None
            if isinstance(elem, ColumnClause):
                if id(elem) in count_star:
                    continue
                if elem.is_literal:
                    return None
                names.add(elem.name)
            elif isinstance(name := getattr(elem, "element", None), str):
                # Window partition_by / order_by specs wrap the column name in .element.
                names.add(name)
    return names


def _used_columns(steps: Sequence[Step]) -> set[str] | None:
    """Return the DB column names used before the first step that bounds
    the input columns, or None when every column must be kept."""
    if not any(s.bounds_columns for s in steps):
        return None
    used: set[str] = set()
    bounded = False
    for step in steps:
        if not bounded and isinstance(step, UDFSignal):
            return None
        cols = step.used_columns()
        if cols is None or "sys__rand" in cols:
            return None
        if not bounded or isinstance(step, (SQLFilter, SQLOrderBy, SQLDistinct)):
            # Clauses without a subquery boundary still see columns the select dropped.
            used |= cols
        bounded = bounded or step.bounds_columns
    return used


def _has_udf_checkpoint(step: RowGenerator, hasher: Any) -> bool:
    h = hasher.copy()
    h.update(step.hash().encode("utf-8"))
    return bool(
        step._find_udf_checkpoint(step.session.get_or_create_job(), h.hexdigest())
    )


def _project_step(
    step: Step,
    later_steps: Sequence[Step],
    hasher: Any,
    checkpoints_enabled: bool,
) -> Step:
    if (
        isinstance(step, RowGenerator)
        and getattr(step.udf.inner, "projectable", False)
        and (used := _used_columns(later_steps)) is not None
        and (projected := step.udf.inner.project(used)) is not None
    ):
        assert projected.output is not None
        udf = attrs.evolve(
            step.udf, inner=projected, output=projected.output.to_udf_spec()
        )
        narrow = attrs.evolve(step, udf=udf)
        # Keep the read this chain already ran on, so the steps after it stay cached.
        if (
            checkpoints_enabled
            and not _has_udf_checkpoint(narrow, hasher)
            and _has_udf_checkpoint(step, hasher)
        ):
            return step
        return narrow
    return step


class DatasetQuery:
    def __init__(
        self,
        name: str | None = None,
        version: str | None = None,
        project_name: str | None = None,
        namespace_name: str | None = None,
        catalog: "Catalog | None" = None,
        session: Session | None = None,
        in_memory: bool = False,
        update: bool = False,
        include_incomplete: bool = False,
        steps: Sequence[Step] = (),
        checkpoints_enabled: bool = True,
    ) -> None:
        self.session = Session.get(session, catalog=catalog, in_memory=in_memory)
        if catalog is not None and catalog is not self.session.catalog:
            self.session = self.session._with_catalog(catalog)
        self.catalog = self.session.catalog
        self.steps: list[Step] = list(steps)
        self._chunk_index: int | None = None
        self._chunk_total: int | None = None
        self.temp_table_names: list[str] = []
        self.dependencies: set[DatasetDependencyType] = set()
        self.table = self.get_table()
        self.starting_step: QueryStep | None = None
        self.delta_spec: DeltaSpec | None = None
        self.name: str | None = None
        self.version: str | None = None
        self.feature_schema: dict | None = None
        self.column_types: dict[str, Any] | None = None
        self.before_steps: list[Callable] = []
        self.listing_fn: Callable | None = None
        self.update = update
        self.checkpoints_enabled = checkpoints_enabled

        self.list_ds_name: str | None = None
        self.listing_ds: list[DatasetRecord] = []  # shared with clones
        self.dialect = self.catalog.warehouse.db.dialect

        if name is None:
            return

        self._init_rooted_query(
            name,
            version=version,
            project_name=project_name,
            namespace_name=namespace_name,
            update=update,
            include_incomplete=include_incomplete,
        )

    def _init_rooted_query(
        self,
        name: str,
        version: str | None = None,
        project_name: str | None = None,
        namespace_name: str | None = None,
        update: bool = False,
        include_incomplete: bool = False,
    ) -> None:
        self.name = name
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

    @property
    def _starting_step_hash(self) -> str:
        if self.starting_step:
            return self.starting_step.hash()
        if self.list_ds_name:
            raise RuntimeError(
                "Cannot compute hash: listing not resolved. "
                "Call resolve_listing() first."
            )
        return ""

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

        hasher.update(self._starting_step_hash.encode("utf-8"))
        if self.delta_spec is not None:
            hasher.update(self.delta_spec.hash().encode("utf-8"))

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

    def resolve_listing(self) -> None:
        """Runs listing pre-step if needed"""
        if self.list_ds_name and not self.starting_step:
            if not self.listing_ds:
                self.listing_ds.append(
                    ensure_listing(
                        self.list_ds_name,
                        self.catalog,
                        self.listing_fn,
                        update=self.update,
                    )
                )

            # at this point we know what is our starting listing dataset name
            self._set_starting_step(self.listing_ds[0])

    def resolve_all_listings(self) -> None:
        """Recursively resolve listings for this query and all sub-queries."""
        self.resolve_listing()
        for step in self.steps:
            for sub_query in step.sub_queries:
                sub_query.resolve_all_listings()

    def apply_steps(self) -> QueryGenerator:
        """
        Apply the steps in the query and return the resulting
        sqlalchemy.SelectBase.
        """
        self.resolve_all_listings()

        hasher = hashlib.sha256()
        hasher.update(self._starting_step_hash.encode("utf-8"))
        if self.delta_spec is not None:
            hasher.update(self.delta_spec.hash().encode("utf-8"))

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

        if query.starting_step is not None:
            result = query.starting_step.apply()
            self.dependencies.update(result.dependencies)
        elif query.steps:

            def q(*columns):
                return sqlalchemy.select(*columns)

            result = step_result(q, ())
        else:
            raise RuntimeError("DatasetQuery has no starting dataset or steps")

        _hash = hasher.hexdigest()
        # Stage flags reset at the start of every apply_steps run; they only
        # describe the in-flight query under construction here.
        state = QueryState(self.catalog.warehouse)
        for i, step in enumerate(query.steps):
            hash_input = _hash
            step = _project_step(
                step, query.steps[i + 1 :], hasher, self.checkpoints_enabled
            )
            hasher.update(step.hash().encode("utf-8"))
            _hash = hasher.hexdigest()
            hash_output = _hash

            result = step.apply(
                result.query_generator,
                self.temp_table_names,
                state,
                hash_input=hash_input,
                hash_output=hash_output,
                checkpoints_enabled=self.checkpoints_enabled,
            )  # a chain of steps linked by results
            self.dependencies.update(result.dependencies)
            # Most non-SQLClause steps materialize a fresh relation, so any
            # prior query-state flags no longer apply. Steps that preserve the
            # in-flight selectable, or manage flags themselves, opt out.
            if step.resets_query_state_after_apply:
                state.reset()

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
        result: ResultIter | None = None
        try:
            query = self.apply_steps().select()
            selected_columns = [c.name for c in query.selected_columns]
            result = ResultIter(
                self.catalog.warehouse.dataset_rows_select(query, **kwargs),
                selected_columns,
            )
            yield result
        finally:
            if result is not None:
                result.close()
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
        obj.steps = self.steps.copy()
        if new_table:
            obj.table = self.get_table()
        obj.dependencies = set()
        obj.temp_table_names = []
        return obj

    def delta_sources(self) -> list["DatasetQuery"]:
        sources: list[DatasetQuery] = []
        for step in self.steps:
            sources.extend(step.collect_delta_sources())

        if self.delta_spec is not None:
            sources.append(self)

        return list({id(source): source for source in sources}.values())

    def replace_sources(self, replacements: dict[int, "DatasetQuery"]) -> "Self":
        replacement = replacements.get(id(self))
        rewritten = (replacement if replacement is not None else self).clone(
            new_table=False
        )
        rewritten.table = self.table
        steps = rewritten.steps.copy() if replacement is not None else []
        steps.extend(step.replace_sources(replacements) for step in self.steps)
        rewritten.steps = steps
        return cast("Self", rewritten)

    @detach
    def select(self, *args, implicit_sys: bool = False, **kwargs) -> "Self":
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
        query.steps.append(SQLSelect((*args, *named_args), implicit_sys=implicit_sys))
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

    def union(self, dataset_query: "DatasetQuery") -> "DatasetQuery":
        left = self.clone()
        right = dataset_query.clone()
        return DatasetQuery(
            session=self.session,
            steps=[SQLUnion(left, right)],
            checkpoints_enabled=self.checkpoints_enabled,
        )

    def join(
        self,
        dataset_query: "DatasetQuery",
        predicates: JoinPredicateType | Sequence[JoinPredicateType],
        inner=False,
        full=False,
        rname="right_",
    ) -> "DatasetQuery":
        left = self.clone(new_table=False)
        if self.table.name == dataset_query.table.name:
            # for use case where we join with itself, e.g dogs.join(dogs, "name")
            right = dataset_query.clone(new_table=True)
        else:
            right = dataset_query.clone(new_table=False)

        predicates = (
            predicates
            if isinstance(predicates, (str, ColumnClause, ColumnElement))
            else tuple(predicates)
        )
        return DatasetQuery(
            session=self.session,
            steps=[SQLJoin(self.catalog, left, right, predicates, inner, full, rname)],
            checkpoints_enabled=self.checkpoints_enabled,
        )

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
        ephemeral: bool = False,
        prefetch: int | None = None,
        namespace: str | None = None,
        project: str | None = None,
    ) -> "Self":
        """
        Adds one or more signals based on the results from the provided UDF.

        Parallel can optionally be specified as >= 1 for parallel processing with a
        specific number of processes, or set to -1 for the default of
        the CPUs available to the process, respecting affinity and container quotas.

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
                self.session,
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
    def subtract(
        self,
        dq: "DatasetQuery",
        on: Sequence[tuple[str, str]],
        presence: Sequence[tuple[KeyPresence, KeyPresence]] = (),
    ) -> "Self":
        query = self.clone()
        query.steps.append(Subtract(dq, self.catalog, on=on, presence=presence))
        return query

    @detach
    def generate(
        self,
        udf: "UDFAdapter",
        partition_by: PartitionByType | None = None,
        is_aggregator: bool = False,
        # Parameters from Settings
        cache: bool = False,
        parallel: int | None = None,
        workers: bool | int = False,
        min_task_size: int | None = None,
        batch_size: int | None = None,
        # Parameters are unused, kept only to match the signature of Settings.to_dict:
        ephemeral: bool = False,
        prefetch: int | None = None,
        namespace: str | None = None,
        project: str | None = None,
    ) -> "Self":
        query = self.clone()
        steps = query.steps
        steps.append(
            RowGenerator(
                udf,
                self.session,
                partition_by=partition_by,
                is_aggregator=is_aggregator,
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
                                    versions=[dep.version],
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
        query_script: str | None = None,
        **kwargs,
    ) -> "Self":
        """Save the query as a dataset.

        A taken explicit version fails before any work, without writing. The
        steps and final table materialization run into a temporary table first,
        then the version is created, the table is renamed into place, and the
        version is completed.
        """
        job = self.session.get_or_create_job()
        job_id = job.id if job else None
        if query_script is None:
            query_script = job.query if job else ""

        project = project or self.catalog.metastore.default_project
        if not name and version:
            raise RuntimeError("Cannot set version for temporary datasets")
        if name and version:
            self._check_explicit_version(name, version, project)
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

            warehouse = self.catalog.warehouse
            temp_table_name = warehouse.temp_table_name()
            temp_table = warehouse.create_dataset_rows_table(
                temp_table_name, columns=columns
            )
            self.temp_table_names.append(temp_table_name)
            warehouse.insert_into(temp_table, query.select())

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
                query_script=query_script,
                **kwargs,
            )
            version = dataset.versions[0].version
            try:
                warehouse.rename_table(
                    temp_table, warehouse.dataset_table_name(dataset, version)
                )
                self.temp_table_names.remove(temp_table_name)

                # Link this dataset version to the job that created it
                if job_id:
                    self.catalog.metastore.link_dataset_version_to_job(
                        dataset.get_version(version).id, job_id, is_creator=True
                    )
                if dependencies:
                    self._set_dependencies(dependencies)
                self._add_dependencies(dataset, version)  # type: ignore [arg-type]

                # Mark as COMPLETE only after all operations succeed.
                self.catalog.complete_dataset_version(
                    dataset, version, expected_job_id=job_id
                )
            except BaseException:
                # Give back the reservation, as if it was never made. It is
                # removed only while still CREATED and ours: an interrupt right
                # after completion must not remove a published version.
                with contextlib.suppress(Exception):
                    self.catalog.release_reservation(dataset, version, job_id)
                raise
        finally:
            self.cleanup()
        return self.__class__(
            name=name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            version=version,
            session=self.session,
        )

    def _check_explicit_version(
        self, name: str, version: str, project: Project
    ) -> None:
        """Fail before any work if an explicit version is taken or removed."""
        try:
            existing = self.catalog.get_dataset(
                name,
                namespace_name=project.namespace.name,
                project_name=project.name,
                versions=[version],
                include_incomplete=True,
            )
        except DatasetNotFoundError:
            return
        if not existing.has_version(version):
            return
        if existing.get_version(version).is_removed:
            raise DatasetInvalidVersionError(
                f"Version {version} of dataset {name} was "
                "removed. Pick a different version."
            )
        raise DatasetInvalidVersionError(
            f"Dataset {name} already has version {version}"
        )

    def _set_dependencies(self, dependencies: list[DatasetDependency]) -> None:
        """Override the dependencies collected from the steps."""
        self.dependencies = {
            (
                self.catalog.get_dataset(
                    dep.name,
                    namespace_name=dep.namespace,
                    project_name=dep.project,
                    versions=[dep.version],
                    include_incomplete=False,
                ),
                dep.version,
            )
            for dep in dependencies
        }

    @property
    def is_ordered(self) -> bool:
        return isinstance(self.last_step, SQLOrderBy)

    @property
    def last_step(self) -> Step | None:
        return self.steps[-1] if self.steps else None
