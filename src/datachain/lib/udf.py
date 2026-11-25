import hashlib
import inspect
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from graphlib import CycleError, TopologicalSorter
from itertools import islice
from operator import itemgetter
from typing import TYPE_CHECKING, Any, TypeVar

import attrs
from fsspec import Callback
from fsspec.callbacks import DEFAULT_CALLBACK

from datachain.asyn import AsyncMapper
from datachain.cache import temporary_cache
from datachain.error import RowEncodeError
from datachain.hash_utils import hash_callable
from datachain.lib.data_model import union_arms
from datachain.lib.file import File, FileError
from datachain.lib.settings import DEFAULT_PREFETCH
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.utils import AbstractUDF, DataChainParamsError, type_to_str
from datachain.query.batch import (
    Batch,
    BatchingStrategy,
    NoBatching,
    Partition,
    RowsOutputBatch,
)
from datachain.query.schema import DEFAULT_DELIMITER
from datachain.utils import TIME_ZERO, safe_closing

logger = logging.getLogger("datachain")

if TYPE_CHECKING:
    from collections import abc
    from contextlib import AbstractContextManager
    from typing import Self

    import pyarrow as pa

    from datachain.cache import Cache
    from datachain.catalog import Catalog
    from datachain.data_storage import AbstractWarehouse
    from datachain.data_storage.codec import RowCodec
    from datachain.lib.settings import Settings
    from datachain.lib.signal_schema import KeyPresence
    from datachain.lib.udf_signature import UdfSignature
    from datachain.query.batch import RowsOutput

T = TypeVar("T", bound=Sequence[Any])


@dataclass
class BindContext:
    """Context handed to a ``BoundSpec`` when it is attached to a verb. Carries the
    chain settings and the target UDF class; new fields (catalog, session, ...) can
    be added here without changing the ``bind`` signature."""

    settings: "Settings"
    target: Any = None  # the verb's UDF class (Mapper/Generator/Aggregator)


class BoundSpec(ABC):
    """A UDF spec that resolves itself against the chain when attached to a verb.
    ``DataChain._udf_to_obj`` calls ``bind`` with a ``BindContext`` to get the
    concrete per-row callable, so a spec can read ``.settings(...)`` and choose its
    shape from the target verb at build time."""

    @abstractmethod
    def bind(self, ctx: BindContext) -> Callable: ...

    @abstractmethod
    def input_columns(self) -> list[str]:
        """Return the chain columns this spec reads, in the order the bound
        callable expects them. Must be stable across ``bind()``."""

    @property
    @abstractmethod
    def output_count(self) -> int:
        """How many chain columns the bound callable produces per row."""


def reject_var_params(
    udf_func: "Callable | UDFBase", label: str, *, columns_explicit: bool = False
) -> None:
    """Reject user callables the single-signal ``.map()`` call site can't invoke.

    Values are passed positionally, so a signature that can't accept positional
    args (only ``**kwargs``, or only keyword-only params) is rejected in both
    modes. Without ``columns_explicit`` (no ``params=``), ``*args`` / ``**kwargs``
    are also rejected because we can't infer column routing from them.
    For UDFBase subclasses, the check runs against ``.process``."""
    target = udf_func.process if isinstance(udf_func, AbstractUDF) else udf_func
    params = list(inspect.signature(target).parameters.values())
    if not columns_explicit:
        var_params = [
            f"*{p.name}"
            if p.kind is inspect.Parameter.VAR_POSITIONAL
            else f"**{p.name}"
            for p in params
            if p.kind
            in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
        ]
        if var_params:
            raise DataChainParamsError(
                f"{label} uses {var_params}; list the input column names "
                "as regular params"
            )
    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    }
    if params and not any(p.kind in positional_kinds for p in params):
        raise DataChainParamsError(
            f"{label} has no positional parameters; list the input column "
            "names as regular params"
        )


class UdfError(DataChainParamsError):
    """Exception raised for UDF-related errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.__class__.__name__!s}: {self.message!s}"

    def __reduce__(self) -> tuple[type, tuple]:
        return self.__class__, (self.message,)


class JsonSerializationError(UdfError):
    def __init__(self, message: str, column_name: str, value_repr: str) -> None:
        self.column_name = column_name
        self.value_repr = value_repr
        super().__init__(message)

    def __reduce__(self) -> tuple[type, tuple]:
        return self.__class__, (self.message, self.column_name, self.value_repr)


class UdfRunError(Exception):
    """Exception raised when UDF execution fails."""

    def __init__(
        self,
        error: Exception | str,
        stacktrace: str | None = None,
        udf_name: str | None = None,
    ) -> None:
        self.error = error
        self.stacktrace = stacktrace
        self.udf_name = udf_name
        super().__init__(str(error))

    def __str__(self) -> str:
        if isinstance(self.error, UdfRunError):
            return str(self.error)
        if isinstance(self.error, Exception):
            return f"{self.error.__class__.__name__!s}: {self.error!s}"
        return f"{self.__class__.__name__!s}: {self.error!s}"

    def __reduce__(self) -> tuple[type, tuple]:
        """Custom reduce method for pickling."""
        return self.__class__, (self.error, self.stacktrace, self.udf_name)


ColumnType = Any

# Specification for the output of a UDF
UDFOutputSpec = Mapping[str, ColumnType]

# Result type when calling the UDF wrapper around the actual Python function / class
# implementing it: the output values in their stored form, then the sys values.
UDFResult = tuple[Any, ...]
# A gen or agg result: the rows of one part (an input row, a partition) and its key
UDFPart = tuple[Any, "Iterable[UDFResult | pa.RecordBatch]"]

PROGRESS_INTERVAL = 0.1  # seconds


class _Progress:
    """Adds counts to a callback every ``PROGRESS_INTERVAL`` rather than per row:
    a callback call costs about as much as a plain row's own work."""

    __slots__ = ("callback", "pending", "report_at")

    def __init__(self, callback: Callback) -> None:
        self.callback = callback
        self.pending = 0
        self.report_at = 0.0

    def add(self, count: int) -> None:
        self.pending += count
        if (now := time.monotonic()) >= self.report_at:
            self.callback.relative_update(self.pending)
            self.pending, self.report_at = 0, now + PROGRESS_INTERVAL

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, *exc) -> None:
        if self.pending:
            self.callback.relative_update(self.pending)


def output_value_error(
    error: RowEncodeError,
    names: Sequence[str],
    schema: SignalSchema,
    udf_kind: str | None = None,
    num_values: int | None = None,
) -> UdfError:
    """The error for a value an output column cannot take."""
    name = names[error.column]
    expected = type_to_str(schema.get_column_type(name))
    cause = error.__cause__
    if isinstance(cause, JsonSerializationError):
        msg = (
            f"UDF returned an invalid value for output column {name!r}. "
            f"Expected JSON-serializable {expected}. {cause.message}"
        )
    else:
        msg = (
            f"UDF returned an invalid value for output column {name!r}. "
            f"Expected {expected}, got {error.value!r} "
            f"(type: {type(error.value).__name__})."
        )
    width = len(names)
    if udf_kind is not None and num_values is not None and num_values != width:
        value_word = "value" if num_values == 1 else "values"
        are_word = "is" if width == 1 else "are"
        msg += (
            f" Note: UDF call returned {num_values} {value_word} "
            f"while {width} {are_word} expected per output definition"
        )
        if udf_kind in ("agg", "gen"):
            msg += (
                f", {udf_kind}() UDFs usually use yield and have return type Iterator."
            )
        else:
            msg += "."
    return UdfError(msg)


class OutputRows:
    """A UDF's results as stored rows, compiled once per run: a result is
    flattened into leaf values by the output schema, then encoded by the
    warehouse's row codec. This is the only place UDF output values convert."""

    def __init__(self, udf: "UDFBase", warehouse: "AbstractWarehouse") -> None:
        spec = udf.output.to_udf_spec()
        self.names = list(spec)
        self.width = len(self.names)
        self.schema = udf.output
        self.udf_kind = udf.udf_kind
        self.flatten = udf.output.row_flattener()
        self.codec: RowCodec = warehouse.row_codec(spec.values())

    def __call__(self, result: Any) -> tuple:
        """One UDF result as a stored row."""
        return self.encode(self.flatten(result))

    def encode(self, values: Sequence[Any]) -> tuple:
        """Leaf values as a stored row: missing trailing values are filled in,
        extra ones dropped."""
        num_values = len(values)
        if num_values != self.width:
            values = (*values[: self.width], *(None,) * (self.width - num_values))
        try:
            return self.codec.encode(values)
        except RowEncodeError as e:
            raise output_value_error(
                e, self.names, self.schema, self.udf_kind, num_values
            ) from e.__cause__

    def encode_batch(self, batch: "pa.RecordBatch") -> "pa.RecordBatch | list[tuple]":
        """An Arrow batch of leaf columns as stored columns or tuples."""
        try:
            return self.codec.encode_arrow(batch)
        except RowEncodeError as e:
            raise output_value_error(e, self.names, self.schema) from e.__cause__


@attrs.define(slots=False)
class UDFAdapter:
    inner: "UDFBase"
    output: UDFOutputSpec
    batch_size: int | None = None
    batch: int | None = None

    def hash(self, include_body: bool = True) -> str:
        inner_hash = self.inner.hash(include_body=include_body)
        if self.batch is None:
            return inner_hash
        return hashlib.sha256(f"{inner_hash}:batched".encode()).hexdigest()

    def get_batching(self, use_partitioning: bool = False) -> BatchingStrategy:
        if use_partitioning:
            return Partition()

        if self.batch is None:
            return NoBatching()
        if self.batch >= 1:
            return Batch(self.batch)
        raise ValueError(f"invalid batch size {self.batch}")

    def run(
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[RowsOutput]",
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult] | UDFPart]:
        yield from self.inner.run(
            udf_fields,
            udf_inputs,
            catalog,
            cache,
            download_cb,
            processed_cb,
        )

    @property
    def prefetch(self) -> int:
        return self.inner.prefetch


class UDFBase(AbstractUDF):
    """Base class for stateful user-defined functions.

    Any class that inherits from it must have a `process()` method that takes input
    params from one or more rows in the chain and produces the expected output.

    Optionally, the class may include these methods:
    - `setup()` to run code on each  worker before `process()` is called.
    - `teardown()` to run code on each  worker after `process()` completes.

    Example:
        ```py
        import datachain as dc
        import open_clip

        class ImageEncoder(dc.Mapper):
            def __init__(self, model_name: str, pretrained: str):
                self.model_name = model_name
                self.pretrained = pretrained

            def setup(self):
                self.model, _, self.preprocess = (
                    open_clip.create_model_and_transforms(
                        self.model_name, self.pretrained
                    )
                )

            def process(self, file) -> list[float]:
                img = file.get_value()
                img = self.preprocess(img).unsqueeze(0)
                emb = self.model.encode_image(img)
                return emb[0].tolist()

        (
            dc.read_storage(
                "gs://datachain-demo/fashion-product-images/images", type="image"
            )
            .limit(5)
            .map(
                ImageEncoder("ViT-B-32", "laion2b_s34b_b79k"),
                params=["file"],
                output={"emb": list[float]},
            )
            .show()
        )
        ```
    """

    is_input_batched = False
    is_output_batched = False
    udf_kind = "map"  # named in output errors: map, gen or agg
    prefetch: int = 0
    prefetch_bytes: int | None = None
    _prefetch_min_rows: int = 1
    # Class-level default so subclasses that skip super().__init__() still
    # have this attribute; _MultiSignalMapper sets params/output per entry
    # but doesn't call _init(), so _func would otherwise be missing.
    _func: Callable | None = None

    def __init__(self):
        self.params: SignalSchema | None = None
        self.output = None
        self._func = None

    def hash(self, include_body: bool = True) -> str:
        """
        Creates SHA hash of this UDF function. It takes into account function,
        inputs and outputs.

        For function-based UDFs, hashes self._func.
        For class-based UDFs, hashes the process method.

        When include_body=False, the function body is excluded (identity-only:
        __module__ + __qualname__ + defaults). Lambdas always include their
        bytecode since they share the name '<lambda>'.
        """
        # Hash user code: either _func (function-based) or process method (class-based)
        func_to_hash = self._func or self.process

        parts = [
            hash_callable(func_to_hash, include_body=include_body),
            self.params.hash() if self.params else "",
            self.output.hash(),
        ]

        return hashlib.sha256(
            b"".join([bytes.fromhex(part) for part in parts])
        ).hexdigest()

    def process(self, *args, **kwargs):
        """Processing function that needs to be defined by user"""
        if not self._func:
            raise NotImplementedError("UDF processing is not implemented")
        return self._func(*args, **kwargs)

    def setup(self):
        """Initialization process executed on each worker before processing begins.
        This is needed for tasks like pre-loading ML models prior to scoring.
        """

    def teardown(self):
        """Teardown process executed on each process/worker after processing ends.
        This is needed for tasks like closing connections to end-points.
        """

    @contextmanager
    def _teardown_on_exit(self) -> Iterator[None]:
        """Run teardown() after processing, also when it fails or stops early.
        A teardown error after a processing error is logged, so the processing
        error is the one raised."""
        try:
            yield
        except BaseException:
            try:
                self.teardown()
            except Exception:
                logger.exception(
                    "Failed to tear down UDF %r after an error", self.verbose_name
                )
            raise
        self.teardown()

    def _init(
        self,
        sign: "UdfSignature",
        params: "SignalSchema",
        func: Callable | None,
    ):
        self.params = params
        self.output = sign.output_schema
        self._func = func

    @classmethod
    def _create(
        cls,
        sign: "UdfSignature",
        params: "SignalSchema",
    ) -> "Self":
        if isinstance(sign.func, AbstractUDF):
            if not isinstance(sign.func, cls):  # type: ignore[unreachable]
                raise UdfError(
                    f"cannot create UDF: provided UDF '{type(sign.func).__name__}'"
                    f" must be a child of target class '{cls.__name__}'",
                )
            result = sign.func
            func = None
        else:
            result = cls()
            func = sign.func

        result._init(sign, params, func)
        return result

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def verbose_name(self):
        """Returns the name of the function or class that implements the UDF."""
        if self._func and callable(self._func):
            if hasattr(self._func, "__name__"):
                return self._func.__name__
            if hasattr(self._func, "__class__") and hasattr(
                self._func.__class__, "__name__"
            ):
                return self._func.__class__.__name__
        return "<unknown>"

    @property
    def signal_names(self) -> Iterable[str]:
        return self.output.to_udf_spec().keys()

    def to_udf_wrapper(
        self,
        batch_size: int | None = None,
        batch: int | None = None,
    ) -> UDFAdapter:
        return UDFAdapter(
            self,
            self.output.to_udf_spec(),
            batch_size,
            batch,
        )

    def run(
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[Any]",
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult] | UDFPart]:
        raise NotImplementedError

    @property
    def _has_file_inputs(self) -> bool:
        return bool(
            self.params
            and any(
                inspect.isclass(arm) and issubclass(arm, File)
                for name, param_type in self.params.values.items()
                if name not in self.params.setup_func
                for arm in union_arms(param_type)[0]
            )
        )

    @property
    def _effective_prefetch(self) -> int:
        """Prefetch only when an input can carry a file."""
        return self.prefetch if self._has_file_inputs else 0

    def read_columns(self) -> list[str]:
        """The db columns `run()` reads, besides sys__id and the partition id."""
        assert self.params is not None
        return self.params.read_columns()

    def _input_reader(
        self,
        udf_fields: "Sequence[str]",
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback,
        include_id: bool = True,
    ) -> Callable[[Sequence[Any]], Sequence[Any]]:
        """A function reading the UDF's arguments (after ``sys__id`` when
        `include_id`) from an input row of `udf_fields`, compiled once per run."""
        assert self.params is not None
        id_names = ["sys__id"] if include_id else []
        columns = self.params.plain_columns()
        if columns is not None:
            positions = {name: i for i, name in enumerate(udf_fields)}
            names = [*id_names, *columns]
            if not names:
                return lambda row: ()
            getter = itemgetter(*(positions[c] for c in names))
            return getter if len(names) > 1 else lambda row: (getter(row),)
        read = self.params.row_reader(udf_fields)
        set_streams = self.params.set_file_streams
        id_pos = udf_fields.index("sys__id") if include_id else None

        def prepare(row: Sequence[Any]) -> Sequence[Any]:
            objs = read(row)
            set_streams(objs, catalog, cache, download_cb)
            return (row[id_pos], *objs) if id_pos is not None else objs

        return prepare


def noop(*args, **kwargs):
    pass


def _row_files(row: Iterable, batch_columns: set[int] | None) -> Iterator[File]:
    """Files in scalar arguments and batched data columns."""
    for i, obj in enumerate(row):
        if isinstance(obj, File):
            yield obj
        elif batch_columns is not None and i in batch_columns and isinstance(obj, list):
            yield from (item for item in obj if isinstance(item, File))


_EMPTY_ETAGS = frozenset(
    {
        "d41d8cd98f00b204e9800998ecf8427e",  # MD5(b"")
        "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",  # Git blob SHA-1(b"")
    }
)


def _row_prefetch_targets(
    row: Iterable,
    batch_columns: set[int] | None,
    max_size: int | None = None,
    custom_prefetch: bool = False,
) -> tuple[list[File], list[File]]:
    """Files to fetch for ``row`` and whole-file cache targets to budget/evict."""
    to_fetch: list[File] = []
    targets: list[File] = []
    for obj in _row_files(row, batch_columns):
        if not obj.path:
            continue
        target = obj._prefetch_target()
        if not target.path:
            continue
        if target is obj:
            target = obj.model_copy()
        if max_size is None or 0 < target.size <= max_size:
            targets.append(target)
            to_fetch.append(obj if custom_prefetch else target)
        elif custom_prefetch:
            to_fetch.append(obj)
    return to_fetch, targets


def _remove_target(target: File) -> None:
    catalog = target._catalog
    assert catalog is not None
    try:
        catalog.cache.remove(target)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to remove prefetched item %r: %s", target.name, e)


class _PrefetchBudget:
    """Listed-byte read-ahead target, allowing the legacy minimum row count."""

    def __init__(self, max_rows: int, max_bytes: int | None, min_rows: int = 1) -> None:
        self.max_rows, self.max_bytes = max_rows, max_bytes
        self.min_rows = min(max(1, min_rows), max_rows)
        self.rows = self.bytes = 0
        self.targets: Counter[str] = Counter()
        self.target_sizes: dict[str, int] = {}
        self.stopped = False
        self.changed = threading.Condition()

    def _target_size(self, target: File) -> int:
        if target.size > 0:
            return target.size
        if target.size == 0 and (
            target.last_modified != TIME_ZERO or target.etag in _EMPTY_ETAGS
        ):
            return 0
        return self.max_bytes or 0

    def targets_for(self, targets: Iterable[File]) -> dict[str, tuple[File, int]]:
        by_oid: dict[str, tuple[File, int]] = {}
        for target in targets:
            oid = target.get_hash()
            size = self._target_size(target)
            prev = by_oid.get(oid)
            if prev is None or size > prev[1]:
                by_oid[oid] = (target, size)
        return by_oid

    def _added_bytes(self, item: Mapping[str, tuple[File, int]]) -> int:
        return sum(size for oid, (_, size) in item.items() if not self.targets[oid])

    def stop(self) -> None:
        with self.changed:
            self.stopped = True
            self.changed.notify_all()

    def take(self, item: Mapping[str, tuple[File, int]]) -> None:
        with self.changed:
            while not self.stopped and (
                self.rows >= self.max_rows
                or (
                    self.rows >= self.min_rows
                    and self.max_bytes is not None
                    and (added := self._added_bytes(item)) > 0
                    and self.bytes + added > self.max_bytes
                )
            ):
                self.changed.wait()
            self.rows += 1
            for oid, (_, size) in item.items():
                if not self.targets[oid]:
                    self.bytes += size
                    self.target_sizes[oid] = size
                self.targets[oid] += 1

    def give_back(
        self, item: Mapping[str, tuple[File, int]], remove: bool = False
    ) -> None:
        with self.changed:
            self.rows -= 1
            for oid, (target, _) in item.items():
                self.targets[oid] -= 1
                if not self.targets[oid]:
                    del self.targets[oid]
                    self.bytes -= self.target_sizes.pop(oid)
                    if remove:
                        _remove_target(target)
            self.changed.notify_all()


async def _prefetch_input(
    files: Iterable[File],
    download_cb: Callback | None = None,
    after_prefetch: "Callable[[], None]" = noop,
    prefetch_file: "Callable[[File, Callback | None], abc.Awaitable[bool]] | None" = (
        None
    ),
) -> None:
    for obj in files:
        try:
            prefetched = (
                await prefetch_file(obj, download_cb)
                if prefetch_file is not None
                else await obj._prefetch(download_cb)
            )
            if prefetched:
                after_prefetch()
        except FileError as e:
            logger.warning("Skipping prefetch for '%s/%s': %s", obj.source, obj.path, e)


def _prefetch_inputs(
    prepared_inputs: "Iterable[T]",
    prefetch: int = 0,
    download_cb: Callback | None = None,
    after_prefetch: Callable[[], None] | None = None,
    remove_prefetched: bool = False,
    batch_columns: set[int] | None = None,
    max_size: int | None = None,
    max_bytes: int | None = None,
    min_rows: int = 2 * DEFAULT_PREFETCH + 1,
    prefetch_file: "Callable[[File, Callback | None], abc.Awaitable[bool]] | None" = (
        None
    ),
) -> "abc.Generator[T, None, None]":
    if not prefetch:
        yield from prepared_inputs
        return

    if after_prefetch is None:
        after_prefetch = noop
        if download_cb and hasattr(download_cb, "increment_file_count"):
            increment_file_count: Callable[[], None] = download_cb.increment_file_count
            after_prefetch = increment_file_count

    f = partial(
        _prefetch_input,
        download_cb=download_cb,
        after_prefetch=after_prefetch,
        prefetch_file=prefetch_file,
    )
    # prefetch=N holds N rows downloading, N downloaded and one in the UDF.
    budget = _PrefetchBudget(2 * prefetch + 1, max_bytes, min_rows)
    custom_prefetch = prefetch_file is not None

    async def fetch(
        item: tuple[T, list[File], dict[str, tuple[File, int]]],
    ) -> tuple[T, dict[str, tuple[File, int]]]:
        row, files, targets = item
        await f(files)
        return row, targets

    def admitted() -> Iterator[tuple[T, list[File], dict[str, tuple[File, int]]]]:
        # Waiting here, before the read-ahead takes a row, leaves the rest of
        # the input to `parallel` peers.
        with safe_closing(prepared_inputs):
            for row in prepared_inputs:
                files, raw_targets = _row_prefetch_targets(
                    row, batch_columns, max_size, custom_prefetch
                )
                targets = budget.targets_for(raw_targets)
                budget.take(targets)
                yield row, files, targets

    mapper = AsyncMapper(fetch, admitted(), workers=prefetch, on_shutdown=budget.stop)
    with closing(mapper.iterate()) as row_iter:
        for row, targets in row_iter:
            try:
                yield row  # type: ignore[misc]
            finally:
                budget.give_back(targets, remove=remove_prefetched)


def _get_cache(
    cache: "Cache", prefetch: int = 0, use_cache: bool = False
) -> "AbstractContextManager[Cache]":
    tmp_dir = cache.tmp_dir
    assert tmp_dir
    if not use_cache:
        # cache=False = "don't write to persistent"; reads still hit it.
        if prefetch:
            # Temp cache for new writes (evicted at end), persistent as
            # read-only fallback for reads.
            return temporary_cache(
                tmp_dir, prefix="prefetch-", fallback=cache.as_readonly()
            )
        return nullcontext(cache.as_readonly())
    return nullcontext(cache)


def _batch_results(result: Any, expected: int, label: str) -> list[Any]:
    """Read one result per input row, consuming at most one extra result."""
    if isinstance(
        result, (str, bytes, bytearray, Mapping, AbstractSet)
    ) or not isinstance(result, Iterable):
        raise UdfError(
            f"{label} must return a list or iterator of row results, "
            f"got {type(result).__name__}. output= describes one row"
        )
    with safe_closing(result):
        results = list(islice(result, expected + 1))
    if len(results) != expected:
        got = f"more than {expected}" if len(results) > expected else len(results)
        raise UdfError(
            f"{label} must return one output per input row: "
            f"got {got} outputs for {expected} inputs"
        )
    return results


class Mapper(UDFBase):
    """Inherit from this class to pass to `DataChain.map()`.

    For lists of rows per call, use `BatchMapper` with `DataChain.batch_map()`.
    """

    prefetch: int = 2

    def run(
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[Sequence[Any]]",
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult]]:
        self.setup()
        # Prepared columns holding one value per row: sys__id and data columns.
        batch_columns: set[int] | None = None
        if self.is_input_batched:
            assert self.params is not None
            batch_columns = {0} | {
                i
                for i, name in enumerate(self.params.values, 1)
                if name not in self.params.setup_func
            }
        label = (
            f"batch_map() function {(self.verbose_name if self._func else self.name)!r}"
        )
        prefetch = self._effective_prefetch
        output_row = OutputRows(self, catalog.warehouse)

        def _prepare_rows(udf_inputs) -> "abc.Generator[Sequence[Any], None, None]":
            prepare = self._input_reader(udf_fields, catalog, cache, download_cb)
            with safe_closing(udf_inputs):
                for row in udf_inputs:
                    if batch_columns is None:
                        yield prepare(row)
                        continue
                    yield [
                        list(col) if i in batch_columns else col[0]
                        for i, col in enumerate(zip(*map(prepare, row), strict=True))
                    ]

        prepared_inputs = _prepare_rows(udf_inputs)
        prepared_inputs = _prefetch_inputs(
            prepared_inputs,
            prefetch,
            download_cb=download_cb,
            remove_prefetched=bool(prefetch) and not cache,
            batch_columns=batch_columns,
            max_bytes=self.prefetch_bytes,
            min_rows=self._prefetch_min_rows,
        )

        def _results(progress) -> Iterator[tuple]:
            for id_, *udf_args in prepared_inputs:
                if batch_columns is None:
                    row = (*output_row(self.process(*udf_args)), id_)
                    progress.add(1)
                    yield row
                    continue
                self._batch_length = len(id_)
                results = _batch_results(self.process(*udf_args), len(id_), label)
                progress.add(len(results))
                # lazy: rows encoded before an invalid one are still written
                for row_id, result in zip(id_, results, strict=True):
                    yield (*output_row(result), row_id)

        with (
            self._teardown_on_exit(),
            closing(prepared_inputs),
            _Progress(processed_cb) as progress,
        ):
            # one lazy result for all rows: no per-row hand-off to the writer
            yield _results(progress)


class BatchMapper(Mapper):
    """Inherit from this class to pass to `DataChain.batch_map()`.

    `process()` receives a list of up to `batch` values for each column
    parameter, even with `batch=1`; resources from `chain.setup()` are passed
    unchanged. It returns `list[T]` or `Iterator[T]` with one result per input
    row, in input order, and the output column has type `T`. Any other number
    of results raises `UdfError`. `setup()` and `teardown()` run once per worker
    process.
    """

    is_input_batched = True


def _make_bound_adapter(fn: Callable, cols: list[str]) -> Callable:
    """Wrap a positional bound-spec callable so it can be invoked with
    column-name kwargs, uniformly with the other multi-signal entries."""

    def adapter(**kw: Any) -> Any:
        return fn(*(kw[c] for c in cols))

    return adapter


class _MultiSignalMapper(Mapper):
    """Mapper that runs N user functions per row, yielding N output signals.

    Implements `.map(a=f1, b=f2, ...)` as a single UDF stage: each row is
    iterated once, all functions run, and no intermediate column is
    materialized between them.

    If a function's parameter name matches another function's output name
    in the same call, the dependent function receives the producer's
    result instead of a row column. Execution order is a topological sort
    of that dependency graph; the order the user wrote the kwargs is
    irrelevant. Cycles raise ``ValueError`` at construction time.
    """

    def __init__(
        self,
        signal_map: "dict[str, Callable | UDFBase]",
        *,
        bound_columns: dict[str, list[str]] | None = None,
        batched: bool = False,
    ):
        super().__init__()
        self.is_input_batched = batched
        self._method_name = "batch_map" if batched else "map"
        self._signal_map = signal_map
        self._bound_columns = bound_columns or {}
        output_names = set(signal_map)
        self._per_func_params: dict[str, list[str]] = {}
        # For each function: which of its params come from another
        # function's output (dependencies) vs from an input row column.
        deps: dict[str, set[str]] = {}
        for name, fn in signal_map.items():
            if name in self._bound_columns:
                params = list(self._bound_columns[name])
            else:
                params = self._resolve_sig_params(name, fn)
            if name in params:
                raise DataChainParamsError(
                    f"{self._method_name}() entry {name!r} declares a parameter "
                    f"named {name!r} "
                    "- an entry can't read a column with the same name as its "
                    "own output. Rename the output or the parameter."
                )
            self._per_func_params[name] = params
            deps[name] = {p for p in params if p in output_names}

        try:
            self._exec_order = list(TopologicalSorter(deps).static_order())
        except CycleError as e:
            raise ValueError(
                f"Cyclic dependency between {self._method_name} functions: {e.args[1]}"
            ) from e

        # combined_params = union of params NOT resolved from other outputs,
        # in first-seen order. Consumed positionally by the UDF layer.
        seen: set[str] = set()
        self.combined_params: list[str] = []
        for name in signal_map:
            for p in self._per_func_params[name]:
                if p in output_names:
                    continue
                if p not in seen:
                    seen.add(p)
                    self.combined_params.append(p)

        # Pre-build one uniform kwargs-callable per entry so process() has a
        # single call path regardless of entry kind.
        self._callers: dict[str, Callable[..., Any]] = {}
        for name, fn in signal_map.items():
            if isinstance(fn, UDFBase):
                self._callers[name] = fn.process
            elif name in self._bound_columns:
                self._callers[name] = _make_bound_adapter(
                    fn, self._per_func_params[name]
                )
            else:
                self._callers[name] = fn

    def _resolve_sig_params(self, name: str, fn: "Callable | UDFBase") -> list[str]:
        """Derive parameter names from an entry's signature, rejecting shapes we
        can't call by kwarg."""
        if isinstance(fn, UDFBase):
            if not isinstance(fn, Mapper):
                raise DataChainParamsError(
                    f"{self._method_name}() entry {name!r} is a "
                    f"{type(fn).__name__}; only "
                    "Mapper subclasses (or plain callables) are supported "
                    "in multi-signal .map() - use .gen() / .agg() for "
                    "other UDF types"
                )
            sig_target = fn.process
        else:
            sig_target = fn
        sig_params = list(inspect.signature(sig_target).parameters.values())
        bad = [
            p.name
            for p in sig_params
            if p.kind
            not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if bad:
            raise DataChainParamsError(
                f"{self._method_name}() function {name!r} has parameters that can't "
                f"be passed by name ({bad}); multi-signal .{self._method_name}() calls "
                "each function with keyword arguments"
            )
        return [p.name for p in sig_params]

    def _init(self, sign, params, func):
        super()._init(sign, params, func)
        # Give each inner UDFBase entry its own params/output so framework
        # attrs like self.signal_names, self.output, self.params work.
        # Params can be outer-input columns or sibling-entry outputs; both
        # flow into process() so both must show up in fn.params.
        output_values = sign.output_schema.values
        for name, fn in self._signal_map.items():
            if isinstance(fn, UDFBase):
                fn_param_types = {}
                for p in self._per_func_params[name]:
                    if p in params.values:
                        fn_param_types[p] = params.values[p]
                    elif p in output_values:
                        fn_param_types[p] = output_values[p]
                fn.params = SignalSchema(fn_param_types)
                fn.output = SignalSchema({name: output_values[name]})

    def process(self, *args):
        row_by_name = dict(zip(self.combined_params, args, strict=True))
        results: dict[str, Any] = {}
        for name in self._exec_order:
            params = self._per_func_params[name]
            kw = {p: (results[p] if p in results else row_by_name[p]) for p in params}
            if self.is_input_batched:
                assert self.params is not None
                kw = {
                    p: list(value)
                    if p in results or p not in self.params.setup_func
                    else value
                    for p, value in kw.items()
                }
            result = self._callers[name](**kw)
            if self.is_input_batched:
                result = _batch_results(
                    result, self._batch_length, f"batch_map() entry {name!r}"
                )
            results[name] = result
        if self.is_input_batched:
            return list(zip(*(results[name] for name in self._signal_map), strict=True))
        # Output order follows the user's declared kwarg order, not exec order.
        return tuple(results[name] for name in self._signal_map)

    def setup(self) -> None:
        for name in self._signal_map:
            fn = self._signal_map[name]
            if isinstance(fn, UDFBase):
                fn.setup()

    def teardown(self) -> None:
        for name in self._signal_map:
            fn = self._signal_map[name]
            if isinstance(fn, UDFBase):
                fn.teardown()

    @property
    def verbose_name(self) -> str:
        # The base property reads self._func, which is unset here.
        return ", ".join(self._signal_map)

    def hash(self, include_body: bool = True) -> str:
        # cache key must vary with the wrapped functions; the base
        # implementation would hash this class's process method, which is
        # identical across instances.
        parts = [
            fn.hash(include_body=include_body)
            if isinstance(fn, UDFBase)
            else hash_callable(fn, include_body=include_body)
            for fn in self._signal_map.values()
        ]
        # _bound_columns carries the chain-column routing for BoundSpec entries;
        # the bound closures don't encode it in their bytecode, so we mix it in
        # here to keep different routings distinct.
        routing = repr(sorted((k, tuple(v)) for k, v in self._bound_columns.items()))
        parts.append(hashlib.sha256(routing.encode()).hexdigest())
        parts.append(self.params.hash() if self.params else "")
        parts.append(self.output.hash())
        return hashlib.sha256(b"".join([bytes.fromhex(p) for p in parts])).hexdigest()


def _key_presence_values(
    row: Sequence[Any], specs: Sequence[tuple[int, str | None] | None]
) -> tuple[bool, ...]:
    return tuple(
        spec is None
        or (row[spec[0]] is not None and (spec[1] is None or row[spec[0]] == spec[1]))
        for spec in specs
    )


class Generator(UDFBase):
    """Inherit from this class to pass to `DataChain.gen()`."""

    is_output_batched = True
    udf_kind = "gen"
    prefetch: int = 2
    prefetch_max_size: int | None = None
    # (input, output) columns of a delta key each output row must keep
    delta_key: "Sequence[tuple[str, str]]" = ()
    delta_key_presence: "Sequence[tuple[KeyPresence, KeyPresence]]" = ()

    def read_columns(self) -> list[str]:
        keys = [spec[0] for spec, _ in self.delta_key_presence if spec is not None]
        return [*super().read_columns(), *(c for c, _ in self.delta_key), *keys]

    async def _prefetch_file(
        self, file: File, download_cb: Callback | None = None
    ) -> bool:
        target = file._prefetch_target()
        if not target.path or (
            self.prefetch_max_size is not None
            and not 0 < target.size <= self.prefetch_max_size
        ):
            return False
        return await target._prefetch(download_cb)

    def run(  # noqa: C901
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[Sequence[Any]]",
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[UDFPart]:
        self.setup()
        signal_names = list(self.signal_names)
        key_inputs = [udf_fields.index(column) for column, _ in self.delta_key]
        key_outputs = [signal_names.index(column) for _, column in self.delta_key]
        presence_inputs = [
            (udf_fields.index(spec[0]), spec[1]) if spec is not None else None
            for spec, _ in self.delta_key_presence
        ]
        presence_outputs = [
            (signal_names.index(spec[0]), spec[1]) if spec is not None else None
            for _, spec in self.delta_key_presence
        ]
        key_columns = [column for _, column in self.delta_key]
        key_columns += [
            spec[0]
            for left, right in self.delta_key_presence
            if (spec := right or left) is not None
        ]
        input_keys: dict[Any, tuple] = {}
        prefetch = self._effective_prefetch
        output_rows = OutputRows(self, catalog.warehouse)

        def _prepare_rows(udf_inputs) -> "abc.Generator[Sequence[Any], None, None]":
            prepare = self._input_reader(udf_fields, catalog, cache, download_cb)
            with safe_closing(udf_inputs):
                for row in udf_inputs:
                    prepared = prepare(row)
                    if key_columns:
                        input_keys[prepared[0]] = tuple(
                            row[i] for i in key_inputs
                        ) + _key_presence_values(row, presence_inputs)
                    yield prepared

        def _check_delta_key(key, output):
            values = tuple(output[i] for i in key_outputs) + _key_presence_values(
                output, presence_outputs
            )
            for expected, value, column in zip(key, values, key_columns, strict=True):
                if value != expected:
                    from datachain.delta import delta_key_changed_error

                    raise delta_key_changed_error(
                        column.split(DEFAULT_DELIMITER)[0], "gen()"
                    )

        def _stored_rows(result_objs, key) -> "Iterator[UDFResult | pa.RecordBatch]":
            pa = sys.modules.get("pyarrow")  # batches come only from arrow readers
            for result_obj in result_objs:
                if pa is not None and isinstance(result_obj, pa.RecordBatch):
                    if key is not None:
                        columns = (c.to_pylist() for c in result_obj.columns)
                        for values in zip(*columns, strict=True):
                            _check_delta_key(key, values)
                    encoded = output_rows.encode_batch(result_obj)
                    if isinstance(encoded, pa.RecordBatch):
                        yield encoded
                    else:
                        yield from encoded
                    continue
                values = output_rows.flatten(result_obj)
                if key is not None:
                    _check_delta_key(key, values)
                yield output_rows.encode(values)

        def _process_row(row_id, row):
            key = input_keys.pop(row_id) if key_columns else None
            result = self.process(*row)
            if result is None:
                result = iter(())
            with safe_closing(result) as result_objs:
                yield from _stored_rows(result_objs, key)

        prepared_inputs = _prepare_rows(udf_inputs)
        prepared_inputs = _prefetch_inputs(
            prepared_inputs,
            prefetch,
            download_cb=download_cb,
            remove_prefetched=bool(prefetch) and not cache,
            max_size=self.prefetch_max_size,
            max_bytes=self.prefetch_bytes,
            min_rows=self._prefetch_min_rows,
            prefetch_file=self._prefetch_file,
        )

        with (
            self._teardown_on_exit(),
            closing(prepared_inputs),
            _Progress(processed_cb) as progress,
        ):
            for row_id, *row in prepared_inputs:
                yield row_id, _process_row(row_id, row)
                progress.add(1)


class Aggregator(UDFBase):
    """Inherit from this class to pass to `DataChain.agg()`."""

    is_input_batched = True
    is_output_batched = True
    udf_kind = "agg"

    def run(
        self,
        udf_fields: Sequence[str],
        udf_inputs: Iterable[RowsOutputBatch],
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[UDFPart]:
        from datachain.data_storage.schema import PARTITION_COLUMN_ID

        self.setup()
        output_row = OutputRows(self, catalog.warehouse)
        prepare = self._input_reader(
            udf_fields, catalog, cache, download_cb, include_id=False
        )

        def _output(result_objs: Any) -> Iterator[tuple]:
            if result_objs is None:
                return
            with safe_closing(result_objs):
                yield from map(output_row, result_objs)

        # Check if partition_id is available (when partition_by is used)
        partition_id_idx = None
        if PARTITION_COLUMN_ID in udf_fields:
            partition_id_idx = list(udf_fields).index(PARTITION_COLUMN_ID)

        with self._teardown_on_exit():
            for batch in udf_inputs:
                # Get partition_id from first row if available (all rows in batch
                # share it). This is used to track which partition produced each
                # output for checkpoints
                input_id = None
                if partition_id_idx is not None:
                    input_id = batch[0][partition_id_idx]

                prepared_rows = [prepare(row) for row in batch]
                batched_args = zip(*prepared_rows, strict=False)
                # Convert aggregated column values to lists. This keeps behavior
                # consistent with the type hints promoted in the public API.
                udf_args = [
                    list(arg) if isinstance(arg, tuple) else arg for arg in batched_args
                ]
                output = _output(self.process(*udf_args))
                processed_cb.relative_update(len(batch))
                yield input_id, output
