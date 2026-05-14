import hashlib
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import attrs
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from pydantic import BaseModel

from datachain.asyn import AsyncMapper
from datachain.cache import temporary_cache
from datachain.dataset import RowDict
from datachain.hash_utils import hash_callable
from datachain.lib.convert.flatten import flatten
from datachain.lib.file import DataModel, File, FileError
from datachain.lib.utils import AbstractUDF, DataChainParamsError
from datachain.query.batch import (
    Batch,
    BatchingStrategy,
    NoBatching,
    Partition,
    RowsOutputBatch,
)
from datachain.utils import safe_closing, with_last_flag

logger = logging.getLogger("datachain")

if TYPE_CHECKING:
    from collections import abc
    from contextlib import AbstractContextManager

    from typing_extensions import Self

    from datachain.cache import Cache
    from datachain.catalog import Catalog
    from datachain.lib.signal_schema import SignalSchema
    from datachain.lib.udf_signature import UdfSignature
    from datachain.query.batch import RowsOutput

T = TypeVar("T", bound=Sequence[Any])


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

# Result type when calling the UDF wrapper around the actual
# Python function / class implementing it.
UDFResult = dict[str, Any]


@attrs.define(slots=False)
class UDFAdapter:
    inner: "UDFBase"
    output: UDFOutputSpec
    batch_size: int | None = None
    batch: int = 1

    def hash(self, include_body: bool = True) -> str:
        return self.inner.hash(include_body=include_body)

    def get_batching(self, use_partitioning: bool = False) -> BatchingStrategy:
        if use_partitioning:
            return Partition()

        if self.batch == 1:
            return NoBatching()
        if self.batch > 1:
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
    ) -> Iterator[Iterable[UDFResult]]:
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
    prefetch: int = 0

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
        batch: int = 1,
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
    ) -> Iterator[Iterable[UDFResult]]:
        raise NotImplementedError

    def _flatten_row(self, row):
        if len(self.output.values) > 1 and not isinstance(row, BaseModel):
            flat = []
            for obj in row:
                flat.extend(self._obj_to_list(obj))
            return tuple(flat)
        return row if isinstance(row, tuple) else tuple(self._obj_to_list(row))

    @staticmethod
    def _obj_to_list(obj):
        return flatten(obj) if isinstance(obj, BaseModel) else [obj]

    def _parse_row(
        self, row_dict: RowDict, catalog: "Catalog", cache: bool, download_cb: Callback
    ) -> list[Any]:
        assert self.params
        row = [row_dict[p] for p in self.params.to_udf_spec()]
        obj_row = self.params.row_to_objs(row)
        for obj in obj_row:
            self._set_stream_recursive(obj, catalog, cache, download_cb)
        return obj_row

    def _set_stream_recursive(
        self, obj: Any, catalog: "Catalog", cache: bool, download_cb: Callback
    ) -> None:
        """Recursively set the catalog stream on all File objects within an object."""
        if isinstance(obj, File):
            obj._set_stream(catalog, caching_enabled=cache, download_cb=download_cb)

        # Check all fields for nested File objects, but only for DataModel objects
        if isinstance(obj, DataModel):
            for field_name in type(obj).model_fields:
                field_value = getattr(obj, field_name, None)
                if isinstance(field_value, DataModel):
                    self._set_stream_recursive(field_value, catalog, cache, download_cb)

    def _prepare_row(
        self, row, udf_fields, catalog, cache, download_cb, include_id=False
    ):
        row_dict = RowDict(zip(udf_fields, row, strict=False))
        udf_input = self._parse_row(row_dict, catalog, cache, download_cb)
        if include_id:
            return row_dict["sys__id"], *udf_input
        return udf_input


def noop(*args, **kwargs):
    pass


async def _prefetch_input(
    row: T,
    download_cb: Callback | None = None,
    after_prefetch: "Callable[[], None]" = noop,
) -> T:
    for obj in row:
        if isinstance(obj, File) and obj.path:
            try:
                if await obj._prefetch(download_cb):
                    after_prefetch()
            except FileError as e:
                logger.warning(
                    "Skipping prefetch for '%s/%s': %s", obj.source, obj.path, e
                )
    return row


def _remove_prefetched(row: T) -> None:
    for obj in row:
        if isinstance(obj, File):
            catalog = obj._catalog
            assert catalog is not None
            try:
                catalog.cache.remove(obj)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to remove prefetched item %r: %s", obj.name, e)


def _prefetch_inputs(
    prepared_inputs: "Iterable[T]",
    prefetch: int = 0,
    download_cb: Callback | None = None,
    after_prefetch: Callable[[], None] | None = None,
    remove_prefetched: bool = False,
) -> "abc.Generator[T, None, None]":
    if not prefetch:
        yield from prepared_inputs
        return

    if after_prefetch is None:
        after_prefetch = noop
        if download_cb and hasattr(download_cb, "increment_file_count"):
            increment_file_count: Callable[[], None] = download_cb.increment_file_count
            after_prefetch = increment_file_count

    f = partial(_prefetch_input, download_cb=download_cb, after_prefetch=after_prefetch)
    mapper = AsyncMapper(f, prepared_inputs, workers=prefetch)
    with closing(mapper.iterate()) as row_iter:
        for row in row_iter:
            try:
                yield row  # type: ignore[misc]
            finally:
                if remove_prefetched:
                    _remove_prefetched(row)


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


class Mapper(UDFBase):
    """Inherit from this class to pass to `DataChain.map()`."""

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

        def _prepare_rows(udf_inputs) -> "abc.Generator[Sequence[Any], None, None]":
            with safe_closing(udf_inputs):
                for row in udf_inputs:
                    yield self._prepare_row(
                        row, udf_fields, catalog, cache, download_cb, include_id=True
                    )

        prepared_inputs = _prepare_rows(udf_inputs)
        prepared_inputs = _prefetch_inputs(
            prepared_inputs,
            self.prefetch,
            download_cb=download_cb,
            remove_prefetched=bool(self.prefetch) and not cache,
        )

        with closing(prepared_inputs):
            for id_, *udf_args in prepared_inputs:
                result_objs = self.process(*udf_args)
                udf_output = self._flatten_row(result_objs)
                output = [
                    {"sys__id": id_}
                    | dict(zip(self.signal_names, udf_output, strict=False))
                ]
                processed_cb.relative_update(1)
                yield output

        self.teardown()


class Generator(UDFBase):
    """Inherit from this class to pass to `DataChain.gen()`."""

    is_output_batched = True
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

        def _prepare_rows(udf_inputs) -> "abc.Generator[Sequence[Any], None, None]":
            with safe_closing(udf_inputs):
                for row in udf_inputs:
                    yield self._prepare_row(
                        row, udf_fields, catalog, cache, download_cb, include_id=True
                    )

        def _process_row(row):
            row_id, *row = row
            has_output = False
            result = self.process(*row)
            if result is None:
                result = iter(())
            with safe_closing(result) as result_objs:
                for result_obj, is_last in with_last_flag(result_objs):
                    has_output = True
                    udf_output = self._flatten_row(result_obj)
                    udf_output = dict(zip(self.signal_names, udf_output, strict=False))
                    udf_output["sys__input_id"] = row_id
                    udf_output["sys__partial"] = not is_last
                    udf_output["sys__empty"] = None
                    yield udf_output
            if not has_output:
                # Marker: records that this input was processed but yielded nothing.
                yield {
                    "sys__input_id": row_id,
                    "sys__partial": False,
                    "sys__empty": True,
                }

        prepared_inputs = _prepare_rows(udf_inputs)
        prepared_inputs = _prefetch_inputs(
            prepared_inputs,
            self.prefetch,
            download_cb=download_cb,
            remove_prefetched=bool(self.prefetch) and not cache,
        )

        with closing(prepared_inputs):
            for row in prepared_inputs:
                yield _process_row(row)
                processed_cb.relative_update(1)

        self.teardown()


class Aggregator(UDFBase):
    """Inherit from this class to pass to `DataChain.agg()`."""

    is_input_batched = True
    is_output_batched = True

    def run(
        self,
        udf_fields: Sequence[str],
        udf_inputs: Iterable[RowsOutputBatch],
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult]]:
        from datachain.data_storage.schema import PARTITION_COLUMN_ID

        self.setup()

        # Check if partition_id is available (when partition_by is used)
        partition_id_idx = None
        if PARTITION_COLUMN_ID in udf_fields:
            partition_id_idx = list(udf_fields).index(PARTITION_COLUMN_ID)

        for batch in udf_inputs:
            # Get partition_id from first row if available (all rows in batch share it)
            # This is used to track which partition produced each output for checkpoints
            input_id = None
            if partition_id_idx is not None:
                input_id = batch[0][partition_id_idx]

            prepared_rows = [
                self._prepare_row(row, udf_fields, catalog, cache, download_cb)
                for row in batch
            ]
            batched_args = zip(*prepared_rows, strict=False)
            # Convert aggregated column values to lists. This keeps behavior
            # consistent with the type hints promoted in the public API.
            udf_args = [
                list(arg) if isinstance(arg, tuple) else arg for arg in batched_args
            ]
            result_objs = self.process(*udf_args)
            if result_objs is None:
                result_objs = iter(())
            udf_outputs = (self._flatten_row(row) for row in result_objs)

            def _process_partition(udf_outputs, input_id):
                has_output = False
                for row, is_last in with_last_flag(udf_outputs):
                    has_output = True
                    udf_output = dict(zip(self.signal_names, row, strict=False))
                    udf_output["sys__input_id"] = input_id
                    udf_output["sys__partial"] = not is_last
                    udf_output["sys__empty"] = None
                    yield udf_output
                if not has_output:
                    yield {
                        "sys__input_id": input_id,
                        "sys__partial": False,
                        "sys__empty": True,
                    }

            output = _process_partition(udf_outputs, input_id)
            processed_cb.relative_update(len(batch))
            yield output

        self.teardown()
