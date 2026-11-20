import copy
import hashlib
import os
import sys
import tempfile
from collections.abc import Collection, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from fsspec.implementations.local import LocalFileSystem
from pyarrow._csv import ParseOptions
from pyarrow.dataset import CsvFileFormat, Dataset, ParquetFileFormat, dataset
from pyarrow.fs import FileInfo, FileType, FSSpecHandler, PyFileSystem

from datachain import json
from datachain.cache import Cache, ReadonlyCache
from datachain.catalog.catalog import clone_catalog_with_cache
from datachain.data_storage.codec import to_json_text
from datachain.fs.reference import ReferenceFileSystem
from datachain.lib.arrow_filter import to_arrow_filter
from datachain.lib.convert.flatten import classify_field, flatten, iter_field_columns
from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.convert.unflatten import ABSENT, unflatten_to_json_pos
from datachain.lib.data_model import (
    NULLABLE_SCALARS,
    TYPE_TAG_FIELD,
    arm_for_tag,
    arm_selector,
    dict_to_data_model,
    union_layout,
    unwrap_optional,
)
from datachain.lib.file import ArrowRow, File
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema, SignalSchemaError
from datachain.lib.udf import Generator
from datachain.lib.utils import normalize_col_names, unique_name
from datachain.progress import tqdm
from datachain.sql.types import Array, Binary, Boolean, DateTime, Float, Int64, String
from datachain.utils import safe_closing

SCHEMA_READ_THREADS = 32

if TYPE_CHECKING:
    from datasets.features.features import Features
    from pydantic import BaseModel

    from datachain.client import Client
    from datachain.lib.data_model import DataType, UnionLayout
    from datachain.lib.dc import DataChain

    # A parent's tag column, its layout, and the arm holding the path read.
    ParentTag = tuple[str, UnionLayout, Any]


DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY = b"DataChain SignalSchema"

_MIXED_FILES = (
    "With columns or output, files written by DataChain can't be read with other files"
)


def fix_pyarrow_format(format, parse_options=None):
    # Re-init invalid row handler: https://issues.apache.org/jira/browse/ARROW-17641
    if (
        format
        and isinstance(format, CsvFileFormat)
        and parse_options
        and isinstance(parse_options, ParseOptions)
    ):
        format.parse_options = parse_options
    return format


def is_parquet(format) -> bool:
    return format == "parquet" or isinstance(format, ParquetFileFormat)


def is_lance(format) -> bool:
    return format == "lance"


# CSV options that decide where a file's rows and columns start.
_CSV_READ_OPTIONS = (
    "column_names",
    "autogenerate_column_names",
    "skip_rows",
    "skip_rows_after_names",
)
_CSV_PARSE_OPTIONS = (
    "delimiter",
    "quote_char",
    "double_quote",
    "escape_char",
    "newlines_in_values",
    "ignore_empty_lines",
)


# A partial read prefetches files up to this size and reads larger ones in parts.
PARTIAL_READ_PREFETCH_MAX_SIZE = 8 * 1024 * 1024
# Row groups smaller than this on average are scanned in one pass rather than
# reading filter columns and rest columns separately per group.
SMALL_ROW_GROUP_BYTES = 16 * 2**20


class _PartialReadHandler(FSSpecHandler):
    """Lets pyarrow read the exact object a File describes.

    A File can come from a listing that is hours old or saved in a dataset, so the
    object at its path may have changed since. pyarrow's default handler opens
    whatever is at the path now. This handler opens the File through its client,
    which reads the File's version id when it has one.

    Parquet is read without read-ahead, so reading some columns downloads only
    those columns. CSV and JSON are read from the start, so they keep read-ahead.
    """

    def __init__(self, client: "Client", file: File, block_size: int | None):
        super().__init__(client.fs)
        self.client = client
        self.file = file
        self.block_size = block_size

    def get_file_info(self, paths):
        # Describes the recorded file without looking up the latest object, which a
        # versioned bucket may have deleted since.
        return [FileInfo(path, FileType.File) for path in paths]

    def open_input_file(self, path):
        f = self.client.open_object(
            self.file,
            use_cache=False,
            cb=self.file._download_cb,
            block_size=self.block_size,
        )
        return pa.PythonFile(f, mode="r")


def pyarrow_fs(file: File, format):
    client = file._catalog.get_client(file.source)  # type: ignore[union-attr]
    if isinstance(client.fs, LocalFileSystem):
        client.check_unchanged(file)
        return client.fs
    block_size = 0 if is_parquet(format) else None
    return PyFileSystem(_PartialReadHandler(client, file, block_size=block_size))


_LANCE_TABLE_DIRS = frozenset(
    ("data", "_versions", "_transactions", "_deletions", "_indices")
)


def _is_lance_table_file(file: File, client: "Client") -> bool:
    fs_path = file.get_fs_path().replace("\\", "/").rstrip("/")
    parts = fs_path.split("/")
    stop = 2 if "://" in fs_path else -1
    for i in range(len(parts) - 2, stop, -1):
        if parts[i] in _LANCE_TABLE_DIRS:
            if i > stop + 1 and parts[i - 1].endswith(".lance"):
                return True
            prefix = f"{'/'.join(parts[:i])}/"
            if client.fs.isdir(f"{prefix}_versions") and (
                parts[i] == "data" or client.fs.isdir(f"{prefix}data")
            ):
                return True
    return False


def _find_lance_blob(fields: Sequence[pa.Field], prefix: str = "") -> str | None:
    for f in fields:
        name = (
            prefix.rstrip(".") if f.name == "item" and prefix else f"{prefix}{f.name}"
        )
        if (f.metadata or {}).get(b"lance-encoding:blob") == b"true":
            return name
        children = [f.type.field(i) for i in range(f.type.num_fields)]
        if blob := _find_lance_blob(children, f"{name}."):
            return blob
    return None


def _lance_file_reader():
    try:
        from lance.file import LanceFileReader
    except ImportError as exc:
        raise ImportError(
            f"Missing dependency for Lance files: {exc}\n"
            "To install run:\n\n"
            "  pip install 'datachain[lance]'\n"
        ) from exc
    return LanceFileReader


def _local_lance_path(file: File) -> str:
    catalog = file._catalog
    assert catalog is not None
    client = catalog.get_client(file.source)
    if _is_lance_table_file(file, client):
        raise ValueError(
            f"'{file.path}' is inside a Lance table directory; "
            "read_lance reads standalone .lance data files"
        )
    if (path := file.get_local_path()) is not None:
        return path
    if isinstance(client.fs, LocalFileSystem):
        client.check_unchanged(file)
        return file.get_fs_path()
    if isinstance(cache := catalog.cache, ReadonlyCache):
        file = file.model_copy()
        file._catalog = clone_catalog_with_cache(
            catalog, Cache(cache.cache_dir, cache.tmp_dir)
        )
    file.ensure_cached()
    path = file.get_local_path()
    assert path is not None
    return path


_LANCE_JSON_META = {b"datachain:json": b"true"}


def _encode_lance_col(name: str) -> str:
    return name.replace("%", "%25").replace(".", "%2E")


def _decode_lance_col(name: str) -> str:
    return name.replace("%2E", ".").replace("%25", "%")


def write_lance(
    target: File,
    signal_schema: SignalSchema,
    column_chunks: Iterable[Sequence[list[Any]]],
    *,
    fs_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    _lance_file_reader()
    from lance.file import LanceFileWriter

    pq_schema = parquet_schema(signal_schema)
    lance_schema = pa.schema(
        [
            pa.field(_encode_lance_col(f.name), pa.string(), metadata=_LANCE_JSON_META)
            if isinstance(f.type, pa.JsonType)
            else f.with_name(_encode_lance_col(f.name))
            for f in pq_schema
        ]
    )
    catalog = target._catalog
    assert catalog is not None
    client = catalog.get_client(target.source)
    fs = (
        client.create_fs(**{**catalog.client_config, **fs_kwargs})
        if fs_kwargs
        else client.fs
    )
    dst_path = target.get_fs_path()
    local = isinstance(fs, LocalFileSystem)
    if local:
        staging = os.path.dirname(os.path.abspath(dst_path))
        os.makedirs(staging, exist_ok=True)
    else:
        staging = catalog.cache.tmp_dir
    null_free = set(pq_schema.names)
    with tempfile.TemporaryDirectory(dir=staging) as tmp_dir:
        staged = os.path.join(tmp_dir, "data.lance")
        with LanceFileWriter(staged, lance_schema, **kwargs) as writer:
            for chunk in column_chunks:
                arrays = [
                    _to_arrow(values, f.type)
                    for values, f in zip(chunk, pq_schema, strict=True)
                ]
                null_free -= {
                    f.name
                    for f, arr in zip(pq_schema, arrays, strict=True)
                    if arr.null_count
                }
                writer.write_batch(
                    pa.Table.from_arrays(
                        [getattr(a, "storage", a) for a in arrays], schema=lance_schema
                    )
                )
            values = _optional_unless_null_free(signal_schema.values, null_free)
            if values != signal_schema.values:
                signal_schema = SignalSchema(values)
            writer.add_schema_metadata(
                DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY.decode("utf-8"),
                json.dumps(signal_schema.serialize(), ensure_ascii=False),
            )
        if local:
            os.replace(staged, dst_path)
        else:
            fs.put_file(staged, dst_path)


class _LanceDataset:
    """Adapter reading a single Lance v2 data file via `LanceFileReader`."""

    def __init__(self, file: File, schema: pa.Schema | None = None):
        self._readers: dict[tuple[str, ...], Any] = {}
        reader_cls = _lance_file_reader()
        self._path = _local_lance_path(file)
        if not os.path.getsize(self._path):
            raise ValueError(f"Lance file '{file.path}' is 0 bytes")
        self._base_reader = reader_cls(self._path)
        meta = self._base_reader.metadata()
        if (blob := _find_lance_blob(list(meta.schema))) is not None:
            self.close()
            raise ValueError(
                f"Lance blob column '{blob}' in '{file.path}' is not "
                "supported; read_lance reads standalone .lance data files"
            )
        self.num_rows: int = meta.num_rows
        self._dc = _written_by_datachain(meta.schema)
        self._physical_schema: pa.Schema = (
            pa.schema(
                [
                    pa.field(
                        _decode_lance_col(f.name),
                        pa.json_()
                        if (f.metadata or {}).get(b"datachain:json") == b"true"
                        else f.type,
                    )
                    for f in meta.schema
                ],
                metadata=meta.schema.metadata,
            )
            if self._dc
            else meta.schema
        )
        self.schema: pa.Schema = schema or self._physical_schema

    def _raw_name(self, col: str) -> str:
        return _encode_lance_col(col) if self._dc else col

    def _reader(self, present: tuple[str, ...]) -> Any:
        # Lance's projection parser rejects empty column names.
        if "" in present or list(present) == self._physical_schema.names:
            return self._base_reader
        if (reader := self._readers.get(present)) is None:
            escaped = [f"`{self._raw_name(c).replace('`', '``')}`" for c in present]
            reader = _lance_file_reader()(self._path, columns=escaped)
            self._readers[present] = reader
        return reader

    def _read(
        self,
        columns: Sequence[str],
        take: pa.Array | None = None,
        *,
        start: int,
        n: int,
    ) -> pa.Table:
        present = tuple(c for c in columns if c in self._physical_schema.names)
        raw = None
        if present and n:
            r = self._reader(present)
            if take is not None:
                idx = (pc.add(take, start) if start else take).to_pylist()
                raw = r.take_rows(idx, batch_size=len(idx)).to_table()
            else:
                raw = r.read_range(start, n, batch_size=n).to_table()
        rows = n if take is None else len(take)
        fields = [self.schema.field(c) for c in columns]
        return pa.Table.from_arrays(
            [
                raw.column(self._raw_name(f.name)).cast(f.type)
                if raw is not None and f.name in present
                else pa.nulls(rows, f.type)
                for f in fields
            ],
            schema=pa.schema(fields, metadata=self.schema.metadata),
        )

    def chunks(self, step: int):
        for start in range(0, self.num_rows, step):
            n = min(step, self.num_rows - start)
            yield start, n, partial(self._read, start=start, n=n)

    def to_batches(self, **kw: Any) -> Iterator[pa.RecordBatch]:
        columns = list(kw.get("columns") or self.schema.names)
        step = kw.get("batch_size") or ArrowGenerator.DEFAULT_BATCH_SIZE
        for _, _, read in self.chunks(step):
            yield from read(columns).to_batches(step)

    def filtered_batches(
        self,
        filter_expr: pc.Expression,
        filter_columns: Sequence[str],
        columns: Sequence[str],
        batch_size: int,
    ) -> Iterator[tuple[pa.Array, pa.RecordBatch]]:
        keys = [c for c in filter_columns if c in self.schema.names]
        rest = [c for c in columns if c not in keys]
        index = unique_name("index", self.schema.names)
        with tqdm(desc="Parsed by pyarrow", unit="rows", leave=False) as pbar:
            for start, num_rows, read in self.chunks(ArrowGenerator.DEFAULT_BATCH_SIZE):
                hits = (
                    read(keys)
                    .append_column(index, pa.arange(0, num_rows))
                    .filter(filter_expr)
                )
                pbar.update(num_rows)
                if not hits.num_rows:
                    continue
                cols = dict(zip(hits.column_names, hits.columns, strict=True))
                if rest:
                    take = hits[index] if hits.num_rows < num_rows else None
                    data = read(rest, take)
                    cols.update(zip(rest, data.columns, strict=True))
                cols[index] = pc.add(hits[index], start)
                table = pa.table(
                    [cols[c] for c in [*columns, index]], [*columns, index]
                )
                for batch in table.to_batches(batch_size):
                    yield batch.column(index), batch.drop_columns([index])

    def take(self, indices: Sequence[int]) -> pa.Table:
        if any(i < 0 or i >= self.num_rows for i in indices):
            raise IndexError(f"Row index out of range ({self.num_rows} rows)")
        rows = sorted(set(indices))
        table = self._read(
            self.schema.names, pa.array(rows, pa.int64()), start=0, n=len(rows)
        )
        return table.take([rows.index(i) for i in indices])

    def close(self) -> None:
        self._readers.clear()
        self._base_reader = None


def read_batches(
    ds: Dataset | _LanceDataset, nrows: int | None = None, **kw: Any
) -> Iterator[tuple[int, pa.RecordBatch]]:
    """Yield the file's batches up to nrows rows, each with its first row's
    index."""
    start, remaining = 0, nrows
    with (
        tqdm(desc="Parsed by pyarrow", unit="rows", total=nrows, leave=False) as pbar,
        safe_closing(ds.to_batches(**kw)) as batches,
    ):
        for batch in batches:
            if remaining is not None:
                batch = batch.slice(0, remaining)
                remaining -= batch.num_rows
            yield start, batch
            start += batch.num_rows
            pbar.update(batch.num_rows)
            if remaining == 0:
                break


def _scan_groups(
    frag: Any,
    groups: Sequence[Any],
    starts: np.ndarray,
    schema: pa.Schema,
    columns: list[str],
    filter: pc.Expression,
    batch_size: int,
    index: str,
    pbar: Any,
) -> Iterator[tuple[pa.Array, pa.RecordBatch]]:
    part = frag.subset(row_group_ids=[rg.id for rg in groups])
    rgs = iter(groups)
    start = left = 0
    with safe_closing(
        part.to_batches(schema=schema, columns=columns, batch_size=batch_size)
    ) as batches:
        for batch in batches:
            while left == 0:
                rg = next(rgs)
                start, left = int(starts[rg.id]), rg.num_rows
            indices = pa.arange(start, start + batch.num_rows)
            start += batch.num_rows
            left -= batch.num_rows
            pbar.update(batch.num_rows)
            hits = batch.append_column(index, indices).filter(filter)
            if hits.num_rows:
                yield hits.column(index), hits.drop_columns([index])


def _scan_group_rest(
    part: Any,
    hits: pa.Table,
    start: int,
    schema: pa.Schema,
    columns: list[str],
    keys: list[str],
    rest: list[str],
    batch_size: int,
    index: str,
) -> Iterator[tuple[pa.Array, pa.RecordBatch]]:
    key_cols = {c: hits[c].combine_chunks() for c in keys}
    indices = pc.add(hits[index], start).combine_chunks()
    pos = hits[index].to_numpy()
    with safe_closing(
        part.to_batches(schema=schema, columns=rest, batch_size=batch_size)
    ) as batches:
        offset = lo = 0
        while lo < len(pos):
            data = next(batches)
            end = offset + data.num_rows
            hi = int(np.searchsorted(pos, end))
            if hi > lo:
                n = hi - lo
                data = data.take(pa.array(pos[lo:hi] - offset))
                read = {
                    **{c: col.slice(lo, n) for c, col in key_cols.items()},
                    **dict(zip(rest, data.columns, strict=True)),
                }
                yield (
                    indices.slice(lo, n),
                    pa.RecordBatch.from_arrays(
                        [read[c] for c in columns], names=columns
                    ),
                )
                lo = hi
            offset = end


def filter_batches(
    ds: Dataset,
    filter: pc.Expression,
    filter_columns: Sequence[str],
    **kw: Any,
) -> Iterator[tuple[pa.Array, pa.RecordBatch]]:
    """Yield the rows the filter keeps. Small row groups are scanned in one
    pass; large ones read filter columns first, then scan four groups at a time
    after two match until a window has no match."""
    (frag,) = ds.get_fragments()
    schema, columns = ds.schema, kw.get("columns") or ds.schema.names
    keys = [c for c in filter_columns if c in schema.names]
    rest = [c for c in columns if c not in keys]
    batch_size = kw["batch_size"]
    starts = np.cumsum([0, *(rg.num_rows for rg in frag.row_groups)])
    groups = frag.subset(filter, schema=schema).row_groups
    if not groups:
        return
    small = sum(rg.total_byte_size for rg in groups) < SMALL_ROW_GROUP_BYTES * len(
        groups
    )
    index = unique_name("index", schema.names)
    with tqdm(desc="Parsed by pyarrow", unit="rows", leave=False) as pbar:
        if not rest or small:
            yield from _scan_groups(
                frag, groups, starts, schema, columns, filter, batch_size, index, pbar
            )
            return
        i = matched = 0
        while i < len(groups):
            if matched == 2:
                window = groups[i : i + 4]
                i += len(window)
                matched = 0
                for indices, batch in _scan_groups(
                    frag,
                    window,
                    starts,
                    schema,
                    columns,
                    filter,
                    batch_size,
                    index,
                    pbar,
                ):
                    if batch.num_rows:
                        matched = 2
                    yield indices, batch
                continue
            rg = groups[i]
            i += 1
            part = frag.subset(row_group_ids=[rg.id])
            hits = (
                part.to_table(schema=schema, columns=keys)
                .append_column(index, pa.arange(0, rg.num_rows))
                .filter(filter)
            )
            pbar.update(rg.num_rows)
            if not hits.num_rows:
                continue
            matched += 1
            yield from _scan_group_rest(
                part,
                hits,
                int(starts[rg.id]),
                schema,
                columns,
                keys,
                rest,
                batch_size,
                index,
            )


class ArrowGenerator(Generator):
    # Rows per scanned batch: small enough to bound memory over many files.
    DEFAULT_BATCH_SIZE = 2**14
    # A predicate that every row SQL filters in matches, the columns it reads,
    # and the hash of the SQL it was translated from.
    filter: pc.Expression | None = None
    filter_columns: tuple[str, ...] = ()
    filter_hash: str = ""

    def __init__(
        self,
        input_schema: pa.Schema | None,
        output_schema: type["BaseModel"],
        source: bool = True,
        nrows: int | None = None,
        columns: list[str] | None = None,
        written_by_datachain: bool | None = None,
        output_from_types: bool = False,
        prefetch_max_size: int | None = None,
        projectable: bool = False,
        **kwargs,
    ):
        """
        Generator for getting rows from tabular files.

        Parameters:

        input_schema : Optional pyarrow schema for validation.
        output_schema : Pydantic model of the output.
        source : Whether to include info about the source file.
        nrows : Optional row limit per file.
        columns : Optional columns to read from each file by name, in output order.
            For files written by DataChain, signal paths such as "p.x".
        written_by_datachain : Whether every file must be written by DataChain, or
            none. Not checked if None.
        output_from_types : Whether output_schema was built from types alone, with no
            validators, so rows of plain values can skip it.
        prefetch_max_size : Files larger than this, or of unknown size, aren't
            prefetched.
        projectable : Whether the query planner may narrow this read to the
            columns that later steps use.
        kwargs: Parameters to pass to pyarrow.dataset.dataset.
        """
        super().__init__()
        self.prefetch_max_size = prefetch_max_size
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.source = source
        self.nrows = nrows
        self.columns = columns
        self.written_by_datachain = written_by_datachain
        self.output_from_types = output_from_types
        # Every file of a read of DataChain files is read by its column names, even
        # one whose metadata a tool such as Polars dropped when rewriting it.
        self.datachain_read = _written_by_datachain(input_schema)
        self.projectable = projectable and not self.datachain_read
        self.parse_options = kwargs.pop("parse_options", None)
        self.kwargs = kwargs

    def hash(self, include_body: bool = True) -> str:
        # The read schema, columns and pushed filter decide what is read, so reads
        # that differ only in them must not reuse each other's results.
        schema = self.input_schema.to_string() if self.input_schema else ""
        read = schema + json.dumps(self.columns) + self.filter_hash
        return hashlib.sha256((super().hash(include_body) + read).encode()).hexdigest()

    def project(
        self, names: set[str] | None, filters: Sequence[Any] = ()
    ) -> "ArrowGenerator | None":
        """Return a copy reading only the top-level signals used by ``names`` (all
        if None) and the rows ``filters`` may keep, or None if it reads the same."""
        if (
            not self.projectable
            or self.delta_key
            or not self.output_schema.model_fields
        ):
            return None
        assert self.input_schema is not None
        assert self.output is not None
        kept = {
            name: anno
            for name, anno in self.output.values.items()
            if names is None
            or not names.isdisjoint(SignalSchema({name: anno}).db_signals())
        }
        data = {k: v for k, v in kept.items() if k in self.output_schema.model_fields}
        raw = normalize_col_names(self.input_schema.names)
        if not data:
            schema = self.input_schema
            pick = min(
                self.output_schema.model_fields,
                key=lambda k: (
                    (0, t.bit_width)
                    if pa.types.is_primitive(t := schema.field(raw[k]).type)
                    else (1, 0)
                ),
            )
            kept[pick] = data[pick] = self.output.values[pick]
        fmt = self.kwargs.get("format")
        pushed = (
            to_arrow_filter(filters, self.input_schema, data)
            if is_parquet(fmt) or is_lance(fmt)
            else None
        )
        if len(kept) == len(self.output.values) and pushed is None:
            return None
        projected = copy.copy(self)
        projected.columns = [raw[k] for k in data]
        projected.output_schema = dict_to_data_model(
            self.output_schema.__name__, data, optional=False
        )
        projected.source = len(kept) > len(data)
        projected.output = SignalSchema(kept)
        if pushed is not None:
            projected.filter, projected.filter_columns, projected.filter_hash = pushed
        if is_parquet(fmt):
            projected.prefetch_max_size = PARTIAL_READ_PREFETCH_MAX_SIZE
        return projected

    def process(self, file: File):
        if file._caching_enabled:
            file.ensure_cached()
        kwargs = self.kwargs
        if format := kwargs.get("format"):
            kwargs["format"] = fix_pyarrow_format(format, self.parse_options)

        # With datachain_read, each file is read with its own columns and types.
        in_schema = None if self.datachain_read else self.input_schema
        with safe_closing(_file_dataset(file, kwargs, in_schema)) as ds:
            yield from self._rows(file, ds)

    def _rows(self, file: File, ds: Dataset | _LanceDataset):
        if self.datachain_read:
            _check_type_tags(file, ds.schema)

        use_datachain_schema = _written_by_datachain(ds.schema)
        if self.written_by_datachain not in (None, use_datachain_schema):
            kind = "is" if use_datachain_schema else "isn't"
            raise ValueError(
                f"{_MIXED_FILES}, and '{file.path}' {kind} written by DataChain"
            )

        source_kwargs = self._source_kwargs() if self.source else None
        schema = ds.schema
        kw: dict[str, Any] = {
            "batch_size": min(self.DEFAULT_BATCH_SIZE, self.nrows or sys.maxsize),
            "batch_readahead": 1,
        }
        if (names := self._flat_columns(schema)) is not None:
            kw["columns"] = names
            yield from self._column_batches(self._batches(ds, kw), file, source_kwargs)
            return
        if self.datachain_read:
            for index, obj in enumerate(self._signal_objects(ds, kw)):
                yield self._process_record(obj, file, index, source_kwargs)
            return
        if self.columns is not None:
            present = set(schema.names)
            kw["columns"] = [c for c in self.columns if c in present]
            # Decode Hugging Face features only for the columns being read.
            fields = [schema.field(c) for c in kw["columns"]]
            schema = pa.schema(fields, metadata=schema.metadata)
        hf_schema = _get_hf_schema(schema)
        for start, batch in self._batches(ds, kw):
            # Maps are read as dicts. Duplicate keys raise an error.
            records = batch.to_pylist(maps_as_pydicts="strict")
            indices = (
                range(start, start + len(records))
                if isinstance(start, int)
                else start.to_pylist()
            )
            for index, record in zip(indices, records, strict=True):
                obj = self._process_non_datachain_record(record, hf_schema)
                yield self._process_record(obj, file, index, source_kwargs)

    def _batches(
        self, ds: Dataset | _LanceDataset, kw: dict[str, Any]
    ) -> Iterator[tuple[int | pa.Array, pa.RecordBatch]]:
        if self.filter is not None:
            if isinstance(ds, _LanceDataset):
                columns = kw.get("columns") or ds.schema.names
                return ds.filtered_batches(
                    self.filter, self.filter_columns, columns, kw["batch_size"]
                )
            return filter_batches(ds, self.filter, self.filter_columns, **kw)
        return read_batches(ds, self.nrows, **kw)

    def _source_kwargs(self) -> dict[str, Any]:
        """Return the read options that each row's source keeps."""
        kwargs = self.kwargs
        # Can't JSON-serialize CsvFileFormat; may lose formatting options.
        if isinstance(format := kwargs.get("format"), CsvFileFormat):
            # Keep the options that split the file into the same rows and
            # columns, so the row can be found again.
            read = format.default_fragment_scan_options.read_options
            parse = format.parse_options
            # The format doesn't return its handler (ARROW-17641).
            handler = getattr(self.parse_options, "invalid_row_handler", None)
            csv = {
                "read_options": {n: getattr(read, n) for n in _CSV_READ_OPTIONS},
                "parse_options": {n: getattr(parse, n) for n in _CSV_PARSE_OPTIONS},
                # A read that finished skipped every invalid row, so the
                # handler can be replaced by skipping them all.
                "skip_invalid_rows": handler is not None,
            }
            # A copy, so the files read after this one keep their CSV options.
            kwargs = {**kwargs, "format": "csv", "csv": csv}
        return kwargs

    def _flat_columns(self, schema: pa.Schema) -> list[str] | None:
        """Return the columns that the output fields take, when each takes its
        values as pyarrow gives them. Then no model is built per row. Else None."""
        if not self.output_from_types:
            return None
        if self.datachain_read:
            # Its columns don't follow the output fields: its own source comes first.
            return None
        fields = list(self.output_schema.model_fields.values())
        names = (self.columns or schema.names)[: len(fields)]
        for name, field in zip(names, fields, strict=True):
            i = schema.get_field_index(name)
            if i < 0 or not _is_plain_type(
                col_type := schema.field(i).type, top_level=True
            ):
                return None
            if classify_field(field.annotation).inner != arrow_type_mapper(col_type):
                return None
        return names

    def _column_batches(
        self,
        batches: Iterator[tuple[int | pa.Array, pa.RecordBatch]],
        file: File,
        source_kwargs: dict[str, Any] | None,
    ) -> Iterator[pa.RecordBatch]:
        """The batches as the output's leaf columns, with no row built in Python."""
        if source_kwargs is not None:
            # Every row's source is the same but its index: the file's columns,
            # the index, then the read options.
            source = ArrowRow(file=file, index=0, kwargs=source_kwargs)
            *file_values, _, stored_options = map(_json_text, flatten(source))
        for start, batch in batches:
            if not batch.num_rows:
                continue
            columns = [_microseconds(c) for c in batch.columns]
            if source_kwargs is not None:
                n = batch.num_rows
                indices = (
                    pa.array(np.arange(start, start + n, dtype=np.int64))
                    if isinstance(start, int)
                    else start
                )
                columns = [
                    *(pa.repeat(v, n) for v in file_values),
                    indices,
                    pa.repeat(stored_options, n),
                    *columns,
                ]
            yield pa.RecordBatch.from_arrays(
                columns, names=[str(i) for i in range(len(columns))]
            )

    def _signal_objects(
        self, ds: Dataset | _LanceDataset, kw: dict[str, Any]
    ) -> Iterator["BaseModel"]:
        """Read each row of a file written by DataChain into the output model by
        column position, as rows are read from the DB. A column the file lacks
        leaves its field out, so the field takes its default."""
        model, schema = self.output_schema, ds.schema
        paths = self.columns or list(model.model_fields)
        parents: dict[str, ParentTag] = {}
        if self.columns is not None and (
            signal_schema := _get_datachain_schema(schema)
        ):
            parents = _parent_tags(signal_schema, schema.names, paths)
        names = [
            ".".join(column)
            for info, path in zip(model.model_fields.values(), paths, strict=True)
            for column in iter_field_columns(info.annotation, (path,))
        ]
        wanted = {*names, *(tag for tag, _, _ in parents.values())}
        kw["columns"] = [c for c in schema.names if c in wanted]
        json_cols = json_columns(schema)
        for _, batch in read_batches(ds, self.nrows, **kw):
            values = {
                name: column_values(column, name in json_cols)
                for name, column in zip(batch.schema.names, batch.columns, strict=True)
            }
            for path, (tag, layout, arm) in parents.items():
                # The parent's tag stands in for the path's own, set where it
                # selects the arm holding the path.
                values[f"{path}.{TYPE_TAG_FIELD}"] = [
                    t if arm_for_tag(layout, t) is arm else None for t in values[tag]
                ]
            absent = [ABSENT] * batch.num_rows
            # An output with no fields still takes one object per row.
            columns = [values.get(n, absent) for n in names] or [absent]
            for row in zip(*columns, strict=True):
                yield model(**unflatten_to_json_pos(model, row, 0, _build_model)[0])

    def _process_record(
        self,
        obj: "BaseModel",
        file: File,
        index: int,
        source_kwargs: dict[str, Any] | None,
    ):
        vals = [obj]
        if source_kwargs is not None:
            arrow_file = ArrowRow(file=file, index=index, kwargs=source_kwargs)
            vals = [arrow_file, *vals]

        if self.output is not None and len(self.output.values) == 1:
            # A single output signal takes its value itself, not a list of values.
            ((name, signal_type),) = self.output.values.items()
            row = vals[0]
            return row if type(row) is signal_type else getattr(row, name)
        return vals

    def _process_non_datachain_record(
        self,
        record: dict[str, Any],
        hf_schema: tuple["Features", dict[str, "DataType"]] | None,
    ):
        names = self.columns or list(record)
        fields = self.output_schema.model_fields
        values = {}
        for (field, field_info), name in zip(fields.items(), names, strict=False):
            value = record.get(name)
            anno = field_info.annotation
            if hf_schema and (feat := hf_schema[0].get(name)) is not None:
                from datachain.lib.hf import convert_feature

                value = convert_feature(value, feat, anno)
            elif value is not None and ModelStore.is_pydantic(anno):
                value = anno(**value)  # type: ignore[misc]
            values[field] = value
        return self.output_schema(**values)


def column_values(column: pa.Array, from_json: bool) -> list[Any]:
    """The column's values, parsed where `from_json` and it holds JSON text. Maps
    are read as dicts, and duplicate keys raise an error."""
    values = column.to_pylist(maps_as_pydicts="strict")
    if from_json and isinstance(column.type, pa.JsonType):
        return [None if v is None else json.loads(v) for v in values]
    return values


def json_columns(schema: pa.Schema | None) -> set[str]:
    """The columns files written by DataChain with this schema store as JSON."""
    signal_schema = _get_datachain_schema(schema)
    if signal_schema is None:
        return set()
    fields = parquet_schema(signal_schema)
    return {f.name for f in fields if isinstance(f.type, pa.JsonType)}


def _optional_unless_null_free(
    signals: dict[str, Any], null_free: Collection[str]
) -> dict[str, Any]:
    """Make each signal optional unless its column, or its union tag, is in
    `null_free`. A model is left as declared: its fields have their own columns."""
    return {
        path: anno
        if ModelStore.is_pydantic(anno)
        or path in null_free
        or f"{path}.{TYPE_TAG_FIELD}" in null_free
        else anno | None
        for path, anno in signals.items()
    }


def _dataset_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    parse_options = kwargs.pop("parse_options", None)
    if format := kwargs.get("format"):
        kwargs["format"] = fix_pyarrow_format(format, parse_options)
    return kwargs


def _file_dataset(
    file: File, kwargs: dict[str, Any], schema: pa.Schema | None = None
) -> Dataset | _LanceDataset:
    if is_lance(kwargs.get("format")):
        return _LanceDataset(file, schema)
    fs_path = file.get_fs_path()
    if cache_path := file.get_local_path():
        fs: Any = ReferenceFileSystem({fs_path: [cache_path]})
    else:
        fs = pyarrow_fs(file, kwargs.get("format"))
    return dataset(fs_path, schema=schema, filesystem=fs, **kwargs)


def _file_datasets(chain: "DataChain", **kwargs) -> Iterator[Dataset | _LanceDataset]:
    kwargs = _dataset_kwargs(kwargs)
    for (file,) in chain.to_iter("file"):
        yield _file_dataset(file, kwargs)  # type: ignore[arg-type]


def file_schemas(chain: "DataChain", **kwargs) -> list[pa.Schema]:
    """Return the schema of each file in the chain, by path, so the first file
    gives the saved DataChain schema, as in `read_columns`."""
    kwargs = _dataset_kwargs(kwargs)
    files = chain.order_by("file.path").to_values("file")

    def file_schema(file: File) -> pa.Schema:
        with safe_closing(_file_dataset(file, kwargs)) as ds:
            schema = ds.schema
        _check_type_tags(file, schema)
        return schema

    # one remote footer read per file
    with ThreadPoolExecutor(max(1, min(SCHEMA_READ_THREADS, len(files)))) as pool:
        schemas = list(pool.map(file_schema, files))
    if not schemas:
        raise ValueError(
            "Cannot infer schema (no files to process or can't access them)"
        )
    return schemas


def infer_output(
    schemas: list[pa.Schema],
    col_names: Sequence[str] | None,
    columns: Sequence[str] | None,
    replace_source: bool,
) -> tuple[pa.Schema, dict[str, type], list[str] | None]:
    """Return the schema to read the files with, the output types, and, for files
    written by DataChain, the signal each output field reads (None when the fields
    take the columns read, in order).

    Files written by DataChain pick signals as `select()` does. With `columns` or
    `col_names`, they can't be mixed with other files.
    """
    kinds = {_written_by_datachain(s) for s in schemas}
    if (columns is not None or col_names) and len(kinds) > 1:
        raise ValueError(_MIXED_FILES)
    signal_schema = _get_datachain_schema(schemas[0])
    if not signal_schema:
        schema, names = merge_schemas(schemas, columns)
        return schema, schema_to_output(schema, col_names or names)[0], None
    for s in schemas[1:]:
        other = _get_datachain_schema(s)
        non_null = (
            {k for k, v in other.values.items() if not unwrap_optional(v)[1]}
            if other
            else ()
        )
        signal_schema = SignalSchema(
            _optional_unless_null_free(signal_schema.values, non_null)
        )

    signals = _signal_paths(signal_schema, columns, replace_source)
    if columns is None:
        schema = _unify_datachain_schemas(schemas)
    else:
        columns = list(signals)  # as stored, e.g. "u.Model.x" for "u.x"
        schema = _picked_schema(schemas, columns)
    if col_names:
        _check_fields_fit(len(col_names), len(signals))
        names = list(normalize_col_names(col_names))
        return schema, dict(zip(names, signals.values(), strict=True)), list(signals)
    if columns is None:
        return schema, signals, None
    # Picked signals keep their names, as select() returns them.
    output = signal_schema.to_partial(*columns).values
    return schema, output, list(output)


def read_columns(
    chain: "DataChain",
    columns: Sequence[str] | None,
    model: type["BaseModel"],
    replace_source: bool,
    **kwargs,
) -> tuple[list[str], pa.Schema | None]:
    """Return the columns the output fields read, in order, and for files written
    by DataChain, the schema to read them with (None for other files).

    Like pyarrow, they come from the first file (by path). Each file is read by
    these names with its own types. Partition and pandas index columns at the end
    are left out when output has no fields for them.
    """
    with safe_closing(
        next(_file_datasets(chain.order_by("file.path").limit(1), **kwargs))
    ) as ds:
        schema = ds.schema
        fragments = ds.get_fragments() if isinstance(ds, Dataset) else None
    fields = len(model.model_fields)
    if signal_schema := _get_datachain_schema(schema):
        signals = _signal_paths(signal_schema, columns, replace_source)
        _check_fields_fit(fields, len(signals))
        for (field, info), (path, stored) in zip(
            model.model_fields.items(), signals.items(), strict=True
        ):
            _check_output_shape(field, info.annotation, path, stored)
        return list(signals), _picked_schema([schema], list(signals))

    if columns is not None:
        schema, _ = merge_schemas([schema], columns)
    elif len(schema) > fields and fragments is not None:
        physical = next(fragments).physical_schema
        extra = set(schema.names) - set(physical.names)
        index = (schema.pandas_metadata or {}).get("index_columns", [])
        extra |= {c for c in index if isinstance(c, str)}
        if set(schema.names[fields:]) <= extra:
            schema = pa.schema(list(schema)[:fields])
    _check_fields_fit(fields, len(schema))
    return schema.names, None


def _check_fields_fit(fields: int, read: int) -> None:
    if fields != read:
        raise ValueError(
            f"output has {fields} fields but {read} columns are read. "
            "output names the columns read, in order; pick columns with columns=."
        )


def merge_schemas(
    schemas: list[pa.Schema], columns: Sequence[str] | None = None
) -> tuple[pa.Schema, list[str] | None]:
    """Merge the files' schemas. With `columns`, keep only those, in that order.

    Column names can be written as in the file or as DataChain cleans them, e.g.
    "Unit Price (USD)" or "unit_price_usd". With `columns`, also returns the
    cleaned names of the kept columns, cleaned over the whole header as in a
    full read.
    """
    _check_columns(columns)
    if columns is None:
        return pa.unify_schemas(schemas), None
    names = list(dict.fromkeys(n for s in schemas for n in s.names))
    cleaned = normalize_col_names(names)
    raw = {n: n for n in names} | cleaned
    if missing := [c for c in columns if c not in raw]:
        raise ValueError(f"Columns {missing} not found. Available columns: {names}")
    selected = [raw[c] for c in columns]
    if len(set(selected)) < len(selected):
        raise ValueError(f"Columns {list(columns)} name a column more than once")
    # Only the selected columns need compatible types across files.
    wanted = set(selected)
    merged = pa.unify_schemas(
        [pa.schema([f for f in s if f.name in wanted]) for s in schemas]
    )
    schema = pa.schema(
        [merged.field(c) for c in selected], metadata=schemas[0].metadata
    )
    clean = {r: c for c, r in cleaned.items()}
    return schema, [clean[c] for c in selected]


def _check_columns(columns: Sequence[str] | None) -> None:
    if columns is not None and (isinstance(columns, str) or not columns):
        raise ValueError("columns must be a non-empty list of column names")


def _signal_paths(
    signal_schema: SignalSchema, columns: Sequence[str] | None, replace_source: bool
) -> dict[str, "DataType"]:
    """Return the signals read from files written by DataChain, with the types they
    read as.

    They are `columns`, or else the top-level signals. With `replace_source`, a
    stored `source` from an earlier read is left out, as the reader adds its own.
    A signal inside an optional model or a union is None where that parent
    doesn't hold it, so it reads as optional.
    """
    _check_columns(columns)
    if columns is None:
        paths = [
            name
            for name, anno in signal_schema.values.items()
            if not (replace_source and _is_read_source(name, anno))
        ]
    else:
        paths = list(columns)
    try:
        signals = signal_schema.resolve(*paths).values
    except SignalSchemaError as e:
        available = list(signal_schema.values)
        raise ValueError(f"{e}. Available signals: {available}") from e
    # A readable path such as "u.x" resolves to the same signal as "u.Model.x".
    if len(signals) < len(paths):
        raise ValueError(f"Columns {list(paths)} name a column more than once")
    return {
        path: anno | None if signal_schema.tagged_parent(path) else anno  # type: ignore[misc]
        for path, anno in signals.items()
    }


def _is_read_source(name: str, anno: Any) -> bool:
    """Whether a stored signal is the `source` an earlier tabular read added."""
    return name == "source" and unwrap_optional(anno)[0] is ArrowRow


def _check_output_shape(field: str, anno: Any, path: str, stored: Any) -> None:
    """Raise unless the output field has the shape of the signal it reads: a model,
    optional where the signal is, a union of the same types, or a plain value,
    down through the fields it declares and the union arms' fields."""
    output_kind, stored_kind = classify_field(anno), classify_field(stored)
    if output_kind.is_model != stored_kind.is_model:
        kind = "a model" if output_kind.is_model else "not a model"
        raise ValueError(f"output field {field} is {kind}, unlike {path}")
    output_layout, stored_layout = union_layout(anno), union_layout(stored)
    if not output_kind.is_model:
        if (arms := _union_arms(output_layout)) != _union_arms(stored_layout):
            kind = f"a union of {arms}" if arms else "not a union"
            raise ValueError(f"output field {field} is {kind}, unlike {path}")
        if output_layout is None:
            return
    if stored_kind.is_optional and not output_kind.is_optional:
        raise ValueError(
            f"output field {field} must be optional, as {path} can be None"
        )
    # A union reads its own tag, so it may be optional where the signal isn't.
    if output_kind.is_model and output_kind.is_optional and not stored_kind.is_optional:
        raise ValueError(
            f"output field {field} can't be optional, as {path} is never None"
        )
    if output_layout and stored_layout and not output_kind.is_model:
        for arm, stored_arm in zip(output_layout.arms, stored_layout.arms, strict=True):
            name = arm_selector(arm)
            _check_output_shape(f"{field}.{name}", arm, f"{path}.{name}", stored_arm)
        return
    fields = stored_kind.inner.model_fields
    for name, info in output_kind.inner.model_fields.items():
        if name in fields:
            _check_output_shape(
                f"{field}.{name}",
                info.annotation,
                f"{path}.{name}",
                fields[name].annotation,
            )


def _union_arms(layout: "UnionLayout | None") -> str | None:
    """Return the arms of a tagged union, which name its columns, else None."""
    return None if layout is None else ", ".join(map(arm_selector, layout.arms))


def _in_signals(column: str, signals: Sequence[str]) -> bool:
    """Whether a stored column belongs to one of the signals, e.g. "p.x" to "p"."""
    return any(column == s or column.startswith(f"{s}.") for s in signals)


def _picked_schema(schemas: list[pa.Schema], signals: Sequence[str]) -> pa.Schema:
    """Unify the files' schemas over the picked signals, as only these need
    compatible types across files."""
    picked = [
        pa.schema([f for f in s if _in_signals(f.name, signals)], metadata=s.metadata)
        for s in schemas
    ]
    return _unify_datachain_schemas(picked)


def _unify_datachain_schemas(schemas: list[pa.Schema]) -> pa.Schema:
    """Unify the schemas of files written by DataChain. A column that one file
    stores as JSON text unifies as JSON, as each file is read with its own types."""
    json_cols = {f.name for s in schemas for f in s if isinstance(f.type, pa.JsonType)}
    return pa.unify_schemas(
        [
            pa.schema(
                [f.with_type(pa.json_()) if f.name in json_cols else f for f in s],
                metadata=s.metadata,
            )
            for s in schemas
        ]
    )


def _parent_tags(
    signal_schema: SignalSchema, columns: Sequence[str], paths: Sequence[str]
) -> dict[str, "ParentTag"]:
    """Map each path of the file that only its parent's ``_type_tag`` tells is
    present to that tag column, the parent's layout and the arm holding the path.

    A column, or a model with its own tag, is null where its parent is absent.
    """
    names = set(columns)
    parents = {}
    for path in paths:
        if path in names or f"{path}.{TYPE_TAG_FIELD}" in names:
            continue
        # A path the file lacks is left out of the row, so no tag decides it.
        if not any(c.startswith(f"{path}.") for c in names):
            continue
        if (found := signal_schema.tagged_parent(path)) is not None:
            parent, layout, arm = found
            parents[path] = (f"{parent}.{TYPE_TAG_FIELD}", layout, arm)
    return parents


def schema_to_output(
    schema: pa.Schema, col_names: Sequence[str] | None = None
) -> tuple[dict[str, type], list[str]]:
    """
    Generate UDF output schema from pyarrow schema.
    Returns a tuple of output schema and original column names (since they may be
    normalized in the output dict).
    """
    signal_schema = _get_datachain_schema(schema)
    if signal_schema:
        return signal_schema.values, list(signal_schema.values)

    if col_names and (len(schema) != len(col_names)):
        raise ValueError(
            "Error generating output from Arrow schema - "
            f"Schema has {len(schema)} columns but got {len(col_names)} column names."
        )
    if not col_names:
        col_names = schema.names or []

    normalized_col_dict = normalize_col_names(col_names)
    col_names = list(normalized_col_dict)

    hf_schema = _get_hf_schema(schema)
    if hf_schema:
        return {
            column: hf_type
            for hf_type, column in zip(hf_schema[1].values(), col_names, strict=False)
        }, list(normalized_col_dict.values())

    output = {}
    for field, column in zip(schema, col_names, strict=False):
        output[column] = _arrow_field_type_mapper(field, column)

    return output, list(normalized_col_dict.values())


def arrow_type_mapper(col_type: pa.DataType, column: str = "") -> type:  # noqa: PLR0911
    """Convert pyarrow types to basic types."""
    if pa.types.is_timestamp(col_type):
        return datetime
    if (
        pa.types.is_binary(col_type)
        or pa.types.is_large_binary(col_type)
        or pa.types.is_fixed_size_binary(col_type)
    ):
        return bytes
    if pa.types.is_floating(col_type):
        return float
    if pa.types.is_integer(col_type):
        return int
    if pa.types.is_boolean(col_type):
        return bool
    if pa.types.is_date(col_type):
        return datetime
    if (
        pa.types.is_string(col_type)
        or pa.types.is_large_string(col_type)
        or isinstance(col_type, pa.JsonType)
    ):
        return str
    if (
        pa.types.is_list(col_type)
        or pa.types.is_large_list(col_type)
        or pa.types.is_fixed_size_list(col_type)
    ):
        item_type = _arrow_field_type_mapper(
            col_type.value_field, column, nullable_scalars_only=True
        )
        return list[item_type]  # type: ignore[return-value, valid-type]
    if pa.types.is_struct(col_type):
        type_dict = {}
        for field in col_type:
            type_dict[field.name] = _arrow_field_type_mapper(field, field.name)
        return dict_to_data_model(f"ArrowDataModel_{column}", type_dict)
    if pa.types.is_map(col_type):
        # JSON storage keeps these values, and the dict type restores the keys.
        if _is_scalar_type(col_type.key_type) and _is_scalar_type(col_type.item_type):
            key_type = arrow_type_mapper(col_type.key_type, column)
            item_type = _arrow_field_type_mapper(col_type.item_field, column)
            return dict[key_type, item_type]  # type: ignore[return-value, valid-type]
        return dict
    if isinstance(col_type, pa.lib.DictionaryType):
        return arrow_type_mapper(col_type.value_type)  # type: ignore[return-value]
    if pa.types.is_null(col_type):
        return str  # use strings for null columns
    raise TypeError(f"{col_type!r} datatypes not supported, column: {column}")


# The SQL types of the signals' Python types, see `python_to_sql`.
_ARROW_TYPES: dict[Any, pa.DataType] = {
    String: pa.string(),
    Boolean: pa.bool_(),
    Int64: pa.int64(),
    Float: pa.float64(),
    DateTime: pa.timestamp("us", tz="UTC"),
    Binary: pa.binary(),
}


def _arrow_type(sql_type: Any) -> pa.DataType:
    """The Arrow type a column of this SQL type is written as. A column holding
    JSON at any depth, or of a type with no Arrow type here, is JSON text, which
    keeps any value."""
    if isinstance(sql_type, Array):
        item = _arrow_type(sql_type.item_type)
        return item if isinstance(item, pa.JsonType) else pa.list_(item)
    key = sql_type if isinstance(sql_type, type) else type(sql_type)
    return _ARROW_TYPES.get(key, pa.json_())


def parquet_schema(signal_schema: SignalSchema) -> pa.Schema:
    """The Arrow schema of the signals' columns, as Parquet stores them."""
    fields = []
    for path, anno, has_subtree, _ in signal_schema.get_flat_tree(
        include_sentinels=True
    ):
        if has_subtree:
            continue
        try:
            sql_type = python_to_sql(anno)
        except TypeError:
            sql_type = None
        fields.append(pa.field(".".join(path), _arrow_type(sql_type)))
    return pa.schema(fields)


def _to_arrow(values: list[Any], arrow_type: pa.DataType) -> pa.Array:
    """The values as an Arrow array of the type. A value the type can't hold as
    it is raises: a typed conversion rejects values of another kind, and a safe
    cast of the inferred array rejects fractions an int would truncate."""
    if isinstance(arrow_type, pa.JsonType):
        texts = [None if v is None else to_json_text(v) for v in values]
        return pa.array(texts, arrow_type)
    pa.array(values).cast(arrow_type)
    return pa.array(values, arrow_type)


def _arrow_field_type_mapper(
    field: pa.Field, column: str = "", *, nullable_scalars_only: bool = False
) -> type:
    dtype = arrow_type_mapper(field.type, column)
    if not field.nullable:
        return dtype
    if nullable_scalars_only and dtype not in NULLABLE_SCALARS:
        # https://github.com/datachain-ai/datachain/issues/1873
        return dtype
    if ModelStore.is_pydantic(dtype):
        return dtype
    return dtype | None  # type: ignore[return-value]


def _is_scalar_type(col_type: pa.DataType) -> bool:
    return (
        pa.types.is_integer(col_type)
        or pa.types.is_floating(col_type)
        or pa.types.is_boolean(col_type)
        or pa.types.is_string(col_type)
        or pa.types.is_large_string(col_type)
    )


def _is_plain_type(col_type: pa.DataType, top_level: bool = False) -> bool:
    """Whether pyarrow gives the column's values as pydantic and the DB keep them,
    typed as `arrow_type_mapper` types them. A top-level column counts after
    `_microseconds`."""
    if (
        pa.types.is_dictionary(col_type)
        or pa.types.is_list(col_type)
        or pa.types.is_large_list(col_type)
        or pa.types.is_fixed_size_list(col_type)
    ):
        return _is_plain_type(col_type.value_type)
    if pa.types.is_timestamp(col_type):
        # pyarrow gives nanoseconds as pandas Timestamps.
        return top_level or col_type.unit != "ns"
    return (
        pa.types.is_integer(col_type)
        or pa.types.is_float32(col_type)
        or pa.types.is_float64(col_type)
        or pa.types.is_boolean(col_type)
        or pa.types.is_string(col_type)
        or pa.types.is_large_string(col_type)
        or pa.types.is_binary(col_type)
        or pa.types.is_large_binary(col_type)
        or pa.types.is_fixed_size_binary(col_type)
        or pa.types.is_null(col_type)
    )


def _microseconds(column: pa.Array) -> pa.Array:
    """A nanosecond timestamp column floored to microseconds, as the DB stores it."""
    col_type = column.type
    if not pa.types.is_timestamp(col_type) or col_type.unit != "ns":
        return column
    naive = column.cast(pa.timestamp("ns"))
    micros = naive.cast(pa.timestamp("us"), safe=False)
    # A cast truncates towards zero. Correct negative fractions without flooring
    # in ns, which would overflow int64 at the earliest representable timestamp.
    rounded_up = pc.greater(micros.cast(pa.timestamp("ns")), naive).cast(pa.int64())
    return pc.subtract(micros.cast(pa.int64()), rounded_up).cast(
        pa.timestamp("us", tz=col_type.tz)
    )


def _json_text(value: Any) -> Any:
    """Return a dict or list as the JSON text the DB stores, other values as they
    are."""
    if not isinstance(value, (dict, list)):
        return value
    return to_json_text(value)


def _get_hf_schema(
    schema: "pa.Schema",
) -> tuple["Features", dict[str, "DataType"]] | None:
    if schema.metadata and b"huggingface" in schema.metadata:
        from datachain.lib.hf import get_output_schema, schema_from_arrow

        features = schema_from_arrow(schema)
        return features, get_output_schema(features)[0]
    return None


def _written_by_datachain(schema: "pa.Schema | None") -> TypeGuard["pa.Schema"]:
    if schema is None or not schema.metadata:
        return False
    return DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY in schema.metadata


def _get_datachain_schema(schema: "pa.Schema | None") -> SignalSchema | None:
    """Return a restored SignalSchema from parquet metadata, if any is found."""
    if _written_by_datachain(schema):
        serialized_signal_schema = json.loads(
            schema.metadata[DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY]
        )
        return SignalSchema.deserialize(serialized_signal_schema)
    return None


def _check_type_tags(file: File, schema: pa.Schema) -> None:
    """Raise for a file written by DataChain before union types: its schema has
    optional models or unions, but the file has no ``_type_tag`` for them."""
    if not (signal_schema := _get_datachain_schema(schema)):
        return
    names = set(schema.names)
    for path, *_ in signal_schema.get_flat_tree(include_sentinels=True):
        if path[-1] == TYPE_TAG_FIELD and (tag := ".".join(path)) not in names:
            raise ValueError(
                f"'{file.path}' has no {tag}, as it was written before union "
                "types. Export it again to read it."
            )


def _build_model(model: type["BaseModel"], values: dict[str, Any]) -> "BaseModel":
    return model(**values)
