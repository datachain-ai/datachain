import hashlib
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any, TypeGuard

import pyarrow as pa
from fsspec.implementations.local import LocalFileSystem
from pyarrow._csv import ParseOptions
from pyarrow.dataset import CsvFileFormat, Dataset, ParquetFileFormat, dataset
from pyarrow.fs import FileInfo, FileType, FSSpecHandler, PyFileSystem

from datachain import json
from datachain.fs.reference import ReferenceFileSystem
from datachain.lib.convert.flatten import classify_field
from datachain.lib.data_model import (
    NULLABLE_SCALARS,
    TYPE_TAG_FIELD,
    arm_for_tag,
    arm_selector,
    dict_to_data_model,
    union_layout,
)
from datachain.lib.file import ArrowRow, File
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema, SignalSchemaError
from datachain.lib.udf import Generator
from datachain.lib.utils import normalize_col_names
from datachain.progress import tqdm

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
        return client.fs
    block_size = 0 if is_parquet(format) else None
    return PyFileSystem(_PartialReadHandler(client, file, block_size=block_size))


class ArrowGenerator(Generator):
    DEFAULT_BATCH_SIZE = 2**17  # same as `pyarrow._dataset._DEFAULT_BATCH_SIZE`

    def __init__(
        self,
        input_schema: pa.Schema | None = None,
        output_schema: type["BaseModel"] | None = None,
        source: bool = True,
        nrows: int | None = None,
        columns: list[str] | None = None,
        written_by_datachain: bool | None = None,
        prefetch_max_size: int | None = None,
        **kwargs,
    ):
        """
        Generator for getting rows from tabular files.

        Parameters:

        input_schema : Optional pyarrow schema for validation.
        output_schema : Optional pydantic model for validation.
        source : Whether to include info about the source file.
        nrows : Optional row limit per file.
        columns : Optional columns to read from each file by name, in output order.
            For files written by DataChain, signal paths such as "p.x".
        written_by_datachain : Whether every file must be written by DataChain, or
            none. Not checked if None.
        prefetch_max_size : Files larger than this, or of unknown size, aren't
            prefetched.
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
        self.parse_options = kwargs.pop("parse_options", None)
        self.kwargs = kwargs

    def hash(self, include_body: bool = True) -> str:
        # The read schema and columns decide what is read, so reads that differ
        # only in them must not reuse each other's results.
        schema = self.input_schema.to_string() if self.input_schema else ""
        read = schema + json.dumps(self.columns)
        return hashlib.sha256((super().hash(include_body) + read).encode()).hexdigest()

    def process(self, file: File):
        if file._caching_enabled:
            file.ensure_cached()
        if cache_path := file.get_local_path():
            fs_path = file.path
            fs = ReferenceFileSystem({fs_path: [cache_path]})
        else:
            client = file._catalog.get_client(file.source)  # type: ignore[union-attr]
            # The File may come from a reused listing or a saved dataset, so the
            # object may have changed since. A download into the cache checks
            # this; a read that skips the cache checks it here.
            client.check_unchanged(file)
            fs = pyarrow_fs(file, self.kwargs.get("format"))
            fs_path = file.get_fs_path()

        kwargs = self.kwargs
        if format := kwargs.get("format"):
            kwargs["format"] = fix_pyarrow_format(format, self.parse_options)

        if self.columns is not None and _written_by_datachain(self.input_schema):
            # Each file keeps its own types, which other files may differ from. A
            # picked field missing from it reads as null, as in a full read.
            ds = dataset(fs_path, filesystem=fs, **kwargs)
            present = set(ds.schema.names)
            if missing := [f for f in self.input_schema if f.name not in present]:
                fields = [*ds.schema, *missing]
                ds = ds.replace_schema(pa.schema(fields, metadata=ds.schema.metadata))
        else:
            ds = dataset(fs_path, schema=self.input_schema, filesystem=fs, **kwargs)

        use_datachain_schema = _written_by_datachain(ds.schema)
        if self.written_by_datachain not in (None, use_datachain_schema):
            raise ValueError(_MIXED_FILES)

        schema = ds.schema
        kw: dict[str, Any] = {}
        if self.nrows:
            kw = {"batch_size": min(self.DEFAULT_BATCH_SIZE, self.nrows)}
        parents: dict[str, ParentTag] = {}
        if (
            self.columns is not None
            and self.output_schema
            and (signal_schema := _get_datachain_schema(schema))
        ):
            signals = _model_signals(self.output_schema, self.columns)
            parents = _parent_tags(signal_schema, schema.names, self.columns)
            tags = {tag for tag, _, _ in parents.values()}
            kw["columns"] = [
                c for c in schema.names if c in tags or _in_signals(c, signals)
            ]
        elif self.columns is not None:
            present = set(schema.names)
            kw["columns"] = [c for c in self.columns if c in present]
            # Decode Hugging Face features only for the columns being read.
            fields = [schema.field(c) for c in kw["columns"]]
            schema = pa.schema(fields, metadata=schema.metadata)
        hf_schema = _get_hf_schema(schema)

        def iter_records():
            for record_batch in ds.to_batches(**kw):
                yield from record_batch.to_pylist()

        it = islice(iter_records(), self.nrows)
        with tqdm(
            it, desc="Parsed by pyarrow", unit="rows", total=self.nrows, leave=False
        ) as pbar:
            for index, record in enumerate(pbar):
                yield self._process_record(
                    record, file, index, hf_schema, use_datachain_schema, parents
                )

    def _process_record(
        self,
        record: dict[str, Any],
        file: File,
        index: int,
        hf_schema: tuple["Features", dict[str, "DataType"]] | None,
        use_datachain_schema: bool,
        parents: dict[str, "ParentTag"],
    ):
        if use_datachain_schema and self.output_schema:
            model = self.output_schema
            vals = [
                _nested_model_instantiate(
                    record, model, paths=self.columns, parents=parents
                )
            ]
        else:
            vals = self._process_non_datachain_record(record, hf_schema)

        if self.source:
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
            arrow_file = ArrowRow(file=file, index=index, kwargs=kwargs)

            if self.output_schema and hasattr(vals[0], "source"):
                # if we are reading parquet file written by datachain it might have
                # source inside of it already, so we should not duplicate it, instead
                # we are re-creating it of the self.source flag
                vals[0].source = arrow_file  # type: ignore[attr-defined]

                return vals
            return [arrow_file, *vals]

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
        if not self.output_schema:
            return [record.get(name) for name in names]

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
        return [self.output_schema(**values)]


def _file_datasets(chain: "DataChain", **kwargs) -> Iterator[Dataset]:
    parse_options = kwargs.pop("parse_options", None)
    if format := kwargs.get("format"):
        kwargs["format"] = fix_pyarrow_format(format, parse_options)
    for (file,) in chain.to_iter("file"):
        fs = pyarrow_fs(file, kwargs.get("format"))  # type: ignore[arg-type]
        yield dataset(file.get_fs_path(), filesystem=fs, **kwargs)  # type: ignore[union-attr]


def file_schemas(chain: "DataChain", **kwargs) -> list[pa.Schema]:
    """Return the schema of each file in the chain, by path, so the first file
    gives the saved DataChain schema, as in `read_columns`."""
    schemas = [
        ds.schema for ds in _file_datasets(chain.order_by("file.path"), **kwargs)
    ]
    if not schemas:
        raise ValueError(
            "Cannot infer schema (no files to process or can't access them)"
        )
    return schemas


def infer_output(
    schemas: list[pa.Schema],
    col_names: Sequence[str] | None,
    columns: Sequence[str] | None,
    source: bool,
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

    signals = _signal_paths(signal_schema, columns, source)
    if columns is None:
        schema = pa.unify_schemas(schemas)
    else:
        columns = list(signals)  # as stored, e.g. "u.Model.x" for "u.x"
        # Only the picked signals need compatible types across files.
        picked = [
            pa.schema(
                [f for f in s if _in_signals(f.name, columns)], metadata=s.metadata
            )
            for s in schemas
        ]
        schema = pa.unify_schemas(picked)
    if col_names:
        _check_fields_fit(len(col_names), len(signals))
        names = list(normalize_col_names(col_names))
        return schema, dict(zip(names, signals.values(), strict=True)), list(signals)
    if columns is None:
        return schema, signal_schema.values, None
    # Picked signals keep their names, as select() returns them.
    output = signal_schema.to_partial(*columns).values
    return schema, output, list(output)


def read_columns(
    chain: "DataChain",
    columns: Sequence[str] | None,
    model: type["BaseModel"],
    source: bool,
    **kwargs,
) -> tuple[list[str], bool]:
    """Return the columns the output fields read, in order, and whether they are
    signals of files written by DataChain.

    Like pyarrow, they come from the first file (by path). Each file is read by
    these names with its own types. Partition and pandas index columns at the end
    are left out when output has no fields for them.
    """
    ds = next(_file_datasets(chain.order_by("file.path").limit(1), **kwargs))
    schema = ds.schema
    fields = len(model.model_fields)
    if signal_schema := _get_datachain_schema(schema):
        signals = _signal_paths(signal_schema, columns, source)
        _check_fields_fit(fields, len(signals))
        for (field, info), (path, stored) in zip(
            model.model_fields.items(), signals.items(), strict=True
        ):
            _check_output_shape(field, info.annotation, path, stored)
        return list(signals), True

    if columns is not None:
        schema, _ = merge_schemas([schema], columns)
    elif len(schema) > fields:
        physical = next(ds.get_fragments()).physical_schema
        extra = set(schema.names) - set(physical.names)
        index = (schema.pandas_metadata or {}).get("index_columns", [])
        extra |= {c for c in index if isinstance(c, str)}
        if set(schema.names[fields:]) <= extra:
            schema = pa.schema(list(schema)[:fields])
    _check_fields_fit(fields, len(schema))
    return schema.names, False


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
    signal_schema: SignalSchema, columns: Sequence[str] | None, source: bool
) -> dict[str, "DataType"]:
    """Return the signals read from files written by DataChain, with the types they
    read as.

    They are `columns`, or else the top-level signals. With `source`, the reader
    adds its own `source`, so a stored one is left out and can't be picked. A
    signal inside an optional model or a union is None where that parent doesn't
    hold it, so it reads as optional.
    """
    _check_columns(columns)
    if columns is None:
        paths = [s for s in signal_schema.values if not (source and s == "source")]
    elif source and any(_in_signals(c, ["source"]) for c in columns):
        raise ValueError(
            "columns can't pick source with source=True, the reader adds its own. "
            "Pass source=False to read the stored one."
        )
    elif len(set(columns)) < len(columns):
        raise ValueError(f"Columns {list(columns)} name a column more than once")
    else:
        paths = list(columns)
    try:
        signals = signal_schema.resolve(*paths).values
    except SignalSchemaError as e:
        available = list(signal_schema.values)
        raise ValueError(f"{e}. Available signals: {available}") from e
    return {
        path: anno | None if signal_schema.tagged_parent(path) else anno  # type: ignore[misc]
        for path, anno in signals.items()
    }


def _check_output_shape(field: str, anno: Any, path: str, stored: Any) -> None:
    """Raise unless the output field has the shape of the signal it reads: a model,
    optional where the signal is, a union of the same types, or a plain value,
    down through the fields it declares and the union arms' fields."""
    output_kind, stored_kind = classify_field(anno), classify_field(stored)
    if output_kind.is_model != stored_kind.is_model:
        kind = "a model" if output_kind.is_model else "not a model"
        raise ValueError(f"output field {field} is {kind}, unlike {path}")
    if not output_kind.is_model:
        output_layout, stored_layout = union_layout(anno), union_layout(stored)
        if (arms := _union_arms(output_layout)) != _union_arms(stored_layout):
            kind = f"a union of {arms}" if arms else "not a union"
            raise ValueError(f"output field {field} is {kind}, unlike {path}")
        if output_layout and stored_layout:
            for arm, stored_arm in zip(
                output_layout.arms, stored_layout.arms, strict=True
            ):
                name = arm_selector(arm)
                _check_output_shape(
                    f"{field}.{name}", arm, f"{path}.{name}", stored_arm
                )
        return
    if output_kind.is_optional != stored_kind.is_optional:
        if stored_kind.is_optional:
            reason = f"must be optional, as {path} can be None"
        else:
            reason = f"can't be optional, as {path} is never None"
        raise ValueError(f"output field {field} {reason}")
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


def _parent_tags(
    signal_schema: SignalSchema, columns: Sequence[str], paths: Sequence[str]
) -> dict[str, "ParentTag"]:
    """Map each path that only its parent's ``_type_tag`` tells is present to that
    tag column, the parent's layout and the arm holding the path.

    A column, or a model with its own tag, is null where its parent is absent.
    """
    names = set(columns)
    parents = {}
    for path in paths:
        if path in names or f"{path}.{TYPE_TAG_FIELD}" in names:
            continue
        if (found := signal_schema.tagged_parent(path)) is not None:
            parent, layout, arm = found
            parents[path] = (f"{parent}.{TYPE_TAG_FIELD}", layout, arm)
    return parents


def _model_signals(model: type["BaseModel"], paths: Sequence[str]) -> list[str]:
    """Return the signals that the model's fields read, each from its path. An
    optional model also reads its ``_type_tag``."""
    read = []
    for info, path in zip(model.model_fields.values(), paths, strict=True):
        kind = classify_field(info.annotation)
        if kind.is_model:
            fields = [f"{path}.{field}" for field in kind.inner.model_fields]
            read += _model_signals(kind.inner, fields)
            if kind.is_optional:
                read.append(f"{path}.{TYPE_TAG_FIELD}")
        else:
            read.append(path)
    return read


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
    from datetime import datetime

    if pa.types.is_timestamp(col_type):
        return datetime
    if pa.types.is_binary(col_type):
        return bytes
    if pa.types.is_floating(col_type):
        return float
    if pa.types.is_integer(col_type):
        return int
    if pa.types.is_boolean(col_type):
        return bool
    if pa.types.is_date(col_type):
        return datetime
    if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
        return str
    if pa.types.is_list(col_type):
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
        return dict
    if isinstance(col_type, pa.lib.DictionaryType):
        return arrow_type_mapper(col_type.value_type)  # type: ignore[return-value]
    if pa.types.is_null(col_type):
        return str  # use strings for null columns
    raise TypeError(f"{col_type!r} datatypes not supported, column: {column}")


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


def _get_datachain_schema(schema: "pa.Schema") -> SignalSchema | None:
    """Return a restored SignalSchema from parquet metadata, if any is found."""
    if _written_by_datachain(schema):
        serialized_signal_schema = json.loads(
            schema.metadata[DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY]
        )
        return SignalSchema.deserialize(serialized_signal_schema)
    return None


def _union_value(
    column_values: dict[str, Any], layout: "UnionLayout", prefix: str
) -> Any:
    """Reconstruct a tagged-union value from flat columns, hydrating the active arm."""
    arm = arm_for_tag(layout, column_values[f"{prefix}.{TYPE_TAG_FIELD}"])
    if arm is None:
        return None
    arm_prefix = f"{prefix}.{arm_selector(arm)}" if layout.use_slots else prefix
    if (fr := ModelStore.to_pydantic(arm)) is not None:
        return _nested_model_instantiate(column_values, fr, arm_prefix)
    return column_values.get(arm_prefix)


def _nested_model_instantiate(
    column_values: dict[str, Any],
    model: type["BaseModel"],
    prefix: str = "",
    paths: Sequence[str] | None = None,
    parents: dict[str, "ParentTag"] | None = None,
) -> "BaseModel":
    """Instantiate the given model and all sub-models/fields based on the provided
    column values. With `paths`, each field is read from its path, in order,
    instead of from its own name, and `parents` maps a path to the parent tag
    that tells it is present."""
    vals_dict: dict[str, Any] = {}
    for i, (field, field_info) in enumerate(model.model_fields.items()):
        anno = field_info.annotation
        if paths is not None:
            cur_path = paths[i]
            if parents and (parent := parents.get(cur_path)):
                tag, parent_layout, arm = parent
                if arm_for_tag(parent_layout, column_values[tag]) is not arm:
                    vals_dict[field] = None
                    continue
                # Present by its parent's tag, even with all of its fields null.
                anno = classify_field(anno).inner
        else:
            cur_path = f"{prefix}.{field}" if prefix else field
        if (layout := union_layout(anno)) is not None:
            vals_dict[field] = _union_value(column_values, layout, cur_path)
            continue
        kind = classify_field(anno)
        if kind.is_model:
            vals_dict[field] = _nested_model_instantiate(
                column_values,
                kind.inner,
                prefix=cur_path,
            )
        elif cur_path in column_values:
            vals_dict[field] = column_values[cur_path]
    return model(**vals_dict)
