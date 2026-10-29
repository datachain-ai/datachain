import hashlib
import math
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from fsspec.implementations.local import LocalFileSystem
from pyarrow._csv import ParseOptions
from pyarrow.dataset import CsvFileFormat, Dataset, dataset
from pyarrow.fs import FileInfo, FileType, FSSpecHandler, PyFileSystem

from datachain import json
from datachain.fs.reference import ReferenceFileSystem
from datachain.lib.convert.flatten import classify_field
from datachain.lib.data_model import (
    NULLABLE_SCALARS,
    dict_to_data_model,
    optional_tag_is_absent,
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
    from datachain.lib.data_model import DataType
    from datachain.lib.dc import DataChain


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


class _PartialReadHandler(FSSpecHandler):
    """Lets pyarrow open the exact File that the reader got from a listing.

    `read_parquet`, `read_csv` and `parse_tabular` run over File rows that
    `read_storage` lists: one per object, with its path, ETag and, in a bucket
    that keeps versions, its version. A listing is reused for hours or saved in
    a dataset, so the object at the path may have changed since. pyarrow's
    default handler opens whatever the path holds now; this one opens the File
    through its client, which opens the recorded version.

    It also opens without read-ahead, so reading some columns fetches only those
    columns; pyarrow merges nearby reads itself. HTTP keeps read-ahead for servers
    without range support.
    """

    def __init__(self, client: "Client", file: File):
        super().__init__(client.fs)
        self.client = client
        self.file = file

    def get_file_info(self, paths):
        # Describes the recorded file without looking up the latest object, which a
        # versioned bucket may have deleted since.
        return [FileInfo(path, FileType.File) for path in paths]

    def open_input_file(self, path):
        f = self.client.open_object(
            self.file, use_cache=False, cb=self.file._download_cb, block_size=0
        )
        return pa.PythonFile(f, mode="r")


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
        **kwargs,
    ):
        """
        Generator for getting rows from tabular files.

        Parameters:

        input_schema : Optional pyarrow schema for validation.
        output_schema : Optional pydantic model for validation.
        source : Whether to include info about the source file.
        nrows : Optional row limit.
        columns : Optional columns to read from each file by name, in output order.
            For files written by DataChain, signal paths such as "p.x".
        written_by_datachain : Whether every file must be written by DataChain, or
            none. Not checked if None.
        kwargs: Parameters to pass to pyarrow.dataset.dataset.
        """
        super().__init__()
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
            fs, fs_path = client.fs, file.get_fs_path()
            parquet = self.kwargs.get("format") == "parquet"
            if parquet and not isinstance(fs, LocalFileSystem):
                fs = PyFileSystem(_PartialReadHandler(client, file))

        kwargs = self.kwargs
        if format := kwargs.get("format"):
            kwargs["format"] = fix_pyarrow_format(format, self.parse_options)

        input_schema = self.input_schema
        if (
            input_schema is None
            or self.columns is None
            or not _written_by_datachain(input_schema)
        ):
            ds = dataset(fs_path, schema=input_schema, filesystem=fs, **kwargs)
        else:
            # A picked field missing from this file reads as null, as in a full
            # read. The fields that decide an optional model keep this file's own
            # types, which other files may type differently.
            ds = dataset(fs_path, filesystem=fs, **kwargs)
            picked = set(input_schema.names)
            extra = [f for f in ds.schema if f.name not in picked]
            ds = ds.replace_schema(
                pa.schema([*input_schema, *extra], metadata=ds.schema.metadata)
            )

        use_datachain_schema = _written_by_datachain(ds.schema)
        if self.written_by_datachain not in (None, use_datachain_schema):
            raise ValueError(_MIXED_FILES)

        schema = ds.schema
        kw: dict[str, Any] = {}
        if self.nrows:
            kw = {"batch_size": min(self.DEFAULT_BATCH_SIZE, self.nrows)}
        if self.columns is not None and use_datachain_schema and self.output_schema:
            read = _model_signals(self.output_schema, self.columns)
            kw["columns"] = [c for c in schema.names if _in_signals(c, read)]
        elif self.columns is not None:
            present = set(schema.names)
            kw["columns"] = [c for c in self.columns if c in present]
            # Decode Hugging Face features only for the columns being read.
            fields = [schema.field(c) for c in kw["columns"]]
            schema = pa.schema(fields, metadata=schema.metadata)
        hf_schema = _get_hf_schema(schema)
        under = _columns_under(kw.get("columns", ds.schema.names))

        def iter_records():
            for record_batch in ds.to_batches(**kw):
                yield from record_batch.to_pylist()

        it = islice(iter_records(), self.nrows)
        with tqdm(
            it, desc="Parsed by pyarrow", unit="rows", total=self.nrows, leave=False
        ) as pbar:
            for index, record in enumerate(pbar):
                yield self._process_record(
                    record, file, index, hf_schema, use_datachain_schema, under
                )

    def _process_record(
        self,
        record: dict[str, Any],
        file: File,
        index: int,
        hf_schema: tuple["Features", dict[str, "DataType"]] | None,
        use_datachain_schema: bool,
        under: dict[str, list[str]],
    ):
        if use_datachain_schema and self.output_schema:
            model = self.output_schema
            vals = [_nested_model_instantiate(record, model, under, paths=self.columns)]
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
        vals = [record.get(name) for name in names]
        if not self.output_schema:
            return vals

        fields = self.output_schema.model_fields
        vals_dict = {}
        for (field, field_info), name, val in zip(
            fields.items(), names, vals, strict=False
        ):
            anno = field_info.annotation
            if hf_schema and (feat := hf_schema[0].get(name)) is not None:
                from datachain.lib.hf import convert_feature

                vals_dict[field] = convert_feature(val, feat, anno)
            elif val is None:
                vals_dict[field] = val
            elif ModelStore.is_pydantic(anno):
                vals_dict[field] = anno(**val)  # type: ignore[misc]
            else:
                vals_dict[field] = val
        return [self.output_schema(**vals_dict)]


def _file_datasets(chain: "DataChain", **kwargs) -> Iterator[Dataset]:
    parse_options = kwargs.pop("parse_options", None)
    if format := kwargs.get("format"):
        kwargs["format"] = fix_pyarrow_format(format, parse_options)
    for (file,) in chain.to_iter("file"):
        yield dataset(file.get_fs_path(), filesystem=file.get_fs(), **kwargs)  # type: ignore[union-attr]


def file_schemas(chain: "DataChain", **kwargs) -> list[pa.Schema]:
    """Return the schema of each file in the chain."""
    schemas = [ds.schema for ds in _file_datasets(chain, **kwargs)]
    if not schemas:
        raise ValueError(
            "Cannot infer schema (no files to process or can't access them)"
        )
    return schemas


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
        names = list(signals)
    elif columns is not None:
        names = merge_schemas([schema], columns)[0].names
    else:
        names = schema.names
        if len(names) > fields:
            physical = next(ds.get_fragments()).physical_schema
            extra = set(names) - set(physical.names)
            index = (schema.pandas_metadata or {}).get("index_columns", [])
            extra |= {c for c in index if isinstance(c, str)}
            if set(names[fields:]) <= extra:
                names = names[:fields]
    _check_fields_fit(fields, len(names))
    if signal_schema:
        for (field, info), (path, stored) in zip(
            model.model_fields.items(), signals.items(), strict=True
        ):
            is_model = classify_field(info.annotation).is_model
            if is_model != classify_field(stored).is_model:
                kind = "a model" if is_model else "not a model"
                raise ValueError(f"output field {field} is {kind}, unlike {path}")
    return names, signal_schema is not None


def _check_fields_fit(fields: int, read: int) -> None:
    if fields != read:
        raise ValueError(
            f"output has {fields} fields but {read} columns are read. "
            "output names the columns read, in order; pick columns with columns=."
        )


def merge_schemas(
    schemas: list[pa.Schema],
    columns: Sequence[str] | None = None,
    source: bool = False,
    with_output: bool = False,
) -> tuple[pa.Schema, list[str] | None]:
    """Merge the files' schemas. With `columns`, keep only those, in that order.

    Column names can be written as in the file or as DataChain cleans them, e.g.
    "Unit Price (USD)" or "unit_price_usd". With `columns`, also returns the
    cleaned names of the kept columns, cleaned over the whole header as in a
    full read. Files written by DataChain pick signals instead and keep their
    names. With `columns` or `with_output`, they can't be mixed with other files.
    """
    if columns is not None and (isinstance(columns, str) or not columns):
        raise ValueError("columns must be a non-empty list of column names")
    kinds = {_written_by_datachain(s) for s in schemas}
    if (columns is not None or with_output) and len(kinds) > 1:
        raise ValueError(_MIXED_FILES)
    if columns is None:
        return pa.unify_schemas(schemas), None
    if signal_schema := _get_datachain_schema(schemas[0]):
        return _pick_signals(schemas, signal_schema, columns, source), None

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


def _pick_signals(
    schemas: list[pa.Schema],
    signal_schema: SignalSchema,
    columns: Sequence[str],
    source: bool,
) -> pa.Schema:
    """Pick signals from files written by DataChain, as `select()` does.

    A nested name like "p.x" keeps its parent as a partial model. A signal is
    stored as its nested `signal.field` columns.
    """
    _signal_paths(signal_schema, columns, source)
    picked = signal_schema.to_partial(*columns)
    # Only the picked signals need compatible types across files.
    merged = pa.unify_schemas(
        [pa.schema([f for f in s if _in_signals(f.name, columns)]) for s in schemas]
    )
    metadata = {
        **schemas[0].metadata,
        DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY: json.dumps(picked.serialize()),
    }
    return merged.with_metadata(metadata)


def _signal_paths(
    signal_schema: SignalSchema, columns: Sequence[str] | None, source: bool
) -> dict[str, "DataType"]:
    """Return the signals read from files written by DataChain, with their types.

    They are `columns`, or else the top-level signals. With `source`, the reader
    adds its own `source`, so a stored one is left out and can't be picked.
    """
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
        return signal_schema.resolve(*paths).values
    except SignalSchemaError as e:
        signals = list(signal_schema.values)
        raise ValueError(f"{e}. Available signals: {signals}") from e


def _in_signals(column: str, signals: Sequence[str]) -> bool:
    """Whether a stored column belongs to one of the signals, e.g. "p.x" to "p"."""
    return any(column == s or column.startswith(f"{s}.") for s in signals)


def _model_signals(model: type["BaseModel"], paths: Sequence[str]) -> list[str]:
    """Return the signals that the model's fields read, each from its path.

    An optional model is read whole: Parquet stores no marker of whether it is
    present, so a full read decides it from all of the model's fields. Reading
    them all here gives the same answer when only some fields are picked.
    """
    read = []
    for info, path in zip(model.model_fields.values(), paths, strict=True):
        kind = classify_field(info.annotation)
        if kind.is_model and not kind.is_optional:
            fields = [f"{path}.{field}" for field in kind.inner.model_fields]
            read += _model_signals(kind.inner, fields)
        else:
            read.append(path)
    return read


def infer_output(
    schema: pa.Schema,
    col_names: Sequence[str] | None,
    columns: Sequence[str] | None,
    source: bool,
) -> tuple[dict[str, type], list[str] | None]:
    """Generate UDF output from the merged schema, `col_names` naming its columns.

    Also returns the signals that the output fields take, in order, from files
    written by DataChain, or None when the output reads all columns by name.
    """
    signal_schema = _get_datachain_schema(schema)
    if not signal_schema:
        return schema_to_output(schema, col_names)[0], None
    if not col_names:
        # Picked signals keep their names, as select() returns them.
        output = signal_schema.values
        return output, (list(output) if columns is not None else None)
    signals = _signal_paths(signal_schema, columns, source)
    _check_fields_fit(len(col_names), len(signals))
    names = list(normalize_col_names(col_names))
    return dict(zip(names, signals.values(), strict=True)), list(signals)


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


def _written_by_datachain(schema: "pa.Schema") -> bool:
    return DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY in (schema.metadata or {})


def _get_datachain_schema(schema: "pa.Schema") -> SignalSchema | None:
    """Return a restored SignalSchema from parquet metadata, if any is found."""
    if schema.metadata and DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY in schema.metadata:
        serialized_signal_schema = json.loads(
            schema.metadata[DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY]
        )
        return SignalSchema.deserialize(serialized_signal_schema)
    return None


def _columns_under(columns: Sequence[str]) -> dict[str, list[str]]:
    """Map each model path to the stored columns under it, e.g. "p" to ["p.x"]."""
    under: dict[str, list[str]] = {}
    for column in columns:
        parts = column.split(".")
        for i in range(1, len(parts)):
            under.setdefault(".".join(parts[:i]), []).append(column)
    return under


def _subtree_all_none(column_values: dict[str, Any], columns: Sequence[str]) -> bool:
    """True when every one of the model's stored ``columns`` is None, including
    fields the output leaves out. NaN / [] / {} count as None-ish so type defaults
    don't read as present."""
    sentinel = f".{SignalSchema._OPTIONAL_SENTINEL_FIELD}"
    for key in columns:
        if key.endswith(sentinel):
            continue
        val = column_values.get(key)
        if val is not None and not _is_nan(val) and val not in ([], {}):
            return False
    return True


def _is_nan(val: Any) -> bool:
    return isinstance(val, float) and math.isnan(val)


def _optional_absent(
    column_values: dict[str, Any], prefix: str, columns: Sequence[str]
) -> bool:
    """Whether the ``Optional[DataModel]`` at ``prefix``, stored as ``columns``, is
    absent. Uses the ``_type_tag`` discriminator when present (0 = present arm),
    else the all-leaves-None heuristic."""
    tag_key = f"{prefix}.{SignalSchema._OPTIONAL_SENTINEL_FIELD}"
    if tag_key in column_values:
        return optional_tag_is_absent(column_values[tag_key])
    return _subtree_all_none(column_values, columns)


def _nested_model_instantiate(
    column_values: dict[str, Any],
    model: type["BaseModel"],
    under: dict[str, list[str]],
    prefix: str = "",
    paths: Sequence[str] | None = None,
) -> "BaseModel":
    """Instantiate the given model and all sub-models/fields based on the provided
    column values. `under` maps each model path to its stored columns. With
    `paths`, each field is read from its path, in order, instead of from its own
    name."""
    vals_dict: dict[str, Any] = {}
    for i, (field, field_info) in enumerate(model.model_fields.items()):
        kind = classify_field(field_info.annotation)
        if paths is not None:
            cur_path = paths[i]
        else:
            cur_path = f"{prefix}.{field}" if prefix else field
        if kind.is_model:
            columns = under.get(cur_path, [])
            if kind.is_optional and _optional_absent(column_values, cur_path, columns):
                # Absent Optional[DataModel] parent -> None.
                vals_dict[field] = None
            else:
                vals_dict[field] = _nested_model_instantiate(
                    column_values,
                    kind.inner,
                    under,
                    prefix=cur_path,
                )
        elif cur_path in column_values:
            vals_dict[field] = column_values[cur_path]
    return model(**vals_dict)
