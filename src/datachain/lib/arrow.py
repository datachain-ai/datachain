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
from datachain.lib.convert.flatten import classify_field, iter_flat_columns
from datachain.lib.data_model import (
    NULLABLE_SCALARS,
    dict_to_data_model,
    optional_tag_is_absent,
)
from datachain.lib.file import ArrowRow, File
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema
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
        kwargs: Parameters to pass to pyarrow.dataset.dataset.
        """
        super().__init__()
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.source = source
        self.nrows = nrows
        self.columns = columns
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

        ds = dataset(fs_path, schema=self.input_schema, filesystem=fs, **kwargs)

        use_datachain_schema = (
            bool(ds.schema.metadata)
            and DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY in ds.schema.metadata
        )

        if self.columns is not None and use_datachain_schema:
            raise ValueError(
                f"'{file.path}' was written by DataChain, so it can't be read "
                "by column names with other files. Read it separately."
            )

        schema = ds.schema
        kw: dict[str, Any] = {}
        if self.nrows:
            kw = {"batch_size": min(self.DEFAULT_BATCH_SIZE, self.nrows)}
        if self.columns is not None:
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
                    record, file, index, hf_schema, use_datachain_schema
                )

    def _process_record(
        self,
        record: dict[str, Any],
        file: File,
        index: int,
        hf_schema: tuple["Features", dict[str, "DataType"]] | None,
        use_datachain_schema: bool,
    ):
        if use_datachain_schema and self.output_schema:
            vals = [_nested_model_instantiate(record, self.output_schema)]
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
    chain: "DataChain", columns: Sequence[str] | None, fields: int, **kwargs
) -> list[str] | None:
    """Return the columns to read from each file when output gives their types.

    Like pyarrow, they come from the first file (by path). Each file is read by
    these names with its own types. Partition and pandas index columns at the end
    are left out when output has no fields for them.
    """
    ds = next(_file_datasets(chain.order_by("file.path").limit(1), **kwargs))
    schema = ds.schema
    if columns is not None:
        schema, _ = merge_schemas([schema], columns)
    elif _get_datachain_schema(schema):
        return None
    elif len(schema) > fields:
        physical = next(ds.get_fragments()).physical_schema
        extra = set(schema.names) - set(physical.names)
        index = (schema.pandas_metadata or {}).get("index_columns", [])
        extra |= {c for c in index if isinstance(c, str)}
        if set(schema.names[fields:]) <= extra:
            schema = pa.schema(list(schema)[:fields])
    if len(schema) != fields:
        raise ValueError(
            f"output has {fields} fields but {len(schema)} columns are read. "
            "output names the columns read, in order; pick columns with columns=."
        )
    return schema.names


def merge_schemas(
    schemas: list[pa.Schema], columns: Sequence[str] | None = None
) -> tuple[pa.Schema, list[str] | None]:
    """Merge the files' schemas. With `columns`, keep only those, in that order.

    Column names can be written as in the file or as DataChain cleans them, e.g.
    "Unit Price (USD)" or "unit_price_usd". With `columns`, also returns the
    cleaned names of the kept columns, cleaned over the whole header as in a
    full read.
    """
    if columns is None:
        return pa.unify_schemas(schemas), None
    if isinstance(columns, str) or not columns:
        raise ValueError("columns must be a non-empty list of column names")

    if any(_get_datachain_schema(s) for s in schemas):
        raise ValueError("columns can't be used with files written by DataChain")
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


def _get_datachain_schema(schema: "pa.Schema") -> SignalSchema | None:
    """Return a restored SignalSchema from parquet metadata, if any is found."""
    if schema.metadata and DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY in schema.metadata:
        serialized_signal_schema = json.loads(
            schema.metadata[DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY]
        )
        return SignalSchema.deserialize(serialized_signal_schema)
    return None


def _subtree_all_none(
    column_values: dict[str, Any], model: type["BaseModel"], prefix: str
) -> bool:
    """True when every scalar leaf under ``model`` (at ``prefix``) is None. NaN /
    [] / {} count as None-ish so type defaults don't read as present."""
    for col in iter_flat_columns(model):
        if col.is_sentinel:
            continue
        val = column_values.get(f"{prefix}." + ".".join(col.path))
        if val is not None and not _is_nan(val) and val not in ([], {}):
            return False
    return True


def _is_nan(val: Any) -> bool:
    return isinstance(val, float) and math.isnan(val)


def _optional_absent(
    column_values: dict[str, Any], inner: type["BaseModel"], prefix: str
) -> bool:
    """Whether the ``Optional[DataModel]`` at ``prefix`` is absent. Uses the
    ``_type_tag`` discriminator when present (0 = present arm), else the
    all-leaves-None heuristic."""
    tag_key = f"{prefix}.{SignalSchema._OPTIONAL_SENTINEL_FIELD}"
    if tag_key in column_values:
        return optional_tag_is_absent(column_values[tag_key])
    return _subtree_all_none(column_values, inner, prefix)


def _nested_model_instantiate(
    column_values: dict[str, Any], model: type["BaseModel"], prefix: str = ""
) -> "BaseModel":
    """Instantiate the given model and all sub-models/fields based on the provided
    column values."""
    vals_dict: dict[str, Any] = {}
    for field, field_info in model.model_fields.items():
        kind = classify_field(field_info.annotation)
        cur_path = f"{prefix}.{field}" if prefix else field
        if kind.is_model:
            if kind.is_optional and _optional_absent(
                column_values, kind.inner, cur_path
            ):
                # Absent Optional[DataModel] parent -> None.
                vals_dict[field] = None
            else:
                vals_dict[field] = _nested_model_instantiate(
                    column_values,
                    kind.inner,
                    prefix=cur_path,
                )
        elif cur_path in column_values:
            vals_dict[field] = column_values[cur_path]
    return model(**vals_dict)
