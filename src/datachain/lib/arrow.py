import dataclasses
import math
import warnings
from collections.abc import Sequence
from datetime import datetime
from itertools import islice
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

import pyarrow as pa
from pyarrow._csv import ParseOptions
from pyarrow.dataset import CsvFileFormat, dataset
from pyarrow.lib import type_for_alias
from pydantic import AliasChoices

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
    from pydantic.fields import FieldInfo

    from datachain.lib.data_model import DataType
    from datachain.lib.dc import DataChain
    from datachain.lib.dc.utils import OutputType


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


class ArrowGenerator(Generator):
    DEFAULT_BATCH_SIZE = 2**17  # same as `pyarrow._dataset._DEFAULT_BATCH_SIZE`

    def __init__(
        self,
        input_schema: pa.Schema | None = None,
        output_schema: type["BaseModel"] | None = None,
        source: bool = True,
        nrows: int | None = None,
        source_columns: Sequence[str] | None = None,
        cast_types: dict[str, pa.DataType] | None = None,
        **kwargs,
    ):
        """
        Generator for getting rows from tabular files.

        Parameters:

        input_schema : Optional pyarrow schema for validation.
        output_schema : Optional pydantic model for validation.
        source : Whether to include info about the source file.
        nrows : Optional row limit.
        source_columns : Optional file column for each `output_schema` field, in
            field order. When omitted, fields take the file's columns in order.
        cast_types : Optional types to cast columns to after reading them.
        kwargs: Parameters to pass to pyarrow.dataset.dataset.
        """
        super().__init__()
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.source = source
        self.nrows = nrows
        self.source_columns = source_columns
        self.cast_types = cast_types or {}
        self.parse_options = kwargs.pop("parse_options", None)
        self.kwargs = kwargs

    def process(self, file: File):
        if file._caching_enabled:
            file.ensure_cached()
        if cache_path := file.get_local_path():
            fs_path = file.path
            fs = ReferenceFileSystem({fs_path: [cache_path]})
        else:
            fs, fs_path = file.get_fs(), file.get_fs_path()

        kwargs = self.kwargs
        if format := kwargs.get("format"):
            kwargs["format"] = fix_pyarrow_format(format, self.parse_options)

        ds = dataset(fs_path, schema=self.input_schema, filesystem=fs, **kwargs)

        hf_schema = _get_hf_schema(ds.schema)
        use_datachain_schema = (
            bool(ds.schema.metadata)
            and DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY in ds.schema.metadata
        )

        kw: dict[str, Any] = {}
        if self.nrows:
            kw["batch_size"] = min(self.DEFAULT_BATCH_SIZE, self.nrows)
        if self.source_columns is not None and not use_datachain_schema:
            # A file may lack some columns when several files are read together.
            present = set(ds.schema.names)
            columns = dict.fromkeys(self.source_columns)
            kw["columns"] = [c for c in columns if c in present]

        def iter_records():
            for record_batch in ds.to_batches(**kw):
                if self.cast_types:
                    record_batch = _cast_batch(record_batch, self.cast_types)
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
        elif self.source_columns is not None:
            vals = self._process_record_by_name(record, hf_schema)
        else:
            vals = self._process_non_datachain_record(record, hf_schema)

        if self.source:
            kwargs: dict = self.kwargs
            # Can't serialize CsvFileFormat; may lose formatting options.
            if isinstance(kwargs.get("format"), CsvFileFormat):
                kwargs["format"] = "csv"
            arrow_file = ArrowRow(file=file, index=index, kwargs=kwargs)

            if self.output_schema and hasattr(vals[0], "source"):
                # if we are reading parquet file written by datachain it might have
                # source inside of it already, so we should not duplicate it, instead
                # we are re-creating it of the self.source flag
                vals[0].source = arrow_file  # type: ignore[attr-defined]

                return vals
            return [arrow_file, *vals]

        return vals

    def _process_record_by_name(
        self,
        record: dict[str, Any],
        hf_schema: tuple["Features", dict[str, "DataType"]] | None,
    ):
        assert self.output_schema
        assert self.source_columns is not None
        fields = self.output_schema.model_fields
        vals_dict = {}
        for (field, field_info), column in zip(
            fields.items(), self.source_columns, strict=True
        ):
            if column not in record:
                continue
            vals_dict[field] = _convert_value(
                record[column], field_info.annotation, hf_schema, column
            )
        return [self.output_schema.model_validate(vals_dict, by_name=True)]

    def _process_non_datachain_record(
        self,
        record: dict[str, Any],
        hf_schema: tuple["Features", dict[str, "DataType"]] | None,
    ):
        vals = list(record.values())
        if not self.output_schema:
            return vals

        fields = self.output_schema.model_fields
        vals_dict = {}
        for i, ((field, field_info), val) in enumerate(
            zip(fields.items(), vals, strict=False)
        ):
            column = list(hf_schema[0])[i] if hf_schema else ""
            vals_dict[field] = _convert_value(
                val, field_info.annotation, hf_schema, column
            )
        return [self.output_schema(**vals_dict)]


def _convert_value(
    val: Any,
    anno: Any,
    hf_schema: tuple["Features", dict[str, "DataType"]] | None,
    column: str,
) -> Any:
    if hf_schema:
        from datachain.lib.hf import convert_feature

        return convert_feature(val, hf_schema[0][column], anno)
    if ModelStore.is_pydantic(anno):
        return anno(**val)  # type: ignore[misc]
    return val


def _cast_batch(batch: pa.RecordBatch, types: dict[str, pa.DataType]) -> pa.RecordBatch:
    arrays = [
        col.cast(types[name]) if name in types else col
        for name, col in zip(batch.schema.names, batch.columns, strict=True)
    ]
    return pa.RecordBatch.from_arrays(arrays, names=batch.schema.names)


def file_schemas(chain: "DataChain", **kwargs) -> list[pa.Schema]:
    """Return the schema of each file in the chain."""
    parse_options = kwargs.pop("parse_options", None)
    if format := kwargs.get("format"):
        kwargs["format"] = fix_pyarrow_format(format, parse_options)

    schemas = []
    for (file,) in chain.to_iter("file"):
        ds = dataset(file.get_fs_path(), filesystem=file.get_fs(), **kwargs)  # type: ignore[union-attr]
        schemas.append(ds.schema)
    if not schemas:
        raise ValueError(
            "Cannot infer schema (no files to process or can't access them)"
        )
    return schemas


def infer_schema(chain: "DataChain", **kwargs) -> pa.Schema:
    return pa.unify_schemas(file_schemas(chain, **kwargs))


@dataclasses.dataclass
class TabularRead:
    """How `parse_tabular` reads files: what to read and what to output."""

    output: "dict[str, DataType] | type[BaseModel]"
    # The schema to read every file with. None reads each file with its own.
    schema: pa.Schema | None = None
    # The file column for each output field. None takes columns in file order.
    source_columns: list[str] | None = None
    # Column types to apply when `schema` is None.
    column_types: dict[str, pa.DataType] = dataclasses.field(default_factory=dict)


def plan_tabular_read(
    schemas: list[pa.Schema],
    output: "OutputType",
    columns: Sequence[str] | None,
    column_types: dict[str, Any] | None,
    is_csv: bool,
) -> TabularRead:
    """Resolve the columns and types to read from files with these schemas.

    Names in `output`, `columns` and `column_types` can be a file's column name or
    DataChain's cleaned version of it, e.g. "Unit Price (USD)" or "unit_price_usd".
    """
    signal_schema = _get_datachain_schema(schemas[0])
    if signal_schema and column_types:
        raise ValueError("column_types can't be used with files written by DataChain")
    names = (
        list(signal_schema.values)
        if signal_schema
        else list(dict.fromkeys(n for s in schemas for n in s.names))
    )
    lookup = _ColumnLookup(names)
    types = lookup.resolve_types(column_types or {})

    if isinstance(output, Sequence):
        if unknown := [n for n in output if lookup.find(n) is None]:
            _warn_positional_output(unknown, names)
            schema = _with_types(pa.unify_schemas(schemas), types)
            out, _ = schema_to_output(schema, output)
            return TabularRead(out, schema, schema.names)
        columns, output = output, None

    if output is None:
        schema = _with_types(pa.unify_schemas(schemas), types)
        out, originals = schema_to_output(schema)
        if columns is None:
            return TabularRead(out, schema, originals)
        selected = lookup.select(columns)
        by_column = dict(zip(originals, out.items(), strict=True))
        out = dict(by_column[c] for c in selected)
        if not signal_schema:
            schema = pa.schema(
                [schema.field(c) for c in selected], metadata=schema.metadata
            )
        return TabularRead(out, schema, selected)

    spec: dict[str, DataType] | type[BaseModel] | None = (
        output if isinstance(output, dict) else ModelStore.to_pydantic(output)
    )
    if spec is None:
        raise ValueError(f"output can't be {output!r}")
    if signal_schema:
        # Files written by DataChain store their own schema and match it by name.
        return TabularRead(spec)

    fields = _output_fields(spec)
    found = [
        next((c for n in candidates if (c := lookup.find(n)) is not None), None)
        for _, candidates, _ in fields
    ]
    if None in found:
        unknown = [f[1][-1] for f, c in zip(fields, found, strict=True) if c is None]
        _warn_positional_output(unknown, names)
        return TabularRead(spec, column_types=types)

    resolved = [c for c in found if c is not None]
    if len(set(resolved)) < len(resolved):
        raise ValueError(
            f"output names the same column twice: {[f[0] for f in fields]}"
        )
    if is_csv:
        # Read text columns as text, so values like "02134" keep their zeros.
        for (_, _, anno), column in zip(fields, resolved, strict=True):
            if column not in types and _is_str(anno):
                types[column] = pa.string()
    return TabularRead(spec, source_columns=resolved, column_types=types)


class _ColumnLookup:
    """Finds a file's columns by their name or by DataChain's cleaned name."""

    def __init__(self, names: list[str]):
        self.names = names
        self.raw = normalize_col_names(names)
        self.cleaned = {raw: clean for clean, raw in self.raw.items()}

    def find(self, name: str) -> str | None:
        if name in self.cleaned:
            return name
        return self.raw.get(name)

    def select(self, names: Sequence[str]) -> list[str]:
        if missing := [n for n in names if self.find(n) is None]:
            raise ValueError(
                f"Columns {missing} not found. Available columns: {self.names}"
            )
        return list(dict.fromkeys(self.find(n) for n in names))  # type: ignore[misc]

    def resolve_types(self, types: dict[str, Any]) -> dict[str, pa.DataType]:
        if unknown := [n for n in types if self.find(n) is None]:
            warnings.warn(
                f"column_types names {unknown} are not columns of the file and are "
                f"ignored. Available columns: {self.names}",
                stacklevel=4,
            )
        return {
            self.find(n): to_arrow_type(t)  # type: ignore[misc]
            for n, t in types.items()
            if self.find(n) is not None
        }


def _warn_positional_output(unknown: list[str], names: list[str]) -> None:
    warnings.warn(
        f"output names {unknown} are not columns of the file, so output is applied "
        "to the columns in order. Applying output by position is deprecated and "
        "will become an error. To name columns by position, pass column_names=. "
        f"Available columns: {names}",
        FutureWarning,
        stacklevel=4,
    )


def _output_fields(
    output: "dict[str, DataType] | type[BaseModel]",
) -> list[tuple[str, list[str], Any]]:
    """Return each output field's name, the names it may read, and its type."""
    if isinstance(output, dict):
        return [(name, [name], typ) for name, typ in output.items()]
    return [
        (name, _field_names(name, info), info.annotation)
        for name, info in output.model_fields.items()
    ]


def _field_names(field: str, field_info: "FieldInfo") -> list[str]:
    """Return the column names a field reads from: its aliases, then its name."""
    alias = field_info.validation_alias
    if isinstance(alias, AliasChoices):
        aliases = [a for a in alias.choices if isinstance(a, str)]
    elif isinstance(alias, str):
        aliases = [alias]
    else:
        aliases = []
    return [*(a for a in aliases if a != field), field]


def _is_str(anno: Any) -> bool:
    args = [a for a in get_args(anno) if a is not type(None)]
    return anno is str or (get_origin(anno) in (Union, UnionType) and args == [str])


def _with_types(schema: pa.Schema, types: dict[str, pa.DataType]) -> pa.Schema:
    if not types:
        return schema
    return pa.schema(
        [f.with_type(types[f.name]) if f.name in types else f for f in schema],
        metadata=schema.metadata,
    )


def to_arrow_type(typ: Any) -> pa.DataType:
    """Return the pyarrow type for a pyarrow type, a type name or a Python type."""
    if isinstance(typ, pa.DataType):
        return typ
    if isinstance(typ, str):
        return type_for_alias(typ)
    if typ in _ARROW_TYPES:
        return _ARROW_TYPES[typ]
    raise ValueError(f"Can't read a column as {typ!r}")


_ARROW_TYPES: dict[Any, pa.DataType] = {
    str: pa.string(),
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
    bytes: pa.binary(),
    datetime: pa.timestamp("us"),
}


def with_column_types(format: Any, types: dict[str, pa.DataType]) -> Any:
    """Return a CSV format that parses these columns with these types."""
    if format == "csv":
        format = CsvFileFormat()
    opts = format.default_fragment_scan_options
    convert = opts.convert_options
    convert.column_types = {**dict(convert.column_types), **types}
    return CsvFileFormat(
        parse_options=format.parse_options,
        read_options=opts.read_options,
        convert_options=convert,
    )


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
