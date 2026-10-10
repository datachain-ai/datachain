import math
import warnings
from collections.abc import Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from pyarrow._csv import ParseOptions
from pyarrow.dataset import CsvFileFormat, dataset
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
        source_columns: Sequence[str | Sequence[str]] | None = None,
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
            field order, or several to try in turn: each file uses the first it
            has. When omitted, fields take the file's columns in order.
        kwargs: Parameters to pass to pyarrow.dataset.dataset.
        """
        super().__init__()
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.source = source
        self.nrows = nrows
        self.source_columns = (
            None
            if source_columns is None
            else [[c] if isinstance(c, str) else list(c) for c in source_columns]
        )
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
        columns: list[str | None] = []
        if self.source_columns is not None and not use_datachain_schema:
            # Files read together may lack some columns, or name them differently.
            present = set(ds.schema.names)
            columns = [
                next((c for c in candidates if c in present), None)
                for candidates in self.source_columns
            ]
            kw["columns"] = [c for c in dict.fromkeys(columns) if c is not None]

        def iter_records():
            for record_batch in ds.to_batches(**kw):
                yield from record_batch.to_pylist()

        it = islice(iter_records(), self.nrows)
        with tqdm(
            it, desc="Parsed by pyarrow", unit="rows", total=self.nrows, leave=False
        ) as pbar:
            for index, record in enumerate(pbar):
                yield self._process_record(
                    record, file, index, hf_schema, use_datachain_schema, columns
                )

    def _process_record(
        self,
        record: dict[str, Any],
        file: File,
        index: int,
        hf_schema: tuple["Features", dict[str, "DataType"]] | None,
        use_datachain_schema: bool,
        columns: list[str | None],
    ):
        if use_datachain_schema and self.output_schema:
            vals = [_nested_model_instantiate(record, self.output_schema)]
        elif self.source_columns is not None:
            vals = self._process_record_by_name(record, hf_schema, columns)
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
        columns: list[str | None],
    ):
        assert self.output_schema
        fields = self.output_schema.model_fields
        vals_dict = {}
        for (field, field_info), column in zip(fields.items(), columns, strict=True):
            if column is not None:
                vals_dict[field] = _convert_value(
                    record[column], field_info.annotation, hf_schema, column
                )
        # Columns are already matched to fields, so aliases must not apply again.
        return [
            self.output_schema.model_validate(vals_dict, by_name=True, by_alias=False)
        ]

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


def output_columns(
    schemas: list[pa.Schema], output: "dict[str, DataType] | type[BaseModel]"
) -> list[list[str]] | None:
    """Return the file columns each field of a dict or model output can read.

    A field's name can be a file's column name or DataChain's cleaned version of
    it, e.g. "Unit Price (USD)" or "unit_price_usd". A model field lists its
    aliases in the model's order, and each file uses the first one it has.
    Returns None when some name isn't a column, so the output keeps its old
    meaning: fields take the file's columns in order.
    """
    if _get_datachain_schema(schemas[0]):
        # Files written by DataChain store their own schema and match it by name.
        return None
    names = list(dict.fromkeys(n for s in schemas for n in s.names))
    lookup = _ColumnLookup(names)
    fields = _output_fields(output)
    found = [
        list(dict.fromkeys(c for n in candidates if (c := lookup.find(n)) is not None))
        for _, candidates in fields
    ]
    if not all(found):
        unknown = [f[0] for f, c in zip(fields, found, strict=True) if not c]
        warnings.warn(
            f"output names {unknown} are not columns of the file, so output is "
            "applied to the columns in order. Applying output by position is "
            "deprecated and will become an error. Name output by the file's "
            "columns and rename them after reading, e.g. with mutate(). "
            f"Available columns: {names}",
            FutureWarning,
            stacklevel=4,
        )
        return None
    if isinstance(output, dict) and len({c[0] for c in found}) < len(found):
        raise ValueError(f"output names the same column twice: {list(output)}")
    return found


class _ColumnLookup:
    """Finds a file's columns by their name or by DataChain's cleaned name."""

    def __init__(self, names: list[str]):
        self.raw = normalize_col_names(names)
        self.cleaned = {raw: clean for clean, raw in self.raw.items()}

    def find(self, name: str) -> str | None:
        if name in self.cleaned:
            return name
        return self.raw.get(name)


def _output_fields(
    output: "dict[str, DataType] | type[BaseModel]",
) -> list[tuple[str, list[str]]]:
    """Return each output field's name and the column names it may read."""
    if isinstance(output, dict):
        return [(name, [name]) for name in output]
    by_alias = output.model_config.get("validate_by_alias", True)
    return [
        (name, _field_names(name, info) if by_alias else [name])
        for name, info in output.model_fields.items()
    ]


def _field_names(field: str, field_info: "FieldInfo") -> list[str]:
    """Return the column names a model field reads, in the model's alias order."""
    alias = field_info.validation_alias
    if isinstance(alias, AliasChoices):
        names = [a for a in alias.choices if isinstance(a, str)]
    elif isinstance(alias, str):
        names = [alias]
    else:
        names = []
    return names if field in names else [*names, field]


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
