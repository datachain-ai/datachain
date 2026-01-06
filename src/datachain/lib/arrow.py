from collections.abc import Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from pyarrow._csv import ParseOptions
from pyarrow.dataset import CsvFileFormat, dataset
from tqdm.auto import tqdm

from datachain import json
from datachain.fs.reference import ReferenceFileSystem
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.file import ArrowRow, File
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf import Generator
from datachain.lib.utils import normalize_col_names

if TYPE_CHECKING:
    from datasets.features.features import Features
    from pydantic import BaseModel

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
        **kwargs,
    ):
        """
        Generator for getting rows from tabular files.

        Parameters:

        input_schema : Optional pyarrow schema for validation.
        output_schema : Optional pydantic model for validation.
        source : Whether to include info about the source file.
        nrows : Optional row limit.
        kwargs: Parameters to pass to pyarrow.dataset.dataset.
        """
        super().__init__()
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.source = source
        self.nrows = nrows
        self.parse_options = kwargs.pop("parse_options", None)
        self.kwargs = kwargs

    def process(self, file: File):
        if file._caching_enabled:
            file.ensure_cached()
            cache_path = file.get_local_path()
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

        kw = {}
        if self.nrows:
            kw = {"batch_size": min(self.DEFAULT_BATCH_SIZE, self.nrows)}

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
            anno = field_info.annotation
            if hf_schema:
                from datachain.lib.hf import convert_feature

                feat = list(hf_schema[0].values())[i]
                vals_dict[field] = convert_feature(val, feat, anno)
            elif ModelStore.is_pydantic(anno):
                vals_dict[field] = anno(**val)  # type: ignore[misc]
            else:
                vals_dict[field] = val
        return [self.output_schema(**vals_dict)]


def infer_schema(chain: "DataChain", **kwargs) -> pa.Schema:
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
    return pa.unify_schemas(schemas)


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
        dtype = arrow_type_mapper(field.type, column)
        if field.nullable and not ModelStore.is_pydantic(dtype):
            dtype = dtype | None  # type: ignore[assignment]
        output[column] = dtype

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
        return list[arrow_type_mapper(col_type.value_type)]  # type: ignore[return-value, misc]
    if pa.types.is_struct(col_type):
        type_dict = {}
        for field in col_type:
            dtype = arrow_type_mapper(field.type, field.name)
            if field.nullable and not ModelStore.is_pydantic(dtype):
                dtype = dtype | None  # type: ignore[assignment]
            type_dict[field.name] = dtype
        return dict_to_data_model(f"ArrowDataModel_{column}", type_dict)
    if pa.types.is_map(col_type):
        return dict
    if isinstance(col_type, pa.lib.DictionaryType):
        return arrow_type_mapper(col_type.value_type)  # type: ignore[return-value]
    if pa.types.is_null(col_type):
        return str  # use strings for null columns
    raise TypeError(f"{col_type!r} datatypes not supported, column: {column}")


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


def _nested_model_instantiate(
    column_values: dict[str, Any], model: type["BaseModel"], prefix: str = ""
) -> "BaseModel":
    """Instantiate the given model and all sub-models/fields based on the provided
    column values."""
    vals_dict = {}
    for field, field_info in model.model_fields.items():
        anno = field_info.annotation
        cur_path = f"{prefix}.{field}" if prefix else field
        if ModelStore.is_pydantic(anno):
            vals_dict[field] = _nested_model_instantiate(
                column_values,
                anno,  # type: ignore[arg-type]
                prefix=cur_path,
            )
        elif cur_path in column_values:
            vals_dict[field] = column_values[cur_path]
    return model(**vals_dict)
