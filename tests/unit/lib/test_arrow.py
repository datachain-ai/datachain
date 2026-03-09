from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

import datachain as dc
from datachain.lib.arrow import (
    ArrowGenerator,
    arrow_type_mapper,
    infer_schema,
    schema_to_output,
)
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.file import ArrowRow, File
from datachain.lib.hf import HFClassLabel


@pytest.mark.parametrize("cache", [True, False])
def test_arrow_generator(tmp_path, catalog, cache):
    ids = [12345, 67890, 34, 0xF0123]
    texts = ["28", "22", "we", "hello world"]
    df = pd.DataFrame({"id": ids, "text": texts})

    name = "111.parquet"
    pq_path = tmp_path / name
    df.to_parquet(pq_path)
    stream = File(path=pq_path.as_posix(), source="file://")
    stream._set_stream(catalog, caching_enabled=cache)

    func = ArrowGenerator()
    objs = list(func.process(stream))

    assert len(objs) == len(ids)
    for o, id, text in zip(objs, ids, texts, strict=False):
        assert isinstance(o[0], ArrowRow)
        file_vals = o[0].read()
        assert file_vals["id"] == id
        assert file_vals["text"] == text
        assert o[1] == id
        assert o[2] == text


def test_arrow_generator_no_source(tmp_path, catalog):
    ids = [12345, 67890, 34, 0xF0123]
    texts = ["28", "22", "we", "hello world"]
    df = pd.DataFrame({"id": ids, "text": texts})

    name = "111.parquet"
    pq_path = tmp_path / name
    df.to_parquet(pq_path)
    stream = File(path=pq_path.as_posix(), source="file://")
    stream._set_stream(catalog, caching_enabled=False)

    func = ArrowGenerator(source=False)
    objs = list(func.process(stream))

    for o, id, text in zip(objs, ids, texts, strict=False):
        assert o[0] == id
        assert o[1] == text


def test_arrow_generator_output_schema(tmp_path, catalog):
    ids = [12345, 67890, 34, 0xF0123]
    texts = ["28", "22", "we", "hello world"]
    dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}, {"a": 7, "b": 8}]
    df = pd.DataFrame({"id": ids, "text": texts, "dict": dicts})
    table = pa.Table.from_pandas(df)

    name = "111.parquet"
    pq_path = tmp_path / name
    pq.write_table(table, pq_path)
    stream = File(path=pq_path.as_posix(), source="file://")
    stream._set_stream(catalog, caching_enabled=False)

    output, original_names = schema_to_output(table.schema)
    output_schema = dict_to_data_model("", output, original_names)
    func = ArrowGenerator(output_schema=output_schema)
    objs = list(func.process(stream))

    assert len(objs) == len(ids)
    for o, id, text, dict in zip(objs, ids, texts, dicts, strict=False):
        assert isinstance(o[0], ArrowRow)
        assert o[1].id == id
        assert o[1].text == text
        assert o[1].dict.a == dict["a"]
        assert o[1].dict.b == dict["b"]


def test_arrow_generator_hf(tmp_path, catalog):
    ds = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"]})
    ds = ds.class_encode_column("pokemon")

    name = "111.parquet"
    pq_path = tmp_path / name
    ds.to_parquet(pq_path)
    stream = File(path=pq_path.as_posix(), source="file:///")
    stream._set_stream(catalog, caching_enabled=False)

    output, original_names = schema_to_output(ds._data.schema, ["col"])

    output_schema = dict_to_data_model("", output, original_names)
    func = ArrowGenerator(output_schema=output_schema)
    for obj in func.process(stream):
        assert isinstance(obj[1].col, HFClassLabel)


@pytest.mark.parametrize("cache", [True, False])
def test_arrow_generator_partitioned(tmp_path, catalog, cache):
    pq_path = tmp_path / "parquets"
    pylist = [
        {"first_name": "Alice", "age": 25, "city": "New York"},
        {"first_name": "Bob", "age": 30, "city": "Los Angeles"},
        {"first_name": "Charlie", "age": 35, "city": "Chicago"},
    ]
    table = pa.Table.from_pylist(pylist)
    pq.write_to_dataset(table, pq_path, partition_cols=["first_name"])

    output, original_names = schema_to_output(table.schema)
    output_schema = dict_to_data_model("", output, original_names)
    func = ArrowGenerator(
        table.schema, output_schema=output_schema, partitioning="hive"
    )

    for path in pq_path.rglob("*.parquet"):
        stream = File(path=path.as_posix(), source="file://")
        stream._set_stream(catalog, caching_enabled=cache)

        (o,) = list(func.process(stream))
        assert isinstance(o[0], ArrowRow)
        assert dict(o[1]) in pylist


@pytest.mark.parametrize(
    "col_type,expected",
    (
        (pa.timestamp("us"), datetime),
        (pa.binary(), bytes),
        (pa.float32(), float),
        (pa.float64(), float),
        (pa.float16(), float),
        (pa.int8(), int),
        (pa.int16(), int),
        (pa.int32(), int),
        (pa.int64(), int),
        (pa.uint8(), int),
        (pa.bool_(), bool),
        (pa.date32(), datetime),
        (pa.string(), str),
        (pa.large_string(), str),
        (pa.map_(pa.string(), pa.int32()), dict),
        (pa.dictionary(pa.int64(), pa.string()), str),
        (pa.list_(pa.string()), list[str]),
        (pa.null(), str),
    ),
)
def test_arrow_type_mapper(col_type, expected):
    assert arrow_type_mapper(col_type) == expected


def test_arrow_type_mapper_struct():
    col_type = pa.struct({"x": pa.int32(), "y": pa.string(), "z": pa.null()})
    fields = arrow_type_mapper(col_type).model_fields
    assert list(fields.keys()) == ["x", "y", "z"]
    dtypes = [field.annotation for field in fields.values()]
    assert dtypes == [int | None, str | None, str | None]


def test_arrow_type_error():
    col_type = pa.union(
        [pa.field("a", pa.binary(10)), pa.field("b", pa.string())],
        mode=pa.lib.UnionMode_DENSE,
    )
    with pytest.raises(TypeError):
        arrow_type_mapper(col_type)


def test_schema_to_output():
    schema = pa.schema(
        [
            ("some_int", pa.int32()),
            ("some_string", pa.string()),
            ("strict_int", pa.int32(), False),
        ]
    )

    output, original_names = schema_to_output(schema)

    assert original_names == ["some_int", "some_string", "strict_int"]
    assert output == {
        "some_int": int | None,
        "some_string": str | None,
        "strict_int": int,
    }


def test_parquet_convert_column_names():
    schema = pa.schema(
        [
            ("UpperCaseCol", pa.int32()),
            ("dot.notation.col", pa.int32()),
            ("with-dashes", pa.int32()),
            ("with spaces", pa.int32()),
            ("with-multiple--dashes", pa.int32()),
            ("with__underscores", pa.int32()),
            ("__leading__underscores", pa.int32()),
            ("trailing__underscores__", pa.int32()),
        ]
    )

    output, original_names = schema_to_output(schema)

    assert original_names == [
        "UpperCaseCol",
        "dot.notation.col",
        "with-dashes",
        "with spaces",
        "with-multiple--dashes",
        "with__underscores",
        "__leading__underscores",
        "trailing__underscores__",
    ]
    assert list(output) == [
        "uppercasecol",
        "dot_notation_col",
        "with_dashes",
        "with_spaces",
        "with_multiple_dashes",
        "with_underscores",
        "leading_underscores",
        "trailing_underscores",
    ]


def test_parquet_missing_column_names():
    schema = pa.schema(
        [
            ("", pa.int32()),
            ("", pa.int32()),
        ]
    )

    output, original_names = schema_to_output(schema)

    assert original_names == ["", ""]
    assert list(output) == ["c0", "c1"]


def test_parquet_override_column_names():
    schema = pa.schema([("some_int", pa.int32()), ("some_string", pa.string())])
    col_names = ["n1", "n2"]

    output, original_names = schema_to_output(schema, col_names)

    assert original_names == ["n1", "n2"]
    assert output == {
        "n1": int | None,
        "n2": str | None,
    }


def test_parquet_override_column_names_invalid():
    schema = pa.schema([("some_int", pa.int32()), ("some_string", pa.string())])
    col_names = ["n1", "n2", "n3"]
    with pytest.raises(ValueError):
        schema_to_output(schema, col_names)


def test_infer_schema_no_files(test_session):
    schema = {"file": File, "my_col": int}
    chain = dc.read_records([], schema=schema, session=test_session, in_memory=True)
    with pytest.raises(ValueError):
        infer_schema(chain)
