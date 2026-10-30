from datetime import datetime
from typing import get_args

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from pyarrow.dataset import CsvFileFormat, JsonFileFormat, ParquetFileFormat
from pydantic import field_validator

import datachain as dc
from datachain.lib.arrow import (
    ArrowGenerator,
    arrow_type_mapper,
    file_schemas,
    is_parquet,
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
    stream = File(path=name, source=f"file://{tmp_path}")
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
    stream = File(path=name, source=f"file://{tmp_path}")
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
    stream = File(path=name, source=f"file://{tmp_path}")
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
    stream = File(path=name, source=f"file://{tmp_path}")
    stream._set_stream(catalog, caching_enabled=False)

    output, original_names = schema_to_output(ds._data.schema, ["col"])

    output_schema = dict_to_data_model("", output, original_names)
    func = ArrowGenerator(output_schema=output_schema)
    for obj in func.process(stream):
        assert isinstance(obj[1].col, HFClassLabel)


@pytest.mark.parametrize("cache", [True, False])
def test_read_parquet_partitioned(tmp_path, test_session, cache):
    pylist = [
        {"first_name": "Alice", "age": 25, "city": "New York"},
        {"first_name": "Bob", "age": 30, "city": "Los Angeles"},
        {"first_name": "Charlie", "age": 35, "city": "Chicago"},
    ]
    pq_path = tmp_path / "parquets"
    table = pa.Table.from_pylist(pylist)
    pq.write_to_dataset(table, pq_path, partition_cols=["first_name"])

    chain = dc.read_parquet(
        pq_path.as_uri(),
        partitioning="hive",
        settings={"cache": cache},
        session=test_session,
    )

    rows = chain.to_list("first_name", "age", "city", "source")
    assert sorted(row[:3] for row in rows) == [tuple(r.values()) for r in pylist]
    for name, *_, source in rows:
        assert f"first_name={name}/" in source.file.path
        assert source.index == 0
        assert source.kwargs["partitioning"] == "hive"


class PlainTypes(dc.DataModel):
    i8: int | None = None
    u32: int | None = None
    f32: float | None = None
    f64: float | None = None
    b: bool | None = None
    s: str | None = None
    ls: str | None = None
    bin: bytes | None = None
    ts: datetime | None = None
    tz: datetime | None = None


@pytest.fixture
def plain_types_parquet(tmp_path):
    when = datetime(2026, 1, 2, 3, 4, 5, 6)
    table = pa.table(
        {
            "i8": pa.array([-1, None, 3], pa.int8()),
            "u32": pa.array([2**32 - 1, 0, None], pa.uint32()),
            "f32": pa.array([1.5, None, float("nan")], pa.float32()),
            "f64": pa.array([None, 2.25, -0.0], pa.float64()),
            "b": pa.array([True, False, None]),
            "s": pa.array(["x", None, ""]),
            "ls": pa.array([None, "ü", "z"], pa.large_string()),
            "bin": pa.array([b"\x00", None, b""]),
            "ts": pa.array([when, None, None]),
            "tz": pa.array([None, when, None], pa.timestamp("ms", tz="UTC")),
        }
    )
    path = tmp_path / "plain.parquet"
    pq.write_table(table, path)
    return path.as_uri()


@pytest.mark.parametrize("column", ["", "p"])
def test_read_parquet_plain_types_as_an_output_model_does(
    plain_types_parquet, test_session, column
):
    output = {name: info.annotation for name, info in PlainTypes.model_fields.items()}
    by_types, by_model = (
        dc.read_parquet(
            plain_types_parquet, output=out, column=column, session=test_session
        )
        for out in (output, PlainTypes)
    )

    prefix = f"{column}." if column else ""
    signals = [*(prefix + name for name in PlainTypes.model_fields), "source"]
    rows = by_types.order_by("source.index").to_list(*signals)
    assert rows == by_model.order_by("source.index").to_list(*signals)
    assert [row[:2] for row in rows] == [(-1, 2**32 - 1), (None, 0), (3, None)]
    assert [row[-1].index for row in rows] == [0, 1, 2]


class Doubled(dc.DataModel):
    a: int | None = None

    @field_validator("a")
    @classmethod
    def double(cls, value):
        return value * 2


def test_read_parquet_output_model_runs_its_validators(tmp_path, test_session):
    path = tmp_path / "a.parquet"
    pq.write_table(pa.table({"a": [1, 2]}), path)
    chain = dc.read_parquet(path.as_uri(), output=Doubled, session=test_session)
    assert sorted(chain.to_values("a")) == [2, 4]


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
        (pa.list_(pa.string()), list[str | None]),
        (pa.list_(pa.field("item", pa.string(), nullable=False)), list[str]),
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


def test_arrow_type_mapper_list_struct_models_distinct_per_column():
    users = pa.list_(pa.struct({"name": pa.string()}))
    orders = pa.list_(pa.struct({"amount": pa.int32()}))

    (users_model,) = get_args(arrow_type_mapper(users, "users"))
    (orders_model,) = get_args(arrow_type_mapper(orders, "orders"))

    assert users_model.__name__ != orders_model.__name__
    assert "users" in users_model.__name__


def test_arrow_type_mapper_nested_list_only_nullable_scalars():
    nested = pa.list_(pa.list_(pa.string()))

    assert arrow_type_mapper(nested) == list[list[str | None]]


def test_read_parquet_multiple_struct_list_columns_roundtrip(test_session, tmp_path):
    df = pd.DataFrame(
        {
            "users": [[{"name": "alice"}, {"name": "bob"}]],
            "orders": [[{"amount": 5}]],
        }
    )
    path = tmp_path / "data.parquet"
    df.to_parquet(path)

    dc.read_parquet(path.as_posix(), session=test_session).save("two_structs")
    back = dc.read_dataset("two_structs", session=test_session)

    (users,) = back.to_values("users")
    (orders,) = back.to_values("orders")
    assert [user.name for user in users] == ["alice", "bob"]
    assert [order.amount for order in orders] == [5]


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


def test_file_schemas_no_files(test_session):
    schema = {"file": File, "my_col": int}
    chain = dc.read_records([], schema=schema, session=test_session, in_memory=True)
    with pytest.raises(ValueError):
        file_schemas(chain)


@pytest.mark.parametrize(
    "format,expected",
    [
        ("parquet", True),
        (ParquetFileFormat(), True),
        (ParquetFileFormat(buffer_size=4096), True),
        ("csv", False),
        (CsvFileFormat(), False),
        (JsonFileFormat(), False),
    ],
)
def test_is_parquet(format, expected):
    assert is_parquet(format) is expected
