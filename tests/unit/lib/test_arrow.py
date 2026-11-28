import asyncio
import multiprocessing
import os
import threading
import weakref
from datetime import UTC, datetime
from typing import get_args

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from fsspec.implementations.http import HTTPFile
from pyarrow import feather
from pyarrow.dataset import CsvFileFormat, JsonFileFormat, ParquetFileFormat
from pydantic import field_validator

import datachain as dc
from datachain.lib import arrow as arrow_mod
from datachain.lib import parquet_ranges
from datachain.lib.arrow import (
    ArrowGenerator,
    arrow_type_mapper,
    file_schemas,
    is_parquet,
    schema_to_output,
)
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.file import ArrowRow, File, FileError
from datachain.lib.hf import HFClassLabel
from datachain.lib.parquet_ranges import ParquetBufferedFile, column_ranges
from datachain.utils import filtered_cloudpickle_dumps
from tests.utils import skip_if_not_sqlite


@pytest.mark.parametrize("source", ["local", "http"])
@pytest.mark.parametrize("format", ["parquet", "csv", "json"])
def test_parse_tabular_rejects_a_file_changed_since_listing(
    tmp_path, test_session, http_server, source, format
):
    path = tmp_path / f"table.{format}"

    def write(value):
        frame = pd.DataFrame({"value": [value]})
        if format == "parquet":
            frame.to_parquet(path, index=False)
        elif format == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_json(path, orient="records", lines=True)

    write(1)
    base = tmp_path.as_uri() if source == "local" else http_server
    files = dc.read_storage(f"{base}/{path.name}", session=test_session)
    files.exec()
    mtime = path.stat().st_mtime + 1
    write(2)
    os.utime(path, (mtime, mtime))
    with pytest.raises(FileNotFoundError, match="changed on the source"):
        files.parse_tabular(format=format).exec()


@pytest.mark.parametrize("cache", [True, False])
def test_read_parquet_source_reads_back_its_row(tmp_path, test_session, cache):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": [34, 12345], "text": ["we", "28"]}), path)
    chain = dc.read_parquet(
        path.as_uri(), settings={"cache": cache}, session=test_session
    )
    rows = chain.order_by("id").to_list("source", "id", "text")
    assert [(s.read(), i, t) for s, i, t in rows] == [
        ({"id": 34, "text": "we"}, 34, "we"),
        ({"id": 12345, "text": "28"}, 12345, "28"),
    ]


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
    func = ArrowGenerator(None, output_schema)
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
    func = ArrowGenerator(None, output_schema)
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
    assert sorted(row[:3] for row in rows) == sorted(tuple(r.values()) for r in pylist)
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
    cat: str | None = None
    empty: str | None = None
    fl: list[float | None] | None = None
    il: list[int | None] | None = None
    sl: list[str | None] | None = None


@pytest.fixture
def plain_types_parquet(tmp_path):
    when = datetime(2026, 1, 2, 3, 4, 5, 6)
    table = pa.table(
        {
            "i8": pa.array([-1, 2, 3], pa.int8()),
            "u32": pa.array([2**32 - 1, 0, None], pa.uint32()),
            "f32": pa.array([1.5, None, -2.5], pa.float32()),
            "f64": pa.array([None, 2.25, -0.0], pa.float64()),
            "b": pa.array([True, False, None]),
            "s": pa.array(["x", None, ""]),
            "ls": pa.array([None, "ü", "z"], pa.large_string()),
            "bin": pa.array([b"\x00", None, b""]),
            "ts": pa.array([when, None, None]),
            "tz": pa.array(
                [None, when.replace(tzinfo=UTC), None], pa.timestamp("ms", tz="UTC")
            ),
            "cat": pa.array(["a", None, "a"]).dictionary_encode(),
            "empty": pa.array([None, None, None]),
            "fl": pa.array([[1.5, -2.0], None, []]),
            "il": pa.array([[1], [2, None], None]),
            "sl": pa.array([None, ["x", None], ["ü"]]),
        }
    )
    path = tmp_path / "plain.parquet"
    pq.write_table(table, path, row_group_size=1)
    return path


@pytest.mark.parametrize("source", [True, False])
@pytest.mark.parametrize("column", ["", "p"])
def test_read_parquet_dict_output_matches_model_output(
    plain_types_parquet, test_session, column, source
):
    output = {name: info.annotation for name, info in PlainTypes.model_fields.items()}
    prefix = f"{column}." if column else ""
    signals = [prefix + name for name in PlainTypes.model_fields]
    if source:
        signals.append("source")
    by_types, by_model = (
        dc.read_parquet(
            plain_types_parquet.as_uri(),
            output=out,
            column=column,
            source=source,
            session=test_session,
        )
        .order_by(prefix + "i8")
        .to_list(*signals)
        for out in (output, PlainTypes)
    )

    assert by_types == by_model
    assert [row[:2] for row in by_types] == [(-1, 2**32 - 1), (2, 0), (3, None)]
    if source:
        assert [row[-1].index for row in by_types] == [0, 1, 2]


def test_read_parquet_dict_output_converts_values_to_its_types(tmp_path, test_session):
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"i": [0, 1], "s": ["1.5", "2"]}), path)
    chain = dc.read_parquet(
        path.as_uri(), output={"i": bool, "s": float}, session=test_session
    )
    rows = sorted(chain.to_list("i", "s"))
    assert [tuple(map(repr, row)) for row in rows] == [
        ("False", "1.5"),
        ("True", "2.0"),
    ]


def test_read_parquet_dict_output_after_file_without_column_reads_none(
    tmp_path, test_session
):
    pq.write_table(pa.table({"a": [1], "b": ["x"]}), tmp_path / "1.parquet")
    pq.write_table(pa.table({"a": [2], "c": ["y"]}), tmp_path / "2.parquet")
    chain = dc.read_parquet(
        f"{tmp_path.as_uri()}/",
        output={"a": int, "b": str},
        source=False,
        session=test_session,
    )
    assert chain.order_by("a").to_list() == [(1, "x"), (2, None)]


def test_read_parquet_after_to_parquet_reads_its_signals(tmp_path, test_session):
    pq.write_table(pa.table({"text": ["hello", "world"]}), tmp_path / "plain.parquet")
    path = tmp_path / "exported.parquet"
    dc.read_parquet(
        (tmp_path / "plain.parquet").as_uri(), session=test_session
    ).to_parquet(path)
    chain = dc.read_parquet(path.as_uri(), session=test_session)
    assert chain.order_by("text").to_values("text") == ["hello", "world"]


@skip_if_not_sqlite  # ClickHouse driver shifts pre-1970 sub-second timestamps by +1s
def test_read_parquet_ns_timestamp_before_1970_drops_nanoseconds(
    tmp_path, test_session
):
    path = tmp_path / "ns.parquet"
    pq.write_table(pa.table({"t": pa.array([-1], pa.timestamp("ns", tz="UTC"))}), path)

    chain = dc.read_parquet(path.as_uri(), session=test_session)

    assert chain.to_values("t") == [
        datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
    ]


class Declared(dc.DataModel):
    a: int | None = None


@pytest.mark.parametrize("base", [Declared, dict_to_data_model("Inferred", {"a": int})])
def test_read_parquet_output_model_runs_its_validators(tmp_path, test_session, base):
    class Doubled(base):
        @field_validator("a")
        @classmethod
        def double(cls, value):
            return value * 2

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
        (pa.map_(pa.string(), pa.int32()), dict[str, int | None]),
        (pa.map_(pa.string(), pa.struct([("x", pa.int64())])), dict),
        (pa.map_(pa.string(), pa.list_(pa.int64(), 2)), dict),
        (pa.map_(pa.decimal128(5, 2), pa.int64()), dict),
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


def test_read_parquet_nested_lists_keep_their_values(tmp_path, test_session):
    path = tmp_path / "nested.parquet"
    values = [[[1], [2, None]], [[], [3]]]
    nested = pa.list_(pa.list_(pa.int64()))
    pq.write_table(pa.table({"i": [0, 1], "v": pa.array(values, nested)}), path)
    chain = dc.read_parquet(path.as_uri(), session=test_session)
    assert chain.order_by("i").to_values("v") == values


@skip_if_not_sqlite  # ClickHouse arrays hold no NULL
def test_read_parquet_nested_list_keeps_a_null_inner_list(tmp_path, test_session):
    path = tmp_path / "nested.parquet"
    nested = pa.list_(pa.list_(pa.int64()))
    pq.write_table(pa.table({"v": pa.array([[[1], None]], nested)}), path)
    chain = dc.read_parquet(path.as_uri(), session=test_session)
    assert chain.to_values("v") == [[[1], None]]


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


def test_read_parquet_map_columns_as_dicts(test_session, tmp_path):
    counts = pa.map_(pa.string(), pa.int64())
    table = pa.table(
        {
            "counts": pa.array([[("a", 1), ("b", 2)]], counts),
            "event": pa.array([{"tags": [("k", 3)]}], pa.struct([("tags", counts)])),
            "history": pa.array([[[("c", 4)]]], pa.list_(counts)),
            "nested": pa.array([[("d", [("e", 5)])]], pa.map_(pa.string(), counts)),
        }
    )
    path = tmp_path / "maps.parquet"
    pq.write_table(table, path)

    chain = dc.read_parquet(path.as_posix(), source=True, session=test_session)

    ((counts, tags, history, nested, source),) = chain.to_list(
        "counts", "event.tags", "history", "nested", "source"
    )
    assert (counts, tags, history) == ({"a": 1, "b": 2}, {"k": 3}, [{"c": 4}])
    assert nested == {"d": {"e": 5}}
    assert source.read() == {
        "counts": {"a": 1, "b": 2},
        "event": {"tags": {"k": 3}},
        "history": [{"c": 4}],
        "nested": {"d": {"e": 5}},
    }


@pytest.mark.parametrize(
    "values,col_type",
    [
        ([[("a", 1), ("a", 2)]], pa.map_(pa.string(), pa.int64())),
        (
            [[("d", [("e", 1), ("e", 2)])]],
            pa.map_(pa.string(), pa.map_(pa.string(), pa.int64())),
        ),
    ],
)
def test_read_parquet_maps_with_duplicate_keys_raise(
    test_session, tmp_path, values, col_type
):
    path = tmp_path / "maps.parquet"
    pq.write_table(pa.table({"m": pa.array(values, col_type)}), path)

    with pytest.raises(KeyError, match="duplicate keys"):
        dc.read_parquet(path.as_posix(), session=test_session).exec()
    with pytest.raises(KeyError, match="duplicate keys"):
        ArrowRow(file=File.at(path.as_posix()), index=0, kwargs={}).read()


def test_read_parquet_maps_of_structs(test_session, tmp_path):
    amounts = pa.map_(pa.string(), pa.struct([("amount", pa.int64())]))
    table = pa.table(
        {
            "orders": pa.array([[("a", {"amount": 1}), ("b", None)]], amounts),
            "history": pa.array([[[("c", {"amount": 2})]]], pa.list_(amounts)),
        }
    )
    path = tmp_path / "maps.parquet"
    pq.write_table(table, path)

    chain = dc.read_parquet(path.as_posix(), session=test_session)

    assert chain.to_list("orders", "history") == [
        ({"a": {"amount": 1}, "b": None}, [{"c": {"amount": 2}}])
    ]


def test_read_parquet_map_keys_keep_their_type(test_session, tmp_path):
    table = pa.table(
        {
            "ints": pa.array([[(1, 10)]], pa.map_(pa.int64(), pa.int64())),
            "flags": pa.array([[(True, 3)]], pa.map_(pa.bool_(), pa.int64())),
        }
    )
    path = tmp_path / "maps.parquet"
    pq.write_table(table, path)

    chain = dc.read_parquet(path.as_posix(), session=test_session)

    assert chain.to_list("ints", "flags") == [({1: 10}, {True: 3})]


def test_read_parquet_maps_of_fixed_and_large_lists(test_session, tmp_path):
    fixed = pa.map_(pa.string(), pa.list_(pa.int64(), 2))
    large = pa.map_(pa.string(), pa.large_list(pa.int64()))
    table = pa.table(
        {
            "fixed": pa.array([[("k", [1, 2])]], fixed),
            "large": pa.array([[("k", [3])]], large),
        }
    )
    path = tmp_path / "maps.parquet"
    pq.write_table(table, path)

    chain = dc.read_parquet(path.as_posix(), session=test_session)

    assert chain.to_list("fixed", "large") == [({"k": [1, 2]}, {"k": [3]})]


# The 1969 value floors to a whole second because the ClickHouse driver stores
# pre-1970 timestamps that have microseconds one second late.
NANOSECONDS_1969 = pd.Timestamp("1969-12-31 23:59:59.0000005")
FLOORED_1969 = datetime(1969, 12, 31, 23, 59, 59, tzinfo=UTC)


def test_read_parquet_nanosecond_timestamps_floored_to_microseconds(
    test_session, tmp_path
):
    timestamps = pa.array(
        [NANOSECONDS_1969, pd.Timestamp("2024-01-01 00:00:00.123456789")],
        type=pa.timestamp("ns"),
    )
    table = pa.table(
        {
            "ts": timestamps,
            "utc": timestamps.cast(pa.timestamp("ns", tz="UTC")),
            "event": pa.StructArray.from_arrays([timestamps], ["ts"]),
            "history": pa.ListArray.from_arrays([0, 1, 2], timestamps),
        }
    )
    path = tmp_path / "data.parquet"
    pq.write_table(table, path)

    chain = dc.read_parquet(path.as_posix(), session=test_session).order_by("ts")

    expected = [FLOORED_1969, datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=UTC)]
    assert chain.to_values("ts") == expected
    assert chain.to_values("utc") == expected
    assert [e.ts for e in chain.to_values("event")] == expected
    assert [h[0] for h in chain.to_values("history")] == expected


def test_parse_tabular_nanosecond_timestamp_dictionary_floored_to_microseconds(
    test_session, tmp_path
):
    timestamps = pa.array([NANOSECONDS_1969], type=pa.timestamp("ns"))
    path = tmp_path / "data.arrow"
    feather.write_feather(pa.table({"ts": timestamps.dictionary_encode()}), path)

    chain = dc.read_storage(path.as_posix(), session=test_session).parse_tabular(
        format="ipc"
    )

    assert chain.to_values("ts") == [FLOORED_1969]


def test_read_csv_fractional_timestamps_floored_to_microseconds(test_session, tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("ts\n2024-01-01 00:00:00.5\n2024-01-01 00:00:00.123456789\n")

    chain = dc.read_csv(path.as_posix(), session=test_session)

    assert chain.order_by("ts").to_values("ts") == [
        datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 0, 0, 500000, tzinfo=UTC),
    ]


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
    chain = dc.read_records([], schema=schema, session=test_session)
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


@skip_if_not_sqlite
@pytest.mark.parametrize("format", ["parquet", "csv", "json"])
@pytest.mark.parametrize("source", [False, True])
def test_tabular_load_uses_native_arrow(tmp_path, test_session_tmpfile, format, source):
    path = tmp_path / f"native.{format}"
    frame = pd.DataFrame({"n": list(range(2048)), "text": ["one"] * 2048})
    if format == "parquet":
        frame.to_parquet(path, index=False)
    elif format == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_json(path, orient="records", lines=True)
    result = (
        dc.read_storage(path.as_uri(), session=test_session_tmpfile)
        .parse_tabular(format=format, source=source)
        .save("native")
    )
    assert result.order_by("n").to_list("n", "text") == [
        (i, "one") for i in range(2048)
    ]
    if source:
        assert [
            row.read() for row in result.order_by("n").limit(3).to_values("source")
        ] == frame.head(3).to_dict("records")


@skip_if_not_sqlite  # includes SQLite's PRAGMA integrity_check
def test_read_parquet_with_parallel_keeps_every_row(tmp_path, test_session_tmpfile):
    for i in range(8):
        pq.write_table(
            pa.table({"n": list(range(i * 200, (i + 1) * 200))}),
            tmp_path / f"part{i}.parquet",
        )
    result = dc.read_parquet(
        f"{tmp_path.as_uri()}/*.parquet",
        session=test_session_tmpfile,
        settings={"parallel": 2},
    ).save("parallel_arrow")
    assert result.order_by("n").to_values("n") == list(range(1600))
    db = test_session_tmpfile.catalog.warehouse.db
    assert db.execute_str("PRAGMA integrity_check").fetchall() == [("ok",)]


def test_parquet_column_ranges():
    table = pa.table(
        {
            "cat": pa.array(["a", "b", "a", "b"]).dictionary_encode(),
            "pad1": [b"x" * 40_000] * 4,
            "nested": pa.array(
                [
                    [{"x": 1, "y": "p"}],
                    [{"x": 2, "y": "q"}],
                    [{"x": 3, "y": "r"}],
                    [{"x": 4, "y": "s"}],
                ]
            ),
            "pad2": [b"y" * 40_000] * 4,
        }
    )
    buf = pa.BufferOutputStream()
    pq.write_table(
        table, buf, row_group_size=2, compression="NONE", use_dictionary=["cat"]
    )
    meta = pq.read_metadata(pa.BufferReader(buf.getvalue()))

    expected = []
    for r in range(meta.num_row_groups):
        group = meta.row_group(r)
        cat = group.column(0)
        assert cat.has_dictionary_page
        assert cat.dictionary_page_offset < cat.data_page_offset
        expected.append(
            (
                cat.dictionary_page_offset,
                cat.dictionary_page_offset + cat.total_compressed_size,
            )
        )
        nx, ny = group.column(2), group.column(3)
        expected.append(
            (
                nx.dictionary_page_offset or nx.data_page_offset,
                (ny.dictionary_page_offset or ny.data_page_offset)
                + ny.total_compressed_size,
            )
        )

    assert column_ranges(meta, ["cat", "nested"]) == expected
    assert column_ranges(meta, ["cat", "nested"], gap=200_000) == [
        (expected[0][0], expected[-1][1])
    ]


@pytest.fixture
def http_requests(monkeypatch):
    calls: list[tuple[str, int, int]] = []
    orig = HTTPFile._fetch_range

    def fetch_range(self, start, end):
        calls.append((self.url, start, end))
        return orig(self, start, end)

    monkeypatch.setattr(HTTPFile, "_fetch_range", fetch_range)
    return calls


def test_read_parquet_columns_falls_back_when_chunks_exceed_budget(
    tmp_path, test_session, http_server, http_requests, monkeypatch
):
    path = tmp_path / "over.parquet"
    table = pa.table({"a": list(range(1000)), "pad": [b"z" * 200] * 1000})
    pq.write_table(
        table, path, row_group_size=500, compression="NONE", use_dictionary=False
    )
    size = path.stat().st_size
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", 4096)
    # Column chunks (~8 KiB) fit under 10 KiB, but 4 KiB footer + chunks exceed it.
    monkeypatch.setattr(parquet_ranges, "PARQUET_RANGE_MAX_BYTES", 10_000)
    chain = dc.read_parquet(
        f"{http_server}/over.parquet",
        columns=["a"],
        settings={"prefetch": 0},
        session=test_session,
    )
    http_requests.clear()

    values = chain.to_values("a")

    ranges = [(s, e) for _, s, e in http_requests]
    assert ranges[:2] == [(size - 4096, size), (size - 65536, size)]
    assert len(ranges) >= 3
    assert sorted(values) == list(range(1000))


def test_read_parquet_columns_falls_back_when_footer_exceeds_budget(
    tmp_path, test_session, http_server, http_requests, monkeypatch
):
    path = tmp_path / "over.parquet"
    table = pa.table({"a": list(range(1000)), "pad": [b"z" * 200] * 1000})
    pq.write_table(
        table, path, row_group_size=500, compression="NONE", use_dictionary=False
    )
    size = path.stat().st_size
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    # Full footer (~500 B) exceeds PARQUET_RANGE_MAX_BYTES when expanding a 64 B sample.
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", 64)
    monkeypatch.setattr(parquet_ranges, "PARQUET_RANGE_MAX_BYTES", 128)
    chain = dc.read_parquet(
        f"{http_server}/over.parquet",
        columns=["a"],
        settings={"prefetch": 0},
        session=test_session,
    )
    http_requests.clear()

    values = chain.to_values("a")

    ranges = [(s, e) for _, s, e in http_requests]
    # 64 B probe does not expand the oversized footer; pyarrow reads its 64 KiB tail.
    assert ranges[:2] == [(size - 64, size), (size - 65536, size)]
    assert sorted(values) == list(range(1000))


def test_read_parquet_columns_shares_byte_budget_across_prefetched_files(
    tmp_path, test_session, http_server, http_requests, monkeypatch
):
    urls = []
    for i in range(6):
        pq.write_table(
            pa.table(
                {
                    "a": [2 * i, 2 * i + 1],
                    "b": [b"x" * 35_000, b"y" * 35_000],
                    "c": [10 * i, 10 * i + 5],
                }
            ),
            tmp_path / f"part{i}.parquet",
            row_group_size=1,
            compression="NONE",
            use_dictionary=False,
        )
        urls.append(f"{http_server}/part{i}.parquet")
    file_size = (tmp_path / "part0.parquet").stat().st_size
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", 65536)
    # prefetch=1 can have 3 files in flight (process, result_queue, worker);
    # the extra 64 KiB lets a 4th file probe its footer if process() never releases.
    monkeypatch.setattr(
        parquet_ranges, "PARQUET_RANGE_MAX_BYTES", 3 * file_size + 65536
    )
    chain = dc.read_parquet(
        urls,
        columns=["a"],
        settings={"prefetch": 1},
        session=test_session,
    )
    http_requests.clear()

    rows = chain.order_by("a").to_values("a")

    assert rows == list(range(12))
    last_calls = [r for r in http_requests if r[0].endswith("part5.parquet")]
    assert len(last_calls) == 2
    assert len(http_requests) == 6 * 2


def test_read_parquet_columns_fetches_ranges_in_process_when_prefetch_budget_full(
    tmp_path, test_session, http_server, http_requests, monkeypatch
):
    urls = []
    for i in range(2):
        pq.write_table(
            pa.table(
                {
                    "a": [2 * i, 2 * i + 1],
                    "b": [b"x" * 70_000, b"y" * 70_000],
                    "c": [10 * i, 10 * i + 5],
                }
            ),
            tmp_path / f"part{i}.parquet",
            row_group_size=1,
            compression="NONE",
            use_dictionary=False,
        )
        urls.append(f"{http_server}/part{i}.parquet")
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", 65536)
    # Holds 1 file (~66 KiB), so a second prefetched file defers to process().
    monkeypatch.setattr(parquet_ranges, "PARQUET_RANGE_MAX_BYTES", 70_000)
    chain = dc.read_parquet(
        urls,
        columns=["a"],
        settings={"prefetch": 2},
        session=test_session,
    )
    http_requests.clear()

    rows = chain.order_by("a").to_values("a")

    assert rows == list(range(4))
    sizes = [
        [end - start for url, start, end in http_requests if url.endswith(name)]
        for name in ("part0.parquet", "part1.parquet")
    ]
    assert sorted(sizes) == [[65536, 77, 77], [65536, 65536, 77, 77]]


def test_read_parquet_columns_releases_budget_on_failed_range_fetch(
    tmp_path, test_session, http_server, monkeypatch
):
    urls = []
    for i in range(2):
        pq.write_table(
            pa.table(
                {
                    "a": [2 * i, 2 * i + 1],
                    "b": [b"x" * 35_000, b"y" * 35_000],
                    "c": [10 * i, 10 * i + 5],
                }
            ),
            tmp_path / f"part{i}.parquet",
            row_group_size=1,
            compression="NONE",
            use_dictionary=False,
        )
        urls.append(f"{http_server}/part{i}.parquet")
    file_size = (tmp_path / "part0.parquet").stat().st_size
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", 65536)
    monkeypatch.setattr(parquet_ranges, "PARQUET_RANGE_MAX_BYTES", file_size)
    chain = dc.read_parquet(
        urls,
        columns=["a"],
        settings={"prefetch": 1},
        session=test_session,
    )
    orig_fetch = HTTPFile._fetch_range
    calls: list[tuple[str, int, int]] = []
    failed_once = False

    def flaky_fetch(self, start, end):
        nonlocal failed_once
        calls.append((self.url, start, end))
        if not failed_once and self.url.endswith("part0.parquet") and start < 100:
            failed_once = True
            raise FileError("transient range failure", http_server, "part0.parquet")
        return orig_fetch(self, start, end)

    monkeypatch.setattr(HTTPFile, "_fetch_range", flaky_fetch)

    rows = chain.order_by("a").to_values("a")

    assert failed_once
    assert rows == list(range(4))
    part1_calls = [r for r in calls if r[0].endswith("part1.parquet")]
    assert len(part1_calls) == 2


def test_read_parquet_without_columns_does_not_prefetch_column_ranges(
    tmp_path, test_session, http_server, http_requests, monkeypatch
):
    path = tmp_path / "full.parquet"
    pq.write_table(pa.table({"a": [1, 2], "b": ["x", "y"]}), path)
    size = path.stat().st_size
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", 64)
    chain = dc.read_parquet(
        f"{http_server}/full.parquet",
        settings={"prefetch": 0},
        session=test_session,
    )
    http_requests.clear()

    assert sorted(chain.to_values("a")) == [1, 2]
    assert (size - 64, size) not in [(s, e) for _, s, e in http_requests]


def test_read_parquet_columns_after_file_without_etag_changed_reads_new_rows(
    tmp_path, test_session, http_server_without_ranges, monkeypatch
):
    path = tmp_path / "mutable.parquet"
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    url = f"{http_server_without_ranges}/mutable.parquet"

    pq.write_table(pa.table({"a": [1], "b": ["x"]}), path)
    assert dc.read_parquet(url, columns=["a"], session=test_session).to_values("a") == [
        1
    ]

    pq.write_table(pa.table({"a": [2], "b": ["y"]}), path)
    assert dc.read_parquet(url, columns=["a"], session=test_session).to_values("a") == [
        2
    ]


def test_read_parquet_columns_merges_trailing_chunk_with_footer_sample(
    tmp_path, test_session, http_server, http_requests, monkeypatch
):
    path = tmp_path / "merge.parquet"
    table = pa.table({"a": [b"a" * 40_000] * 2, "b": [b"x" * 40_000, b"y" * 40_000]})
    pq.write_table(
        table, path, row_group_size=2, compression="NONE", use_dictionary=False
    )
    raw = path.read_bytes()
    meta = pq.read_metadata(path)
    col_b = meta.row_group(0).column(1)
    col_b_end = col_b.data_page_offset + col_b.total_compressed_size
    # Place sample_start inside column b's chunk so its span ends right at sample_start.
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    monkeypatch.setattr(
        parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", len(raw) - (col_b_end - 64)
    )
    chain = dc.read_parquet(
        f"{http_server}/merge.parquet", columns=["b"], session=test_session
    )
    http_requests.clear()

    rows = chain.to_values("b")

    # 1 footer sample + 1 span [col_b_start, sample_start), merged with no 3rd read.
    assert len(http_requests) == 2
    assert sorted(rows) == [b"x" * 40_000, b"y" * 40_000]


@pytest.mark.parametrize(
    "payload,sample_size",
    [
        (b"not a parquet file", 4096),
        (b"PAR1" + b"\x00" * 100 + (50).to_bytes(4, "little") + b"PAR1", 4096),
        (b"PAR1" + b"\x00" * 100 + (50).to_bytes(4, "little") + b"PAR1", 16),
    ],
    ids=["non_parquet", "corrupt_footer", "corrupt_expanded_footer"],
)
def test_read_parquet_columns_invalid_files_fail_like_full_read(
    tmp_path, test_session, http_server, monkeypatch, payload, sample_size
):
    valid = tmp_path / "0_valid.parquet"
    pq.write_table(pa.table({"a": [1]}), valid)
    broken = tmp_path / "1_broken.parquet"
    broken.write_bytes(payload)
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", sample_size)
    urls = [f"{http_server}/0_valid.parquet", f"{http_server}/1_broken.parquet"]

    with pytest.raises((pa.ArrowInvalid, OSError)) as full_exc:
        dc.read_parquet(urls, output={"a": int}, session=test_session).exec()

    with pytest.raises(type(full_exc.value)) as col_exc:
        dc.read_parquet(
            urls, output={"a": int}, columns=["a"], session=test_session
        ).exec()

    assert (
        str(col_exc.value).split(": ")[-1].strip()
        == str(full_exc.value).split(": ")[-1].strip()
    )


def test_parquet_buffered_file_freed_by_refcount(tmp_path, catalog):
    class Parts(list):
        pass

    file = File(path="table.parquet", source=tmp_path.as_uri())
    client = catalog.get_client(file.source)
    parts = Parts([(0, b"PAR1")])
    parts_ref = weakref.ref(parts)
    f = ParquetBufferedFile(client, file, 4, parts)
    file_ref = weakref.ref(f)
    assert f.read() == b"PAR1"
    del f, parts
    assert file_ref() is None
    assert parts_ref() is None


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="requires POSIX fork",
)
def test_read_parquet_columns_in_forked_child_after_parent_read(
    tmp_path, test_session, http_server, monkeypatch
):
    path = tmp_path / "fork.parquet"
    pq.write_table(
        pa.table({"a": [1, 2], "pad": [b"x" * 100_000, b"y" * 100_000]}),
        path,
        row_group_size=1,
        compression="NONE",
        use_dictionary=False,
    )
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", 4096)
    url = f"{http_server}/fork.parquet"

    assert sorted(
        dc.read_parquet(url, columns=["a"], session=test_session).to_values("a")
    ) == [1, 2]

    file = File(source=http_server, path="fork.parquet", size=path.stat().st_size)
    client = test_session.catalog.get_client(file.source)
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=lambda conn: conn.send(
            len(
                parquet_ranges.read_parquet_parts(client, file, ["a"], lambda n: True)[
                    0
                ][1]
            )
        ),
        args=(child_conn,),
    )
    try:
        proc.start()
        child_conn.close()
        proc.join(timeout=10)
        assert proc.exitcode == 0
        assert parent_conn.recv() == 3
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join()
        parent_conn.close()
        child_conn.close()


def test_arrow_generator_is_picklable_and_resets_prefetch_state(tmp_path, catalog):
    path = tmp_path / "picklable.parquet"
    pq.write_table(pa.table({"a": [1, 2]}), path)
    file = File(
        path="picklable.parquet", source=tmp_path.as_uri(), size=path.stat().st_size
    )
    file._set_stream(catalog, caching_enabled=False)

    model = dict_to_data_model("Row", {"a": int})
    gen = ArrowGenerator(
        pa.schema([("a", pa.int64())]),
        model,
        source=False,
        columns=["a"],
        prefetch_max_size=1,
        format="parquet",
    )
    gen._parts[id(file)] = (file, 128, (file.size, [(0, b"PAR1")]))
    gen._parts_bytes = 128

    restored = cloudpickle.loads(filtered_cloudpickle_dumps(gen))
    assert restored._parts == {}
    assert restored._parts_bytes == 0
    assert [row.a for (row,) in restored.process(file)] == [1, 2]


def test_arrow_generator_cancel_and_teardown_release_prefetch_budget(
    tmp_path, catalog, http_server, monkeypatch
):
    path = tmp_path / "cancel.parquet"
    pq.write_table(
        pa.table({"a": [1, 2], "pad": [b"x" * 20_000, b"y" * 20_000]}),
        path,
        row_group_size=1,
        compression="NONE",
        use_dictionary=False,
    )
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", 4096)
    size = path.stat().st_size

    model = dict_to_data_model("Row", {"a": int})
    gen = ArrowGenerator(
        pa.schema([("a", pa.int64()), ("pad", pa.binary())]),
        model,
        columns=["a"],
        prefetch_max_size=1,
        format="parquet",
    )

    f1 = File(source=http_server, path="cancel.parquet", size=size)
    f1._set_stream(catalog, caching_enabled=False)
    assert asyncio.run(gen._prefetch_file(f1)) is True
    assert id(f1) in gen._parts
    assert gen._parts_bytes > 0
    gen.teardown()
    assert gen._parts == {}
    assert gen._parts_bytes == 0

    orig_fetch = HTTPFile._fetch_range
    in_chunk_fetch = threading.Event()
    allow_chunk_finish = threading.Event()

    def slow_chunk_fetch(self, start, end):
        if start < 100:
            in_chunk_fetch.set()
            allow_chunk_finish.wait(timeout=5)
        return orig_fetch(self, start, end)

    monkeypatch.setattr(HTTPFile, "_fetch_range", slow_chunk_fetch)
    f2 = File(source=http_server, path="cancel.parquet", size=size)
    f2._set_stream(catalog, caching_enabled=False)

    async def cancel_during_chunk_fetch():
        task = asyncio.create_task(gen._prefetch_file(f2))
        assert await asyncio.to_thread(in_chunk_fetch.wait, 5)
        assert gen._parts_bytes > 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        allow_chunk_finish.set()
        for _ in range(50):
            if gen._parts_bytes == 0:
                break
            await asyncio.sleep(0.01)
        assert gen._parts == {}
        assert gen._parts_bytes == 0

    asyncio.run(cancel_during_chunk_fetch())


def test_read_parquet_columns_missing_column_fetches_footer_once(
    tmp_path, test_session, http_server, http_requests, monkeypatch
):
    pq.write_table(pa.table({"a": [1]}), tmp_path / "f0.parquet")
    pq.write_table(pa.table({"a": [2], "b": ["x"]}), tmp_path / "f1.parquet")
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    urls = [f"{http_server}/f0.parquet", f"{http_server}/f1.parquet"]
    chain = dc.read_parquet(
        urls,
        columns=["b"],
        settings={"prefetch": 2},
        session=test_session,
    )
    http_requests.clear()

    values = chain.to_values("b")

    assert sorted(values, key=lambda v: v or "") == [None, "x"]
    f0_calls = [r for r in http_requests if r[0].endswith("f0.parquet")]
    assert len(f0_calls) == 1


@pytest.mark.parametrize("prefetch", [2, 0])
def test_read_parquet_columns_fetches_footer_and_chunk_ranges_once_per_file(
    tmp_path, test_session, http_server, http_requests, monkeypatch, prefetch
):
    rng = np.random.default_rng(0)
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    monkeypatch.setattr(parquet_ranges, "PARQUET_FOOTER_SAMPLE_SIZE", 128 * 1024)
    # Footer > 64 KiB so pyarrow alone would read the 64 KiB tail and then the footer.
    table = pa.table(
        {f"c{i}": rng.random(8_000) for i in range(10)}
    ).replace_schema_metadata({b"pad": b"x" * 35_000})
    urls = []
    for i in range(2):
        path = tmp_path / f"f{i}.parquet"
        pq.write_table(table, path, row_group_size=8_000)
        file_size = path.stat().st_size
        urls.append(f"{http_server}/f{i}.parquet")
    local = dc.read_parquet(
        tmp_path.as_uri(), columns=["c1", "c8"], session=test_session
    ).order_by("c1", "c8")
    chain = dc.read_parquet(
        urls,
        columns=["c1", "c8"],
        settings={"prefetch": prefetch},
        session=test_session,
    ).order_by("c1", "c8")
    http_requests.clear()

    assert chain.to_list("c1", "c8") == local.to_list("c1", "c8")
    # Per file: 1 footer sample + 2 non-adjacent column chunk ranges (not 4 sequential).
    assert len(http_requests) == 2 * 3
    fetched_bytes = sum(end - start for _, start, end in http_requests)
    assert 0 < fetched_bytes < 2 * (file_size // 2)
