import os
import weakref
from datetime import UTC, datetime
from typing import get_args

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from pyarrow import feather
from pyarrow.dataset import CsvFileFormat, JsonFileFormat, ParquetFileFormat
from pydantic import field_validator

import datachain as dc
from datachain.lib import arrow as arrow_mod
from datachain.lib.arrow import (
    ArrowGenerator,
    _column_ranges,
    _ParquetBufferedFile,
    arrow_type_mapper,
    file_schemas,
    is_parquet,
    schema_to_output,
)
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.file import ArrowRow, File
from datachain.lib.hf import HFClassLabel
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

    assert _column_ranges(meta, ["cat", "nested"]) == expected
    assert _column_ranges(meta, ["cat", "nested"], gap=200_000) == [
        (expected[0][0], expected[-1][1])
    ]


def test_read_parquet_parts_falls_back_when_over_bound(
    tmp_path, test_session, http_server, monkeypatch, mocker
):
    path = tmp_path / "over.parquet"
    table = pa.table({"a": list(range(1000)), "pad": [b"z" * 100] * 1000})
    pq.write_table(table, path, row_group_size=500, compression="NONE")
    raw = path.read_bytes()
    monkeypatch.setattr(arrow_mod, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    monkeypatch.setattr(arrow_mod, "PARQUET_FOOTER_SAMPLE_SIZE", 4096)
    # Column chunks (~8 KiB) fit under 10 KiB, but footer + chunks exceed it.
    monkeypatch.setattr(arrow_mod, "PARQUET_RANGE_MAX_BYTES", 10_000)
    spy = mocker.spy(arrow_mod, "_read_parquet_parts")

    values = dc.read_parquet(
        f"{http_server}/over.parquet", columns=["a"], session=test_session
    ).to_values("a")

    assert spy.spy_return == (len(raw), [(len(raw) - 4096, raw[-4096:])])
    assert sorted(values) == list(range(1000))

    # Full footer (~500 B) exceeds PARQUET_RANGE_MAX_BYTES when expanding a 64 B sample.
    monkeypatch.setattr(arrow_mod, "PARQUET_FOOTER_SAMPLE_SIZE", 64)
    monkeypatch.setattr(arrow_mod, "PARQUET_RANGE_MAX_BYTES", 128)
    values = dc.read_parquet(
        f"{http_server}/over.parquet", columns=["a"], session=test_session
    ).to_values("a")

    assert spy.spy_return == (len(raw), [])
    assert sorted(values) == list(range(1000))


def test_parquet_buffered_file_freed_by_refcount(catalog):
    class Parts(list):
        pass

    file = File(path="table.parquet", source="file:///tmp")
    client = catalog.get_client(file.source)
    parts = Parts([(0, b"PAR1")])
    parts_ref = weakref.ref(parts)
    f = _ParquetBufferedFile(client, file, 4, parts)
    file_ref = weakref.ref(f)
    assert f.read() == b"PAR1"
    del f, parts
    assert file_ref() is None
    assert parts_ref() is None


def test_parquet_buffered_file_closes_fallback_remote_stream(catalog, mocker):
    import io

    file = File(path="table.parquet", source="file:///tmp")
    client = catalog.get_client(file.source)
    remote = io.BytesIO(b"01234567")
    close_spy = mocker.spy(remote, "close")
    mocker.patch.object(client, "open_object", return_value=remote)
    f = _ParquetBufferedFile(client, file, 8, [(0, b"0123")])
    f.seek(4)
    assert f.read(4) == b"4567"
    f.close()
    assert close_spy.call_count == 1


def test_generator_prefetch_file_checks_archive_target_size(catalog, monkeypatch):
    import asyncio

    parent = File(source="s3://bucket", path="huge.tar", size=100 * 1024 * 1024)
    member = File(
        source="s3://bucket",
        path="small.parquet",
        size=100,
        location=[{"vtype": "tar", "parent": parent.model_dump()}],
    )
    member._set_stream(catalog)
    model = dict_to_data_model("Row", {"a": int})
    gen = ArrowGenerator(
        pa.schema([("a", pa.int64())]),
        model,
        prefetch_max_size=8 * 1024 * 1024,
        format="parquet",
    )
    assert asyncio.run(gen._prefetch_file(member)) is False
    assert "_parquet_parts" not in member.__dict__


def test_parquet_prefetch_deduplicates_versioned_files(
    tmp_path, catalog, http_server, monkeypatch, mocker
):
    import io
    from contextlib import contextmanager

    path = tmp_path / "multi.parquet"
    pq.write_table(
        pa.table({"a": list(range(1000)), "pad": [b"z" * 100] * 1000}),
        path,
        row_group_size=250,
        compression="NONE",
    )
    raw_bytes = path.read_bytes()
    monkeypatch.setattr(arrow_mod, "PARQUET_FOOTER_SAMPLE_SIZE", 4096)

    gen = ArrowGenerator(
        pa.schema([("a", pa.int64()), ("pad", pa.binary())]),
        dict_to_data_model("Row", {"a": int}),
        columns=["a"],
        prefetch_max_size=1,
        format="parquet",
    )

    f1 = File(
        source="s3://bucket", path="multi.parquet", size=len(raw_bytes), etag="e1"
    )
    f2 = File(
        source="s3://bucket", path="multi.parquet", size=len(raw_bytes), etag="e1"
    )
    f1._set_stream(catalog)
    f2._set_stream(catalog)
    client = catalog.get_client(f1.source)
    get_client_patch = mocker.patch.object(catalog, "get_client", return_value=client)

    calls = 0

    @contextmanager
    def fake_open_object(file, **kwargs):
        nonlocal calls
        calls += 1
        stream = io.BytesIO(raw_bytes)
        stream._fetch_range = lambda s, e: raw_bytes[s:e]  # type: ignore[attr-defined]
        yield stream

    mocker.patch.object(client, "open_object", side_effect=fake_open_object)
    gen._prefetch_parquet(f1, ["a"])
    gen._prefetch_parquet(f2, ["a"])
    assert calls == 1
    assert f1.__dict__.pop("_parquet_parts") is f2.__dict__.pop("_parquet_parts")
    mocker.stop(get_client_patch)

    # Overlapping reads of a mutable unversioned HTTP URL must not share cached bytes.
    mutable_path = tmp_path / "mutable.parquet"
    mutable_gen = ArrowGenerator(
        pa.schema([("a", pa.int64())]),
        dict_to_data_model("Row", {"a": int}),
        columns=["a"],
        prefetch_max_size=1,
        format="parquet",
    )
    pq.write_table(pa.table({"a": [1]}), mutable_path)
    u1 = File(
        source=http_server, path="mutable.parquet", size=mutable_path.stat().st_size
    )
    u1._set_stream(catalog, caching_enabled=False)
    mutable_gen._prefetch_parquet(u1, ["a"])

    pq.write_table(pa.table({"a": [2]}), mutable_path)
    u2 = File(
        source=http_server, path="mutable.parquet", size=mutable_path.stat().st_size
    )
    u2._set_stream(catalog, caching_enabled=False)
    mutable_gen._prefetch_parquet(u2, ["a"])

    assert [row.a for _, row in mutable_gen.process(u1)] == [1]
    assert [row.a for _, row in mutable_gen.process(u2)] == [2]


def test_parquet_prefetch_waits_for_sibling_ranges_on_error(
    tmp_path, catalog, monkeypatch, mocker
):
    import io
    import threading
    from contextlib import contextmanager

    path = tmp_path / "multi.parquet"
    pq.write_table(
        pa.table({"a": [1, 2], "pad": [b"x" * 100_000, b"y" * 100_000]}),
        path,
        row_group_size=1,
        compression="NONE",
        use_dictionary=False,
    )
    raw_bytes = path.read_bytes()
    monkeypatch.setattr(arrow_mod, "PARQUET_FOOTER_SAMPLE_SIZE", 4096)

    f1 = File(source="s3://bucket", path="multi.parquet", size=len(raw_bytes))
    f1._set_stream(catalog)
    client = catalog.get_client(f1.source)

    closed_while_running, running, calls = False, 0, 0
    lock = threading.Lock()
    entered_second = threading.Event()
    first_raising = threading.Event()
    release_second = threading.Event()

    @contextmanager
    def error_open_object(file, **kwargs):
        nonlocal closed_while_running
        stream = io.BytesIO(raw_bytes)

        def fetch_range(s, e):
            nonlocal calls, running
            with lock:
                calls, running = calls + 1, running + 1
                my_call = calls
            try:
                if my_call == 1:
                    assert entered_second.wait(timeout=5)
                    first_raising.set()
                    raise OSError("boom")
                entered_second.set()
                assert first_raising.wait(timeout=5)
                release_second.wait(timeout=0.05)
                return raw_bytes[s:e]
            finally:
                with lock:
                    running -= 1

        stream._fetch_range = fetch_range  # type: ignore[attr-defined]
        try:
            yield stream
        finally:
            with lock:
                closed_while_running = running > 0
            release_second.set()

    mocker.patch.object(client, "open_object", side_effect=error_open_object)
    with pytest.raises(OSError, match="boom"):
        arrow_mod._read_parquet_parts(client, f1, ["a"])
    assert calls == 2
    assert entered_second.is_set()
    assert first_raising.is_set()
    assert not closed_while_running


def test_parquet_prefetch_cancel_waits_for_cleanup(
    tmp_path, catalog, monkeypatch, mocker
):
    import asyncio
    import io
    import threading
    import time
    from contextlib import contextmanager

    path = tmp_path / "cancel.parquet"
    table = pa.table({"a": list(range(1000)), "pad": [b"z" * 100] * 1000})
    pq.write_table(table, path, row_group_size=250, compression="NONE")
    raw_bytes = path.read_bytes()
    monkeypatch.setattr(arrow_mod, "PARQUET_FOOTER_SAMPLE_SIZE", 4096)

    model = dict_to_data_model("Row", {"a": int})
    gen = ArrowGenerator(
        pa.schema([("a", pa.int64()), ("pad", pa.binary())]),
        model,
        columns=["a"],
        prefetch_max_size=1,
        format="parquet",
    )
    f1 = File(source="s3://bucket", path="cancel.parquet", size=len(raw_bytes))
    f1._set_stream(catalog)
    client = catalog.get_client(f1.source)
    mocker.patch.object(catalog, "get_client", return_value=client)

    started = threading.Event()
    finished = threading.Event()

    @contextmanager
    def slow_open_object(file, **kwargs):
        stream = io.BytesIO(raw_bytes)

        def fetch_range(s, e):
            started.set()
            time.sleep(0.05)
            return raw_bytes[s:e]

        stream._fetch_range = fetch_range  # type: ignore[attr-defined]
        try:
            yield stream
        finally:
            finished.set()

    mocker.patch.object(client, "open_object", side_effect=slow_open_object)

    async def cancel_prefetch():
        task = asyncio.create_task(gen._prefetch_file(f1))
        await asyncio.to_thread(started.wait, 2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert "_parquet_parts" not in f1.__dict__

    asyncio.run(cancel_prefetch())
