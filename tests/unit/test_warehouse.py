import base64
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.data_storage.codec import (
    RowCodec,
    RowEncodeError,
    TypeCodec,
    check_value,
)
from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.data_storage.writer import MAX_SEQ
from datachain.lib.file import File
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf import JsonSerializationError, OutputRows, UdfError
from datachain.sql.sqlite.base import adapt_datetime
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Int,
    Int64,
    String,
)
from tests.utils import skip_if_not_sqlite


def test_serialize(sqlite_db):
    obj = SQLiteWarehouse(sqlite_db)
    assert obj.db == sqlite_db

    # Test clone
    obj2 = obj.clone()
    try:
        assert isinstance(obj2, SQLiteWarehouse)
        assert obj2.db.db_file == sqlite_db.db_file
        assert obj2.clone_params() == obj.clone_params()
    finally:
        obj2.close_on_exit()

    # Test serialization JSON format
    serialized = obj.serialize()
    assert serialized
    raw = base64.b64decode(serialized.encode())
    data = json.loads(raw.decode())
    assert data["callable"] == "sqlite.warehouse.init_after_clone"
    assert data["args"] == []
    nested = data["kwargs"]["db_clone_params"]
    assert nested["callable"] == "sqlite.from_db_file"
    assert nested["args"] == [":memory:"]
    assert nested["kwargs"] == {}

    obj3 = deserialize(serialized)
    try:
        assert isinstance(obj3, SQLiteWarehouse)
        assert obj3.db.db_file == sqlite_db.db_file
        assert obj3.clone_params() == obj.clone_params()
    finally:
        obj3.close_on_exit()


def test_is_temp_table_name(warehouse):
    assert warehouse.is_temp_table_name("tmp_vc12F") is True
    assert warehouse.is_temp_table_name("udf_jh653") is False
    assert warehouse.is_temp_table_name("ds_my_dataset") is False
    assert warehouse.is_temp_table_name("src_my_bucket") is False
    assert warehouse.is_temp_table_name("ds_ds_my_query_script_1_1") is False


def test_query_count(numbers_table, warehouse):
    query = sa.select(numbers_table.c.number).where(
        numbers_table.c.primality == "prime"
    )
    assert warehouse.query_count(query) == 21


def test_table_rows_count(numbers_table, warehouse):
    assert warehouse.table_rows_count(numbers_table) == 73


def test_dataset_select_paginated(numbers_table, warehouse):
    query = sa.select(numbers_table.c.sys__id, numbers_table.c.number)
    with patch.object(
        type(warehouse),
        attribute="dataset_rows_select",
        wraps=warehouse.dataset_rows_select,
    ) as mock_dataset_rows_select:
        rows = list(warehouse.dataset_select_paginated(query, page_size=13))
        assert mock_dataset_rows_select.call_count == 6  # 6 pages: 13 * 5 + 8
    assert len(rows) == 73
    ids, nums = zip(*rows, strict=False)
    assert len(set(ids)) == 73
    assert set(nums) == set(range(1, 74))


@pytest.mark.parametrize("limit", [7, 8, 9])
def test_dataset_select_paginated_with_limit(limit, numbers_table, warehouse):
    query = sa.select(numbers_table.c.sys__id, numbers_table.c.number).limit(limit)
    with patch.object(
        type(warehouse),
        attribute="dataset_rows_select",
        wraps=warehouse.dataset_rows_select,
    ) as mock_dataset_rows_select:
        rows = list(warehouse.dataset_select_paginated(query, page_size=3))
        assert mock_dataset_rows_select.call_count == 3  # 3 pages: 3 + 3 + [1 | 2 | 3]
    assert len(rows) == limit
    ids, nums = zip(*rows, strict=False)
    assert len(set(ids)) == limit
    assert set(nums) <= set(range(1, 74))


def test_dataset_rows_select(numbers_table, warehouse):
    query = (
        sa.select(numbers_table.c.sys__id, numbers_table.c.number)
        .order_by(numbers_table.c.number)
        .limit(7)
    )
    rows = list(warehouse.dataset_rows_select(query))
    assert len(rows) == 7
    ids, nums = zip(*rows, strict=False)
    assert len(set(ids)) == 7
    assert list(nums) == list(range(1, 8))


def test_dataset_rows_select_from_ids(numbers_table, warehouse):
    query_ids = (
        sa.select(numbers_table.c.sys__id).order_by(numbers_table.c.number).limit(5)
    )
    test_ids = [r[0] for r in warehouse.db.execute(query_ids)]
    assert len(test_ids) == len(set(test_ids)) == 5

    query = sa.select(numbers_table.c.sys__id, numbers_table.c.number)
    rows = list(
        warehouse.dataset_rows_select_from_ids(
            query,
            ids=test_ids,
            is_batched=False,
        )
    )
    assert len(rows) == 5
    ids, nums = zip(*rows, strict=False)
    assert set(ids) == set(test_ids)
    assert set(nums) == {1, 2, 3, 4, 5}


def test_dataset_rows_select_from_ids_batched(numbers_table, warehouse):
    query_ids = (
        sa.select(numbers_table.c.sys__id).order_by(numbers_table.c.number).limit(6)
    )
    test_ids = [r[0] for r in warehouse.db.execute(query_ids)]
    assert len(test_ids) == len(set(test_ids)) == 6

    # Split into two batches: odd and even
    batched_ids = [test_ids[::2], test_ids[1::2]]

    query = sa.select(numbers_table.c.sys__id, numbers_table.c.number)
    batches = list(
        warehouse.dataset_rows_select_from_ids(
            query,
            ids=batched_ids,
            is_batched=True,
        )
    )
    assert len(batches) == 2

    ids, nums = zip(*batches[0], strict=False)
    assert set(ids) == set(batched_ids[0])
    assert set(nums) == {1, 3, 5}

    ids, nums = zip(*batches[1], strict=False)
    assert set(ids) == set(batched_ids[1])
    assert set(nums) == {2, 4, 6}


@pytest.mark.parametrize("is_batched", [True, False])
def test_dataset_rows_select_from_ids_requires_sys_id(
    is_batched, numbers_table, warehouse
):
    # Build a query without sys__id
    query = sa.select(numbers_table.c.number)

    with pytest.raises(RuntimeError, match="sys__id column not found in query"):
        list(
            warehouse.dataset_rows_select_from_ids(
                query,
                ids=[1, 2, 3],
                is_batched=is_batched,
            )
        )


WILDCARD_PATHS = [
    "dir_%1/a.csv",
    "dir_%1/b.csv",
    "dir_%1/nested/c.csv",
    # decoy: unescaped, `dir_%1/` is the pattern `dir?<any>1/`, which matches this
    "dirXX1/d.csv",
]


@pytest.fixture
def wildcard_rows(test_session, warehouse):
    saved = dc.read_values(
        file=[File(path=p) for p in WILDCARD_PATHS], session=test_session
    ).save("wildcard_paths")
    dataset = saved.dataset
    assert dataset is not None
    yield warehouse.dataset_rows(dataset, dataset.latest_version, column="file")
    dc.delete_dataset(dataset.name, force=True, session=test_session)


def test_prefix_match_treats_wildcards_literally(warehouse, wildcard_rows):
    query = wildcard_rows.select(wildcard_rows.c("path")).where(
        warehouse.prefix_match(wildcard_rows.c("path"), "dir_%1/")
    )

    assert sorted(r[0] for r in warehouse.db.execute(query)) == [
        "dir_%1/a.csv",
        "dir_%1/b.csv",
        "dir_%1/nested/c.csv",
    ]


def test_select_node_fields_by_parent_path_tar_wildcards_are_literal(
    warehouse, wildcard_rows
):
    rows = warehouse.select_node_fields_by_parent_path_tar(
        wildcard_rows, "dir_%1", ["path"]
    )

    assert sorted(r[0] for r in rows) == [
        "dir_%1/a.csv",
        "dir_%1/b.csv",
        "dir_%1/nested/c.csv",
    ]


def test_empty_rows_insert_defaults(sqlite_db):
    warehouse = SQLiteWarehouse(sqlite_db)
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    warehouse.insert_stored_rows(table, [], [(), ()])

    query = sa.select(table.c.value)
    assert list(warehouse.dataset_rows_select(query)) == [(None,), (None,)]


def test_output_rows_validate_tuples(sqlite_db):
    warehouse = SQLiteWarehouse(sqlite_db)
    udf = SimpleNamespace(output=SignalSchema({"a": int, "b": str}), udf_kind="map")
    output_rows = OutputRows(udf, warehouse)  # type: ignore[arg-type]

    assert output_rows((1, "x")) == (1, "x")
    with pytest.raises(UdfError, match="output column 'a'"):
        output_rows(("x", "y"))


@skip_if_not_sqlite
def test_insert_into_reads_all_ids_before_inserting_batches(
    warehouse_tmpfile, monkeypatch
):
    warehouse = warehouse_tmpfile
    monkeypatch.setattr(type(warehouse), "INSERT_BATCH_SIZE", 3)
    src, dst, unrelated = (
        warehouse.create_udf_table([sa.Column("value", sa.Integer)]) for _ in range(3)
    )
    warehouse.insert_stored_rows(src, ["value"], [(i,) for i in range(10)])

    # Another connection commits after each batch. With the id cursor still open,
    # the next batch fails with "database is locked".
    with warehouse.clone() as other:
        warehouse.insert_into(
            dst,
            sa.select(src),
            progress_cb=lambda _: other.db.execute(unrelated.insert().values(value=0)),
        )

    assert warehouse.table_rows_count(dst) == 10


@pytest.mark.parametrize("fetch_size,selects", [(100, 1), (3, 3)])
def test_batched_id_fetch_groups_queries_and_preserves_rows(
    numbers_table, warehouse, monkeypatch, fetch_size, selects
):
    monkeypatch.setattr(
        "datachain.data_storage.warehouse.DEFAULT_BATCH_SIZE", fetch_size
    )
    query = sa.select(numbers_table.c.number, numbers_table.c.sys__id)
    rows = list(
        warehouse.dataset_rows_select(query.order_by(numbers_table.c.number).limit(4))
    )
    a, b, c, d = [row[1] for row in rows]
    ids = [[c, a, c], [], [b], [d, a], [0]]
    with patch.object(
        type(warehouse), "dataset_rows_select", wraps=warehouse.dataset_rows_select
    ) as select:
        batches = list(
            warehouse.dataset_rows_select_from_ids(
                query.limit(1).offset(20), iter(ids), True
            )
        )
    assert select.call_count == selects
    assert [[row[0] for row in batch] for batch in batches] == [
        [3, 1],
        [],
        [2],
        [4, 1],
        [],
    ]


def test_batched_id_fetch_preserves_join_fanout(numbers_table, warehouse):
    left = (
        sa.select(numbers_table.c.sys__id, numbers_table.c.primality)
        .where(numbers_table.c.number == 2)
        .subquery("left_rows")
    )
    right = (
        sa.select(numbers_table.c.primality, numbers_table.c.number)
        .where(numbers_table.c.number.in_([2, 3]))
        .subquery("right_rows")
    )
    query = sa.select(left.c.sys__id, right.c.number).select_from(
        left.join(right, left.c.primality == right.c.primality)
    )
    row_id = next(warehouse.dataset_rows_select(query))[0]
    batches = list(
        warehouse.dataset_rows_select_from_ids(query, [[row_id], [], [row_id]], True)
    )
    assert [sorted(row[1] for row in batch) for batch in batches] == [
        [2, 3],
        [],
        [2, 3],
    ]


DATETIMES = [
    datetime(2024, 1, 2, 3, 4, 5),
    datetime(2024, 1, 2, 3, 4, 5, 120),
    datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=-5))),
    datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
    datetime(9, 1, 1),
]


@pytest.mark.parametrize("value", DATETIMES)
def test_codec_stores_datetimes_as_the_adapter_text(sqlite_db, value):
    # Filter parameters still go through adapt_datetime: stored text must match it.
    codec = SQLiteWarehouse(sqlite_db).row_codec([DateTime()])

    assert codec.encode((value,)) == (adapt_datetime(value),)
    encoded = codec.encode_arrow(
        pa.record_batch([pa.array([value, None])], names=["t"])
    )
    assert encoded.column(0).to_pylist() == [adapt_datetime(value), None]


class _StrEnum(StrEnum):
    A = "a"
    B = "b"
    ONE = "1"


class _Mangled(str):
    """A str subclass whose __str__ disagrees with the key json emits."""

    __slots__ = ()

    def __str__(self):
        return "MANGLED"


class _OpaqueStr(str):
    """A str subclass whose equality cannot spot a duplicate key."""

    __slots__ = ()
    __eq__ = object.__eq__
    __hash__ = object.__hash__


# Ported from #1943 (#1914): keys that collide as JSON raise instead of losing a
# value; the codec's JSON encoder is the one place a dict becomes JSON.
@pytest.mark.parametrize(
    "value",
    [
        {"1": "a", 1: "b"},
        {None: "a", "null": "b"},
        {True: "a", "true": "b"},
        {(1, 2): "a", "[1,2]": "b"},
        {"1": "a", _OpaqueStr("1"): "b"},
        {_StrEnum.ONE: "a", 1: "b"},
        {"payload": np.array([{"1": "first", 1: "second"}], dtype=object)},
    ],
    ids=[
        "int-and-str",
        "none-and-null",
        "bool-and-str",
        "tuple-and-str",
        "subclass",
        "enum-value-and-int",
        "inside-a-numpy-object-array",
    ],
)
def test_json_encoding_refuses_colliding_dict_keys(value):
    with pytest.raises(JsonSerializationError, match="would be lost"):
        check_value(JSON())(value)


def test_json_array_items_refuse_colliding_dict_keys():
    with pytest.raises(JsonSerializationError, match="would be lost"):
        check_value(Array(JSON()))([{"1": "first", 1: "second"}])


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"x": "a", 2: "b"}, {"x": "a", "2": "b"}),
        ({1: "a", 2: "b"}, {"1": "a", "2": "b"}),
        ({_StrEnum.A: "x", _StrEnum.B: "y"}, {"a": "x", "b": "y"}),
        ({_StrEnum.A: "x", "_StrEnum.A": "y"}, {"a": "x", "_StrEnum.A": "y"}),
        ({_Mangled("a"): "x"}, {"a": "x"}),
    ],
    ids=["mixed", "all-int", "enum", "enum-and-its-label", "custom-str"],
)
def test_json_encoding_keeps_dict_keys_that_do_not_collide(value, expected):
    assert json.loads(check_value(JSON())(value)) == expected


@pytest.mark.parametrize(
    "value,col_type,expected",
    [
        ([1, 2], Array(Float()), [1.0, 2.0]),
        (("x", "y"), Array(String()), ["x", "y"]),
    ],
    ids=["float-from-int", "tuple-to-list"],
)
def test_check_value_converts_array_items(value, col_type, expected):
    assert check_value(col_type)(value) == expected


def test_check_value_does_not_re_encode_dict_array_items():
    value = [{"a": 1}, {"b": 2}]
    assert check_value(Array(JSON()))(value) == value


def test_encode_arrow_keeps_arrow_batch_when_all_columns_support_it():
    codecs = [
        TypeCodec(
            kept=frozenset({int}),
            encode=int,
            default=False,
            encode_column=lambda c: c.cast(pa.int64()),
        ),
        TypeCodec(
            kept=frozenset({str}),
            encode=str,
            default="",
            encode_column=lambda c: c.cast(pa.string()),
        ),
    ]
    batch = pa.record_batch(
        [pa.array([1, None], type=pa.int64()), pa.array(["a", None])],
        names=["b", "s"],
    )
    encoded = RowCodec(codecs).encode_arrow(batch)
    assert isinstance(encoded, pa.RecordBatch)
    assert [c.to_pylist() for c in encoded.columns] == [[1, 0], ["a", ""]]

    # Missing trailing column stays in Arrow when its codec supports pa.nulls().
    short = pa.record_batch([pa.array([1, None], type=pa.int64())], names=["b"])
    encoded_short = RowCodec(codecs).encode_arrow(short)
    assert isinstance(encoded_short, pa.RecordBatch)
    assert [c.to_pylist() for c in encoded_short.columns] == [[1, 0], ["", ""]]


@skip_if_not_sqlite
@pytest.mark.parametrize("nullable", [False, True])
def test_native_arrow_matches_scalar_codec_storage(warehouse_tmpfile, nullable):
    wh = warehouse_tmpfile
    specs = [
        (String, pa.array(["", "a\x00b", "é", None])),
        (String, pa.array(["one", None, "two", "one"]).dictionary_encode()),
        (String, pa.nulls(4)),
        (DateTime, pa.nulls(4)),
        (Binary, pa.array([b"", b"a\x00b", b"x", None])),
        (Boolean, pa.array([True, False, None, True])),
        (Int64, pa.array([-(2**63), 2**63 - 1, None, 0])),
        (Float, pa.array([float("nan"), float("inf"), None, -0.0])),
        (Float, pa.array([1.25, None, 2.5, 0], type=pa.float32())),
        (
            DateTime,
            pa.array([-1, 0, 1234567, None], type=pa.timestamp("us", "Asia/Kolkata")),
        ),
        (JSON, pa.array(['{"a": 1}', "null", None, "{}"])),
    ]
    columns = [
        sa.Column(f"c{i}", t.as_nullable(t) if nullable else t())
        for i, (t, _) in enumerate(specs)
    ]
    batch = pa.record_batch([a for _, a in specs], names=[c.name for c in columns])
    batch = pa.Table.from_batches([batch] * 512).combine_chunks().to_batches()[0]
    codec = wh.row_codec(columns)
    native = codec.encode_arrow(batch)
    assert isinstance(native, pa.RecordBatch)
    expected = [
        codec.encode(row)
        for row in zip(*(a.to_pylist() for _, a in specs), strict=True)
    ] * 512
    reference = wh.create_dataset_rows_table("scalar_reference", columns=columns)
    actual = wh.create_dataset_rows_table(
        "arrow_actual", columns=[sa.Column(c.name, c.type) for c in columns]
    )
    try:
        with wh.writer(reference, batch.schema.names) as w:
            w.extend(expected)
        with wh.writer(actual, batch.schema.names) as w:
            w.append_arrow(native)
        # Raw sqlite values and storage classes, bypassing registered converters.
        fields = ", ".join(f'"{c.name}", typeof("{c.name}")' for c in columns)
        with closing(sqlite3.connect(wh.db.db_file)) as db:
            actual_sql = f"SELECT {fields} FROM arrow_actual"  # noqa: S608
            expected_sql = f"SELECT {fields} FROM scalar_reference"  # noqa: S608
            assert (
                db.execute(actual_sql).fetchall() == db.execute(expected_sql).fetchall()
            )
    finally:
        wh.db.drop_table(reference)
        wh.db.drop_table(actual)


@skip_if_not_sqlite
def test_encode_arrow_fills_missing_trailing_columns(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    batch = pa.record_batch([[1, 2]], names=["x"])

    rc_arrow = warehouse.row_codec([sa.Column("x", Int()), sa.Column("ts", DateTime())])
    enc_arrow = rc_arrow.encode_arrow(batch)
    assert isinstance(enc_arrow, pa.RecordBatch)
    assert enc_arrow.column(1).to_pylist() == [None, None]

    rc_tuples = warehouse.row_codec(
        [sa.Column("x", Int()), sa.Column("arr", Array(Int()))]
    )
    assert rc_tuples.encode_arrow(batch) == [(1, None), (2, None)]


@skip_if_not_sqlite
def test_encode_arrow_falls_back_for_unsupported_or_out_of_range_values(
    warehouse_tmpfile,
):
    warehouse = warehouse_tmpfile
    dt_codec = warehouse.row_codec([sa.Column("ts", DateTime())])
    int_codec = warehouse.row_codec([sa.Column("x", Int())])

    # Naive timestamp before year 1 falls back to tuple path; aware timestamp clamps.
    far_past = pa.array([-70_000_000_000_000_000], type=pa.int64())
    with pytest.raises(OverflowError):
        dt_codec.encode_arrow(
            pa.record_batch([far_past.view(pa.timestamp("us"))], names=["ts"])
        )
    clamped = dt_codec.encode_arrow(
        pa.record_batch([far_past.view(pa.timestamp("us", "UTC"))], names=["ts"])
    )
    assert isinstance(clamped, pa.RecordBatch)
    assert clamped.column(0).to_pylist() == ["0001-01-01 00:00:00"]

    # Non-timestamp input for DateTime and non-scalar / overflowing uint64 for Int
    # fall back to the tuple path.
    with pytest.raises(RowEncodeError):
        dt_codec.encode_arrow(pa.record_batch([pa.array([1, 2])], names=["ts"]))
    with pytest.raises(RowEncodeError):
        int_codec.encode_arrow(
            pa.record_batch([pa.array([[1]], type=pa.list_(pa.int64()))], names=["x"])
        )
    assert int_codec.encode_arrow(
        pa.record_batch([pa.array([2**63], type=pa.uint64())], names=["x"])
    ) == [(2**63,)]


@skip_if_not_sqlite
def test_encode_arrow_rejects_invalid_utf8_strings(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    str_codec = warehouse.row_codec([sa.Column("s", String())])
    invalid = pa.array([b"\xff"], type=pa.binary()).view(pa.string())
    with pytest.raises(UnicodeDecodeError):
        str_codec.encode_arrow(pa.record_batch([invalid], names=["s"]))


@skip_if_not_sqlite
def test_append_arrow_raises_when_writer_runs_out_of_ids(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("overflow_id", [sa.Column("x", Int)])
    try:
        with warehouse.writer(table, ["x"]) as writer:
            writer.next_id = MAX_SEQ + 1
            with pytest.raises(RuntimeError, match="ran out of ids"):
                writer.append_arrow(pa.record_batch([[1]], names=["x"]))
    finally:
        warehouse.db.drop_table(table)
