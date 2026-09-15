import base64
import datetime
import json
from enum import Enum
from unittest.mock import patch

import numpy as np
import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.lib.file import File
from datachain.lib.udf import JsonSerializationError
from datachain.sql.types import JSON, Array, Float, String
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
    query = sa.select(numbers_table.c.sys__id, numbers_table.c.number).order_by(
        numbers_table.c.number
    )
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
    query = (
        sa.select(numbers_table.c.sys__id, numbers_table.c.number)
        .order_by(numbers_table.c.number)
        .limit(limit)
    )
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
    assert list(nums) == list(range(1, limit + 1))


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


class _StrEnum(str, Enum):
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


@pytest.mark.parametrize(
    "value",
    [
        {"1": "a", 1: "b"},
        {None: "a", "null": "b"},
        {True: "a", "true": "b"},
        {(1, 2): "a", "[1,2]": "b"},
        {"1": "a", _OpaqueStr("1"): "b"},
        {_StrEnum.ONE: "a", 1: "b"},
    ],
    ids=[
        "int-and-str",
        "none-and-null",
        "bool-and-str",
        "tuple-and-str",
        "subclass",
        "enum-value-and-int",
    ],
)
def test_convert_type_refuses_colliding_dict_keys(warehouse, value):
    with pytest.raises(JsonSerializationError, match="would be lost"):
        warehouse.convert_type(value, JSON(), dict, "JSON", "c")


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
def test_convert_type_keeps_dict_keys_that_do_not_collide(warehouse, value, expected):
    got = warehouse.convert_type(value, JSON(), dict, "JSON", "c")
    assert json.loads(got) == expected


@pytest.mark.parametrize(
    "value,col_type,expected",
    [
        ([1, 2], Array(Float()), [1.0, 2.0]),
        (("x", "y"), Array(String()), ["x", "y"]),
    ],
    ids=["float-from-int", "tuple-to-list"],
)
def test_convert_type_converts_array_items(warehouse, value, col_type, expected):
    assert warehouse.convert_type(value, col_type, list, "Array", "c") == expected


@skip_if_not_sqlite
def test_convert_type_keeps_a_numpy_object_array_that_adds_keys(warehouse):
    value = [{"payload": np.array([{"a": 1}], dtype=object)}]

    assert warehouse.convert_type(value, Array(JSON()), list, "Array", "c") == value


@pytest.mark.parametrize(
    "value,col_type,col_python_type,col_type_name",
    [
        (
            [
                {
                    "1": "first",
                    1: "second",
                    "payload": np.array([{"a": 1}], dtype=object),
                }
            ],
            Array(JSON()),
            list,
            "Array",
        ),
        (
            {"payload": np.array([{"1": "first", 1: "second"}], dtype=object)},
            JSON(),
            dict,
            "JSON",
        ),
    ],
    ids=["array-item", "top-level"],
)
def test_convert_type_refuses_a_collision_a_numpy_array_hides(
    warehouse, value, col_type, col_python_type, col_type_name
):
    with pytest.raises(JsonSerializationError, match="would be lost"):
        warehouse.convert_type(value, col_type, col_python_type, col_type_name, "m__d")


@skip_if_not_sqlite
def test_convert_type_names_the_column_for_an_unserializable_array_item(warehouse):
    with pytest.raises(JsonSerializationError) as excinfo:
        warehouse.convert_type([{"k": {1, 2}}], Array(JSON()), list, "Array", "m__d")

    assert excinfo.value.column_name == "m__d"


@skip_if_not_sqlite
def test_convert_type_keeps_array_dict_keys_the_driver_stores_apart(warehouse):
    value = [{(1, 2): "tuple", "[1,2]": "string"}]

    assert warehouse.convert_type(value, Array(JSON()), list, "Array", "c") == value


@skip_if_not_sqlite
@pytest.mark.parametrize(
    "value",
    [
        [{(1, 2): "tuple", "(1, 2)": "string"}],
        [{datetime.date(2020, 1, 2): "date", "2020-01-02": "string"}],
    ],
    ids=["tuple-key", "date-key"],
)
def test_convert_type_refuses_array_dict_keys_the_driver_merges(warehouse, value):
    with pytest.raises(JsonSerializationError, match="would be lost"):
        warehouse.convert_type(value, Array(JSON()), list, "Array", "c")


def test_convert_type_does_not_re_encode_dict_array_items(warehouse):
    value = [{"a": 1}, {"b": 2}]

    got = warehouse.convert_type(value, Array(JSON()), list, "Array", "c")

    if warehouse.python_type(JSON()) is dict:
        assert got == value
    else:
        assert got == ['{"a":1}', '{"b":2}']


@pytest.mark.xfail(
    strict=True,
    reason="_numpy_to_python rebuilds the mapping while materializing an object "
    "array, so np.datetime64('NaT', 'ns') and None collapse onto one None key "
    "there, "
    "before any emitted JSON exists to read. Pre-existing; tracked on #1914.",
)
def test_convert_type_refuses_a_collision_numpy_normalization_hides(warehouse):
    value = {
        "payload": np.array(
            [{np.datetime64("NaT", "ns"): "a", None: "b"}], dtype=object
        )
    }

    with pytest.raises(JsonSerializationError, match="would be lost"):
        warehouse.convert_type(value, JSON(), dict, "JSON", "c")


def test_convert_type_names_the_column_for_a_colliding_dict(warehouse):
    with pytest.raises(JsonSerializationError) as excinfo:
        warehouse.convert_type({"1": "a", 1: "b"}, JSON(), dict, "JSON", "m__d")

    assert excinfo.value.column_name == "m__d"
    assert "both serialize to the JSON key '1'" in str(excinfo.value)
