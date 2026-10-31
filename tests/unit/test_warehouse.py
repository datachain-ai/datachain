import base64
import json
from unittest.mock import patch

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.lib.file import File
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


def test_empty_rows_insert_defaults(sqlite_db):
    warehouse = SQLiteWarehouse(sqlite_db)
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    warehouse.insert_rows(table, [{}, {}])
    warehouse.insert_rows_done(table)

    query = sa.select(table.c.value)
    assert list(warehouse.dataset_rows_select(query)) == [(None,), (None,)]


def test_map_tuples_are_validated(sqlite_db):
    from types import SimpleNamespace

    from datachain.lib.signal_schema import SignalSchema
    from datachain.lib.udf import UdfError
    from datachain.query.dataset import process_udf_outputs
    from datachain.sql.types import Int, String

    warehouse = SQLiteWarehouse(sqlite_db)
    table = warehouse.create_udf_table([sa.Column("a", Int), sa.Column("b", String)])
    inner = SimpleNamespace(
        output=SignalSchema({"a": int, "b": str}),
        is_input_batched=False,
        is_output_batched=False,
    )
    udf = SimpleNamespace(output={"a": Int, "b": String}, inner=inner)

    with pytest.raises(UdfError):
        process_udf_outputs(warehouse, table, iter([[(1, "x", None)]]), udf)


@skip_if_not_sqlite
def test_insert_into_reads_all_ids_before_inserting_batches(
    warehouse_tmpfile, monkeypatch
):
    warehouse = warehouse_tmpfile
    monkeypatch.setattr(type(warehouse), "INSERT_BATCH_SIZE", 3)
    src, dst, unrelated = (
        warehouse.create_udf_table([sa.Column("value", sa.Integer)]) for _ in range(3)
    )
    warehouse.insert_rows(src, [{"value": i} for i in range(10)])
    warehouse.insert_rows_done(src)

    # Another connection commits after each batch. With the id cursor still open,
    # the next batch fails with "database is locked".
    with warehouse.clone() as other:
        warehouse.insert_into(
            dst,
            sa.select(src),
            progress_cb=lambda _: other.db.execute(unrelated.insert().values(value=0)),
        )

    assert warehouse.table_rows_count(dst) == 10


@skip_if_not_sqlite
@pytest.mark.parametrize(
    "kind,direct",
    [(kind, direct) for kind in ("inner", "left", "full") for direct in (False, True)]
    + [("subtract", False)],
)
@pytest.mark.parametrize("computed_side", ["left", "right", "both"])
@pytest.mark.parametrize("typed", [False, True])
def test_join_computed_keys_evaluated_linearly(
    warehouse, numbers_table, kind, computed_side, direct, typed
):
    sqlite_db = warehouse.db
    calls = 0

    def join_key(value):
        nonlocal calls
        calls += 1
        return value

    sqlite_db.db.create_function("test_join_key", 1, join_key, deterministic=True)

    def make_side(table, computed):
        key = table.c.number.label("key")
        if computed and not direct:
            key = sa.func.test_join_key(key, type_=sa.Integer if typed else None).label(
                "key"
            )
        side = sa.select(key).subquery()
        predicate = side.c.key
        if computed and direct:
            predicate = sa.func.test_join_key(
                predicate, type_=sa.Integer if typed else None
            )
        return side, predicate

    left, left_key = make_side(numbers_table, computed_side in ("left", "both"))
    right, right_key = make_side(numbers_table, computed_side in ("right", "both"))
    if kind == "subtract":
        query = warehouse.subtract_query(
            left.select(), right.select(), [("key", "key")]
        )
        expected = []
    else:
        query = warehouse.join(
            left,
            right,
            left_key == right_key,
            inner=kind == "inner",
            full=kind == "full",
            columns=[left.c.key, right.c.key.label("right_key")],
        )
        expected = [(i, i) for i in range(1, 74)]

    assert sorted(sqlite_db.execute(query).fetchall()) == expected
    assert calls == 73 * (2 if computed_side == "both" else 1)


@skip_if_not_sqlite
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "left_value,right_value,expected",
    [(1, "1", []), (1, 1.5, []), (1.5, 1.5, [(1.5, 1.5)])],
)
def test_join_computed_keys_preserves_comparison_semantics(
    sqlite_db, direct, left_value, right_value, expected
):
    warehouse = SQLiteWarehouse(sqlite_db)
    sqlite_db.db.create_function(
        "test_left_key", 1, lambda value: left_value, deterministic=True
    )
    sqlite_db.db.create_function(
        "test_right_key", 1, lambda value: right_value, deterministic=True
    )
    tables = [
        sa.Table(name, sqlite_db.metadata, sa.Column("key", sa.Integer))
        for name in ("join_comparison_left", "join_comparison_right")
    ]
    for table in tables:
        sqlite_db.create_table(table)
        warehouse.insert_rows(table, [{"key": 1}])
        warehouse.insert_rows_done(table)
    left_expression = sa.func.test_left_key(tables[0].c.key, type_=sa.Integer)
    right_expression = sa.func.test_right_key(
        tables[1].c.key, type_=sa.String if isinstance(right_value, str) else sa.Float
    )
    if direct:
        left, right = tables
        columns = [
            left_expression.label("left_key"),
            right_expression.label("right_key"),
        ]
    else:
        left = sa.select(left_expression.label("key")).subquery()
        right = sa.select(right_expression.label("key")).subquery()
        left_expression, right_expression = left.c.key, right.c.key
        columns = [left.c.key, right.c.key.label("right_key")]
    query = warehouse.join(
        left, right, left_expression == right_expression, columns=columns
    )
    assert sqlite_db.execute(query).fetchall() == expected


@skip_if_not_sqlite
@pytest.mark.parametrize("rename", [False, True])
def test_join_plain_keys_do_not_materialize_computed_payload(sqlite_db, rename):
    table = sa.Table(
        "plain_join_keys", sqlite_db.metadata, sa.Column("key", sa.Integer)
    )
    key = table.c.key.label("renamed") if rename else table.c.key
    left = sa.select(key, (table.c.key + 1).label("payload")).subquery()
    right = sa.select(key).subquery()
    key_name = key.name
    query = SQLiteWarehouse(sqlite_db).join(
        left, right, left.c[key_name] == right.c[key_name], columns=[left.c.payload]
    )
    sql, _ = sqlite_db.compile_to_args(query)
    assert "OFFSET" not in sql


@skip_if_not_sqlite
def test_join_expression_on_derived_key_evaluates_each_function_once(
    warehouse, numbers_table
):
    calls = 0

    def join_key(value):
        nonlocal calls
        calls += 1
        return value

    warehouse.db.db.create_function("test_nested_join_key", 1, join_key)
    left = sa.select(
        sa.func.test_nested_join_key(numbers_table.c.number, type_=sa.Integer).label(
            "key"
        )
    ).subquery()
    right = sa.select(numbers_table.c.number).subquery()
    query = warehouse.join(
        left,
        right,
        sa.func.test_nested_join_key(left.c.key, type_=sa.Integer) == right.c.number,
        columns=[left.c.key, right.c.number],
    )
    rows = warehouse.db.execute(query).fetchall()
    assert sorted(rows) == [(i, i) for i in range(1, 74)]
    assert calls == 2 * 73


@skip_if_not_sqlite
@pytest.mark.parametrize("kind", ["inner", "left", "full", "subtract"])
def test_computed_join_does_not_coerce_runtime_text_to_numeric(sqlite_db, kind):
    warehouse = SQLiteWarehouse(sqlite_db)
    left_table = sa.Table(
        "nullable_keys", sqlite_db.metadata, sa.Column("key", sa.Integer)
    )
    right_table = sa.Table("text_keys", sqlite_db.metadata, sa.Column("key", sa.String))
    for table, value in [(left_table, None), (right_table, "missing")]:
        sqlite_db.create_table(table)
        warehouse.insert_rows(table, [{"key": value}])
        warehouse.insert_rows_done(table)
    left = sa.select(
        sa.func.coalesce(left_table.c.key, "missing").label("key")
    ).subquery()
    right = right_table.alias()
    if kind == "subtract":
        query = warehouse.subtract_query(
            left.select(), right.select(), [("key", "key")]
        )
        expected = []
    else:
        query = warehouse.join(
            left,
            right,
            left.c.key == right.c.key,
            inner=kind == "inner",
            full=kind == "full",
            columns=[left.c.key, right.c.key.label("right_key")],
        )
        expected = [("missing", "missing")]
    assert sqlite_db.execute(query).fetchall() == expected
