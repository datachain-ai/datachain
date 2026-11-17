import base64
import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.data_storage.warehouse import AbstractWarehouse
from datachain.lib.file import File
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf import UdfError
from datachain.query.dataset import process_udf_outputs
from datachain.sql.types import Int, String
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


def _nested_scans(db, query):
    sql, params = db.compile_to_args(query)
    plan = db.db.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    loops = Counter(parent for _, parent, _, step in plan if step.startswith("SCAN"))
    return [step for _, parent, _, step in plan if loops[parent] > 1]


@skip_if_not_sqlite
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "kind,computed,lazy",
    [
        ("inner", "left", False),
        ("inner", "right", False),
        ("inner", "both", False),
        ("left", "left", True),
        ("left", "right", False),
        ("left", "both", False),
        ("full", "left", False),
        ("full", "right", False),
        ("full", "both", False),
        ("subtract", "left", True),
        ("subtract", "right", False),
        ("subtract", "both", False),
    ],
)
def test_computed_join_keys_scale_linearly(
    warehouse, numbers_table, kind, computed, lazy, direct
):
    if kind == "subtract" and direct:
        pytest.skip("subtract joins on column names")
    calls = 0

    def join_key(value):
        nonlocal calls
        calls += 1
        return value

    def unused_payload(value):
        raise AssertionError("unused payload must not be evaluated")

    warehouse.db.db.create_function("test_join_key", 1, join_key, deterministic=True)
    warehouse.db.db.create_function("test_unused_payload", 1, unused_payload)

    def side(is_computed):
        number = numbers_table.c.number
        key = sa.func.test_join_key(number) if is_computed and not direct else number
        payload = sa.func.test_unused_payload(number).label("payload")
        side = sa.select(number.label("id"), key.label("key"), payload).subquery()
        if is_computed and direct:
            return side, sa.func.test_join_key(side.c.key)
        return side, side.c.key

    (left, left_key), (right, right_key) = (
        side(computed != "right"),
        side(computed != "left"),
    )
    if kind == "subtract":
        query = warehouse.subtract_query(
            sa.select(left.c.id, left.c.key), sa.select(right.c.key), [("key", "key")]
        )
        expected = []
    else:
        query = warehouse.join(
            left,
            right,
            left_key == right_key,
            inner=kind == "inner",
            full=kind == "full",
            columns=[left.c.id, right.c.id.label("right_id")],
        )
        expected = [(i, i) for i in range(1, 74)]

    assert sorted(warehouse.db.execute(query)) == expected
    assert calls <= 2 * 73
    assert not _nested_scans(warehouse.db, query)
    if kind == "inner" and computed in ("left", "right"):
        # Filtering the non-computed side uses the index on the computed CTE.
        calls = 0
        result = query.subquery()
        plain_id = result.c.right_id if computed == "left" else result.c.id
        filtered = sa.select(result).where(plain_id.in_([1, 2, 3, 4, 5]))
        assert len(list(warehouse.db.execute(filtered))) == 5
        assert calls == 73
        assert not _nested_scans(warehouse.db, filtered)
    if lazy:
        # A filter on the computed side reaches it before its keys are computed.
        calls = 0
        result = query.subquery()
        computed_id = result.c.id if computed == "left" else result.c.right_id
        list(warehouse.db.execute(sa.select(result).where(computed_id <= 5)))
        assert calls <= 10


@pytest.fixture
def computed_join_inputs(test_session):
    calls = Counter()
    warehouse = test_session.catalog.warehouse

    def key(side):
        def value(number):
            calls[side] += 1
            return number

        return value

    for side in ("left", "right"):
        warehouse.db.db.create_function(
            f"test_{side}_key", 1, key(side), deterministic=True
        )
    left = dc.read_values(x=[0, 1, 2], session=test_session).mutate(
        key=sa.func.test_left_key(dc.C.x, type_=sa.Integer())
    )
    right = dc.read_values(y=list(range(12)), session=test_session).mutate(
        key=sa.func.test_right_key(dc.C.y, type_=sa.Integer())
    )
    return left, right, calls, warehouse


@skip_if_not_sqlite
@pytest.mark.parametrize(
    "preview", ["limit", "filter_limit", "union_limit", "limit_before_order", "show"]
)
def test_computed_join_preview_does_not_scan_all_right_keys(
    computed_join_inputs, preview, capsys
):
    left, right, calls, _ = computed_join_inputs
    joined = left.merge(right, on="key", inner=False).select("x", "y")
    if preview == "show":
        joined.show(limit=1)
        assert "0" in capsys.readouterr().out
    else:
        if preview == "filter_limit":
            joined = joined.filter(dc.C.x >= 0)
        elif preview == "union_limit":
            joined = joined.union(joined)
        elif preview == "limit_before_order":
            joined = joined.limit(1).order_by("x", descending=True)
        assert joined.limit(1).to_list() == [(0, 0)]
    assert calls["left"] <= 2
    assert calls["right"] == 12


@skip_if_not_sqlite
@pytest.mark.parametrize(
    "barrier", ["order", "group", "distinct", "map", "join_input", "full_join"]
)
def test_computed_join_limit_keeps_fence_before_blocking_steps(
    computed_join_inputs, barrier
):
    left, right, calls, _ = computed_join_inputs
    joined = left.merge(right, on="key", inner=False).select("x", "y")
    if barrier == "order":
        result = joined.order_by("x", descending=True)
        expected = [(2, 2)]
    elif barrier == "group":
        result = joined.group_by(partition_by="x", total=dc.func.sum("y"))
        expected = [(0, 0)]
    elif barrier == "distinct":
        result = joined.distinct("x")
        expected = [(0, 0)]
    elif barrier == "map":
        result = joined.map(z=lambda x: x + 1, output=int)
        expected = [(0, 0, 1)]
    elif barrier == "join_input":
        result = joined.merge(left.select("x"), on="x").select("x", "y")
        expected = [(0, 0)]
    else:
        result = left.merge(right, on="key", full=True).select("x", "y")
        # Full joins include unmatched rows; without ordering any first row is valid.
        rows = result.limit(1).to_list()
        assert len(rows) == 1
        assert calls["left"] <= 6
        assert calls["right"] <= 24
        return
    assert result.limit(1).to_list() == expected
    assert calls["left"] == 3
    assert calls["right"] == 12


@skip_if_not_sqlite
def test_computed_join_full_read_and_pagination_keep_linear_key_work(
    computed_join_inputs, monkeypatch
):
    left, right, calls, warehouse = computed_join_inputs
    joined = left.merge(right, on="key", inner=False).select("x", "y")
    expected = [(0, 0), (1, 1), (2, 2)]
    assert joined.to_list() == expected
    assert calls["right"] == 12
    calls.clear()
    clone = warehouse.clone

    def clone_with_test_keys(*args, **kwargs):
        wh = clone(*args, **kwargs)
        for side in ("left", "right"):

            def key(number, side=side):
                calls[side] += 1
                return number

            wh.db.db.create_function(f"test_{side}_key", 1, key, deterministic=True)
        return wh

    monkeypatch.setattr(warehouse, "clone", clone_with_test_keys)
    query = joined._query.apply_steps().select()
    try:
        assert list(warehouse.dataset_select_paginated(query, page_size=2)) == expected
        # Both pages retain the build-side fence, despite their internal LIMIT.
        assert calls["right"] == 24
    finally:
        joined._query.cleanup()


@skip_if_not_sqlite
def test_computed_subtract_with_limit_keeps_results(computed_join_inputs):
    left, right, calls, _ = computed_join_inputs
    assert right.subtract(left, on="key").select("y").limit(1).to_list() == [(3,)]
    assert calls["left"] == 3
    assert calls["right"] == 4


@skip_if_not_sqlite
@pytest.mark.parametrize("direct", [False, True])
def test_computed_join_evaluates_key_once_per_row_on_duplicates(test_session, direct):
    calls = 0

    def stem_key(path):
        nonlocal calls
        calls += 1
        return path.split(".")[0]

    test_session.catalog.warehouse.db.db.create_function(
        "test_stem_key", 1, stem_key, deterministic=True
    )
    left = dc.read_values(stem=["x"] * 20, session=test_session)
    right = dc.read_values(path=["x.png"] * 20, session=test_session)
    if direct:
        right_on = sa.func.test_stem_key(right.c("path"), type_=sa.String())
    else:
        right = right.mutate(stem=sa.func.test_stem_key(dc.C.path, type_=sa.String()))
        right_on = "stem"
    rows = left.merge(right, on="stem", right_on=right_on).to_list("stem", "path")
    assert len(rows) == 400
    assert calls == 20


@skip_if_not_sqlite
def test_computed_join_indexes_against_cast_other_key(warehouse, numbers_table):
    calls = 0

    def join_key(value):
        nonlocal calls
        calls += 1
        return str(value)

    warehouse.db.db.create_function(
        "test_str_join_key", 1, join_key, deterministic=True
    )
    number = numbers_table.c.number
    left = sa.select(
        number.label("id"), sa.cast(number, sa.String).label("key")
    ).subquery()
    right = sa.select(
        number.label("id"),
        sa.func.test_str_join_key(number, type_=sa.String).label("key"),
    ).subquery()
    query = warehouse.join(
        left,
        right,
        left.c.key == right.c.key,
        columns=[left.c.id, right.c.id.label("right_id")],
    )
    assert sorted(warehouse.db.execute(query)) == [(i, i) for i in range(1, 74)]
    assert calls == 73
    assert not _nested_scans(warehouse.db, query)


@skip_if_not_sqlite
@pytest.mark.parametrize("kind", ["inner", "left", "full", "subtract"])
@pytest.mark.parametrize("computed", ["left", "right", "both"])
@pytest.mark.parametrize("stored", [sa.Integer, sa.Float, sa.String, sa.LargeBinary])
@pytest.mark.parametrize(
    "declared",
    [
        sa.Integer,
        sa.Float,
        sa.String,
        sa.LargeBinary,
        sa.types.NullType,
        sa.BINARY,
        sa.Numeric,
    ],
)
@pytest.mark.parametrize("cast_key", [False, True])
def test_computed_join_keys_keep_sqlite_comparison_semantics(
    sqlite_db, kind, computed, stored, declared, cast_key
):
    values = [1, 1.0, 1.5, "1", "1.0", "01", "abc", "1e0", b"1", None, 0, True, 2**62]
    sqlite_db.db.create_function("test_value", 1, values.__getitem__)
    table = sa.Table(
        "join_values",
        sqlite_db.metadata,
        sa.Column("id", sa.Integer),
        sa.Column("key", stored),
    )
    sqlite_db.create_table(table)
    sqlite_db.db.executemany(
        "INSERT INTO join_values VALUES (?, ?)", list(enumerate(values))
    )

    def side(is_computed, *where):
        key = (
            sa.func.test_value(table.c.id, type_=declared)
            if is_computed
            else table.c.key
        )
        if is_computed and cast_key:
            key = sa.cast(key, declared) if declared is not sa.types.NullType else key
        return sa.select(table.c.id, key.label("key")).where(*where).subquery()

    left = side(computed != "right")
    right = side(computed != "left", table.c.id % 2 == 0)
    warehouse = SQLiteWarehouse(sqlite_db)
    if kind == "subtract":
        args = (sa.select(left), sa.select(right), [("key", "key")])
        query = warehouse.subtract_query(*args)
        plain = AbstractWarehouse.subtract_query(warehouse, *args)
    else:
        on = left.c.key == right.c.key
        columns = [left.c.id, right.c.id.label("right_id")]
        query = warehouse.join(
            left, right, on, inner=kind == "inner", full=kind == "full", columns=columns
        )
        joins = [sa.join(left, right, on, isouter=kind != "inner")]
        if kind == "full":
            joins.append(sa.join(right, left, on, isouter=True))
        plain = sa.union(*(sa.select(*columns).select_from(j) for j in joins))

    assert sorted(sqlite_db.execute(query), key=repr) == sorted(
        sqlite_db.execute(plain), key=repr
    )


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


@skip_if_not_sqlite
@pytest.mark.parametrize("full", [False, True])
def test_computed_join_keeps_disjunctive_predicate_matches(sqlite_db, full):
    warehouse = SQLiteWarehouse(sqlite_db)
    table = sa.Table(
        "disjunctive_keys", sqlite_db.metadata, sa.Column("key", sa.Integer)
    )
    sqlite_db.create_table(table)
    warehouse.insert_rows(table, [{"key": key} for key in [0, 1, 2]])
    warehouse.insert_rows_done(table)
    left = table.alias()
    right = sa.select((table.c.key + 1).label("key")).subquery()
    on = sa.or_(left.c.key == right.c.key, left.c.key == 0)
    columns = [left.c.key, right.c.key.label("right_key")]
    query = warehouse.join(left, right, on, full=full, columns=columns)
    plain = sa.select(*columns).select_from(sa.join(left, right, on, isouter=True))
    if full:
        plain = sa.union(
            plain,
            sa.select(*columns).select_from(sa.join(right, left, on, isouter=True)),
        )
    assert sorted(sqlite_db.execute(query), key=repr) == sorted(
        sqlite_db.execute(plain), key=repr
    )


@skip_if_not_sqlite
@pytest.mark.parametrize("full,inner", [(False, False), (False, True), (True, False)])
def test_computed_join_type_coercion_keeps_column_affinity(sqlite_db, full, inner):
    warehouse = SQLiteWarehouse(sqlite_db)
    left = sa.Table("text_codes", sqlite_db.metadata, sa.Column("key", sa.String))
    right_table = sa.Table(
        "numeric_codes", sqlite_db.metadata, sa.Column("key", sa.Integer)
    )
    for table, values in [(left, ["01", "1", "1e0"]), (right_table, [1])]:
        sqlite_db.create_table(table)
        warehouse.insert_rows(table, [{"key": value} for value in values])
        warehouse.insert_rows_done(table)
    right = sa.select(
        sa.type_coerce(right_table.c.key, sa.String).label("key")
    ).subquery()
    query = warehouse.join(
        left,
        right,
        left.c.key == right.c.key,
        full=full,
        inner=inner,
        columns=[left.c.key, right.c.key.label("right_key")],
    )
    assert sorted(sqlite_db.execute(query)) == [("01", 1), ("1", 1), ("1e0", 1)]


@skip_if_not_sqlite
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("collation", ["left_column", "right_expression"])
def test_computed_join_preserves_collation_precedence(
    sqlite_db, reverse, full, collation
):
    warehouse = SQLiteWarehouse(sqlite_db)
    left = sa.Table(
        "collated_left",
        sqlite_db.metadata,
        sa.Column(
            "key",
            sa.String(collation="NOCASE" if collation == "left_column" else "BINARY"),
        ),
    )
    right = sa.Table("collated_right", sqlite_db.metadata, sa.Column("key", sa.String))
    for table, value in [(left, "A"), (right, "a")]:
        sqlite_db.create_table(table)
        warehouse.insert_rows(table, [{"key": value}])
        warehouse.insert_rows_done(table)
    key = sa.func.lower(right.c.key, type_=sa.String)
    if collation == "right_expression":
        key = key.collate("NOCASE")
    query = warehouse.join(
        left,
        right,
        key == left.c.key if reverse else left.c.key == key,
        full=full,
        columns=[left.c.key, right.c.key.label("right_key")],
    )
    assert list(sqlite_db.execute(query)) == [("A", "a")]


@skip_if_not_sqlite
def test_full_computed_join_with_reversed_predicate(sqlite_db):
    warehouse = SQLiteWarehouse(sqlite_db)
    left = sa.select(sa.literal("a").label("key")).subquery()
    right = sa.select(sa.literal("a").label("key")).subquery()
    query = warehouse.join(
        left,
        right,
        sa.func.lower(right.c.key, type_=sa.String) == left.c.key,
        full=True,
        columns=[left.c.key, right.c.key.label("right_key")],
    )
    assert list(sqlite_db.execute(query)) == [("a", "a")]
