"""Functional tests for multi-arm ``Union[...]`` support (tagged unions).

Covers every union kind end to end — ``Union[basic, basic]``,
``Union[Model, Model]``, mixed ``Union[basic, Model]`` and nullable
``Union[..., None]`` — through ``read_values`` / ``map`` ingestion, ``save``
round-trips, parquet/JSON export, and the query ops (filter / mutate / select /
group_by / isnone / count / union). The active arm is stored under a ``_type_tag``
discriminator; inactive arms are NULL. These run on both SQLite and ClickHouse
(via ``scripts/run-clickhouse-tests.sh``).

Rows are read with an explicit ``id`` column and ordered by it; ``sys.id`` order
is not stable across backends for generated rows.
"""

import json
from typing import Optional, Union

import pytest

import datachain as dc
from datachain import C, func
from datachain.lib.data_model import DataModel


class _Foo(DataModel):
    a: int = 0
    b: str = ""


class _Bar(DataModel):
    x: float = 0.0


class _Holder(DataModel):
    id: int = 0
    payload: Union[str, int] = 0


def _ordered(chain, *cols):
    return chain.order_by("id").to_list("id", *cols)


# ---- save round-trips ------------------------------------------------------


def test_scalar_union_roundtrip(test_session):
    dc.read_values(
        id=[1, 2, 3, 4],
        value=["hello", 42, "world", 7],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    ).save("u_scalar")
    back = dc.read_dataset("u_scalar", session=test_session)
    assert _ordered(back, "value") == [(1, "hello"), (2, 42), (3, "world"), (4, 7)]


def test_model_union_roundtrip(test_session):
    items = [_Foo(a=1, b="z"), _Bar(x=2.5), _Foo(a=9, b="q")]
    dc.read_values(
        id=[1, 2, 3],
        item=items,
        output={"id": int, "item": Union[_Foo, _Bar]},
        session=test_session,
    ).save("u_models")
    back = dc.read_dataset("u_models", session=test_session)
    assert [v for _, v in _ordered(back, "item")] == items


def test_mixed_union_roundtrip(test_session):
    items = ["txt", _Foo(a=2, b="m"), 5]
    dc.read_values(
        id=[1, 2, 3],
        value=items,
        output={"id": int, "value": Union[str, int, _Foo]},
        session=test_session,
    ).save("u_mixed")
    back = dc.read_dataset("u_mixed", session=test_session)
    assert [v for _, v in _ordered(back, "value")] == items


def test_nullable_union_roundtrip(test_session):
    dc.read_values(
        id=[1, 2, 3, 4],
        value=["a", 3, None, "c"],
        output={"id": int, "value": Union[str, int, None]},
        session=test_session,
    ).save("u_nullable")
    back = dc.read_dataset("u_nullable", session=test_session)
    assert [v for _, v in _ordered(back, "value")] == ["a", 3, None, "c"]


def test_union_float_arm_roundtrip(test_session):
    # A float arm works in a multi-arm union (the _type_tag disambiguates it),
    # unlike the single-arm Optional[float].
    items = [1.5, "txt", 2.0]
    dc.read_values(
        id=[1, 2, 3],
        value=items,
        output={"id": int, "value": Union[str, float]},
        session=test_session,
    ).save("u_float")
    back = dc.read_dataset("u_float", session=test_session)
    assert [v for _, v in _ordered(back, "value")] == items


def test_union_nested_in_model_roundtrip(test_session):
    holders = [_Holder(id=1, payload="x"), _Holder(id=2, payload=99)]
    dc.read_values(
        id=[1, 2],
        h=holders,
        output={"id": int, "h": _Holder},
        session=test_session,
    ).save("u_nested")
    back = dc.read_dataset("u_nested", session=test_session)
    assert [v for _, v in _ordered(back, "h")] == holders


# ---- ingestion via map -----------------------------------------------------


def test_map_returns_union(test_session):
    base = dc.read_values(id=[1, 2, 3, 4], session=test_session)

    def f(id) -> Union[str, int]:
        return "even" if id % 2 == 0 else id

    base.map(r=f, output={"r": Union[str, int]}).save("u_map")
    back = dc.read_dataset("u_map", session=test_session)
    assert _ordered(back, "r") == [(1, 1), (2, "even"), (3, 3), (4, "even")]


# ---- export round-trips ----------------------------------------------------


def test_union_parquet_roundtrip(test_session, tmp_path):
    items = [_Foo(a=1, b="z"), _Bar(x=2.5)]
    path = str(tmp_path / "u.parquet")
    dc.read_values(
        id=[1, 2],
        item=items,
        output={"id": int, "item": Union[_Foo, _Bar]},
        session=test_session,
    ).order_by("id").to_parquet(path)
    back = dc.read_parquet(path, session=test_session)
    assert [v for _, v in _ordered(back, "item")] == items


def test_union_jsonl_export(test_session, tmp_path):
    path = str(tmp_path / "u.jsonl")
    dc.read_values(
        id=[1, 2],
        value=["hello", 42],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    ).order_by("id").to_jsonl(path)
    with open(path) as f:
        rows = [json.loads(line) for line in f.read().splitlines() if line]
    rows.sort(key=lambda r: r["id"])
    assert rows == [{"id": 1, "value": "hello"}, {"id": 2, "value": 42}]


# ---- query ops -------------------------------------------------------------


def _nullable_union(test_session):
    return dc.read_values(
        id=[1, 2, 3, 4, 5],
        value=["hi", 42, "yo", 7, None],
        output={"id": int, "value": Union[str, int, None]},
        session=test_session,
    )


def test_union_isnone(test_session):
    chain = _nullable_union(test_session)
    assert chain.filter(func.isnone("value")).count() == 1
    assert chain.filter(func.not_(func.isnone("value"))).count() == 4


def test_union_count_present(test_session):
    chain = _nullable_union(test_session)
    assert chain.group_by(c=func.count("value")).to_values("c") == [4]


def test_union_filter_on_arm(test_session):
    chain = _nullable_union(test_session)
    # int sorts before str, so value._0 is the int arm.
    assert chain.filter(C("value._0") == 42).to_values("value") == [42]
    assert chain.filter(C("value._1") == "hi").to_values("value") == ["hi"]


def test_union_mutate_on_arm(test_session):
    chain = _nullable_union(test_session)
    # value._0 is the int arm: present only for int rows, NULL elsewhere.
    assert _ordered(chain.mutate(z=C("value._0")), "z") == [
        (1, None),
        (2, 42),
        (3, None),
        (4, 7),
        (5, None),
    ]


def test_union_select_keeps_whole_signal(test_session):
    chain = _nullable_union(test_session)
    # A union is atomic: selecting any part keeps the whole value.
    assert [v for _, v in _ordered(chain.select("id", "value._0"), "value")] == [
        "hi",
        42,
        "yo",
        7,
        None,
    ]


def test_union_distinct_on_arm(test_session):
    chain = _nullable_union(test_session)
    assert chain.distinct("value._1").count() == 3  # 'hi', 'yo', NULL


def test_readable_arm_access_across_ops(test_session):
    # A readable arm path (value.int) must work in every op, not only filter/mutate.
    chain = dc.read_values(
        id=[1, 2, 3, 4],
        value=["hi", 42, "yo", 7],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    )
    assert chain.filter(C("value.int") == 42).to_values("id") == [2]
    assert chain.distinct("value.int").count() == 3  # 42, 7, NULL (str rows)
    assert chain.order_by("value.int").to_values("value") == [7, 42, "hi", "yo"]
    # selecting an arm field keeps the whole (atomic) union
    assert _ordered(chain.select("id", "value.int"), "value") == [
        (1, "hi"),
        (2, 42),
        (3, "yo"),
        (4, 7),
    ]


def test_union_combination_same_type(test_session):
    left = dc.read_values(
        id=[1, 2],
        value=["a", 1],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    )
    right = dc.read_values(
        id=[3, 4],
        value=[2, "b"],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    )
    left.union(right).save("u_union_op")
    back = dc.read_dataset("u_union_op", session=test_session)
    assert _ordered(back, "value") == [(1, "a"), (2, 1), (3, 2), (4, "b")]


def test_merge_keeps_union_on_unmatched_rows(test_session):
    # An outer-join unmatched row pads the right columns with NULL; the union on the
    # left must survive (the _type_tag is nullable precisely for this).
    left = dc.read_values(
        id=[1, 2, 3],
        value=["a", 1, "b"],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    )
    right = dc.read_values(
        id=[1, 2], k=["x", "y"], output={"id": int, "k": str}, session=test_session
    )
    merged = left.merge(right, on="id", inner=False)
    assert _ordered(merged, "value") == [(1, "a"), (2, 1), (3, "b")]


def test_union_with_missing_signal_names_the_signal(test_session):
    # The mismatch error reports the signal name, not internal arm-slot columns.
    from datachain.query.dataset import UnionSchemaMismatchError

    left = dc.read_values(
        id=[1],
        value=["a"],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    )
    right = dc.read_values(
        id=[2], other=["z"], output={"id": int, "other": str}, session=test_session
    )
    with pytest.raises(UnionSchemaMismatchError) as exc:
        left.union(right).count()
    msg = str(exc.value)
    assert "value" in msg and "_0" not in msg and "_1" not in msg


def test_to_database_readable_arm_columns(test_session):
    import sqlite3

    conn = sqlite3.connect(":memory:")
    dc.read_values(
        id=[1, 2, 3],
        value=["a", 1, "b"],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    ).to_database("t", conn)
    cols = [d[0] for d in conn.execute("SELECT * FROM t").description]
    # readable arm names, no internal slots or discriminator
    assert {"value__int", "value__str"} <= set(cols)
    assert not any("_0" in c or "_1" in c or "_type_tag" in c for c in cols)


def test_union_optional_nested_in_model_roundtrip(test_session):
    class H(DataModel):
        v: Optional[Union[str, int]] = None

    items = [H(v="a"), H(v=5), H(v=None)]
    dc.read_values(
        id=[1, 2, 3],
        h=items,
        output={"id": int, "h": H},
        session=test_session,
    ).save("u_opt_nested")
    back = dc.read_dataset("u_opt_nested", session=test_session)
    assert [h.v for _, h in _ordered(back, "h")] == ["a", 5, None]


def test_union_schema_canonical_order_both_spellings(test_session):
    # Union[str, int] and Union[int, str] are the same type; the persisted schema
    # and values must match regardless of how the union is written.
    a = dc.read_values(
        id=[1, 2],
        value=["x", 1],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    )
    b = dc.read_values(
        id=[1, 2],
        value=["x", 1],
        output={"id": int, "value": Union[int, str]},
        session=test_session,
    )
    assert a.signals_schema.db_signals() == b.signals_schema.db_signals()
    assert _ordered(a, "value") == _ordered(b, "value") == [(1, "x"), (2, 1)]


# ---- collection arms (list / dict) -----------------------------------------
