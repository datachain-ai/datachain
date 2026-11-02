from typing import ClassVar

import pytest

import datachain as dc
from datachain import C, DataModel, func
from datachain.lib.utils import DataChainParamsError


class SubtractKey(DataModel):
    a: int
    b: str


class ReorderedSubtractKey(DataModel):
    b: str
    a: int


class Box(DataModel):
    size: int | None


class Bag(DataModel):
    name: str


class NullableKey(DataModel):
    a: int | None = None
    b: str | None = None


class ReorderedNullableKey(DataModel):
    b: str | None = None
    a: int | None = None


@pytest.mark.parametrize("column", ["__dc_present_0", "__DC_PRESENT_0"])
def test_subtract_preserves_columns_with_presence_label_names(test_session, column):
    left = dc.read_values(
        key=[None, NullableKey()],
        **{column: [10, 20]},
        output={"key": NullableKey | None, column: int},
        session=test_session,
    )
    right = dc.read_values(
        key=[NullableKey()],
        **{column: [30], "_" + column: [40]},
        output={"key": NullableKey | None, column: int, "_" + column: int},
        session=test_session,
    )
    assert left.subtract(right, on="key").to_list("key", column) == [(None, 10)]


class Holder(DataModel):
    key: NullableKey | None = None


class PlainHolder(DataModel):
    key: ReorderedNullableKey = ReorderedNullableKey()


def _holder_version_2() -> type[DataModel]:
    class Holder(DataModel):  # the same union arm name, with a plain key
        _version: ClassVar[int] = 2
        key: NullableKey = NullableKey()

    return Holder


@pytest.mark.parametrize(
    "right_model,left_optional,right_optional",
    [
        (ReorderedSubtractKey, False, False),
        (ReorderedSubtractKey, True, False),
        (ReorderedSubtractKey, False, True),
        (SubtractKey, True, True),
    ],
)
def test_subtract_matches_model_key_fields_by_name(
    test_session, right_model, left_optional, right_optional
):
    def chain(name, model, optional, keys):
        values = [model(a=a, b=b) for a, b in keys] + ([None] if optional else [])
        return dc.read_values(
            **{name: values},
            output={name: model | None if optional else model},
            session=test_session,
        )

    left = chain("key", SubtractKey, left_optional, [(1, "x"), (2, "y")])
    right = chain("other", right_model, right_optional, [(1, "x")])

    result = left.subtract(right, on="key", right_on="other").to_values("key")
    assert sorted(
        (None if key is None else (key.a, key.b) for key in result), key=repr
    ) == [(2, "y")] + ([None] if left_optional and not right_optional else [])


def test_subtract_union_key_distinguishes_empty_arm_from_none(test_session):
    def chain(name, values):
        return dc.read_values(
            id=list(range(len(values))),
            **{name: values},
            output={"id": int, name: Box | Bag | None},
            session=test_session,
        )

    left = chain("item", [Box(size=None), None, Bag(name="x")])
    right = chain("other", [None, Bag(name="x")])
    assert left.subtract(right, on="item", right_on="other").to_values("id") == [0]


@pytest.mark.parametrize("right_model", [NullableKey, ReorderedNullableKey])
@pytest.mark.parametrize("right_optional", [False, True])
def test_subtract_optional_model_key_keeps_none_apart_from_empty_model(
    test_session, right_model, right_optional
):
    def chain(name, values, output):
        return dc.read_values(
            **{name: values}, output={name: output}, session=test_session
        )

    left = chain("key", [None, NullableKey()], NullableKey | None)
    empty = chain(
        "other", [right_model()], right_model | None if right_optional else right_model
    )
    none = chain("key", [None], NullableKey | None)

    assert left.subtract(empty, on="key", right_on="other").to_values("key") == [None]
    assert left.subtract(none, on="key", right_on="key").to_values("key") == [
        NullableKey()
    ]
    assert empty.subtract(none, on="other", right_on="key").to_values("other") == [
        right_model()
    ]


def test_subtract_nested_optional_model_key(test_session):
    holders = dc.read_values(
        id=[0, 1, 2],
        holder=[None, Holder(), Holder(key=NullableKey())],
        output={"id": int, "holder": Holder | None},
        session=test_session,
    )
    plain = dc.read_values(
        holder=[PlainHolder()], output={"holder": PlainHolder}, session=test_session
    )

    def subtract(left, right):
        return left.subtract(right, on="holder", right_on="holder")

    assert sorted(subtract(holders, plain).to_values("id")) == [0, 1]
    assert subtract(plain, holders).empty
    assert subtract(plain, holders.filter(C("id") < 2)).to_values("holder") == [
        PlainHolder()
    ]


def test_subtract_union_key_compares_optional_model_inside_arm(test_session):
    def chain(values, arm):
        return dc.read_values(
            id=list(range(len(values))),
            item=values,
            output={"id": int, "item": arm | Bag},
            session=test_session,
        )

    def subtract(left, right):
        return sorted(left.subtract(right, on="item", right_on="item").to_values("id"))

    items = chain([Bag(name="x"), Holder(), Holder(key=NullableKey())], Holder)
    assert subtract(items, chain([Holder(key=NullableKey())], Holder)) == [0, 1]

    holder_v2 = _holder_version_2()
    plain = chain([Bag(name="x"), holder_v2()], holder_v2)
    assert subtract(items, plain) == [1]
    assert subtract(plain, items.filter(C("id") < 2)) == [1]


def test_subtract_handles_duplicates_with_some_different_values(test_session):
    base = dc.read_values(
        session_id=[1, 1, 2],
        value=[10, 20, 30],
        session=test_session,
        in_memory=True,
    ).mutate(rnd=(func.rand() % 3) + 1)

    other = base.filter(C("session_id") == 1)
    result = base.subtract(other, on="session_id")

    counts = [result.count() for _ in range(5)]
    assert counts == [1, 1, 1, 1, 1]
    assert result.to_values("session_id") == [2]


def test_subtract_defaults_to_common_columns(test_session):
    base = dc.read_values(
        a=[1, 1, 2],
        b=[1, 2, 1],
        c=[10, 20, 30],
        session=test_session,
        in_memory=True,
    )

    other = dc.read_values(
        a=[1, 9],
        b=[1, 9],
        d=["x", "y"],
        session=test_session,
        in_memory=True,
    )

    result = base.subtract(other)

    remaining = sorted(result.to_list("a", "b"))
    assert remaining == [(1, 2), (2, 1)]


def test_subtract(test_session):
    chain1 = dc.read_values(a=[1, 1, 2], b=["x", "y", "z"], session=test_session)
    chain2 = dc.read_values(a=[1, 2], b=["x", "y"], session=test_session)
    assert set(chain1.subtract(chain2, on=["a", "b"]).to_list()) == {(1, "y"), (2, "z")}
    assert set(chain1.subtract(chain2, on=["b"]).to_list()) == {(2, "z")}
    assert not set(chain1.subtract(chain2, on=["a"]).to_list())
    assert set(chain1.subtract(chain2).to_list()) == {(1, "y"), (2, "z")}
    assert chain1.subtract(chain1).count() == 0

    chain3 = dc.read_values(a=[1, 3], c=["foo", "bar"], session=test_session)
    assert set(chain1.subtract(chain3, on="a").to_list()) == {(2, "z")}
    assert set(chain1.subtract(chain3).to_list()) == {(2, "z")}

    chain4 = dc.read_values(d=[1, 2, 3], e=["x", "y", "z"], session=test_session)
    chain5 = dc.read_values(a=[1, 2], b=["x", "y"], session=test_session)

    assert set(chain4.subtract(chain5, on="d", right_on="a").to_list()) == {(3, "z")}


def test_subtract_duplicated_rows(test_session):
    chain1 = dc.read_values(id=[1, 1], name=["1", "1"], session=test_session)
    chain2 = dc.read_values(id=[2], name=["2"], session=test_session)
    sub = chain1.subtract(chain2, on="id")
    assert set(sub.to_list()) == {(1, "1"), (1, "1")}


def test_subtract_error(test_session):
    chain1 = dc.read_values(a=[1, 1, 2], b=["x", "y", "z"], session=test_session)
    chain2 = dc.read_values(a=[1, 2], b=["x", "y"], session=test_session)
    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=[])
    with pytest.raises(TypeError):
        chain1.subtract(chain2, on=42)

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on="")

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on="a", right_on="")

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=["a", "b"], right_on=["c", ""])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=["a", "b"], right_on=[])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, on=["a", "b"], right_on=["d"])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on=[])

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on="")

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on=42)

    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain2, right_on=["a"])

    with pytest.raises(TypeError):
        chain1.subtract(chain2, on=42, right_on=42)

    chain3 = dc.read_values(c=["foo", "bar"], session=test_session)
    with pytest.raises(DataChainParamsError):
        chain1.subtract(chain3)


def test_subtract_chained(test_session):
    base = dc.read_values(
        x=[1, 2, 3, 4, 5],
        session=test_session,
        in_memory=True,
    )

    remove1 = dc.read_values(x=[1], session=test_session, in_memory=True)
    remove2 = dc.read_values(x=[3], session=test_session, in_memory=True)
    remove3 = dc.read_values(x=[5], session=test_session, in_memory=True)

    result = (
        base.subtract(remove1, on="x")
        .subtract(remove2, on="x")
        .subtract(remove3, on="x")
    )

    assert sorted(result.to_values("x")) == [2, 4]


def test_subtract_after_mutate(test_session):
    base = dc.read_values(
        x=[1, 2, 3, 4],
        session=test_session,
        in_memory=True,
    ).mutate(y=C("x") * 10)

    remove = dc.read_values(
        x=[2, 4],
        session=test_session,
        in_memory=True,
    )

    result = base.subtract(remove, on="x")

    assert sorted(result.to_list("x", "y")) == [(1, 10), (3, 30)]


def test_subtract_after_group_by(test_session):
    base = dc.read_values(
        category=["a", "a", "b", "b", "c"],
        value=[1, 2, 3, 4, 5],
        session=test_session,
        in_memory=True,
    ).group_by(
        total=func.sum(C("value")),
        partition_by="category",
    )

    remove = dc.read_values(
        category=["b"],
        session=test_session,
        in_memory=True,
    )

    result = base.subtract(remove, on="category")

    remaining = sorted(result.to_list("category", "total"))
    assert remaining == [("a", 3), ("c", 5)]


def test_subtract_after_map(test_session):
    def double(x: int) -> int:
        return x * 2

    base = dc.read_values(
        x=[1, 2, 3, 4, 5],
        session=test_session,
        in_memory=True,
    ).map(doubled=double, params=["x"])

    remove = dc.read_values(
        doubled=[4, 8],
        session=test_session,
        in_memory=True,
    )

    result = base.subtract(remove, on="doubled")

    assert sorted(result.to_values("doubled")) == [2, 6, 10]


def test_subtract_preserves_sys_columns(test_session):
    base = dc.read_values(
        x=[1, 2, 3],
        session=test_session,
        in_memory=True,
    )

    remove = dc.read_values(
        x=[2],
        session=test_session,
        in_memory=True,
    )

    result = base.subtract(remove, on="x")

    # Check signal schema contains sys columns
    sys_schema = result.signals_schema.resolve("sys.id", "sys.rand").values
    assert sys_schema["sys.id"] is int
    assert sys_schema["sys.rand"] is int

    # Check actual data has sys__id and sys__rand columns
    rows = list(result.to_list("sys.id", "sys.rand", "x"))
    assert len(rows) == 2

    for sys_id, sys_rand, x in rows:
        assert isinstance(sys_id, int)
        assert isinstance(sys_rand, int)
        assert x in [1, 3]


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("left_optional", [False, True])
@pytest.mark.parametrize("right_optional", [False, True])
def test_subtract_nullable_model_uses_same_equality_with_default_keys(
    test_session, implicit, left_optional, right_optional
):
    left = dc.read_values(
        key=[None, NullableKey()] if left_optional else [NullableKey()],
        output={"key": NullableKey | None if left_optional else NullableKey},
        session=test_session,
    )
    right = dc.read_values(
        key=[None] if right_optional else [ReorderedNullableKey()],
        output={
            "key": ReorderedNullableKey | None
            if right_optional
            else ReorderedNullableKey
        },
        session=test_session,
    )
    kwargs = {} if implicit else {"on": "key"}
    expected = [NullableKey()] if right_optional else ([None] if left_optional else [])
    assert left.subtract(right, **kwargs).to_values("key") == expected


def test_subtract_implicit_keys_keep_partial_model_field_matching(test_session):
    class PartialKey(DataModel):
        a: int

    left = dc.read_values(key=[SubtractKey(a=1, b="x")], session=test_session)
    right = dc.read_values(key=[PartialKey(a=1)], session=test_session)
    assert left.subtract(right).empty
