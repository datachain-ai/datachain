import pytest

import datachain as dc
from datachain import C, func
from datachain.lib.utils import DataChainParamsError


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
