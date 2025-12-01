import pytest

import datachain as dc
from datachain import func
from datachain.lib.utils import DataChainParamsError


def test_aggregate_avg(test_session):
    class Data(dc.DataModel):
        i: int
        f: float

    ds = (
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(i=10, f=1.0),
                Data(i=20, f=2.0),
                Data(i=30, f=3.0),
            ),
            ii=(20, 40, 60),
            ff=(2.0, 4.0, 6.0),
            session=test_session,
        ).group_by(
            t1=func.avg("data.i"),
            t2=func.avg(dc.C("data.f")),
            t3=func.avg(dc.C("ii")),
            t4=func.avg("ff"),
        )
    ).to_list("t1", "t2", "t3", "t4")

    assert ds == [(20.0, 2.0, 40.0, 4.0)]


def test_aggregate_count(test_session):
    class Data(dc.DataModel):
        i: int
        f: float
        s: str

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(i=10, f=1.0, s="a"),
                Data(i=20, f=2.0, s="b"),
                Data(i=30, f=3.0, s="c"),
            ),
            ii=(20, 40, 60),
            ff=(2.0, 4.0, 6.0),
            ss=("x", "y", "z"),
            session=test_session,
        )
        .group_by(
            t1=func.count("data.i"),
            t2=func.count("data.f"),
            t3=func.count(dc.C("data.s")),
            t4=func.count("ii"),
            t5=func.count(dc.C("ff")),
            t6=func.count("ss"),
        )
        .to_list("t1", "t2", "t3", "t4", "t5", "t6")
    )

    assert ds == [(3, 3, 3, 3, 3, 3)]


def test_aggregate_sum(test_session):
    class Data(dc.DataModel):
        i: int
        f: float

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(i=10, f=1.0),
                Data(i=20, f=2.0),
                Data(i=30, f=3.0),
            ),
            ii=(20, 40, 60),
            ff=(2.0, 4.0, 6.0),
            session=test_session,
        )
        .group_by(
            t1=func.sum("data.i"),
            t2=func.sum(dc.C("data.f")),
            t3=func.sum(dc.C("ii")),
            t4=func.sum("ff"),
        )
        .to_list("t1", "t2", "t3", "t4")
    )

    assert ds == [(60, 6.0, 120, 12.0)]


def test_aggregate_min_max(test_session):
    class Data(dc.DataModel):
        i: int
        f: float
        s: str

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(i=10, f=1.0, s="a"),
                Data(i=20, f=2.0, s="b"),
                Data(i=30, f=3.0, s="c"),
            ),
            ii=(20, 40, 60),
            ff=(2.0, 4.0, 6.0),
            ss=("x", "y", "z"),
            session=test_session,
        )
        .group_by(
            t1=func.min(dc.C("data.i")),
            t2=func.min("data.f"),
            t3=func.min("data.s"),
            t4=func.max("data.i"),
            t5=func.max("data.f"),
            t6=func.max(dc.C("data.s")),
            t7=func.min("ii"),
            t8=func.min(dc.C("ff")),
            t9=func.min("ss"),
            t10=func.max(dc.C("ii")),
            t11=func.max("ff"),
        )
        .to_list("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11")
    )

    assert ds == [(10, 1.0, "a", 30, 3.0, "c", 20, 2.0, "x", 60, 6.0)]


def test_aggregate_any_value(test_session):
    class Data(dc.DataModel):
        i: int
        f: float
        s: str

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(i=10, f=1.0, s="a"),
                Data(i=20, f=2.0, s="b"),
                Data(i=30, f=3.0, s="c"),
            ),
            ii=(20, 40, 60),
            ff=(2.0, 4.0, 6.0),
            ss=("x", "y", "z"),
            session=test_session,
        )
        .group_by(
            t1=func.any_value("data.i"),
            t2=func.any_value("data.f"),
            t3=func.any_value(dc.C("data.s")),
            t4=func.any_value(dc.C("ii")),
            t5=func.any_value(dc.C("ff")),
            t6=func.any_value("ss"),
            partition_by="id",
        )
        .order_by("id")
        .to_list("t1", "t2", "t3", "t4", "t5", "t6")
    )

    # any_value can return any value from the group,
    # so we just check that it returns a value
    assert all(isinstance(x[0], int) and x[0] in (10, 20, 30) for x in ds)
    assert all(isinstance(x[1], float) and x[1] in (1.0, 2.0, 3.0) for x in ds)
    assert all(isinstance(x[2], str) and x[2] in ("a", "b", "c") for x in ds)
    assert all(isinstance(x[3], int) and x[3] in (20, 40, 60) for x in ds)
    assert all(isinstance(x[4], float) and x[4] in (2.0, 4.0, 6.0) for x in ds)
    assert all(isinstance(x[5], str) and x[5] in ("x", "y", "z") for x in ds)


def test_funcs_disallow_complex_object_collect(test_session):
    class Rec(dc.DataModel):
        i: int

    ds = dc.read_values(
        id=(1, 2),
        rec=(Rec(i=1), Rec(i=2)),
        session=test_session,
    )

    with pytest.raises(DataChainParamsError, match="doesn't support complex object"):
        ds.group_by(res=func.collect("rec"), partition_by="id")


def test_funcs_disallow_complex_object_min(test_session):
    class Rec(dc.DataModel):
        i: int

    ds = dc.read_values(
        id=(1, 2),
        rec=(Rec(i=1), Rec(i=2)),
        session=test_session,
    )

    with pytest.raises(DataChainParamsError, match="doesn't support complex object"):
        ds.group_by(res=func.min("rec"), partition_by="id")
