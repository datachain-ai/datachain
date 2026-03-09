import pytest

import datachain as dc
from datachain import func
from datachain.lib.utils import DataChainParamsError
from tests.utils import skip_if_not_sqlite


def test_conditional_and_or_not(test_session):
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
            session=test_session,
        )
        .mutate(
            t1=func.and_(dc.C("data.i") > 15, dc.C("data.f") > 1.5),
            t2=func.and_(dc.C("data.i") > 15, dc.C("data.f") > 2.5),
            t3=func.or_(dc.C("data.i") > 15, dc.C("data.f") > 1.5),
            t4=func.or_(dc.C("data.i") > 15, dc.C("data.f") > 2.5),
            t5=func.not_(dc.C("data.i") > 15),
        )
        .order_by("id")
    ).to_list("t1", "t2", "t3", "t4", "t5")

    assert ds == [(0, 0, 0, 0, 1), (1, 0, 1, 1, 0), (1, 1, 1, 1, 0)]


def test_conditional_case(test_session):
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
            session=test_session,
        )
        .mutate(
            t1=func.case(
                (dc.C("data.i") < 15, "small"),
                (dc.C("data.i") < 25, "medium"),
                else_="large",
            ),
            t2=func.case(
                (dc.C("data.f") < 1.5, "low"),
                (dc.C("data.f") < 2.5, "medium"),
                else_="high",
            ),
        )
        .order_by("id")
        .to_list("t1", "t2")
    )

    assert ds == [
        ("small", "low"),
        ("medium", "medium"),
        ("large", "high"),
    ]


def test_conditional_ifelse(test_session):
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
            session=test_session,
        )
        .mutate(
            t1=func.ifelse(dc.C("data.i") > 15, "large", "small"),
            t2=func.ifelse(dc.C("data.f") > 1.5, "high", "low"),
            t3=func.ifelse(dc.C("data.s") == "b", "middle", "other"),
        )
        .order_by("id")
        .to_list("t1", "t2", "t3")
    )

    assert ds == [
        ("small", "low", "other"),
        ("large", "high", "middle"),
        ("large", "high", "other"),
    ]


@skip_if_not_sqlite  # ClickHouse does not support NULL out of the box
def test_conditional_isnone(test_session):
    class Data(dc.DataModel):
        i: int
        f: float | None
        s: str | None

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(i=10, f=1.0, s="a"),
                Data(i=20, f=None, s="b"),
                Data(i=30, f=3.0, s=None),
            ),
            session=test_session,
        )
        .mutate(
            t1=func.isnone("data.i"),
            t2=func.isnone("data.f"),
            t3=func.isnone("data.s"),
        )
        .order_by("id")
        .to_list("t1", "t2", "t3")
    )

    assert ds == [
        (False, False, False),
        (False, True, False),
        (False, False, True),
    ]


def test_conditional_greatest_least(test_session):
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
            session=test_session,
        )
        .mutate(
            t1=func.greatest("data.i", 15),
            t2=func.greatest(dc.C("data.f"), 1.5),
            t3=func.least(dc.C("data.i"), 25),
            t4=func.least("data.f", 2.5),
            t5=func.greatest(dc.C("ii"), 30),
            t6=func.greatest("ff", 3.5),
            t7=func.least("ii", 50),
            t8=func.least(dc.C("ff"), 3.5),
            t9=func.greatest(dc.C("data.i"), "ii", 15, 30),
            t10=func.least("data.f", dc.C("ff"), 1.5, 3.5),
        )
        .order_by("id")
        .to_list("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10")
    )

    assert ds == [
        (15, 1.5, 10, 1.0, 30, 3.5, 20, 2.0, 30, 1.0),
        (20, 2.0, 20, 2.0, 40, 4.0, 40, 3.5, 40, 1.5),
        (30, 3.0, 25, 2.5, 60, 6.0, 50, 3.5, 60, 1.5),
    ]


def test_conditional_funcs_disallow_complex_object_greatest(test_session):
    class Rec(dc.DataModel):
        i: int

    ds = dc.read_values(id=[1, 2], rec=[Rec(i=1), Rec(i=2)], session=test_session)

    with pytest.raises(DataChainParamsError, match="doesn't support complex object"):
        ds.mutate(t=func.greatest("rec", 1))
