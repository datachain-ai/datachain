import pytest
from sqlalchemy import Label

import datachain as dc
from datachain import func
from datachain.func import (
    and_,
    bit_hamming_distance,
    byte_hamming_distance,
    case,
    ifelse,
    int_hash_64,
    isnone,
    literal,
    not_,
    or_,
)
from datachain.func.array import contains
from datachain.func.random import rand
from datachain.func.string import length as strlen
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.utils import DataChainParamsError
from datachain.sql.sqlite.base import (
    sqlite_bit_hamming_distance,
    sqlite_byte_hamming_distance,
    sqlite_int_hash_64,
)
from tests.utils import skip_if_not_sqlite


@pytest.fixture()
def chain():
    return dc.read_values(
        num=list(range(1, 6)),
        val=["x" * i for i in range(1, 6)],
    )


def test_db_cols():
    rnd = rand()
    assert rnd._db_cols == []
    assert rnd._db_col_type(SignalSchema({})) is None


def test_label():
    rnd = rand()
    assert rnd.col_label is None
    assert rnd.label("test2") == "test2"

    f = rnd.label("test")
    assert f.col_label == "test"
    assert f.label("test2") == "test2"


def test_col_name():
    rnd = rand()
    assert rnd.get_col_name() == "rand"
    assert rnd.label("test").get_col_name() == "test"
    assert rnd.get_col_name("test2") == "test2"


def test_result_type():
    rnd = rand()
    assert rnd.get_result_type(SignalSchema({})) is int


def test_get_column():
    rnd = rand()
    col = rnd.get_column(SignalSchema({}))
    assert isinstance(col, Label)
    assert col.name == "rand"


def test_get_column_disallow_complex_object_in_sql_funcs():
    class Rec(dc.DataModel):
        i: int

    schema = SignalSchema({"rec": Rec})

    with pytest.raises(DataChainParamsError, match="doesn't support complex object"):
        func.min("rec").get_column(schema)

    with pytest.raises(DataChainParamsError, match="doesn't support complex object"):
        func.collect("rec").get_column(schema)


def test_add():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 + 1
    assert str(f) == "add()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 1 + rnd2
    assert str(f) == "add()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 + rnd2
    assert str(f) == "add()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_add_mutate(chain):
    res = chain.mutate(test=strlen("val") + 1).order_by("num").to_values("test")
    assert res == [2, 3, 4, 5, 6]

    res = chain.mutate(test=1 + strlen("val")).order_by("num").to_values("test")
    assert res == [2, 3, 4, 5, 6]

    res = (
        chain.mutate(test=strlen("val") + strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [2, 4, 6, 8, 10]


def test_sub():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 - 1
    assert str(f) == "sub()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 1 - rnd2
    assert str(f) == "sub()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 - rnd2
    assert str(f) == "sub()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_sub_mutate(chain):
    res = chain.mutate(test=strlen("val") - 1).order_by("num").to_values("test")
    assert res == [0, 1, 2, 3, 4]

    res = chain.mutate(test=5 - strlen("val")).order_by("num").to_values("test")
    assert res == [4, 3, 2, 1, 0]

    res = (
        chain.mutate(test=strlen("val") - strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [0, 0, 0, 0, 0]


def test_mul():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 * 2
    assert str(f) == "mul()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 * rnd2
    assert str(f) == "mul()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 * rnd2
    assert str(f) == "mul()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_mul_mutate(chain):
    res = chain.mutate(test=strlen("val") * 2).order_by("num").to_values("test")
    assert res == [2, 4, 6, 8, 10]

    res = chain.mutate(test=3 * strlen("val")).order_by("num").to_values("test")
    assert res == [3, 6, 9, 12, 15]

    res = (
        chain.mutate(test=strlen("val") * strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [1, 4, 9, 16, 25]


def test_truediv():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 / 2
    assert str(f) == "div()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 1 / rnd2
    assert str(f) == "div()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 / rnd2
    assert str(f) == "div()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_truediv_mutate(chain):
    res = chain.mutate(test=strlen("val") / 2).order_by("num").to_values("test")
    assert res == [0.5, 1.0, 1.5, 2.0, 2.5]

    res = chain.mutate(test=10 / strlen("val")).order_by("num").to_values("test")
    assert res == [10.0, 5.0, 10 / 3, 2.5, 2.0]

    res = (
        chain.mutate(test=strlen("val") / strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_floordiv():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 // 2
    assert str(f) == "floordiv()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 1 // rnd2
    assert str(f) == "floordiv()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 // rnd2
    assert str(f) == "floordiv()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_floordiv_mutate(chain):
    res = chain.mutate(test=strlen("val") // 2).order_by("num").to_values("test")
    assert res == [0, 1, 1, 2, 2]

    res = chain.mutate(test=10 // strlen("val")).order_by("num").to_values("test")
    assert res == [10, 5, 3, 2, 2]

    res = (
        chain.mutate(test=strlen("val") // strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [1, 1, 1, 1, 1]


def test_mod():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 % 2
    assert str(f) == "mod()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 10 % rnd2
    assert str(f) == "mod()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 % rnd2
    assert str(f) == "mod()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_mod_mutate(chain):
    res = chain.mutate(test=strlen("val") % 2).order_by("num").to_values("test")
    assert res == [1, 0, 1, 0, 1]

    res = chain.mutate(test=10 % strlen("val")).order_by("num").to_values("test")
    assert res == [0, 0, 1, 2, 0]

    res = (
        chain.mutate(test=strlen("val") % strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [0, 0, 0, 0, 0]


def test_and():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 & 2
    assert str(f) == "and()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 & rnd2
    assert str(f) == "and()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 & rnd2
    assert str(f) == "and()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_and_mutate(chain):
    res = chain.mutate(test=strlen("val") & 2).order_by("num").to_values("test")
    assert res == [0, 2, 2, 0, 0]

    res = chain.mutate(test=2 & strlen("val")).order_by("num").to_values("test")
    assert res == [0, 2, 2, 0, 0]

    res = (
        chain.mutate(test=strlen("val") & strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [1, 2, 3, 4, 5]


def test_or():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 | 2
    assert str(f) == "or()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 | rnd2
    assert str(f) == "or()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 | rnd2
    assert str(f) == "or()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_or_mutate(chain):
    res = chain.mutate(test=strlen("val") | 2).order_by("num").to_values("test")
    assert res == [3, 2, 3, 6, 7]

    res = chain.mutate(test=2 | strlen("val")).order_by("num").to_values("test")
    assert res == [3, 2, 3, 6, 7]

    res = (
        chain.mutate(test=strlen("val") | strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [1, 2, 3, 4, 5]


@skip_if_not_sqlite
def test_or_func_mutate(chain):
    res = chain.mutate(
        test=ifelse(or_(dc.C("num") < 3, dc.C("num") > 4), "Match", "Not Match")
    )
    assert list(res.order_by("num").to_values("test")) == [
        "Match",
        "Match",
        "Not Match",
        "Not Match",
        "Match",
    ]


@skip_if_not_sqlite
def test_and_func_mutate(chain):
    res = chain.mutate(
        test=ifelse(and_(dc.C("num") > 1, dc.C("num") < 4), "Match", "Not Match")
    )
    assert list(res.order_by("num").to_values("test")) == [
        "Not Match",
        "Match",
        "Match",
        "Not Match",
        "Not Match",
    ]


@skip_if_not_sqlite
def test_not_func_mutate(chain):
    res = chain.mutate(test=ifelse(not_(dc.C("num") > 3), "Match", "Not Match"))
    assert list(res.order_by("num").to_values("test")) == [
        "Match",
        "Match",
        "Match",
        "Not Match",
        "Not Match",
    ]


def test_xor():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 ^ 2
    assert str(f) == "xor()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 ^ rnd2
    assert str(f) == "xor()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 ^ rnd2
    assert str(f) == "xor()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_xor_mutate(chain):
    res = chain.mutate(test=strlen("val") ^ 2).order_by("num").to_values("test")
    assert res == [3, 0, 1, 6, 7]

    res = chain.mutate(test=2 ^ strlen("val")).order_by("num").to_values("test")
    assert res == [3, 0, 1, 6, 7]

    res = (
        chain.mutate(test=strlen("val") ^ strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [0, 0, 0, 0, 0]


def test_rshift():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 >> 2
    assert str(f) == "rshift()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 >> rnd2
    assert str(f) == "rshift()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 >> rnd2
    assert str(f) == "rshift()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_rshift_mutate(chain):
    res = chain.mutate(test=strlen("val") >> 2).order_by("num").to_values("test")
    assert res == [0, 0, 0, 1, 1]

    res = chain.mutate(test=2 >> strlen("val")).order_by("num").to_values("test")
    assert res == [1, 0, 0, 0, 0]

    res = (
        chain.mutate(test=strlen("val") >> strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [0, 0, 0, 0, 0]


def test_lshift():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 << 2
    assert str(f) == "lshift()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 << rnd2
    assert str(f) == "lshift()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 << rnd2
    assert str(f) == "lshift()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_lshift_mutate(chain):
    res = chain.mutate(test=strlen("val") << 2).order_by("num").to_values("test")
    assert res == [4, 8, 12, 16, 20]

    res = chain.mutate(test=2 << strlen("val")).order_by("num").to_values("test")
    assert res == [4, 8, 16, 32, 64]

    res = (
        chain.mutate(test=strlen("val") << strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [2, 8, 24, 64, 160]


def test_lt():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 < 1
    assert str(f) == "lt()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 > 1
    assert str(f) == "gt()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 < rnd2
    assert str(f) == "lt()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_lt_mutate(chain):
    res = chain.mutate(test=strlen("val") < 3).order_by("num").to_values("test")
    assert res == [1, 1, 0, 0, 0]

    res = chain.mutate(test=strlen("val") > 3).order_by("num").to_values("test")
    assert res == [0, 0, 0, 1, 1]

    res = (
        chain.mutate(test=strlen("val") < strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("value", [1, 0.5, "a", True])
def test_mutate_with_literal(chain, value):
    res = chain.mutate(test=value).to_values("test")
    assert res == [value] * 5


def test_le():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 <= 1
    assert str(f) == "le()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 >= 1
    assert str(f) == "ge()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 <= rnd2
    assert str(f) == "le()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_le_mutate(chain):
    res = chain.mutate(test=strlen("val") <= 3).order_by("num").to_values("test")
    assert res == [1, 1, 1, 0, 0]

    res = chain.mutate(test=strlen("val") >= 3).order_by("num").to_values("test")
    assert res == [0, 0, 1, 1, 1]

    res = (
        chain.mutate(test=strlen("val") <= strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [1, 1, 1, 1, 1]


def test_eq():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 == 1
    assert str(f) == "eq()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 == 1
    assert str(f) == "eq()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 == rnd2
    assert str(f) == "eq()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_eq_mutate(chain):
    res = chain.mutate(test=strlen("val") == 2).order_by("num").to_values("test")
    assert res == [0, 1, 0, 0, 0]

    res = chain.mutate(test=strlen("val") == 4).order_by("num").to_values("test")
    assert res == [0, 0, 0, 1, 0]

    res = (
        chain.mutate(test=strlen("val") == strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [1, 1, 1, 1, 1]


def test_ne():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 != 1
    assert str(f) == "ne()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 != 1
    assert str(f) == "ne()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 != rnd2
    assert str(f) == "ne()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_ne_mutate(chain):
    res = chain.mutate(test=strlen("val") != 2).order_by("num").to_values("test")
    assert res == [1, 0, 1, 1, 1]

    res = chain.mutate(test=strlen("val") != 4).order_by("num").to_values("test")
    assert res == [1, 1, 1, 0, 1]

    res = (
        chain.mutate(test=strlen("val") != strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [0, 0, 0, 0, 0]


def test_gt():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 > 1
    assert str(f) == "gt()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 < 1
    assert str(f) == "lt()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 > rnd2
    assert str(f) == "gt()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_gt_mutate(chain):
    res = chain.mutate(test=strlen("val") > 2).order_by("num").to_values("test")
    assert res == [0, 0, 1, 1, 1]

    res = chain.mutate(test=strlen("val") < 4).order_by("num").to_values("test")
    assert res == [1, 1, 1, 0, 0]

    res = (
        chain.mutate(test=strlen("val") > strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [0, 0, 0, 0, 0]


def test_ge():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 >= 1
    assert str(f) == "ge()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 <= 1
    assert str(f) == "le()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 >= rnd2
    assert str(f) == "ge()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_ge_mutate(chain):
    res = chain.mutate(test=strlen("val") >= 2).order_by("num").to_values("test")
    assert res == [0, 1, 1, 1, 1]

    res = chain.mutate(test=strlen("val") <= 4).order_by("num").to_values("test")
    assert res == [1, 1, 1, 1, 0]

    res = (
        chain.mutate(test=strlen("val") >= strlen("val"))
        .order_by("num")
        .to_values("test")
    )
    assert res == [1, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "value,inthash",
    [
        [0, 4761183170873013810],
        [1, 10577349846663553072 - (1 << 64)],
        [5, 15228578409069794350 - (1 << 64)],
        [123456, 13379111408315310133 - (1 << 64)],
    ],
)
def test_sqlite_int_hash_64(value, inthash):
    assert sqlite_int_hash_64(value) == inthash


def test_int_hash_64_mutate(chain):
    res = (
        chain.mutate(test=int_hash_64(strlen("val"))).order_by("num").to_values("test")
    )
    assert [x & (2**64 - 1) for x in res] == [
        10577349846663553072,
        18198135717204167749,
        9624464864560415994,
        7766709361750702608,
        15228578409069794350,
    ]


@pytest.mark.parametrize(
    "value1,value2,distance",
    [
        [0, 0, 0],
        [0, 1, 1],
        [2, 3, 1],
        [2, 4, 2],
        [0, 2**64 - 1, 64],
        [-(2**63), 2**63 - 1, 64],
        [-(2**63), 2**63, 0],
    ],
)
def test_sqlite_bit_hamming_distance(value1, value2, distance):
    assert sqlite_bit_hamming_distance(value1, value2) == distance


def test_bit_hamming_distance_mutate(chain):
    res = (
        chain.mutate(test=bit_hamming_distance(strlen("val"), 5))
        .order_by("num")
        .to_values("test")
    )
    assert res == [1, 3, 2, 1, 0]


@pytest.mark.parametrize(
    "value1,value2,distance",
    [
        ["", "", 0],
        ["", "a", 1],
        ["foo", "foo", 0],
        ["foo", "bar", 3],
        ["foo", "foobar", 3],
        ["karolin", "kathrin", 3],
    ],
)
def test_sqlite_byte_hamming_distance(value1, value2, distance):
    assert sqlite_byte_hamming_distance(value1, value2) == distance


def test_byte_hamming_distance_mutate(chain):
    res = (
        chain.mutate(test=byte_hamming_distance("val", literal("xxx")))
        .order_by("num")
        .to_values("test")
    )
    assert res == [2, 1, 0, 1, 2]


@pytest.mark.parametrize(
    "val,else_,type_",
    [
        ["A", "D", str],
        [1, 2, int],
        [1.5, 2.5, float],
        [True, False, bool],
    ],
)
def test_case_mutate(chain, val, else_, type_):
    res = chain.mutate(test=case((dc.C("num") < 2, val), else_=else_))
    assert list(res.order_by("test").to_values("test")) == sorted(
        [val, else_, else_, else_, else_]
    )
    assert res.schema["test"] == type_


def test_case_mutate_column_as_value(chain):
    res = chain.mutate(test=case((dc.C("num") < 3, dc.C("val")), else_="cc"))
    assert list(res.order_by("num").to_values("test")) == ["x", "xx", "cc", "cc", "cc"]


def test_case_mutate_column_as_value_in_else(chain):
    res = chain.mutate(test=case((dc.C("num") < 3, dc.C("val")), else_=dc.C("val")))
    assert list(res.order_by("num").to_values("test")) == [
        "x",
        "xx",
        "xxx",
        "xxxx",
        "xxxxx",
    ]
    assert res.schema["test"] is str


@pytest.mark.parametrize(
    "val,else_,type_",
    [
        ["A", "D", str],
        [1, 2, int],
        [1.5, 2.5, float],
        [True, False, bool],
    ],
)
def test_nested_case_on_condition_mutate(chain, val, else_, type_):
    res = chain.mutate(
        test=case((case((dc.C("num") < 2, True), else_=False), val), else_=else_)
    )
    assert list(res.order_by("test").to_values("test")) == sorted(
        [val, else_, else_, else_, else_]
    )
    assert res.schema["test"] == type_


@pytest.mark.parametrize(
    "v1,v2,v3,type_",
    [
        ["A", "B", "C", str],
        [1, 2, 3, int],
        [1.5, 2.5, 3.5, float],
        [False, True, True, bool],
    ],
)
def test_nested_case_on_value_mutate(chain, v1, v2, v3, type_):
    res = chain.mutate(
        test=case((dc.C("num") < 4, case((dc.C("num") < 2, v1), else_=v2)), else_=v3)
    )
    assert list(res.order_by("num").to_values("test")) == sorted([v1, v2, v2, v3, v3])
    assert res.schema["test"] == type_


@pytest.mark.parametrize(
    "v1,v2,v3,type_",
    [
        ["A", "B", "C", str],
        [1, 2, 3, int],
        [1.5, 2.5, 3.5, float],
        [False, True, True, bool],
    ],
)
def test_nested_case_on_else_mutate(chain, v1, v2, v3, type_):
    res = chain.mutate(
        test=case((dc.C("num") < 3, v1), else_=case((dc.C("num") < 4, v2), else_=v3))
    )
    assert list(res.order_by("num").to_values("test")) == sorted([v1, v1, v2, v3, v3])
    assert res.schema["test"] == type_


@pytest.mark.parametrize(
    "if_val,else_val,type_",
    [
        ["A", "D", str],
        [1, 2, int],
        [1.5, 2.5, float],
        [True, False, bool],
    ],
)
def test_ifelse_mutate(chain, if_val, else_val, type_):
    res = chain.mutate(test=ifelse(dc.C("num") < 2, if_val, else_val))
    assert list(res.order_by("test").to_values("test")) == sorted(
        [if_val, else_val, else_val, else_val, else_val]
    )
    assert res.schema["test"] == type_


@pytest.mark.parametrize(
    "if_val,else_val,type_,result",
    [
        [dc.C("num"), 0, int, [0, 0, 0, 1, 2]],
        ["a", dc.C("val"), str, ["a", "a", "xxx", "xxxx", "xxxxx"]],
    ],
)
def test_ifelse_mutate_with_columns_as_values(chain, if_val, else_val, type_, result):
    res = chain.mutate(test=ifelse(dc.C("num") < 3, if_val, else_val))
    assert list(res.order_by("test").to_values("test")) == result
    assert res.schema["test"] == type_


@pytest.mark.parametrize("col", ["val", dc.C("val")])
@skip_if_not_sqlite
def test_isnone_mutate(col):
    chain = dc.read_values(
        num=list(range(1, 6)),
        val=[None if i > 3 else "A" for i in range(1, 6)],
    )

    res = chain.mutate(test=isnone(col))
    assert list(res.order_by("test").to_values("test")) == sorted(
        [False, False, False, True, True]
    )
    assert res.schema["test"] is bool


@pytest.mark.parametrize("col", [dc.C("val"), "val"])
@skip_if_not_sqlite
def test_isnone_with_ifelse_mutate(col):
    chain = dc.read_values(
        num=list(range(1, 6)),
        val=[None if i > 3 else "A" for i in range(1, 6)],
    )

    res = chain.mutate(test=ifelse(isnone(col), "NONE", "NOT_NONE"))
    assert (
        list(res.order_by("num").to_values("test")) == ["NOT_NONE"] * 3 + ["NONE"] * 2
    )
    assert res.schema["test"] is str


def test_array_contains():
    chain = dc.read_values(
        arr=[list(range(1, i)) * i for i in range(2, 7)],
        val=list(range(2, 7)),
    )

    assert list(
        chain.mutate(res=contains("arr", 3)).order_by("val").to_values("res")
    ) == [
        0,
        0,
        1,
        1,
        1,
    ]
    assert list(
        chain.mutate(res=contains(dc.C("arr"), 3)).order_by("val").to_values("res")
    ) == [0, 0, 1, 1, 1]
    assert list(
        chain.mutate(res=contains(dc.C("arr"), 10)).order_by("val").to_values("res")
    ) == [0, 0, 0, 0, 0]
    assert list(
        chain.mutate(res=contains(dc.C("arr"), None)).order_by("val").to_values("res")
    ) == [0, 0, 0, 0, 0]
