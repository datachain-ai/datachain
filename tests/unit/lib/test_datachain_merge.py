import math

import pandas as pd
import pytest
from pydantic import BaseModel
from sqlalchemy import func

import datachain as dc
from datachain import DataModel
from datachain.lib.dc import C, DatasetMergeError
from datachain.lib.signal_schema import SignalResolvingError
from datachain.sql.types import Int, String
from tests.utils import skip_if_not_sqlite


class User(BaseModel):
    name: str | None = None
    age: int | None = None


class Player(User):
    weight: float | None = None
    height: int | None = None


class Employee(BaseModel):
    id: int | None = None
    person: User


class TeamMember(BaseModel):
    player: str | None = None
    sport: str | None = None
    weight: float | None = None
    height: float | None = None


employees = [
    Employee(id=151, person=User(name="Alice", age=31)),
    Employee(id=152, person=User(name="Bob", age=27)),
    Employee(id=153, person=User(name="Charlie", age=54)),
    Employee(id=154, person=User(name="David", age=29)),
]
team = [
    TeamMember(player="Alice", sport="volleyball", weight=120.3, height=5.5),
    TeamMember(player="Charlie", sport="football", weight=200.0, height=6.0),
    TeamMember(player="David", sport="football", weight=158.7, height=5.7),
    TeamMember(player="John", sport="basketball", weight=250.3, height=7.0),
]


def test_merge_objects(test_session):
    ch1 = dc.read_values(emp=employees, session=test_session)
    ch2 = dc.read_values(team=team, session=test_session)
    ch = ch1.merge(ch2, "emp.person.name", "team.player")

    str_default = String.default_value(test_session.catalog.warehouse.db.dialect)

    i = 0
    j = 0
    for items in ch.order_by("emp.person.name", "team.player"):
        assert len(items) == 2

        empl, player = items
        assert isinstance(empl, Employee)
        assert empl == employees[i]
        i += 1

        assert isinstance(player, TeamMember)
        if empl.person.name != "Bob":
            assert player.player == team[j].player
            assert player.sport == team[j].sport
            assert math.isclose(player.weight, team[j].weight, rel_tol=1e-7)
            assert math.isclose(player.height, team[j].height, rel_tol=1e-7)
            j += 1
        else:
            assert player.player == str_default
            assert player.sport == str_default
            assert pd.isnull(player.weight)
            assert pd.isnull(player.height)

    assert i == len(employees)
    assert j == len(team) - 1


@pytest.mark.parametrize("multiple_predicates", [True, False])
def test_merge_objects_full_join(test_session, multiple_predicates):
    ch1 = dc.read_values(emp=employees, session=test_session)
    ch2 = dc.read_values(team=team, session=test_session)
    if multiple_predicates:
        ch = ch1.merge(
            ch2,
            ["emp.person.name", "emp.person.name"],
            ["team.player", "team.player"],
            full=True,
        )
    else:
        ch = ch1.merge(ch2, "emp.person.name", "team.player", full=True)

    str_default = String.default_value(test_session.catalog.warehouse.db.dialect)
    int_default = Int.default_value(test_session.catalog.warehouse.db.dialect)

    i = 0
    for items in ch.order_by("emp.person.name", "team.player"):
        assert len(items) == 2

        empl, player = items
        assert isinstance(empl, Employee)
        assert isinstance(player, TeamMember)

        if player.player == "John":
            assert empl.person.name == str_default
            assert empl.person.age == int_default
            continue

        if empl.person.name == "Bob":
            assert player.player == str_default
            assert player.sport == str_default
            assert pd.isnull(player.weight)
            assert pd.isnull(player.height)
            continue

        assert player.player == team[i].player
        assert player.sport == team[i].sport
        assert math.isclose(player.weight, team[i].weight, rel_tol=1e-7)
        assert math.isclose(player.height, team[i].height, rel_tol=1e-7)
        i += 1

    assert i == len(employees) - 1 == len(team) - 1


def test_merge_similar_objects(test_session):
    new_employees = [
        Employee(id=152, person=User(name="Bob", age=27)),
        Employee(id=201, person=User(name="Karl", age=18)),
        Employee(id=154, person=User(name="David", age=29)),
    ]

    ch1 = dc.read_values(emp=employees, session=test_session)
    ch2 = dc.read_values(emp=new_employees, session=test_session)

    rname = "qq"
    ch = ch1.merge(ch2, "emp.person.name", rname=rname)

    assert list(ch.signals_schema.values.keys()) == ["emp", rname + "emp"]

    empl = list(ch.to_list())
    assert len(empl) == 4
    assert len(empl[0]) == 2

    ch_inner = ch1.merge(ch2, "emp.person.name", rname=rname, inner=True)
    assert len(list(ch_inner.to_list())) == 2


@skip_if_not_sqlite
def test_merge_similar_objects_in_memory():
    # Skip if not on SQLite, as in_memory databases are only supported on SQLite
    new_employees = [
        Employee(id=152, person=User(name="Bob", age=27)),
        Employee(id=201, person=User(name="Karl", age=18)),
        Employee(id=154, person=User(name="David", age=29)),
    ]

    ch1 = dc.read_values(emp=employees, in_memory=True)
    # This should use the same session as above (in_memory=True automatically)
    ch2 = dc.read_values(emp=new_employees)
    assert ch1.session.catalog.in_memory is True
    assert ch1.session.catalog.metastore.db.db_file == ":memory:"
    assert ch1.session.catalog.warehouse.db.db_file == ":memory:"
    assert ch2.session.catalog.in_memory is True
    assert ch2.session.catalog.metastore.db.db_file == ":memory:"
    assert ch2.session.catalog.warehouse.db.db_file == ":memory:"

    rname = "qq"
    ch = ch1.merge(ch2, "emp.person.name", rname=rname)
    assert ch.session.catalog.in_memory is True
    assert ch.session.catalog.metastore.db.db_file == ":memory:"
    assert ch.session.catalog.warehouse.db.db_file == ":memory:"

    assert list(ch.signals_schema.values.keys()) == ["emp", rname + "emp"]

    empl = list(ch.to_list())
    assert len(empl) == 4
    assert len(empl[0]) == 2

    ch_inner = ch1.merge(ch2, "emp.person.name", rname=rname, inner=True)
    assert len(list(ch_inner.to_list())) == 2


def test_merge_values(test_session):
    order_ids = [11, 22, 33, 44]
    order_descr = ["water", "water", "paper", "water"]

    delivery_ids = [11, 44]
    delivery_time = [24.0, 16.5]

    ch1 = dc.read_values(id=order_ids, descr=order_descr, session=test_session)
    ch2 = dc.read_values(id=delivery_ids, time=delivery_time, session=test_session)

    ch = ch1.merge(ch2, "id")

    assert list(ch.signals_schema.values.keys()) == [
        "id",
        "descr",
        "right_id",
        "time",
    ]

    i = 0
    j = 0
    sorted_items_list = sorted(ch.to_list(), key=lambda x: x[0])
    for items in sorted_items_list:
        assert len(items) == 4
        id, name, _right_id, time = items

        assert id == order_ids[i]
        assert name == order_descr[i]
        i += 1

        if pd.notnull(time):
            assert id == delivery_ids[j]
            assert time == delivery_time[j]
            j += 1

    assert i == len(order_ids)
    assert j == len(delivery_ids)


def test_merge_nested_key_without_collision_matches_schema(test_session):
    class Item(DataModel):
        score: float

    class Metadata(DataModel):
        score: float

    left = dc.read_values(
        id=[1, 2],
        item=[Item(score=0.5), Item(score=0.7)],
        session=test_session,
    )
    right = dc.read_values(
        id=[1, 2],
        metadata=[Metadata(score=0.3), Metadata(score=0.4)],
        session=test_session,
    )

    merged = left.merge(right, "id", rname="right_")

    schema = merged.signals_schema
    resolved = schema.resolve("id", "item.score", "metadata.score", "right_id")

    assert resolved.values["id"] is int
    assert resolved.values["item.score"] is float
    assert resolved.values["metadata.score"] is float
    assert resolved.values["right_id"] is int

    with pytest.raises(SignalResolvingError):
        schema.resolve("right_metadata.score")

    rows = (
        merged.select("id", "item.score", "right_id", "metadata.score")
        .order_by("id")
        .results()
    )
    assert rows == [(1, 0.5, 1, 0.3), (2, 0.7, 2, 0.4)]


def test_merge_prefixed_root_suffix_matches_schema(test_session):
    class Item(DataModel):
        score: float

    class RightItem(DataModel):
        score: float

    class ItemMetrics(DataModel):
        score: int
        confidence: int

    left = dc.read_values(
        id=[1, 2],
        item=[Item(score=0.5), Item(score=0.7)],
        right_item=[RightItem(score=0.9), RightItem(score=0.95)],
        session=test_session,
    )
    right = dc.read_values(
        id=[1, 2],
        item=[ItemMetrics(score=3, confidence=30), ItemMetrics(score=4, confidence=40)],
        session=test_session,
    )

    merged = left.merge(right, "id", rname="right_")

    schema = merged.signals_schema
    resolved = schema.resolve(
        "id",
        "item.score",
        "right_item.score",
        "right_item_1.score",
        "right_item_1.confidence",
        "right_id",
    )

    assert resolved.values["id"] is int
    assert resolved.values["item.score"] is float
    assert resolved.values["right_item.score"] is float
    assert resolved.values["right_item_1.score"] is int
    assert resolved.values["right_item_1.confidence"] is int
    assert resolved.values["right_id"] is int

    rows = (
        merged.select(
            "id",
            "item.score",
            "right_item.score",
            "right_item_1.score",
            "right_item_1.confidence",
            "right_id",
        )
        .order_by("id")
        .results()
    )
    assert rows == [
        (1, 0.5, 0.9, 3, 30, 1),
        (2, 0.7, 0.95, 4, 40, 2),
    ]


def test_merge_rename_collides_with_existing_column_matches_schema(test_session):
    class ItemFloat(DataModel):
        score: float

    class ItemInt(DataModel):
        score: int

    class RightItemStr(DataModel):
        score: str

    left = dc.read_values(
        id=[1, 2],
        item=[ItemFloat(score=0.5), ItemFloat(score=0.7)],
        session=test_session,
    )
    right = dc.read_values(
        id=[1, 2],
        item=[ItemInt(score=5), ItemInt(score=6)],
        right_item=[RightItemStr(score="high"), RightItemStr(score="low")],
        session=test_session,
    )

    merged = left.merge(right, "id", rname="right_")

    schema = merged.signals_schema
    resolved = schema.resolve(
        "id",
        "item.score",
        "right_item.score",
        "right_item_1.score",
        "right_id",
    )

    assert resolved.values["id"] is int
    assert resolved.values["item.score"] is float
    assert resolved.values["right_item.score"] is str
    assert resolved.values["right_item_1.score"] is int
    assert resolved.values["right_id"] is int

    rows = (
        merged.select(
            "id",
            "item.score",
            "right_item.score",
            "right_item_1.score",
            "right_id",
        )
        .order_by("id")
        .results()
    )
    assert rows == [
        (1, 0.5, "high", 5, 1),
        (2, 0.7, "low", 6, 2),
    ]


def test_merge_multi_conditions(test_session):
    order_ids = [11, 22, 33, 44]
    order_name = ["water", "water", "paper", "water"]
    order_descr = ["still water", "still water", "white paper", "sparkling water"]

    delivery_ids = [11, 44]
    delivery_name = ["water", "unknown"]
    delivery_time = [24.0, 16.5]

    ch1 = dc.read_values(
        id=order_ids, name=order_name, descr=order_descr, session=test_session
    )
    ch2 = dc.read_values(
        id=delivery_ids, d_name=delivery_name, time=delivery_time, session=test_session
    )

    ch = ch1.merge(ch2, ("id", "name"), ("id", C("d_name")))

    res = list(ch.to_list())

    assert len(res) == max(len(employees), len(team))
    success_ids = set()
    for items in res:
        if items[3]:
            success_ids.add(items[0])

    assert success_ids == {11}


def test_merge_errors(test_session):
    ch1 = dc.read_values(emp=employees, session=test_session)
    ch2 = dc.read_values(team=team, session=test_session)

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, "unknown")

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, ["emp.person.name"], "unknown")

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, ["emp.person.name"], ["unknown"])

    with pytest.raises(DatasetMergeError):
        ch1.merge(
            ch2, ("emp.person.age", func.substr(["emp.person.name"], 2)), "unknown"
        )

    ch1.merge(ch2, ["emp.person.name"], ["team.sport"])
    ch1.merge(ch2, ["emp.person.name"], ["team.sport"])

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, ["emp.person.name"], ["team.player", "team.sport"])

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, 33)

    with pytest.raises(DatasetMergeError):
        ch1.merge(ch2, "emp.person.name", True)


def test_merge_with_itself(test_session):
    ch = dc.read_values(emp=employees, session=test_session)
    merged = ch.merge(ch, "emp.id")

    count = 0
    for left, right in merged.order_by("emp.id"):
        assert isinstance(left, Employee)
        assert isinstance(right, Employee)
        assert left == right == employees[count]
        count += 1

    assert count == len(employees)


def test_merge_with_itself_column(test_session):
    ch = dc.read_values(emp=employees, session=test_session)
    merged = ch.merge(ch, C("emp.id"))

    count = 0
    for left, right in merged.order_by("emp.id"):
        assert isinstance(left, Employee)
        assert isinstance(right, Employee)
        assert left == right == employees[count]
        count += 1

    assert count == len(employees)


def test_merge_on_expression(test_session):
    def _get_expr(chain):
        c = chain.c("team.sport")
        return func.substr(c, func.length(c) - 3)

    chain = dc.read_values(team=team, session=test_session)
    right_dc = chain.clone()

    # cross join on "ball" from sport
    merged = chain.merge(right_dc, on=_get_expr(chain), right_on=_get_expr(right_dc))

    cross_team = [
        (left_member, right_member) for left_member in team for right_member in team
    ]

    merged.show()
    count = 0
    for left, right_dc in merged.order_by("team.player", "right_team.player"):
        assert isinstance(left, TeamMember)
        assert isinstance(right_dc, TeamMember)
        left_member, right_member = cross_team[count]
        assert left == left_member
        assert right_dc == right_member
        count += 1

    assert count == len(team) * len(team)


def test_merge_with_drops_sys_columns(test_session):
    left = dc.read_values(id=[1, 1], lval=[10, 20], session=test_session)
    right = dc.read_values(id=[1, 1], rval=["a", "b"], session=test_session)

    merged = left.merge(right, on="id")

    assert "sys" not in merged.signals_schema.values

    cols = merged.settings(sys=True).to_pandas(flatten=True).columns
    assert all(not str(col).startswith("sys") for col in cols)

    ds_name = "merge_left_dups_sys_check_sys"
    merged.save(ds_name)

    df_with_sys = (
        dc.read_dataset(ds_name, session=test_session)
        .settings(sys=True)
        .to_pandas(flatten=True)
    )

    sys_cols = [c for c in df_with_sys.columns if str(c).startswith("sys")]
    assert sys_cols

    def _col(name: str) -> str:
        for col in df_with_sys.columns:
            if str(col) == f"sys.{name}":
                return str(col)
        raise AssertionError(f"Missing sys column for {name}")

    sys_id_col = _col("id")
    sys_rand_col = _col("rand")

    sys_ids = list(df_with_sys[sys_id_col])
    assert len(sys_ids) == len(set(sys_ids))

    sys_rand = list(df_with_sys[sys_rand_col])
    assert len(sys_rand) == len(set(sys_rand))
