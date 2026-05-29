from typing import ClassVar, Literal, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from datachain import DataModel
from datachain.lib.convert.flatten import flatten, flatten_list
from datachain.lib.convert.unflatten import unflatten, unflatten_to_json
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema
from datachain.sql.types import Int64, String


class FileBasic(DataModel):
    parent: str = Field(default="")
    name: str
    size: int = Field(default=0)


class FileInfo(FileBasic):
    location: dict = Field(default={})


class FileInfoEx(DataModel):
    f_info: FileInfo
    type_id: int


class MyNestedClass(DataModel):
    type: int
    Name: str = Field(default="test1")


class MyTest(DataModel):
    ThisIsName: str
    subClass: MyNestedClass  # noqa: N815


def test_flatten_basic():
    vals = flatten(FileBasic(parent="hello", name="world", size=123))
    assert vals == ("hello", "world", 123)


def test_flatten_with_json():
    t1 = FileInfo(parent="prt4", name="test1", size=42, location={"ee": "rr"})
    assert flatten(t1) == ("prt4", "test1", 42, {"ee": "rr"})


def test_flatten_with_empty_json():
    with pytest.raises(ValidationError):
        FileInfo(parent="prt4", name="test1", size=42, location=None)


def test_flatten_with_accepted_empty_json():
    class _Test(DataModel):
        d: dict | None

    assert flatten(_Test(d=None)) == (None,)


def test_flatten_nested():
    t0 = FileInfo(parent="sfo", name="sf", size=567, location={"42": 999})
    t1 = FileInfoEx(f_info=t0, type_id=1849)

    assert flatten(t1) == ("sfo", "sf", 567, {"42": 999}, 1849)


def test_flatten_list_field():
    class Trace(BaseModel):
        x: int
        y: int

    class Nested(BaseModel):
        traces: list[Trace]

    n = Nested(traces=[Trace(x=1, y=1), Trace(x=2, y=2)])
    assert flatten(n) == ([{"x": 1, "y": 1}, {"x": 2, "y": 2}],)


def test_flatten_nested_list_field():
    class Trace(BaseModel):
        x: int
        y: int

    class Nested(BaseModel):
        traces: list[list[Trace]]

    n = Nested(traces=[[Trace(x=1, y=1)], [Trace(x=2, y=2)]])
    assert flatten(n) == ([[{"x": 1, "y": 1}], [{"x": 2, "y": 2}]],)


def test_flatten_multiple_nested_list_field():
    class Trace(BaseModel):
        x: int
        y: int

    class Nested(BaseModel):
        traces: list[list[list[Trace]]]

    n = Nested(
        traces=[
            [[Trace(x=1, y=1)], [Trace(x=2, y=2)]],
            [[Trace(x=3, y=3)], [Trace(x=4, y=4)]],
        ]
    )

    assert flatten(n) == (
        [
            [[{"x": 1, "y": 1}], [{"x": 2, "y": 2}]],
            [[{"x": 3, "y": 3}], [{"x": 4, "y": 4}]],
        ],
    )


def test_flatten_list():
    t1 = FileInfo(parent="p1", name="n4", size=3, location={"a": "b"})
    t2 = FileInfo(parent="p2", name="n5", size=2, location={"c": "d"})

    vals = flatten_list([t1, t2])
    assert vals == ("p1", "n4", 3, {"a": "b"}, "p2", "n5", 2, {"c": "d"})


def test_registry():
    class MyTestRndmz(DataModel):
        name: str
        count: int

    ModelStore.register(MyTestRndmz)
    assert ModelStore.get(MyTestRndmz.__name__) == MyTestRndmz
    assert ModelStore.get(MyTestRndmz.__name__, version=1) == MyTestRndmz
    ModelStore.remove(MyTestRndmz)


def test_registry_versioned():
    class MyTestXYZ(DataModel):
        _version: ClassVar[int] = 42
        name: str
        count: int

    assert ModelStore.get(MyTestXYZ.__name__) == MyTestXYZ
    assert ModelStore.get(MyTestXYZ.__name__, version=1) is None
    assert ModelStore.get(MyTestXYZ.__name__, version=42) == MyTestXYZ
    ModelStore.remove(MyTestXYZ)


def test_model_store_rebuild_all_recursive():
    class Node(DataModel):
        value: int = 0
        child: Optional["Node"] = None

    try:
        ModelStore.rebuild_all()
        root = Node(value=1, child=Node(value=2))
        assert root.child and root.child.value == 2
    finally:
        ModelStore.remove(Node)


def test_inheritance():
    class SubObject(DataModel):
        subname: str

    class SoMyTest1(DataModel):
        name: str
        sub: SubObject

    class SoMyTest2(SoMyTest1):
        pass

    try:
        with pytest.raises(ValueError):
            SoMyTest2()

        obj = SoMyTest2(name="name", sub=SubObject(subname="subname"))
        assert flatten(obj) == ("name", "subname")
    finally:
        ModelStore.remove(SubObject)
        ModelStore.remove(SoMyTest1)
        ModelStore.remove(SoMyTest2)


def test_deserialize_nested():
    class Child(DataModel):
        type: int
        name: str = Field(default="test1")

    class Parent(DataModel):
        name: str
        child: Child

    in_db_map = {
        "name": "a1",
        "child__type": 42,
        "child__name": "a2",
    }

    p = unflatten(Parent, in_db_map)

    assert p.name == "a1"
    assert p.child.type == 42
    assert p.child.name == "a2"


def test_deserialize_nested_with_name_normalization():
    class ChildClass(DataModel):
        type: int
        name: str = Field(default="test1")

    class Parent2(DataModel):
        name: str
        childClass11: ChildClass  # noqa: N815

    in_db_map = {
        "name": "name1",
        "child_class11__type": 12,
        "child_class11__name": "n2",
    }

    p = unflatten(Parent2, in_db_map)

    assert p.name == "name1"
    assert p.childClass11.type == 12
    assert p.childClass11.name == "n2"


def test_type_array_of_floats():
    class _Test(DataModel):
        d: list[float]

    dict_ = {"d": [1, 3, 5]}
    t = _Test(**dict_)
    assert t.d == [1, 3, 5]


def test_unflatten_to_json():
    class _Child(DataModel):
        type: int
        name: str = Field(default="test1")

    class _Parent(DataModel):
        name: str
        child: _Child

    p = _Parent(name="parent1", child=_Child(type=12, name="child1"))

    flt = flatten(p)
    assert unflatten_to_json(_Parent, flt) == {
        "name": "parent1",
        "child": {"type": 12, "name": "child1"},
    }


def test_unflatten_to_json_list():
    class _Child(DataModel):
        type: int
        name: str = Field(default="test1")

    class _Parent(DataModel):
        name: str
        children: list[_Child]

    p = _Parent(
        name="parent1",
        children=[_Child(type=12, name="child1"), _Child(type=13, name="child2")],
    )

    flt = flatten(p)
    json = unflatten_to_json(_Parent, flt)
    assert json == {
        "name": "parent1",
        "children": [{"type": 12, "name": "child1"}, {"type": 13, "name": "child2"}],
    }


def test_unflatten_to_json_dict():
    class _Child(DataModel):
        type: int
        address: str = Field(default="test1")

    class _Parent(DataModel):
        name: str
        children: dict[str, _Child]

    p = _Parent(
        name="parent1",
        children={
            "child1": _Child(type=12, address="sf"),
            "child2": _Child(type=13, address="nyc"),
        },
    )

    flt = flatten(p)
    json = unflatten_to_json(_Parent, flt)
    assert json == {
        "name": "parent1",
        "children": {
            "child1": {"type": 12, "address": "sf"},
            "child2": {"type": 13, "address": "nyc"},
        },
    }


def test_unflatten_to_json_list_of_int():
    class _Child(DataModel):
        types: list[int]
        name: str = Field(default="test1")

    child1 = _Child(name="n1", types=[14])
    assert unflatten_to_json(_Child, flatten(child1)) == {
        "name": "n1",
        "types": [14],
    }

    child2 = _Child(name="qwe", types=[1, 2, 3, 5])
    assert unflatten_to_json(_Child, flatten(child2)) == {
        "name": "qwe",
        "types": [1, 2, 3, 5],
    }


def test_unflatten_to_json_list_of_lists():
    class _Child(DataModel):
        type: int
        name: str = Field(default="test1")

    class _Parent(DataModel):
        name: str
        children: list[_Child]

    class _Company(DataModel):
        name: str
        parents: list[_Parent]

    p = _Company(
        name="Co",
        parents=[_Parent(name="parent1", children=[_Child(type=12, name="child1")])],
    )

    assert unflatten_to_json(_Company, flatten(p)) == {
        "name": "Co",
        "parents": [{"name": "parent1", "children": [{"type": 12, "name": "child1"}]}],
    }


def test_version():
    class _MyCls(BaseModel):
        name: str
        age: int
        _version: ClassVar[int] = 23

    assert ModelStore.get_version(_MyCls) == 23


def test_dict_to_feature():
    data_dict = {"file": FileBasic, "id": int, "type": Literal["text"]}

    cls = dict_to_data_model("val", data_dict)
    assert ModelStore.is_pydantic(cls)

    spec = SignalSchema({"val": cls}).to_udf_spec()
    assert list(spec.keys()) == [
        "val__file__parent",
        "val__file__name",
        "val__file__size",
        "val__id",
        "val__type",
    ]
    # Optional scalar leaves yield a marked type instance (dc_nullable); others
    # stay as the type class. Normalize to the class to compare the SQL types.
    assert [v if isinstance(v, type) else type(v) for v in spec.values()] == [
        String,
        String,
        Int64,
        Int64,
        String,
    ]


# ---------------------------------------------------------------------------
# Optional[DataModel] round-trip via sentinel column.
# ---------------------------------------------------------------------------


class _Addr(DataModel):
    city: str = ""
    zip: str = ""


class _Outer(DataModel):
    name: str
    addr: _Addr | None = None


def test_flatten_optional_datamodel_present():
    out = _Outer(name="Alice", addr=_Addr(city="Berlin", zip="10115"))
    assert flatten(out) == ("Alice", False, "Berlin", "10115")


def test_flatten_optional_datamodel_absent():
    out = _Outer(name="Bob", addr=None)
    assert flatten(out) == ("Bob", True, None, None)


def test_unflatten_optional_datamodel_present():
    row = ("Alice", False, "Berlin", "10115")
    j = unflatten_to_json(_Outer, row)
    assert j == {
        "name": "Alice",
        "addr": {"city": "Berlin", "zip": "10115"},
    }
    obj = _Outer(**j)
    assert obj.addr == _Addr(city="Berlin", zip="10115")


def test_unflatten_optional_datamodel_absent():
    row = ("Bob", True, None, None)
    j = unflatten_to_json(_Outer, row)
    assert j == {"name": "Bob", "addr": None}
    assert _Outer(**j).addr is None


def test_unflatten_ignores_leaf_garbage_when_sentinel_true():
    # sentinel=True hydrates the parent as None even when the leaves hold values.
    row = ("Bob", True, "garbage-city", "garbage-zip")
    j = unflatten_to_json(_Outer, row)
    assert j == {"name": "Bob", "addr": None}


def test_signal_schema_emits_sentinel_column_for_optional_datamodel():
    schema = SignalSchema({"out": _Outer})
    flat = list(schema.get_flat_tree())
    leaf_names = [".".join(p) for p, _, has_subtree, _ in flat if not has_subtree]
    assert "out.name" in leaf_names
    assert "out.addr.is_null" in leaf_names
    assert "out.addr.city" in leaf_names
    assert "out.addr.zip" in leaf_names


def test_to_udf_spec_includes_sentinel_for_optional_datamodel():
    spec = SignalSchema({"out": _Outer}).to_udf_spec()
    assert "out__addr__is_null" in spec
    assert "out__addr__city" in spec


def test_optional_datamodel_roundtrip_at_top_level():
    schema = SignalSchema({"item": Optional[_Addr]})
    flat = [
        ".".join(p)
        for p, _, has_subtree, _ in schema.get_flat_tree()
        if not has_subtree
    ]
    assert "item.is_null" in flat
    assert "item.city" in flat
    # row_to_objs with sentinel=True returns None for top-level Optional[Model].
    assert schema.row_to_objs((True, None, None)) == [None]
    assert schema.row_to_objs((False, "Paris", "75001")) == [
        _Addr(city="Paris", zip="75001")
    ]


def test_nested_optional_datamodel_roundtrip():
    class Mid(DataModel):
        addr: _Addr | None = None
        tag: str = ""

    class Top(DataModel):
        mid: Mid | None = None

    # Outer present, inner absent
    obj = Top(mid=Mid(addr=None, tag="x"))
    flat = flatten(obj)
    # mid_is_null=False, mid_addr_is_null=True, addr_city=None, addr_zip=None, tag="x"
    assert flat == (False, True, None, None, "x")
    rec = unflatten_to_json(Top, flat)
    assert rec == {"mid": {"addr": None, "tag": "x"}}

    # Both absent
    obj2 = Top(mid=None)
    flat2 = flatten(obj2)
    # mid_is_null=True, then 4 placeholder values for the (absent) inner subtree
    assert flat2[0] is True
    assert (
        len(flat2) == 5
    )  # sentinel + (inner sentinel + 2 addr leaves + tag) placeholders
    rec2 = unflatten_to_json(Top, flat2)
    assert rec2 == {"mid": None}
