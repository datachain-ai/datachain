"""Unit tests for multi-arm ``Union[...]`` support (tagged unions).

Covers the type helpers (``union_arms`` canonical ordering, ``union_layout``
classification, ``is_chain_type``), the ``_type_tag`` discriminator and per-arm
slot schema emission, nullability of arm leaves, and flatten/unflatten
round-trips for every union kind (basic/basic, model/model, mixed, nullable).
``X | None`` itself is the single-arm case and lives in ``test_optional.py``.
"""

import copy
from datetime import datetime
from typing import Generic, Literal, Optional, TypeVar, Union

import pytest

from datachain.dataset import RowDict
from datachain.error import OutdatedDatasetFormatError
from datachain.lib.arrow import _union_value
from datachain.lib.convert.flatten import flatten, flatten_value
from datachain.lib.convert.unflatten import unflatten_to_json_pos
from datachain.lib.data_model import (
    DataModel,
    UnionLayout,
    _warn_index_tag,
    arm_selector,
    is_chain_type,
    union_arms,
    union_layout,
    validate_chain_type,
)
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema, SignalSchemaWarning
from datachain.lib.utils import DataChainParamsError
from datachain.sql.types import Int64, String


@pytest.fixture(autouse=True)
def restore_model_store():
    snapshot = copy.deepcopy(ModelStore.store)
    ModelStore.store = {}
    try:
        yield
    finally:
        ModelStore.store = snapshot


class Foo(DataModel):
    a: int = 0
    b: str = ""


class Bar(DataModel):
    x: float = 0.0


@pytest.mark.parametrize(
    "anno,expected_arms,has_none",
    [
        (Union[str, int], [int, str], False),  # noqa: UP007
        (Union[int, str], [int, str], False),  # noqa: UP007
        (Union[str, int, None], [int, str], True),  # noqa: UP007
        (Optional[int | str], [int, str], True),  # noqa: UP045
        (int, [int], False),
        (Optional[int], [int], True),  # noqa: UP045
    ],
)
def test_union_arms_canonical_order(anno, expected_arms, has_none):
    arms, none = union_arms(anno)
    assert arms == expected_arms
    assert none is has_none


def test_union_arms_order_is_serialization_stable():
    assert union_arms(Union[str, int]) == union_arms(Union[int, str])  # noqa: UP007
    assert union_arms(Union[Foo, Bar]) == union_arms(Union[Bar, Foo])  # noqa: UP007


def test_union_layout_multiarm_uses_slots():
    layout = union_layout(str | int)
    assert layout is not None
    assert layout.use_slots
    assert layout.arms == (int, str)
    assert not layout.has_none


def test_union_layout_optional_model_no_slots():
    # Model | None is the single-arm union: tag + direct leaves, no slot prefix.
    layout = union_layout(Foo | None)
    assert layout is not None
    assert not layout.use_slots
    assert layout.has_none


@pytest.mark.parametrize("anno", [int, str, int | None, list[int], dict[str, int]])
def test_union_layout_none_for_non_tagged(anno):
    assert union_layout(anno) is None


def test_union_layout_json_union_not_tagged():
    assert union_layout(dict | list[dict]) is None
    assert union_layout(dict | list[dict] | None) is None


def test_is_chain_type_multiarm_union():
    assert is_chain_type(str | int)
    assert is_chain_type(Foo | Bar)
    assert is_chain_type(str | int | Foo)
    assert is_chain_type(str | int | None)


def test_is_chain_type_answers_instead_of_raising():
    class Human(DataModel):
        name: str = ""

    class Robot(DataModel):
        name: str = ""

    candidates = [int, str, list[Human | Robot], float]
    assert [t for t in candidates if is_chain_type(t)] == [int, str, float]
    for anno in (
        list[Human | Robot],
        dict[str, Human | Robot],
        list[Human | Robot] | None,
    ):
        with pytest.raises(
            DataChainParamsError, match="Put the Union inside a DataModel"
        ):
            validate_chain_type(anno)


def test_same_shaped_arms_rejected_inside_a_collection():
    class Human(DataModel):
        label: str = ""

    class Machine(DataModel):
        label: str = ""

    class Holder(DataModel):
        items: list[Human | Machine] = []  # noqa: RUF012

    class DeepHolder(DataModel):
        items: list[Holder] = []  # noqa: RUF012

    class KeyedHolder(DataModel):
        items: dict[str, Human | Machine] = {}  # noqa: RUF012

    assert is_chain_type(Human | Machine)
    for anno in (Holder, DeepHolder, KeyedHolder):
        with pytest.raises(DataChainParamsError, match="declare the same field names"):
            validate_chain_type(anno)

    with pytest.raises(DataChainParamsError, match=r"`Holder\.items`"):
        validate_chain_type(DeepHolder)


def test_arms_stored_alike_in_json_rejected():
    class Holder(DataModel):
        items: list[datetime | str] = []  # noqa: RUF012

    with pytest.raises(DataChainParamsError, match="stored alike in JSON"):
        validate_chain_type(Holder)
    with pytest.raises(DataChainParamsError, match="stored alike in JSON"):
        validate_chain_type(list[datetime | str])
    validate_chain_type(list[int | str])


def test_union_dict_key_rejected():
    class Holder(DataModel):
        items: dict[int | str, str] = {}  # noqa: RUF012

    with pytest.raises(DataChainParamsError, match="Union dict key"):
        validate_chain_type(Holder)
    with pytest.raises(DataChainParamsError, match="Union dict key"):
        validate_chain_type(dict[int | str, str])
    validate_chain_type(dict[str, int | str])


def test_arm_fields_subset_of_another_rejected():
    class Small(DataModel):
        x: int = 0

    class Big(DataModel):
        x: int = 0
        y: str | None = None

    class Holder(DataModel):
        items: list[Small | Big] = []  # noqa: RUF012

    with pytest.raises(DataChainParamsError, match="subset of the other"):
        validate_chain_type(Holder)
    # as a signal, each arm keeps its own columns, so the shapes stay apart
    validate_chain_type(Small | Big)


def test_same_shaped_arms_rejected_whatever_the_field_order():
    class Human(DataModel):
        label: str = ""

    class Machine(DataModel):
        label: str = ""

    class Inner(DataModel):
        pet: Human | Machine

    class Wrapped(DataModel):
        wrapped: list[Inner] = []  # noqa: RUF012
        direct: Inner = Inner(pet=Human())

    with pytest.raises(DataChainParamsError, match=r"`Inner\.pet`"):
        validate_chain_type(Wrapped)


def test_differently_shaped_arms_allowed_inside_a_collection():
    class Human(DataModel):
        label: str = ""

    class Machine(DataModel):
        score: float = 0.0

    class Holder(DataModel):
        items: list[Human | Machine] = []  # noqa: RUF012

    assert is_chain_type(Holder)


def test_literal_discriminator_allows_same_shaped_arms():
    class Human(DataModel):
        kind: Literal["human"] = "human"
        label: str = ""

    class Machine(DataModel):
        kind: Literal["machine"] = "machine"
        label: str = ""

    class Holder(DataModel):
        items: list[Human | Machine] = []  # noqa: RUF012

    assert is_chain_type(Holder)


def test_self_referential_model_field_walk_terminates():
    class Node(DataModel):
        name: str = ""
        child: "Node | None" = None

    Node.model_rebuild()
    validate_chain_type(Node)
    assert is_chain_type(Node)


def test_union_serialization_is_order_independent():
    left, right = SignalSchema({"v": str | int}), SignalSchema({"v": int | str})
    assert left.serialize() == right.serialize() == {"v": "Union[int, str]"}
    assert left.hash() == right.hash()
    assert SignalSchema({"v": str | int | None}).serialize() == {
        "v": "Union[int, str, NoneType]"
    }
    assert SignalSchema.deserialize({"v": "Union[str, int]"}).values["v"] == (int | str)


def test_schema_scalar_union_columns():
    schema = SignalSchema({"value": str | int})
    assert schema.db_signals() == ["value___type_tag", "value__int", "value__str"]
    # The discriminator is hidden from user-facing signals; arm slots are not.
    assert schema.user_signals() == ["value.int", "value.str"]


def test_schema_model_union_columns():
    schema = SignalSchema({"item": Foo | Bar})
    # Bar ("Bar") sorts before Foo ("Foo"); each arm is stored under its model name
    assert schema.db_signals() == [
        "item___type_tag",
        "item__Bar__x",
        "item__Foo__a",
        "item__Foo__b",
    ]


def test_arm_selector_stable_across_reload():
    # Reading a dataset in a process without the model code rebuilds the model with a
    # versioned __name__ but a preserved logical base name; the readable arm path
    # (C("u.Block.x") -> the Block arm) must resolve via the stable name.
    class Reloaded(DataModel):
        a: int = 0

    Reloaded._modelstore_base_name = "Block"
    assert arm_selector(Reloaded) == "Block"
    assert arm_selector(Reloaded) != Reloaded.__name__


def test_union_arm_leaves_are_nullable():
    cols = SignalSchema({"value": str | int}).db_signals(as_columns=True)
    by_name = {c.name: c for c in cols}
    assert by_name["value__int"].type.dc_nullable  # int arm
    assert by_name["value__str"].type.dc_nullable  # str arm
    assert by_name["value___type_tag"].type.dc_nullable


def test_arm_selector_is_the_type_name():
    assert arm_selector(int) == "int"
    assert arm_selector(str) == "str"
    assert arm_selector(Foo) == "Foo"


def _roundtrip(value, anno):
    class _W(DataModel):
        value: anno  # type: ignore[valid-type]

    flat = flatten_value(value, anno)
    back, _ = unflatten_to_json_pos(_W, flat)
    return flat, back["value"]


@pytest.mark.parametrize(
    "value,anno,tag",
    [
        ("hello", str | int, "str"),
        (42, str | int, "int"),
        ("hi", str | int | None, "str"),
        (7, str | int | None, "int"),
        (None, str | int | None, None),  # None -> NULL discriminator
    ],
)
def test_flatten_scalar_union(value, anno, tag):
    flat, restored = _roundtrip(value, anno)
    assert flat[0] == tag
    assert restored == value


@pytest.mark.parametrize("value", [Foo(a=1, b="z"), Bar(x=3.5)])
def test_flatten_model_union(value):
    _, restored = _roundtrip(value, Foo | Bar)
    assert restored == value.model_dump()


def test_flatten_mixed_union():
    for value in ["txt", 5, Foo(a=2, b="m")]:
        _, restored = _roundtrip(value, str | int | Foo)
        expected = value.model_dump() if isinstance(value, DataModel) else value
        assert restored == expected


def test_flatten_bool_not_swallowed_by_int_arm():
    # bool is a subclass of int; exact-type matching must keep them distinct.
    layout = union_layout(int | bool)
    assert layout is not None
    flat_true = flatten_value(True, int | bool)
    flat_one = flatten_value(1, int | bool)
    assert flat_true[0] == "bool"
    assert flat_one[0] == "int"


def test_nested_union_reads_the_tagged_arm():
    class Cat(DataModel):
        name: str = ""

    class Dog(DataModel):
        name: str = ""

    class Holder(DataModel):
        pet: Cat | Dog

    row = flatten(Holder(pet=Dog(name="fido")))
    schema = SignalSchema({"h": Holder})
    (holder,) = schema.row_to_objs(RowDict(zip(schema.to_udf_spec(), row, strict=True)))
    assert holder.pet == Dog(name="fido")


def test_flatten_subclass_matches_narrowest_arm():
    class Animal(DataModel):
        legs: int = 4

    class Zebra(Animal):
        stripes: int = 1

    class BabyZebra(Zebra):
        age: int = 0

    flat, restored = _roundtrip(BabyZebra(stripes=9, age=2), Animal | Zebra)
    assert flat[0] == "Zebra"
    assert restored == {"legs": 4, "stripes": 9}


def test_flatten_ambiguous_diamond_arms_error():
    class A(DataModel):
        a: int = 1

    class B(DataModel):
        b: int = 2

    class AB(A, B):
        pass

    with pytest.raises(DataChainParamsError, match="unrelated union arms A, B"):
        flatten_value(AB(a=10, b=20), A | B)
    assert flatten_value(AB(a=10, b=20), A | B | AB)[0] == "AB"


def test_flatten_datetime_arm():
    now = datetime(2024, 1, 2, 3, 4, 5)
    _, restored = _roundtrip(now, str | datetime)
    assert restored == now


def test_flatten_inactive_arms_are_none():
    # str active -> the int slot and (model) arm leaves are None placeholders.
    flat = flatten_value("hi", int | str | Foo)
    assert flat[0] == "str"
    assert sum(1 for v in flat[1:] if v is not None) == 1


def test_deserialize_union_with_unresolvable_arm_skips_signal():
    serialized = {
        "kept": "int",
        "v": "Union[KnownArm@v1, MissingArm@v1]",
        "_custom_types": {
            "KnownArm@v1": {
                "schema_version": 2,
                "name": "KnownArm@v1",
                "fields": {"p": "int"},
                "bases": [],
                "hidden_fields": [],
            }
        },
    }
    with pytest.warns(SignalSchemaWarning):
        schema = SignalSchema.deserialize(serialized)
    assert "v" not in schema.values
    assert "kept" in schema.values


def test_union_arm_named_like_the_discriminator_rejected():
    class _type_tag(DataModel):  # noqa: N801
        x: int = 0

    with pytest.raises(DataChainParamsError, match="is reserved"):
        SignalSchema({"value": _type_tag | str}).db_signals()


def test_single_arm_reads_the_arm_under_any_tag_name():
    layout = UnionLayout(arms=[Foo], has_none=True, use_slots=False)
    got = _union_value({"v._type_tag": "FooPartial_1a2b_v1", "v.a": 5}, layout, "v")
    assert (got.a, got.b) == (5, "")
    assert _union_value({"v._type_tag": None}, layout, "v") is None
    assert _union_value({"v._type_tag": ""}, layout, "v") is None  # CH NULL tag


def test_arm_name_wins_over_a_field_of_the_same_name():
    class Left(DataModel):
        value: int = 0

    class Right(DataModel):
        Left: str = ""

    schema = SignalSchema({"v": Left | Right})
    assert schema.resolve_arm_path("v.Left.value") == "v.Left.value"
    assert schema.to_db_col("v.Left.value") == "v__Left__value"


def test_same_named_arms_answer_false_instead_of_raising():
    first = type("Dup", (DataModel,), {"__annotations__": {"a": int}, "a": 0})
    second = type("Dup", (DataModel,), {"__annotations__": {"b": int}, "b": 0})

    assert is_chain_type(first | second) is False
    with pytest.raises(DataChainParamsError, match="indistinguishable arms"):
        validate_chain_type(first | second)


def test_parameterized_model_arm_rejected():
    T = TypeVar("T")

    class Box(DataModel, Generic[T]):
        v: int = 0

    assert is_chain_type(Box[int] | str) is False
    with pytest.raises(DataChainParamsError, match="cannot name a column"):
        validate_chain_type(Box[int] | str)


def test_legacy_tag_dataset_rejected():
    from datachain.lib.dc.datasets import _reject_legacy_tags

    schema = SignalSchema({"m": Foo | None})
    assert _reject_legacy_tags("new_ds", schema, {"m___type_tag": String()}) is None
    with pytest.raises(OutdatedDatasetFormatError, match="re-create the dataset"):
        _reject_legacy_tags("old_ds", schema, {"m___type_tag": Int64()})


def test_binary_tag_reads_as_its_name():
    # parquet written by Spark holds string columns as binary
    schema = SignalSchema({"v": int | str})
    row = RowDict(zip(schema.to_udf_spec(), (b"str", None, "a"), strict=True))
    assert schema.row_to_objs(row) == ["a"]


def test_multi_arm_unknown_tag_errors():
    schema = SignalSchema({"v": int | str})
    with pytest.raises(DataChainParamsError, match="not one of the declared Union"):
        schema.row_to_objs(
            RowDict(zip(schema.to_udf_spec(), ("Gone", 42, None), strict=True))
        )
    assert schema.row_to_objs(
        RowDict(zip(schema.to_udf_spec(), (None, None, None), strict=True))
    ) == [None]


@pytest.fixture
def unwarned_index_tag():
    _warn_index_tag.cache_clear()
    yield
    _warn_index_tag.cache_clear()


def test_index_tag_reads_optional_model(unwarned_index_tag):
    layout = UnionLayout(arms=[Foo], has_none=True, use_slots=False)
    with pytest.warns(FutureWarning, match="numeric Optional"):
        got = _union_value({"v._type_tag": 0, "v.a": 7, "v.b": "z"}, layout, "v")
    assert (got.a, got.b) == (7, "z")

    schema = SignalSchema({"m": Foo | None})
    assert schema.row_to_objs(
        RowDict(zip(schema.to_udf_spec(), (0, 7, "z"), strict=True))
    ) == [Foo(a=7, b="z")]
    # any other index is the absent arm
    assert schema.row_to_objs(
        RowDict(zip(schema.to_udf_spec(), (1, 7, "z"), strict=True))
    ) == [None]
    assert _union_value({"v._type_tag": 1, "v.a": 7}, layout, "v") is None


def test_index_tag_rejected_multi_arm(unwarned_index_tag):
    # only the single-arm layout is ever stored with an index
    schema = SignalSchema({"value": int | str})
    with pytest.raises(OutdatedDatasetFormatError, match="unknown _type_tag 0"):
        schema.row_to_objs(
            RowDict(zip(schema.to_udf_spec(), (0, 42, None), strict=True))
        )


def test_order_by_column_none_for_unknown_signal():
    assert SignalSchema({"v": int | str}).order_by_column("nonexistent") is None


def test_union_value_infers_arm_when_tag_absent():
    layout = union_layout(Foo | int)
    fk, ik = arm_selector(Foo), arm_selector(int)

    foo = _union_value({f"v.{fk}.a": 5, f"v.{fk}.b": "z"}, layout, "v")
    assert (foo.a, foo.b) == (5, "z")

    empty_foo = {f"v.{fk}.a": float("nan"), f"v.{fk}.b": None, f"v.{ik}": 9}
    assert _union_value(empty_foo, layout, "v") == 9

    assert _union_value({f"v.{fk}.a": None, f"v.{ik}": None}, layout, "v") is None
