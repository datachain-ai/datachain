"""Unit tests for multi-arm ``Union[...]`` support (tagged unions).

Covers the type helpers (``union_arms`` canonical ordering, ``union_layout``
classification, ``is_chain_type``), the ``_type_tag`` discriminator and per-arm
slot schema emission, nullability of arm leaves, and flatten/unflatten
round-trips for every union kind (basic/basic, model/model, mixed, nullable).
``Optional[X]`` itself is the single-arm case and lives in ``test_optional.py``.
"""

import copy
from datetime import datetime
from typing import Optional, Union

import pytest

from datachain.lib.arrow import _union_value
from datachain.lib.convert.flatten import flatten_value
from datachain.lib.convert.unflatten import unflatten_to_json_pos
from datachain.lib.data_model import (
    DataModel,
    UnionLayout,
    is_chain_type,
    union_arms,
    union_layout,
    union_slot_key,
)
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema, SignalSchemaWarning


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


class Wrap(DataModel):
    """Wrapper used to drive flatten/unflatten of a single union field."""

    value: str | int


# ---- union_arms canonical ordering -----------------------------------------


@pytest.mark.parametrize(
    "anno,expected_arms,has_none",
    [
        (Union[str, int], [int, str], False),
        (Union[int, str], [int, str], False),  # same type, order-independent
        (Union[str, int, None], [int, str], True),
        (Optional[int | str], [int, str], True),
        (int, [int], False),
        (Optional[int], [int], True),
    ],
)
def test_union_arms_canonical_order(anno, expected_arms, has_none):
    arms, none = union_arms(anno)
    assert arms == expected_arms
    assert none is has_none


def test_union_arms_order_is_serialization_stable():
    # The two spellings are the same type; arm order (hence the _type_tag index)
    # must not depend on how the Union was written.
    assert union_arms(Union[str, int]) == union_arms(Union[int, str])
    assert union_arms(Union[Foo, Bar]) == union_arms(Union[Bar, Foo])


# ---- union_layout classification -------------------------------------------


def test_union_layout_multiarm_uses_slots():
    layout = union_layout(Union[str, int])
    assert layout is not None
    assert layout.use_slots
    assert layout.arms == [int, str]
    assert not layout.has_none


def test_union_layout_optional_model_no_slots():
    # Optional[Model] is the single-arm union: tag + direct leaves, no slot prefix.
    layout = union_layout(Optional[Foo])
    assert layout is not None
    assert not layout.use_slots
    assert layout.has_none


@pytest.mark.parametrize("anno", [int, str, Optional[int], list[int], dict[str, int]])
def test_union_layout_none_for_non_tagged(anno):
    assert union_layout(anno) is None


def test_union_layout_json_union_not_tagged():
    # Collection/JSON unions stay single JSON columns, not tagged unions.
    assert union_layout(Union[dict, list[dict]]) is None
    assert union_layout(Union[dict, list[dict], None]) is None


def test_is_chain_type_multiarm_union():
    assert is_chain_type(Union[str, int])
    assert is_chain_type(Union[Foo, Bar])
    assert is_chain_type(Union[str, int, Foo])
    assert is_chain_type(Union[str, int, None])


def test_schema_scalar_union_columns():
    schema = SignalSchema({"value": Union[str, int]})
    # int sorts before str, so _0 = int, _1 = str.
    assert schema.db_signals() == ["value___type_tag", "value___0", "value___1"]
    # The discriminator is hidden from user-facing signals; arm slots are not.
    assert schema.user_signals() == ["value._0", "value._1"]


def test_schema_model_union_columns():
    schema = SignalSchema({"item": Union[Foo, Bar]})
    # Bar ("Bar") sorts before Foo ("Foo").
    assert schema.db_signals() == [
        "item___type_tag",
        "item___0__x",
        "item___1__a",
        "item___1__b",
    ]


def test_arm_selector_stable_across_reload():
    # Reading a dataset in a process without the model code rebuilds the model with a
    # versioned __name__ but a preserved logical base name; the readable arm path
    # (C("u.Block.x") -> the Block arm) must resolve via the stable name.
    class Reloaded(DataModel):
        a: int = 0

    Reloaded._modelstore_base_name = "Block"
    assert SignalSchema._arm_selector(Reloaded) == "Block"
    assert SignalSchema._arm_selector(Reloaded) != Reloaded.__name__


def test_union_arm_leaves_are_nullable():
    cols = SignalSchema({"value": Union[str, int]}).db_signals(as_columns=True)
    by_name = {c.name: c for c in cols}
    assert by_name["value___0"].type.dc_nullable  # int arm
    assert by_name["value___1"].type.dc_nullable  # str arm
    assert by_name["value___type_tag"].type.dc_nullable


def test_union_slot_key():
    assert union_slot_key(0) == "_0"
    assert union_slot_key(3) == "_3"


# ---- flatten / unflatten round-trips ---------------------------------------


def _roundtrip(value, anno):
    class _W(DataModel):
        value: anno  # type: ignore[valid-type]

    flat = flatten_value(value, anno)
    back, _ = unflatten_to_json_pos(_W, flat)
    return flat, back["value"]


@pytest.mark.parametrize(
    "value,anno,tag",
    [
        ("hello", Union[str, int], 1),  # str is arm 1
        (42, Union[str, int], 0),  # int is arm 0
        ("hi", Union[str, int, None], 1),
        (7, Union[str, int, None], 0),
        (None, Union[str, int, None], None),  # None arm -> NULL discriminator
    ],
)
def test_flatten_scalar_union(value, anno, tag):
    flat, restored = _roundtrip(value, anno)
    assert flat[0] == tag
    assert restored == value


@pytest.mark.parametrize("value", [Foo(a=1, b="z"), Bar(x=3.5)])
def test_flatten_model_union(value):
    _, restored = _roundtrip(value, Union[Foo, Bar])
    assert restored == value.model_dump()


def test_flatten_mixed_union():
    for value in ["txt", 5, Foo(a=2, b="m")]:
        _, restored = _roundtrip(value, Union[str, int, Foo])
        expected = value.model_dump() if isinstance(value, DataModel) else value
        assert restored == expected


def test_flatten_bool_not_swallowed_by_int_arm():
    # bool is a subclass of int; exact-type matching must keep them distinct.
    layout = union_layout(Union[int, bool])
    assert layout is not None
    flat_true = flatten_value(True, Union[int, bool])
    flat_one = flatten_value(1, Union[int, bool])
    # bool ("bool") sorts before int ("int"): bool = arm 0, int = arm 1.
    assert flat_true[0] == 0
    assert flat_one[0] == 1


def test_flatten_datetime_arm():
    now = datetime(2024, 1, 2, 3, 4, 5)
    _, restored = _roundtrip(now, Union[str, datetime])
    assert restored == now


def test_flatten_inactive_arms_are_none():
    # str active -> the int slot and (model) arm leaves are None placeholders.
    flat = flatten_value("hi", Union[int, str, Foo])
    assert flat[0] == 2  # Foo=0? no: arms sorted Foo,int,str -> str index 2
    # exactly one arm column is non-None (the active str slot).
    assert sum(1 for v in flat[1:] if v is not None) == 1


def test_deserialize_union_with_unresolvable_arm_skips_signal():
    ser = {
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
        schema = SignalSchema.deserialize(ser)
    assert "v" not in schema.values
    assert "kept" in schema.values


def test_union_value_out_of_range_tag_returns_none():
    layout = UnionLayout(arms=[Foo], has_none=True, use_slots=False)
    assert _union_value({"v._type_tag": 1}, layout, "v") is None
