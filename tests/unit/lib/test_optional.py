"""Unit tests for ``Optional[X]`` / nullable-object support.

Covers the type helpers (``unwrap_optional``, ``is_chain_type`` over Optional),
the ``x: T = None`` validation rule, the ``_is_null`` sentinel (collision-safety,
schema emission), and flatten/unflatten round-trips of ``Optional[DataModel]``.

Extracted from ``test_data_model.py`` and ``test_feature.py`` to keep
nullable-object coverage in one place.
"""

import copy
import io
from typing import Optional, Union

import pytest
from pydantic import BaseModel

from datachain.lib.convert.flatten import flatten
from datachain.lib.convert.unflatten import unflatten_to_json
from datachain.lib.convert.values_to_tuples import _infer_type_from_sequence
from datachain.lib.data_model import DataModel, is_chain_type, unwrap_optional
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.utils import DataChainParamsError


@pytest.fixture(autouse=True)
def restore_model_store():
    snapshot = copy.deepcopy(ModelStore.store)
    ModelStore.store = {}
    try:
        yield
    finally:
        ModelStore.store = snapshot


# -- unwrap_optional -------------------------------------------------------


class _Sample(BaseModel):
    x: int


@pytest.mark.parametrize(
    ("annotation", "expected_inner", "expected_is_optional"),
    [
        (Optional[int], int, True),
        (int | None, int, True),
        (Union[int, None], int, True),
        (Optional[str], str, True),
        (Optional[_Sample], _Sample, True),
        (Optional[list[int]], list[int], True),
        (Optional[dict[str, int]], dict[str, int], True),
        (int, int, False),
        (str, str, False),
        (list[int], list[int], False),
        (_Sample, _Sample, False),
        (Union[int, str], Union[int, str], False),
    ],
)
def test_unwrap_optional(annotation, expected_inner, expected_is_optional):
    inner, is_optional = unwrap_optional(annotation)
    assert is_optional is expected_is_optional
    assert inner == expected_inner


def test_unwrap_optional_collapses_nested():
    # Python collapses Optional[Optional[X]] to Optional[X], so this is mostly
    # a sanity check that the helper still gives one-level unwrap.
    inner, is_optional = unwrap_optional(Union[None, int])
    assert is_optional is True
    assert inner is int


# -- is_chain_type with Optional ------------------------------------------


def test_is_chain_type_optional_basic():
    assert is_chain_type(Optional[int])
    assert is_chain_type(int | None)
    assert is_chain_type(Union[int, None])
    assert is_chain_type(Optional[str])
    assert is_chain_type(Optional[float])
    assert is_chain_type(Optional[bool])


def test_is_chain_type_optional_datamodel():
    class M(DataModel):
        x: int

    assert is_chain_type(Optional[M])
    assert is_chain_type(M | None)


def test_is_chain_type_optional_collections():
    assert is_chain_type(Optional[list[int]])
    assert is_chain_type(Optional[dict[str, int]])
    assert is_chain_type(list[int | None])  # nulls inside list (defer support)
    assert is_chain_type(dict[str, int | None])


# -- validate_default_none -----------------------------------------------


def test_default_none_rejected_for_non_optional():
    with pytest.raises(DataChainParamsError, match="default value `None` requires"):

        class Bad(DataModel):
            x: int = None  # type: ignore[assignment]


def test_default_none_accepted_for_optional_typing():
    class Ok(DataModel):
        x: int | None = None

    assert Ok().x is None


def test_default_none_accepted_for_pep604():
    class Ok(DataModel):
        x: int | None = None

    assert Ok().x is None


def test_default_none_accepted_for_union_none():
    class Ok(DataModel):
        x: int | None = None

    assert Ok().x is None


def test_default_non_none_unaffected():
    class Ok(DataModel):
        x: int = 5
        y: str = "hello"

    assert Ok().x == 5
    assert Ok().y == "hello"


def test_default_none_error_names_field_and_type():
    with pytest.raises(DataChainParamsError) as exc_info:

        class Bad(DataModel):
            count: int = None  # type: ignore[assignment]

    assert "count" in str(exc_info.value)
    assert "Bad" in str(exc_info.value)
    assert "Optional[int]" in str(exc_info.value)


# -- sentinel name does not collide with user fields ---------------------


def test_is_null_field_name_allowed():
    # `is_null` is a normal user field now; the sentinel uses the internal
    # `_is_null` name, which pydantic forbids users from declaring.
    class HasIsNull(DataModel):
        is_null: bool = False
        value: int = 0

    assert HasIsNull(is_null=True).is_null is True


def test_optional_datamodel_sentinel_does_not_collide_with_is_null_field():
    class M(DataModel):
        is_null: bool = False  # same spelling as the old reserved word
        value: int = 0

    leaves = [
        "__".join(p)
        for p, _, has_subtree, _ in SignalSchema({"m": Optional[M]}).get_flat_tree()
        if not has_subtree
    ]
    assert "m__is_null" in leaves  # the user's field
    assert "m___is_null" in leaves  # the internal sentinel, no collision


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
    leaves = [
        ".".join(p)
        for p, _, has_subtree, _ in schema.get_flat_tree()
        if not has_subtree
    ]
    assert "out.name" in leaves
    assert "out.addr._is_null" in leaves
    assert "out.addr.city" in leaves
    assert "out.addr.zip" in leaves

    # The sentinel is an internal column: hidden from user-facing views.
    visible = [
        ".".join(p)
        for p, _, has_subtree, _ in schema.get_flat_tree(include_sentinels=False)
        if not has_subtree
    ]
    assert "out.addr._is_null" not in visible
    assert "out.addr.city" in visible


def test_print_schema_hides_optional_sentinel():
    # print_schema()/print_tree() must not expose the internal _is_null sentinel
    # of an Optional[DataModel] (consistent with to_pandas/to_records output).
    buf = io.StringIO()
    SignalSchema({"out": _Outer}).print_tree(file=buf)
    out = buf.getvalue()
    assert "_is_null" not in out
    assert "city" in out and "addr" in out  # real fields still shown


def test_to_udf_spec_includes_sentinel_for_optional_datamodel():
    spec = SignalSchema({"out": _Outer}).to_udf_spec()
    assert "out__addr___is_null" in spec
    assert "out__addr__city" in spec


def test_infer_optional_datamodel_from_nones():
    # A DataModel column with some None values is inferred as Optional[DataModel]
    # (so the sentinel is emitted) rather than the bare model type. This is the
    # pure-inference half of the func read_values round-trip test.
    seq = [_Addr(city="a"), None, _Addr(city="c")]
    inner, is_optional = unwrap_optional(_infer_type_from_sequence(seq, "item", "ds"))
    assert is_optional and inner is _Addr


def test_optional_datamodel_roundtrip_at_top_level():
    schema = SignalSchema({"item": Optional[_Addr]})
    flat = [
        ".".join(p)
        for p, _, has_subtree, _ in schema.get_flat_tree()
        if not has_subtree
    ]
    assert "item._is_null" in flat
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
