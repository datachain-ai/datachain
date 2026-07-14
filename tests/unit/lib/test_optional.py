"""Unit tests for ``Optional[X]`` / nullable-object support.

Covers the type helpers (``unwrap_optional``, ``is_chain_type`` over Optional),
the ``x: T = None`` validation rule, the ``_type_tag`` discriminator (collision-safety,
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


@pytest.fixture(autouse=True)
def restore_model_store():
    snapshot = copy.deepcopy(ModelStore.store)
    ModelStore.store = {}
    try:
        yield
    finally:
        ModelStore.store = snapshot


class _Sample(BaseModel):
    x: int


@pytest.mark.parametrize(
    ("anno", "expected"),
    [
        (Optional[float], True),
        (Optional[int], True),
        (Optional[str], True),
        (float, False),
        (Optional[list[int]], False),
    ],
)
def test_is_nullable_column_scalar(anno, expected):
    assert SignalSchema({"x": anno}).is_nullable_column("x", anno) is expected


def test_is_nullable_column_float_leaf_under_optional_model():
    class M(DataModel):
        x: float = 0.0

    schema = SignalSchema({"m": Optional[M]})
    assert schema.is_nullable_column("m__x", float) is True


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


def test_default_none_promoted_to_optional():
    # `x: int = None` is auto-promoted to `Optional[int]` (not an error) so the
    # column is genuinely nullable and None round-trips.
    class M(DataModel):
        x: int = None  # type: ignore[assignment]

    inner, is_optional = unwrap_optional(M.model_fields["x"].annotation)
    assert is_optional and inner is int
    assert M().x is None
    assert M(x=5).x == 5


def test_default_none_already_optional_unchanged():
    # The three Optional spellings already work; promotion is a no-op for them.
    class A(DataModel):
        x: Optional[int] = None

    class B(DataModel):
        x: int | None = None

    class C(DataModel):
        x: Union[int, None] = None

    for ok in (A, B, C):
        assert ok().x is None
        inner, is_optional = unwrap_optional(ok.model_fields["x"].annotation)
        assert is_optional and inner is int


def test_default_non_none_unaffected():
    class Ok(DataModel):
        x: int = 5
        y: str = "hello"

    assert Ok().x == 5
    assert Ok().y == "hello"
    # not promoted — still required-with-default, non-optional
    _, x_optional = unwrap_optional(Ok.model_fields["x"].annotation)
    assert x_optional is False


def test_default_none_promotes_only_the_none_field():
    class M(DataModel):
        count: int = None  # type: ignore[assignment]
        name: str = "x"

    assert M().count is None and M().name == "x"
    _, count_opt = unwrap_optional(M.model_fields["count"].annotation)
    _, name_opt = unwrap_optional(M.model_fields["name"].annotation)
    assert count_opt is True and name_opt is False


def test_is_null_field_name_allowed():
    # `is_null` is a normal user field now; the discriminator uses the internal
    # `_type_tag` name, which pydantic forbids users from declaring.
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
    assert "m___type_tag" in leaves  # the internal discriminator, no collision


class _Addr(DataModel):
    city: str = ""
    zip: str = ""


class _Outer(DataModel):
    name: str
    addr: _Addr | None = None


def test_flatten_optional_datamodel_present():
    out = _Outer(name="Alice", addr=_Addr(city="Berlin", zip="10115"))
    assert flatten(out) == ("Alice", 0, "Berlin", "10115")


def test_flatten_optional_datamodel_absent():
    out = _Outer(name="Bob", addr=None)
    assert flatten(out) == ("Bob", 1, None, None)


def test_unflatten_optional_datamodel_present():
    row = ("Alice", 0, "Berlin", "10115")
    j = unflatten_to_json(_Outer, row)
    assert j == {
        "name": "Alice",
        "addr": {"city": "Berlin", "zip": "10115"},
    }
    obj = _Outer(**j)
    assert obj.addr == _Addr(city="Berlin", zip="10115")


def test_unflatten_optional_datamodel_absent():
    row = ("Bob", 1, None, None)
    j = unflatten_to_json(_Outer, row)
    assert j == {"name": "Bob", "addr": None}
    assert _Outer(**j).addr is None


def test_unflatten_ignores_leaf_garbage_when_sentinel_absent():
    # tag=1 (None arm) hydrates the parent as None even when the leaves hold values.
    row = ("Bob", 1, "garbage-city", "garbage-zip")
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
    assert "out.addr._type_tag" in leaves
    assert "out.addr.city" in leaves
    assert "out.addr.zip" in leaves

    # The sentinel is an internal column: hidden from user-facing views.
    visible = [
        ".".join(p)
        for p, _, has_subtree, _ in schema.get_flat_tree(include_sentinels=False)
        if not has_subtree
    ]
    assert "out.addr._type_tag" not in visible
    assert "out.addr.city" in visible


def test_print_schema_hides_optional_sentinel():
    # print_schema()/print_tree() must not expose the internal _type_tag discriminator
    # of an Optional[DataModel] (consistent with to_pandas/to_records output).
    buf = io.StringIO()
    SignalSchema({"out": _Outer}).print_tree(file=buf)
    out = buf.getvalue()
    assert "_type_tag" not in out
    assert "city" in out and "addr" in out  # real fields still shown


def test_get_signals_recognizes_optional_file():
    """get_signals(File) must still find an Optional[File] node. The isclass
    guard (needed because Optional[File] is a Union) would otherwise silently
    skip it, which undercounts File-derived dataset stats (e.g. total size)."""
    from datachain.lib.file import File

    schema = SignalSchema({"plain": File, "maybe": Optional[File]})
    assert set(schema.get_signals(File)) == {"plain", "maybe"}


def test_to_udf_spec_includes_sentinel_for_optional_datamodel():
    spec = SignalSchema({"out": _Outer}).to_udf_spec()
    assert "out__addr___type_tag" in spec
    assert "out__addr__city" in spec


def test_user_signals_hides_optional_sentinel():
    """user_signals() is the user-facing leaf list, so it must not leak the
    internal _type_tag discriminator — otherwise select_except/compare_signals seed it
    and to_partial asserts it against the model's real fields."""
    signals = SignalSchema({"out": _Outer}).user_signals()
    assert not any(s.endswith("_type_tag") for s in signals), signals
    assert "out.addr.city" in signals


def test_select_except_on_optional_datamodel():
    """select_except() on a schema with an Optional[DataModel] must not crash:
    the sentinel used to leak via user_signals() and trip to_partial's
    'Selection should match existing model fields' assertion."""
    schema = SignalSchema({"out": _Outer, "id": int})
    result = schema.select_except_signals("out.addr.city")
    leaves = result.user_signals()
    assert "out.addr.city" not in leaves
    assert "out.addr.zip" in leaves
    assert not any(s.endswith("_type_tag") for s in leaves)


def test_compare_signals_ignores_optional_sentinel():
    a = SignalSchema({"out": _Outer})
    b = SignalSchema({"out": _Outer, "extra": int})
    added, removed = b.compare_signals(a)
    assert added == {"extra"}
    assert all(not s.endswith("_type_tag") for s in added | removed)


def test_create_model_does_not_promote_non_optional_fields():
    """Replayed schemas keep `default=None` fields non-Optional (no promotion, no
    _type_tag added)."""
    schema = SignalSchema({"id": int, "addr": _Addr, "name": str})
    model = schema.create_model("Probe")
    annotations = {f: fi.annotation for f, fi in model.model_fields.items()}
    assert annotations == {"id": int, "addr": _Addr, "name": str}
    # the rebuilt schema must not have gained an Optional[DataModel] sentinel.
    leaves = SignalSchema({"probe": model}).user_signals()
    assert not any(s.endswith("_type_tag") for s in leaves), leaves


def test_infer_optional_datamodel_from_nones():
    seq = [_Addr(city="a"), None, _Addr(city="c")]
    inner, is_optional = unwrap_optional(_infer_type_from_sequence(seq, "item", "ds"))
    assert is_optional and inner is _Addr


@pytest.mark.parametrize(
    ("seq", "expected_inner"),
    [
        ([10, None, 30], int),
        (["x", None, "z"], str),
        ([[1, 2], None], list[int]),
    ],
)
def test_infer_optional_scalar_and_collection_from_nones(seq, expected_inner):
    inner, is_optional = unwrap_optional(_infer_type_from_sequence(seq, "c", "ds"))
    assert is_optional and inner == expected_inner


def test_infer_non_optional_when_no_none():
    typ = _infer_type_from_sequence([1, 2, 3], "c", "ds")
    inner, is_optional = unwrap_optional(typ)
    assert not is_optional and inner is int


def test_optional_datamodel_roundtrip_at_top_level():
    schema = SignalSchema({"item": Optional[_Addr]})
    flat = [
        ".".join(p)
        for p, _, has_subtree, _ in schema.get_flat_tree()
        if not has_subtree
    ]
    assert "item._type_tag" in flat
    assert "item.city" in flat
    # row_to_objs with tag=1 (None arm) returns None for top-level Optional[Model].
    assert schema.row_to_objs((1, None, None)) == [None]
    assert schema.row_to_objs((0, "Paris", "75001")) == [
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
    # tags: mid=0 (present), addr=1 (None); then city=None, zip=None, tag="x"
    assert flat == (0, 1, None, None, "x")
    rec = unflatten_to_json(Top, flat)
    assert rec == {"mid": {"addr": None, "tag": "x"}}

    # Both absent
    obj2 = Top(mid=None)
    flat2 = flatten(obj2)
    # mid tag=1 (None arm), then 4 placeholder values for the absent inner subtree
    assert flat2[0] == 1
    assert (
        len(flat2) == 5
    )  # sentinel + (inner sentinel + 2 addr leaves + tag) placeholders
    rec2 = unflatten_to_json(Top, flat2)
    assert rec2 == {"mid": None}
