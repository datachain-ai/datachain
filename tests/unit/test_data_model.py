import copy
from collections.abc import Mapping, Sequence
from typing import Dict, Generic, List, TypeVar  # noqa: UP035

import pytest

# typing.TypedDict cannot be combined with Generic on Python 3.10
from typing_extensions import TypedDict

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.data_model import (
    DataModel,
    compute_model_fingerprint,
    is_chain_type,
)
from datachain.lib.model_store import ModelStore
from datachain.sql.types import JSON


@pytest.fixture(autouse=True)
def restore_model_store():
    snapshot = copy.deepcopy(ModelStore.store)
    ModelStore.store = {}
    try:
        yield
    finally:
        ModelStore.store = snapshot


def test_compute_model_fingerprint_missing_field():
    class Sample(DataModel):
        a: int

    with pytest.raises(ValueError, match="Field missing not found in Sample"):
        compute_model_fingerprint(Sample, {"missing": None})


def test_compute_model_fingerprint_non_model_child():
    class Sample(DataModel):
        a: int

    with pytest.raises(ValueError, match="Field a in Sample is not a model"):
        compute_model_fingerprint(Sample, {"a": {"child": None}})


def test_compute_model_fingerprint_stable_for_same_selection():
    class Sample(DataModel):
        a: int
        b: int

    sel = {"a": None}
    fp1 = compute_model_fingerprint(Sample, sel)
    fp2 = compute_model_fingerprint(Sample, sel)
    assert fp1 == fp2


def test_compute_model_fingerprint_changes_with_selection():
    class Sample(DataModel):
        a: int
        b: int

    fp_a = compute_model_fingerprint(Sample, {"a": None})
    fp_b = compute_model_fingerprint(Sample, {"b": None})
    assert fp_a != fp_b


def test_compute_model_fingerprint_nested_model():
    class Child(DataModel):
        x: int
        y: int

    class Parent(DataModel):
        child: Child
        z: int

    fp_child_x = compute_model_fingerprint(Parent, {"child": {"x": None}})
    fp_child_y = compute_model_fingerprint(Parent, {"child": {"y": None}})
    fp_child_all = compute_model_fingerprint(Parent, {"child": {"x": None, "y": None}})

    assert fp_child_x != fp_child_y
    assert fp_child_all != fp_child_x
    assert fp_child_all != fp_child_y


def test_compute_model_fingerprint_required_vs_optional_differs():
    class Required(DataModel):
        value: int

    class OptionalField(DataModel):
        value: int | None = None

    fp_required = compute_model_fingerprint(Required, {"value": None})
    fp_optional = compute_model_fingerprint(OptionalField, {"value": None})
    assert fp_required != fp_optional


def test_is_chain_type_handles_generics_that_reject_issubclass():
    T = TypeVar("T")

    class GenericRow(TypedDict, Generic[T]):
        x: int

    # TypedDicts raise on issubclass; validation must answer, not crash.
    assert is_chain_type(GenericRow[int]) is False


@pytest.mark.parametrize(
    "annotation",
    [
        # Abstract origins serialize as a bare "Sequence"/"Mapping", losing their
        # args, so a saved signal would not survive a reload.
        Sequence[int],
        Mapping[str, int],
        # Bare aliases have no args for the SQL layer to work from.
        List,  # noqa: UP006
        Dict,  # noqa: UP006
        tuple,
    ],
)
def test_is_chain_type_rejects_types_that_cannot_round_trip(annotation):
    assert is_chain_type(annotation) is False


@pytest.mark.parametrize("annotation", [list[int], dict[str, int], list[list[int]]])
def test_is_chain_type_accepts_serializable_collections(annotation):
    assert is_chain_type(annotation) is True


@pytest.mark.parametrize(
    "annotation",
    [
        # Wrong arity is silently truncated by `type_to_str`, e.g. `dict[str]`
        # would be written back out as `dict[str, Any]`.
        list[int, str],  # type: ignore[misc]
        dict[str],  # type: ignore[misc]
        dict[str, int, float],  # type: ignore[misc]
        # `python_to_sql` mis-types tuples, so they are not valid root signals.
        tuple[int, ...],
        tuple[int, str],
        # Ellipsis must not be normalised away before the arity is checked.
        list[int, ...],  # type: ignore[misc]
        dict[str, int, ...],  # type: ignore[misc]
        list[..., int],  # type: ignore[misc]
    ],
)
def test_is_chain_type_rejects_malformed_or_mistyped_collections(annotation):
    assert is_chain_type(annotation) is False


def test_python_to_sql_still_accepts_bare_dict_aliases():
    # A `payload: Dict` field must keep mapping to a JSON column.
    assert python_to_sql(Dict) is JSON  # noqa: UP006
