import copy
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Dict, Generic, List, TypeVar  # noqa: UP035

import pytest

# typing.TypedDict cannot be combined with Generic on Python 3.10
from typing_extensions import TypedDict

from datachain.lib.data_model import (
    DataModel,
    compute_model_fingerprint,
    is_chain_type,
)
from datachain.lib.model_store import ModelStore


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
        x: T

    assert is_chain_type(GenericRow[int]) is False


@pytest.mark.parametrize(
    "annotation",
    [
        # Abstract origins serialize as their bare origin name, losing their args,
        # so a saved signal would not survive a reload.
        Sequence[int],
        Mapping[str, int],
        Collection[int],
        Iterable[int],
    ],
    ids=["Sequence", "Mapping", "Collection", "Iterable"],
)
def test_is_chain_type_rejects_abstract_collection_roots(annotation):
    assert is_chain_type(annotation) is False


@pytest.mark.parametrize(
    "annotation",
    [list[int], dict[str, int], list[list[int]]],
    ids=["list", "dict", "nested-list"],
)
def test_is_chain_type_accepts_serializable_collections(annotation):
    assert is_chain_type(annotation) is True


@pytest.mark.parametrize(
    "annotation",
    [
        # `type_to_str` normalises a wrong arity rather than refusing it, so
        # `dict[str]` would be written back out as `dict[str, Any]`.
        list[int, str],  # type: ignore[misc]
        dict[str],  # type: ignore[misc]
        dict[str, int, float],  # type: ignore[misc]
        # `python_to_sql` mis-types tuples, so they are not valid root signals.
        tuple[int, ...],
        tuple[int, str],
        # `set` has no column type, and rejection must hold through nesting
        set[int],
        list[set[int]],
        # both arguments are validated, not just one
        dict[str, set[int]],
        dict[set[int], int],
        # Ellipsis counts toward arity during validation; it must not be
        # normalised away first.
        list[int, ...],  # type: ignore[misc]
        dict[str, int, ...],  # type: ignore[misc]
        list[..., int],  # type: ignore[misc]
    ],
    ids=[
        "list-two-args",
        "dict-one-arg",
        "dict-three-args",
        "variadic-tuple",
        "fixed-tuple",
        "set",
        "nested-set",
        "set-as-dict-value",
        "set-as-dict-key",
        "list-ellipsis",
        "dict-ellipsis",
        "list-leading-ellipsis",
    ],
)
def test_is_chain_type_rejects_unsupported_or_malformed_collection_annotations(
    annotation,
):
    assert is_chain_type(annotation) is False


def test_is_chain_type_accepts_bare_dict():
    assert is_chain_type(dict) is True


@pytest.mark.parametrize(
    "annotation",
    [
        List,  # noqa: UP006
        Dict,  # noqa: UP006
        tuple,
    ],
    ids=["typing-List", "typing-Dict", "tuple"],
)
def test_is_chain_type_rejects_unsupported_bare_collection_roots(annotation):
    assert is_chain_type(annotation) is False
