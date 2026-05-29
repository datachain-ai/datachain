import copy
from typing import Optional, Union

import pytest
from pydantic import BaseModel

from datachain.lib.data_model import (
    DataModel,
    compute_model_fingerprint,
    is_chain_type,
    unwrap_optional,
)
from datachain.lib.model_store import ModelStore
from datachain.lib.utils import DataChainParamsError


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


# -- validate_reserved_names ---------------------------------------------


def test_reserved_is_null_field_rejected():
    with pytest.raises(DataChainParamsError, match="is reserved by DataChain"):

        class Bad(DataModel):
            is_null: bool = False


def test_non_reserved_field_names_ok():
    class Ok(DataModel):
        is_valid: bool = True
        null_count: int = 0
        isnull: bool = False  # close, but not exactly `is_null`

    assert Ok().is_valid is True
