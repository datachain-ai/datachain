import copy

import pytest

from datachain.lib.data_model import DataModel, compute_model_fingerprint
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
