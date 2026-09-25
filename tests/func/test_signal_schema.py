import copy
import sys
import uuid
from collections.abc import Sequence

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, create_model

import datachain as dc
from datachain import DataModel, func
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchemaWarning


def test_partial_collision_on_dataset_reload(test_session):
    """
    Simulate two runs:
    1) Create and save a dataset whose schema includes a partial of Info
       (partition by info.a).
    2) Reset the ModelStore, then create a different partial with the same
       generated name (partition by info.b), and finally read the saved
       dataset back.

    If partial names collide without structural checks, the dataset
    deserialization will reuse the incompatible partial, causing a schema
    mismatch.
    """

    class Info(DataModel):
        a: int
        b: str

    def make_chain():
        return dc.read_values(a=[1, 2], b=["x", "y"], session=test_session).map(
            lambda a, b: Info(a=a, b=b), params=["a", "b"], output={"info": Info}
        )

    # Preserve and restore ModelStore across the test to avoid leaking state.
    original_store = copy.deepcopy(ModelStore.store)
    try:
        # First run: build and save dataset using a partial on info.a.
        ModelStore.store = {}
        ds_name = f"partial-collision-{uuid.uuid4()}"
        make_chain().group_by(cnt=func.count(), partition_by="info.a").save(ds_name)

        partials_run1 = []
        for name, versions in ModelStore.store.items():
            if name.startswith("InfoPartial_"):
                partials_run1.extend(versions.values())
        assert len(partials_run1) == 2
        assert all(set(p.model_fields.keys()) == {"a"} for p in partials_run1)

        # Second run: reset registry and create a different partial with the
        # same base name but a different structure (partition by info.b).
        ModelStore.store = {}
        make_chain().group_by(cnt=func.count(), partition_by="info.b")

        # Now read back the saved dataset; it should bring in the original
        # partial definition and register it in ModelStore.
        dc.read_dataset(ds_name, session=test_session)

        partials = {
            name: model
            for name, versions in ModelStore.store.items()
            if name.startswith("InfoPartial_")
            for model in versions.values()
        }

        # There should be two distinct partial bases (info.a and info.b) registered
        # after reading the dataset.
        fields_by_base: dict[str, set[str]] = {}
        for name, model in partials.items():
            base = name.removesuffix("_v1")
            fields_by_base.setdefault(base, set()).update(model.model_fields.keys())

        assert len(fields_by_base) == 2
        actual_fields = sorted(
            tuple(sorted(fields)) for fields in fields_by_base.values()
        )
        assert actual_fields == [("a",), ("b",)]
    finally:
        ModelStore.store = original_store


@pytest.mark.parametrize("container", ["list", "dict", "nested_list"])
@pytest.mark.parametrize("required", [False, True], ids=["defaulted", "required"])
def test_serialized_aliases_readback(test_session, container, required):
    class Aliased(BaseModel):
        model_config = ConfigDict(serialize_by_alias=True)

        value: int = Field(... if required else 0, alias="externalValue")

    class Wrapper(BaseModel):
        values: list[Aliased]

    aliased = Aliased(externalValue=7)
    item = {
        "list": [aliased],
        "dict": {"a": aliased},
        "nested_list": Wrapper(values=[aliased]),
    }[container]
    dataset_name = f"serialized-alias-{container}-{required}"
    dc.read_values(
        session=test_session,
        settings={"prefetch": False},
        item=[item],
    ).save(dataset_name)

    restored = dc.read_dataset(dataset_name, session=test_session).to_list("item")[0][0]

    if container == "list":
        assert restored[0].value == 7
    elif container == "dict":
        assert restored["a"].value == 7
    else:
        assert restored.values[0].value == 7


def test_root_model_readback(test_session):
    class Scalar(RootModel[int]):
        pass

    dc.read_values(
        session=test_session,
        settings={"prefetch": False},
        item=[Scalar(7)],
    ).save("root-model-readback")

    restored = dc.read_dataset("root-model-readback", session=test_session).to_list(
        "item"
    )[0][0]

    assert restored.root == 7


def test_saved_sequence_model_reads_with_drifted_child(test_session, monkeypatch):
    child_name = "SequenceChildForReadRegression"
    outer_name = "OuterSequenceForReadRegression"
    stored_child = create_model(child_name, __module__=__name__, value=(int, ...))
    stored_outer = create_model(
        outer_name,
        __module__=__name__,
        items=(Sequence[stored_child], ...),  # type: ignore[valid-type]
    )
    dataset_name = f"sequence-child-drift-{uuid.uuid4()}"
    dc.read_values(
        session=test_session,
        settings={"prefetch": False},
        item=[stored_outer(items=[stored_child(value=7)])],
    ).save(dataset_name)

    current_child = create_model(child_name, __module__=__name__, value=(str, ...))
    current_outer = create_model(
        outer_name,
        __module__=__name__,
        items=(Sequence[current_child], ...),  # type: ignore[valid-type]
    )
    monkeypatch.setattr(sys.modules[__name__], child_name, current_child, raising=False)
    monkeypatch.setattr(sys.modules[__name__], outer_name, current_outer, raising=False)
    monkeypatch.setattr(ModelStore, "store", {})

    with pytest.warns(SignalSchemaWarning) as caught_warnings:
        restored = dc.read_dataset(dataset_name, session=test_session).to_list("item")[
            0
        ][0]

    assert any(
        "does not preserve its type arguments" in str(warning.message)
        for warning in caught_warnings
    )
    assert restored.items == [{"value": 7}]


def test_serialized_aliases_readback_for_union_items(test_session):
    class AliasedValue(BaseModel):
        model_config = ConfigDict(serialize_by_alias=True)

        value: int = Field(alias="externalValue")

    class AliasedLabel(BaseModel):
        model_config = ConfigDict(serialize_by_alias=True)

        label: str = Field(alias="externalLabel")

    class Wrapper(BaseModel):
        items: list[AliasedValue | AliasedLabel]

    dataset_name = f"serialized-alias-union-{uuid.uuid4()}"
    dc.read_values(
        session=test_session,
        settings={"prefetch": False},
        item=[
            Wrapper(
                items=[
                    AliasedValue(externalValue=7),
                    AliasedLabel(externalLabel="label"),
                ]
            )
        ],
    ).save(dataset_name)

    restored = dc.read_dataset(dataset_name, session=test_session).to_list("item")[0][0]

    assert isinstance(restored.items[0], AliasedValue)
    assert restored.items[0].value == 7
    assert isinstance(restored.items[1], AliasedLabel)
    assert restored.items[1].label == "label"
