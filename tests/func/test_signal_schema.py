import copy
import uuid

import datachain as dc
from datachain import DataModel, func
from datachain.lib.model_store import ModelStore


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
