from importlib import import_module

import pytest
from pydantic import create_model

from datachain.lib.data_model import DataModel
from datachain.lib.model_store import BUILTIN_MODELS, ModelStore
from datachain.lib.signal_schema import SignalSchema

pytestmark = pytest.mark.usefixtures("empty_model_store")
NAMES = ["BBox", "File", "Image", "Audio", "Video"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("builtin_cached", [False, True])
def test_user_model_restores_its_fields_despite_a_builtin_name(name, builtin_cached):
    user_model = create_model(
        name, __base__=DataModel, __module__="user_models", score=(float, ...)
    )
    saved = SignalSchema({"value": user_model}).serialize()
    ModelStore.store.clear()
    if builtin_cached:
        ModelStore.register(getattr(import_module(BUILTIN_MODELS[name]), name))
    restored = SignalSchema.deserialize(saved).values["value"]
    assert set(restored.model_fields) == {"score"}
    assert restored(score=0.9).score == 0.9


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("restored_user", [False, True])
def test_builtin_model_restores_after_a_colliding_user_model(name, restored_user):
    builtin = getattr(import_module(BUILTIN_MODELS[name]), name)
    saved = SignalSchema({"value": builtin}).serialize()
    user_model = create_model(
        name, __base__=DataModel, __module__="user_models", score=(float, ...)
    )
    user_saved = SignalSchema({"value": user_model}).serialize()
    if restored_user:
        ModelStore.store.clear()
        SignalSchema.deserialize(user_saved)
    assert SignalSchema.deserialize(saved).values["value"] is builtin


def test_reconstructed_user_model_is_reused():
    user_model = create_model(
        "BBox", __base__=DataModel, __module__="user_models", score=(float, ...)
    )
    saved = SignalSchema({"value": user_model}).serialize()
    ModelStore.store.clear()
    first = SignalSchema.deserialize(saved).values["value"]
    assert SignalSchema.deserialize(saved).values["value"] is first


def test_saved_origin_does_not_import_an_arbitrary_datachain_module(mocker):
    user_model = create_model(
        "BBox", __base__=DataModel, __module__="datachain.__main__", score=(float, ...)
    )
    saved = SignalSchema({"value": user_model}).serialize()
    ModelStore.store.clear()
    load = mocker.patch("datachain.lib.model_store.import_module")
    restored = SignalSchema.deserialize(saved).values["value"]
    assert set(restored.model_fields) == {"score"}
    load.assert_not_called()


@pytest.mark.parametrize("name", ["BBox", "File"])
def test_user_subclass_restores_with_its_builtin_parent(name):
    builtin = getattr(import_module(BUILTIN_MODELS[name]), name)
    user_model = create_model(
        name, __base__=builtin, __module__="user_models", score=(float, ...)
    )
    saved = SignalSchema({"value": user_model}).serialize()
    ModelStore.store.clear()
    ModelStore.register(builtin)
    restored = SignalSchema.deserialize(saved).values["value"]
    assert set(restored.model_fields) == {*builtin.model_fields, "score"}
    assert issubclass(restored, builtin)


@pytest.mark.parametrize("name", ["BBox", "File"])
def test_saved_user_model_reads_values_with_a_builtin_in_the_store(name, test_session):
    import datachain as dc

    user_model = create_model(
        name, __base__=DataModel, __module__="user_models", score=(float, ...)
    )
    dc.read_values(value=[user_model(score=0.9)], session=test_session).save("custom")
    ModelStore.store.clear()
    ModelStore.register(getattr(import_module(BUILTIN_MODELS[name]), name))
    values = dc.read_dataset("custom", session=test_session).to_values("value")
    assert [value.model_dump() for value in values] == [{"score": 0.9}]
