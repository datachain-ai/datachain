from importlib import import_module
from importlib.metadata import PackageNotFoundError

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


@pytest.mark.parametrize("name", ["BBox", "File"])
def test_legacy_model_without_origin_does_not_import_a_builtin(name, mocker):
    load = mocker.spy(import_module("datachain.lib.model_store"), "import_module")
    schema = {
        "value": f"{name}@v1",
        "_custom_types": {f"{name}@v1": {"score": "float"}},
    }
    restored = SignalSchema.deserialize(schema).values["value"]
    assert restored(score=0.9).model_dump() == {"score": 0.9}
    load.assert_not_called()


@pytest.mark.parametrize("name", ["BBox", "File", "CustomBox"])
def test_legacy_model_reuses_the_registered_class(name):
    registered = create_model(
        name, __base__=DataModel, __module__="user_models", score=(float, ...)
    )
    schema = {
        "value": f"{name}@v1",
        "_custom_types": {f"{name}@v1": {"score": "float"}},
    }
    assert SignalSchema.deserialize(schema).values["value"] is registered


@pytest.mark.parametrize("name", ["BBox", "File", "CustomBox"])
@pytest.mark.parametrize(
    "saved_module,live_module",
    [("__main__", "user_models"), ("user_models", "__main__")],
)
def test_registered_user_model_is_reused_across_modules(
    name, saved_module, live_module
):
    saved_model = create_model(
        name, __base__=DataModel, __module__=saved_module, score=(float, ...)
    )
    saved = SignalSchema({"value": saved_model}).serialize()
    registered = create_model(
        name, __base__=DataModel, __module__=live_module, score=(float, ...)
    )
    assert SignalSchema.deserialize(saved).values["value"] is registered


@pytest.mark.parametrize("wrappers", [0, 1, 2])
@pytest.mark.parametrize("metadata", [False, True])
def test_builtin_missing_optional_dependency_is_a_miss(mocker, wrappers, metadata):
    error = (
        PackageNotFoundError("datasets")
        if metadata
        else ModuleNotFoundError("No module named 'datasets'", name="datasets")
    )
    for _ in range(wrappers):
        outer = ImportError("Missing dependencies for Hugging Face")
        outer.__cause__ = error
        error = outer
    mocker.patch("datachain.lib.model_store.import_module", side_effect=error)
    assert ModelStore.get("HFAudio", 1) is None


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("model_name", ["McapFile", "HFAudio"])
@pytest.mark.parametrize(
    "error",
    [
        ImportError(
            "cannot import name 'broken_internal_symbol'", name="datachain.lib.file"
        ),
        ModuleNotFoundError(
            "No module named 'datachain.lib.missing'", name="datachain.lib.missing"
        ),
        ModuleNotFoundError("No module named 'datachain'", name="datachain"),
        ModuleNotFoundError("Unidentified import failure"),
        ModuleNotFoundError("No module named 'numpy'", name="numpy"),
        PackageNotFoundError("datachain"),
        PackageNotFoundError(),
    ],
)
def test_unexpected_builtin_import_errors_propagate(mocker, wrapped, model_name, error):
    if wrapped:
        outer = ImportError("Failed to import builtin")
        outer.__cause__ = error
        error = outer
    mocker.patch("datachain.lib.model_store.import_module", side_effect=error)
    with pytest.raises(ImportError) as caught:
        ModelStore.get(model_name, 1)
    assert caught.value is error


@pytest.mark.parametrize("version", [1, 999])
def test_lazy_builtin_lookup_honors_the_requested_version(version):
    from datachain.lib.mcap import McapFile

    assert ModelStore.get("McapFile", version) is (McapFile if version == 1 else None)


@pytest.mark.parametrize("name", ["BBox", "File"])
@pytest.mark.parametrize("builtin_cached", [False, True])
def test_saved_user_model_with_legacy_metadata_reads_values(
    test_session, name, builtin_cached
):
    import datachain as dc

    user_model = create_model(name, __base__=DataModel, score=(float, ...))
    dc.read_values(value=[user_model(score=0.9)], session=test_session).save("legacy")
    catalog = test_session.catalog
    dataset = catalog.get_dataset("legacy", versions=None)
    catalog.metastore.update_dataset_version(
        dataset,
        dataset.latest_version,
        feature_schema={
            "value": f"{name}@v1",
            "_custom_types": {f"{name}@v1": {"score": "float"}},
        },
    )
    ModelStore.store.clear()
    if builtin_cached:
        ModelStore.register(getattr(import_module(BUILTIN_MODELS[name]), name))
    values = dc.read_dataset("legacy", session=test_session).to_values("value")
    assert [value.model_dump() for value in values] == [{"score": 0.9}]


@pytest.mark.parametrize("name", ["BBox", "File"])
def test_legacy_builtin_model_reuses_the_registered_class(name):
    builtin = getattr(import_module(BUILTIN_MODELS[name]), name)
    ModelStore.register(builtin)
    saved = SignalSchema({"value": builtin}).serialize()
    saved["_custom_types"] = {
        name: record["fields"] for name, record in saved["_custom_types"].items()
    }
    assert SignalSchema.deserialize(saved).values["value"] is builtin


@pytest.mark.parametrize("name", ["DatasetInfo", "HFAudio"])
def test_reconstructed_builtin_keeps_its_origin_and_class_identity(name, mocker):
    module = BUILTIN_MODELS[name]
    previous = create_model(
        name, __base__=DataModel, __module__=module, score=(float, ...)
    )
    saved = SignalSchema({"a": previous, "b": previous}).serialize()
    ModelStore.store.clear()

    def load(target):
        if target == "datachain.lib.hf":
            raise ModuleNotFoundError("No module named 'datasets'", name="datasets")
        return import_module(target)

    loader = mocker.patch("datachain.lib.model_store.import_module", side_effect=load)
    schema = SignalSchema.deserialize(saved)
    restored = schema.values["a"]
    assert schema.values["b"] is restored
    assert SignalSchema.deserialize(saved).values["a"] is restored
    resaved = schema.serialize()
    assert resaved["_custom_types"][f"{name}@v1"]["bases"][0][1] == module
    assert SignalSchema.deserialize(resaved).values["a"] is restored
    assert restored(score=0.9).model_dump() == {"score": 0.9}
    loader.assert_called_once_with(module)

    child_name = f"Custom{name}"
    child = create_model(
        child_name, __base__=restored, __module__="user_models", extra=(int, ...)
    )
    assert ModelStore.get_name(child) == f"{child_name}@v1"
    assert ModelStore.get_module(child) == "user_models"
    child_schema = SignalSchema({"value": child}).serialize()
    assert child_schema["value"] == f"{child_name}@v1"
    assert (
        child_schema["_custom_types"][f"{child_name}@v1"]["bases"][0][1]
        == "user_models"
    )
    assert ModelStore.get(name, 1, module=module) is restored
