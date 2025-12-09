import copy

import pytest
from pydantic import Field

from datachain import DataModel
from datachain.lib.data_model import compute_model_fingerprint
from datachain.lib.file import File, TextFile
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import (
    SignalResolvingTypeError,
    SignalSchema,
    SignalSchemaError,
    create_feature_model,
)


class Info(DataModel):
    a: int
    b: int


def _reset_model_store():
    ModelStore.store = {}


@pytest.fixture(autouse=True)
def _autoreset_model_store():
    snapshot = copy.deepcopy(ModelStore.store)
    try:
        ModelStore.store = {}
        yield
    finally:
        ModelStore.store = snapshot


def test_partial_same_selection_reuses_name():
    schema = SignalSchema({"info": Info})

    schema.to_partial("info.a")
    selection = {"a": None}
    fingerprint = compute_model_fingerprint(Info, selection)
    base_partial_name = f"InfoPartial_{fingerprint[:10]}"

    names_after_first = set(ModelStore.store)
    assert names_after_first == {base_partial_name, f"{base_partial_name}_v1"}

    schema.to_partial("info.a")
    names_after_second = set(ModelStore.store)

    assert names_after_first == names_after_second


def test_partial_different_selection_differs():
    schema = SignalSchema({"info": Info})

    schema.to_partial("info.a")
    selection_a = {"a": None}
    fingerprint_a = compute_model_fingerprint(Info, selection_a)
    base_a = f"InfoPartial_{fingerprint_a[:10]}"

    schema.to_partial("info.b")
    selection_b = {"b": None}
    fingerprint_b = compute_model_fingerprint(Info, selection_b)
    base_b = f"InfoPartial_{fingerprint_b[:10]}"

    names = {name for name in ModelStore.store if name.startswith("InfoPartial_")}

    assert names == {base_a, f"{base_a}_v1", base_b, f"{base_b}_v1"}


def test_partial_name_collision_disambiguates():
    schema = SignalSchema({"info": Info})

    # Pre-register a conflicting model using the deterministic base name but
    # wrong fingerprint
    selection = {"a": None}
    fingerprint = compute_model_fingerprint(Info, selection)
    base_name, _ = ModelStore.parse_name_version(ModelStore.get_name(Info))
    colliding_name = f"{base_name}Partial_{fingerprint[:10]}@v1"

    rogue = create_feature_model(
        colliding_name,
        {"a": (int, None)},
        base=DataModel,
    )
    rogue._partial_fingerprint = "wrong"  # type: ignore[attr-defined]
    ModelStore.register(rogue)

    with pytest.raises(SignalSchemaError, match="partial model name collision"):
        schema.to_partial("info.a")


def test_partial_fingerprint_roundtrip_serialization():
    schema = SignalSchema({"info": Info})

    partial_schema = schema.to_partial("info.a")
    partial_model = ModelStore.to_pydantic(partial_schema.values["info"])
    orig_fp = getattr(partial_model, "_partial_fingerprint", None)

    serialized = partial_schema.serialize()
    _reset_model_store()

    roundtrip = SignalSchema.deserialize(serialized)
    rt_model = ModelStore.to_pydantic(roundtrip.values["info"])

    assert getattr(rt_model, "_partial_fingerprint", None) == orig_fp


def test_to_partial():
    schema = SignalSchema({"name": str, "age": float, "f": File})
    partial = schema.to_partial("name", "f.path")
    assert set(partial.values) == {"name", "f"}
    assert partial.values["name"] is str

    file_partial = partial.values["f"]
    assert issubclass(file_partial, DataModel)
    assert file_partial.__name__.startswith("FilePartial")
    assert set(file_partial.model_fields) == {"path"}
    assert file_partial.model_fields["path"].annotation is str

    serialized = partial.serialize()
    assert serialized["name"] == "str"
    assert serialized["f"] == ModelStore.get_name(file_partial)
    assert ModelStore.get_name(file_partial) in serialized["_custom_types"]


def test_to_partial_duplicate():
    schema = SignalSchema({"name": str, "age": float, "f1": File, "f2": File})
    partial = schema.to_partial("age", "f1.path", "f2.source")
    assert set(partial.values) == {"age", "f1", "f2"}
    assert partial.values["age"] is float

    f1_partial = partial.values["f1"]
    f2_partial = partial.values["f2"]

    assert issubclass(f1_partial, DataModel)
    assert issubclass(f2_partial, DataModel)
    assert f1_partial is not f2_partial

    assert f1_partial.__name__.startswith("FilePartial")
    assert f2_partial.__name__.startswith("FilePartial")

    assert set(f1_partial.model_fields) == {"path"}
    assert f1_partial.model_fields["path"].annotation is str

    assert set(f2_partial.model_fields) == {"source"}
    assert f2_partial.model_fields["source"].annotation is str

    serialized = partial.serialize()
    assert serialized["age"] == "float"
    assert serialized["f1"] == ModelStore.get_name(f1_partial)
    assert serialized["f2"] == ModelStore.get_name(f2_partial)
    assert ModelStore.get_name(f1_partial) in serialized["_custom_types"]
    assert ModelStore.get_name(f2_partial) in serialized["_custom_types"]


def test_to_partial_multiple_calls_unique_partial_names():
    schema = SignalSchema({"file": File, "name": str})

    partial1 = schema.to_partial("file.path")
    partial2 = schema.to_partial("file.source")

    file_partial_1 = partial1.values["file"]
    file_partial_2 = partial2.values["file"]

    # Each call should produce a distinct partial model to avoid name collisions
    assert file_partial_1 is not file_partial_2
    assert file_partial_1.__name__ != file_partial_2.__name__

    assert set(file_partial_1.model_fields) == {"path"}
    assert file_partial_1.model_fields["path"].annotation is str

    assert set(file_partial_2.model_fields) == {"source"}
    assert file_partial_2.model_fields["source"].annotation is str

    serialized_1 = partial1.serialize()
    serialized_2 = partial2.serialize()

    assert serialized_1["file"] != serialized_2["file"]
    assert serialized_1["file"] in serialized_1["_custom_types"]
    assert serialized_2["file"] in serialized_2["_custom_types"]


def test_to_partial_nested():
    class Custom(DataModel):
        foo: str
        file: File

    schema = SignalSchema({"name": str, "age": float, "f": File, "custom": Custom})
    partial = schema.to_partial("name", "f.path", "custom.file.source")
    assert set(partial.values) == {"name", "f", "custom"}
    assert partial.values["name"] is str

    f_partial = partial.values["f"]
    assert issubclass(f_partial, DataModel)
    assert set(f_partial.model_fields) == {"path"}
    assert f_partial.model_fields["path"].annotation is str
    assert f_partial.__name__.startswith("FilePartial")

    custom_partial = partial.values["custom"]
    assert issubclass(custom_partial, DataModel)
    assert set(custom_partial.model_fields) == {"file"}
    assert custom_partial.__name__.startswith("CustomPartial")

    nested_file_partial = custom_partial.model_fields["file"].annotation
    assert issubclass(nested_file_partial, DataModel)
    assert nested_file_partial is not f_partial
    assert set(nested_file_partial.model_fields) == {"source"}
    assert nested_file_partial.model_fields["source"].annotation is str
    assert nested_file_partial.__name__.startswith("FilePartial")

    serialized = partial.serialize()
    assert serialized["name"] == "str"
    assert serialized["f"] == ModelStore.get_name(f_partial)
    assert serialized["custom"] == ModelStore.get_name(custom_partial)
    assert ModelStore.get_name(nested_file_partial) in serialized["_custom_types"]


def test_get_file_signal():
    assert SignalSchema({"name": str, "f": File}).get_file_signal() == "f"
    assert SignalSchema({"name": str}).get_file_signal() is None


def test_to_partial_complex_signal_entire_file():
    """Test to_partial with entire complex signal requested."""
    schema = SignalSchema({"file": File, "name": str})
    partial = schema.to_partial("file")

    # Should return the entire File complex signal
    assert partial.values == {"file": File}


def test_to_partial_complex_nested_signal():
    class Custom(DataModel):
        src: File
        type: str

    schema = SignalSchema({"my_col": Custom, "name": str})
    partial = schema.to_partial("my_col.src")

    assert set(partial.values) == {"my_col"}

    custom_partial = partial.values["my_col"]
    assert issubclass(custom_partial, DataModel)
    assert set(custom_partial.model_fields) == {"src"}
    assert custom_partial.model_fields["src"].annotation is File
    assert custom_partial.__name__.startswith("CustomPartial")

    serialized = partial.serialize()
    assert serialized["my_col"] == ModelStore.get_name(custom_partial)
    assert "_custom_types" in serialized


def test_to_partial_complex_deeply_nested_signal():
    """Test to_partial with deeply nested complex signals (3+ levels)."""
    from datachain.lib.file import ImageFile

    class Level1(DataModel):
        image: ImageFile
        name: str

    class Level2(DataModel):
        level1: Level1
        category: str

    class Level3(DataModel):
        level2: Level2
        id: str

    schema = SignalSchema({"deep": Level3, "simple": str})

    # Test deeply nested complex signal
    partial = schema.to_partial("deep.level2.level1.image")

    deep_partial = partial.values["deep"]
    level2_partial = deep_partial.model_fields["level2"].annotation
    level1_partial = level2_partial.model_fields["level1"].annotation

    assert issubclass(level1_partial, DataModel)
    assert set(level1_partial.model_fields) == {"image"}
    assert level1_partial.model_fields["image"].annotation is ImageFile
    assert deep_partial.__name__.startswith("Level3Partial")
    assert level2_partial.__name__.startswith("Level2Partial")
    assert level1_partial.__name__.startswith("Level1Partial")

    serialized = partial.serialize()
    assert serialized["deep"] == ModelStore.get_name(deep_partial)
    assert ModelStore.get_name(level1_partial) in serialized["_custom_types"]


def test_to_partial_complex_nested_multiple_complex_signals():
    """Test to_partial with multiple nested complex signals."""

    class Container(DataModel):
        file1: File
        file2: TextFile
        name: str

    schema = SignalSchema({"container": Container, "simple": str})

    # Request multiple nested complex signals
    partial = schema.to_partial("container.file1", "container.file2")

    assert set(partial.values) == {"container"}

    container_partial = partial.values["container"]
    assert issubclass(container_partial, DataModel)
    assert container_partial.model_fields["file1"].annotation is File
    assert container_partial.model_fields["file2"].annotation is TextFile
    assert container_partial.__name__.startswith("ContainerPartial")

    serialized = partial.serialize()
    assert serialized["container"] == ModelStore.get_name(container_partial)


def test_to_partial_complex_nested_mixed_complex_and_simple():
    """Test to_partial with mix of nested complex signals and simple fields."""

    class Container(DataModel):
        file: File
        name: str
        count: int

    schema = SignalSchema({"container": Container, "simple": str})

    # Request mix of nested complex signal and simple field
    partial = schema.to_partial("container.file", "container.name", "simple")

    assert set(partial.values) == {"container", "simple"}
    assert partial.values["simple"] is str

    container_partial = partial.values["container"]
    assert issubclass(container_partial, DataModel)
    assert container_partial.model_fields["file"].annotation is File
    assert container_partial.model_fields["name"].annotation is str
    assert container_partial.__name__.startswith("ContainerPartial")

    serialized = partial.serialize()
    assert serialized["container"] == ModelStore.get_name(container_partial)
    assert serialized["simple"] == "str"


def test_to_partial_complex_nested_same_type_different_paths():
    """Test to_partial with same complex type accessed via different nested paths."""

    class Container1(DataModel):
        file: File
        name: str

    class Container2(DataModel):
        file: File
        category: str

    schema = SignalSchema({"cont1": Container1, "cont2": Container2})

    # Request same complex type from different nested paths
    partial = schema.to_partial("cont1.file", "cont2.file")

    assert set(partial.values) == {"cont1", "cont2"}

    cont1_partial = partial.values["cont1"]
    cont2_partial = partial.values["cont2"]
    assert issubclass(cont1_partial, DataModel)
    assert issubclass(cont2_partial, DataModel)
    assert cont1_partial is not cont2_partial

    assert cont1_partial.model_fields["file"].annotation is File
    assert cont2_partial.model_fields["file"].annotation is File
    assert cont1_partial.__name__.startswith("Container1Partial")
    assert cont2_partial.__name__.startswith("Container2Partial")

    serialized = partial.serialize()
    assert serialized["cont1"] == ModelStore.get_name(cont1_partial)
    assert serialized["cont2"] == ModelStore.get_name(cont2_partial)


def test_to_partial_complex_signal_file_single_field():
    """Test to_partial with File complex signal - single field."""
    schema = SignalSchema({"name": str, "file": File})
    partial = schema.to_partial("file.path")

    assert set(partial.values) == {"file"}

    file_partial = partial.values["file"]
    assert issubclass(file_partial, DataModel)
    assert set(file_partial.model_fields) == {"path"}
    assert file_partial.model_fields["path"].annotation is str
    assert file_partial.__name__.startswith("FilePartial")

    serialized = partial.serialize()
    assert serialized["file"] == ModelStore.get_name(file_partial)


def test_to_partial_complex_signal_mixed_entire_and_fields():
    """Test to_partial with mix of entire complex signal and specific fields."""
    schema = SignalSchema({"file1": File, "file2": File, "name": str})
    partial = schema.to_partial("file1", "file2.path", "name")

    assert set(partial.values) == {"file1", "file2", "name"}

    assert partial.values["file1"] is File
    assert partial.values["name"] is str

    file2_partial = partial.values["file2"]
    assert issubclass(file2_partial, DataModel)
    assert set(file2_partial.model_fields) == {"path"}
    assert file2_partial.model_fields["path"].annotation is str
    assert file2_partial.__name__.startswith("FilePartial")

    serialized = partial.serialize()
    assert serialized["file1"] == "File@v1"
    assert serialized["file2"] == ModelStore.get_name(file2_partial)
    assert serialized["name"] == "str"
    assert ModelStore.get_name(file2_partial) in serialized["_custom_types"]


def test_to_partial_complex_signal_multiple_entire_files():
    """Test to_partial with multiple entire complex signals."""
    schema = SignalSchema({"file1": File, "file2": File, "name": str})
    partial = schema.to_partial("file1", "file2")

    assert set(partial.values) == {"file1", "file2"}
    assert partial.values["file1"] is File
    assert partial.values["file2"] is File


def test_to_partial_complex_signal_nested_entire():
    """Test to_partial with nested complex signal - entire parent."""

    class Container(DataModel):
        name: str
        file: File

    schema = SignalSchema({"container": Container, "simple": str})
    partial = schema.to_partial("container")

    assert set(partial.values) == {"container"}

    container_type = partial.values["container"]
    assert issubclass(container_type, DataModel)
    assert set(container_type.model_fields) == {"name", "file"}
    assert container_type.model_fields["name"].annotation is str
    assert container_type.model_fields["file"].annotation is File


def test_to_partial_complex_signal_empty_request():
    """Test to_partial with no columns requested."""
    schema = SignalSchema({"file": File, "name": str})
    partial = schema.to_partial()

    # Should return empty schema
    assert partial.values == {}


def test_to_partial_complex_signal_error_invalid_signal():
    """Test to_partial with invalid signal name."""
    schema = SignalSchema({"file": File})

    with pytest.raises(
        SignalSchemaError, match="Column nonexistent not found in the schema"
    ):
        schema.to_partial("nonexistent")


def test_to_partial_complex_signal_error_invalid_field():
    """Test to_partial with invalid field in complex signal."""
    schema = SignalSchema({"file": File})

    with pytest.raises(
        SignalSchemaError,
        match=r"Field nonexistent not found in custom type File",
    ):
        schema.to_partial("file.nonexistent")


def test_to_partial_rejects_non_string_column():
    schema = SignalSchema({"name": str})

    with pytest.raises(
        SignalResolvingTypeError, match=r"to_partial\(\) supports only `str` type"
    ):
        schema.to_partial(123)


def test_to_partial_nested_on_scalar_column():
    schema = SignalSchema({"name": str})

    with pytest.raises(
        SignalSchemaError, match=r"Column name\.path not found in the schema"
    ):
        schema.to_partial("name.path")


def test_to_partial_prefers_whole_selection_over_fields():
    schema = SignalSchema({"file": File})

    partial = schema.to_partial("file", "file.path")

    assert partial.values == {"file": File}


def test_to_partial_propagates_optional_default():
    class WithDefault(DataModel):
        required: int
        optional: str | None = Field(default="fallback")

    schema = SignalSchema({"data": WithDefault})

    partial = schema.to_partial("data.optional")
    partial_model = ModelStore.to_pydantic(partial.values["data"])
    assert partial_model is not None

    assert set(partial_model.model_fields) == {"optional"}

    optional_field = partial_model.model_fields["optional"]

    original_field = WithDefault.model_fields["optional"]

    assert optional_field.default == "fallback"
    assert optional_field.annotation is original_field.annotation


def test_to_partial_does_not_create_model_when_all_fields_selected():
    class WithDefault(DataModel):
        required: int
        optional: str | None = Field(default="fallback")

    schema = SignalSchema({"data": WithDefault})

    partial = schema.to_partial("data.required", "data.optional")

    # When the selection includes all fields, return the original model type.
    assert partial.values == {"data": WithDefault}
