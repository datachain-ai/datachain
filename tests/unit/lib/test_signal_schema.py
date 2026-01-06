import json
from datetime import datetime
from typing import (
    Any,
    Final,
    ForwardRef,
    Optional,
    Union,
    get_args,
    get_origin,
)

import pytest
from pydantic import ValidationError

from datachain import Column, DataModel, Sys, func
from datachain.lib.convert.flatten import flatten
from datachain.lib.file import File, TextFile
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import (
    SetupError,
    SignalRemoveError,
    SignalResolvingError,
    SignalResolvingTypeError,
    SignalSchema,
    SignalSchemaError,
    SignalSchemaWarning,
)
from datachain.lib.utils import DataChainColumnError
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Float32,
    Float64,
    Int,
    Int32,
    Int64,
    String,
    UInt32,
    UInt64,
)


@pytest.fixture
def nested_file_schema():
    class _MyFile(File):
        ref: str
        nested_file: File

    schema = {"name": str, "age": float, "f": File, "my_f": _MyFile}

    return SignalSchema(schema)


class MyType1(DataModel):
    aa: int
    bb: str


class MyType2(DataModel):
    name: str
    deep: MyType1


class MyType3(MyType1):
    name: str


class MyTypeComplex(DataModel):
    name: str
    items: list[MyType1]
    lookup: dict[str, MyType2]


class MyTypeComplexOld(DataModel):
    name: str
    items: list[MyType1]
    lookup: dict[str, MyType2]


def test_deserialize_basic():
    # Make this test robust to other tests clearing the global ModelStore.
    ModelStore.register(File)
    stored = {"name": "str", "count": "int", "file": "File@v1"}
    signals = SignalSchema.deserialize(stored)

    assert len(signals.values) == 3
    assert signals.values.keys() == stored.keys()
    assert list(signals.values.values()) == [str, int, File]


def test_deserialize_error():
    SignalSchema.deserialize({})

    with pytest.raises(SignalSchemaError):
        SignalSchema.deserialize(json.dumps({"name": "str"}))

    with pytest.raises(SignalSchemaError):
        SignalSchema.deserialize({"name": [1, 2, 3]})

    with pytest.raises(SignalSchemaError):
        SignalSchema.deserialize({"name": "Union[str,"})

    with pytest.warns(SignalSchemaWarning):
        # Warn if unknown fields are encountered - don't throw an exception to ensure
        # that all data can be shown.
        SignalSchema.deserialize({"name": "unknown"})


def test_serialize_simple():
    schema = {
        "name": str,
        "age": float,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 2
    assert signals["name"] == "str"
    assert signals["age"] == "float"
    assert "_custom_types" not in signals


def test_serialize_basic():
    schema = {
        "name": str,
        "age": float,
        "f": File,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 4
    assert signals["name"] == "str"
    assert signals["age"] == "float"
    assert signals["f"] == "File@v1"
    assert "File@v1" in signals["_custom_types"]


def test_serialize_warns_for_unresolved_forwardref():
    schema = SignalSchema({"x": ForwardRef("X")})  # type: ignore[arg-type]

    with pytest.warns(SignalSchemaWarning, match=r"Unable to determine name of type"):
        assert schema.serialize()["x"] == "Any"


def test_feature_schema_serialize_optional():
    schema = {
        "name": str | None,
        "feature": MyType1 | None,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Optional[str]"
    assert signals["feature"] == "Optional[MyType1@v1]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {
            "schema_version": 2,
            "fields": {"aa": "int", "bb": "str"},
            "name": "MyType1@v1",
            "bases": [
                ("MyType1", "tests.unit.lib.test_signal_schema", "MyType1@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        }
    }

    deserialized_schema = SignalSchema.deserialize(signals)
    assert deserialized_schema.values == schema


def test_feature_schema_serialize_list():
    schema = {
        "name": str | None,
        "features": list[MyType1],
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Optional[str]"
    assert signals["features"] == "list[MyType1@v1]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {
            "schema_version": 2,
            "fields": {"aa": "int", "bb": "str"},
            "name": "MyType1@v1",
            "bases": [
                ("MyType1", "tests.unit.lib.test_signal_schema", "MyType1@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        }
    }

    deserialized_schema = SignalSchema.deserialize(signals)
    assert deserialized_schema.values == schema


def test_serialize_tuple_and_variadic_tuple():
    schema = {
        "coords": tuple[float, float],
        "direction": Optional[tuple[int, ...]],
    }

    serialized = SignalSchema(schema).serialize()

    assert serialized["coords"] == "tuple[float, float]"
    assert serialized["direction"] == "Optional[tuple[int, ...]]"

    deserialized = SignalSchema.deserialize(serialized)

    coords_type = deserialized.values["coords"]
    assert get_origin(coords_type) is tuple
    assert get_args(coords_type) == (float, float)

    direction_type = deserialized.values["direction"]
    dir_origin = get_origin(direction_type)
    dir_args = get_args(direction_type)
    assert dir_origin in (Union, type(Union[int, None]))  # Optional
    tuple_arg = (
        dir_args[0] if dir_args and dir_args[0] is not type(None) else dir_args[1]
    )
    assert get_origin(tuple_arg) is tuple
    assert get_args(tuple_arg) == (int, Ellipsis)


def test_schema_or_rejects_type_change():
    base = SignalSchema({"foo": int})
    new = SignalSchema({"foo": str})

    with pytest.raises(DataChainColumnError, match="different type"):
        _ = base | new


def test_schema_or_rejects_root_conflict():
    base = SignalSchema({"feature": MyType1})
    new = SignalSchema({"feature.extra": int})

    with pytest.raises(DataChainColumnError, match="root"):
        _ = base | new


def test_schema_or_allows_sys_root():
    base = SignalSchema({"foo": int})
    new = SignalSchema({"sys": Sys})

    combined = base | new
    assert combined.values["sys"] is Sys


def test_schema_or_rejects_sys_override():
    base = SignalSchema({"sys": Sys})
    new = SignalSchema({"sys": dict})

    with pytest.raises(DataChainColumnError, match="different type"):
        _ = base | new


def test_schema_or_allows_identical_signal():
    base = SignalSchema({"foo": int})
    combined = base | SignalSchema({"foo": int})

    assert combined.values["foo"] is int


def test_feature_schema_serialize_list_old():
    schema = {
        "name": str | None,
        "features": list[MyType1],
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Optional[str]"
    assert signals["features"] == "list[MyType1@v1]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {
            "schema_version": 2,
            "fields": {"aa": "int", "bb": "str"},
            "name": "MyType1@v1",
            "bases": [
                ("MyType1", "tests.unit.lib.test_signal_schema", "MyType1@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        }
    }

    new_schema = {
        "name": str | None,
        "features": list[MyType1],
    }

    deserialized_schema = SignalSchema.deserialize(signals)
    assert deserialized_schema.values == new_schema


def test_feature_schema_serialize_nested_types():
    schema = {
        "name": str | None,
        "feature_nested": MyType2 | None,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Optional[str]"
    assert signals["feature_nested"] == "Optional[MyType2@v1]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {
            "schema_version": 2,
            "fields": {"aa": "int", "bb": "str"},
            "name": "MyType1@v1",
            "bases": [
                ("MyType1", "tests.unit.lib.test_signal_schema", "MyType1@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
        "MyType2@v1": {
            "schema_version": 2,
            "fields": {"name": "str", "deep": "MyType1@v1"},
            "name": "MyType2@v1",
            "bases": [
                ("MyType2", "tests.unit.lib.test_signal_schema", "MyType2@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
    }

    deserialized_schema = SignalSchema.deserialize(signals)
    assert deserialized_schema.values == schema


def test_feature_schema_serialize_nested_duplicate_types():
    schema = {
        "name": str | None,
        "feature_nested": MyType2 | None,
        "feature_not_nested": MyType1 | None,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 4
    assert signals["name"] == "Optional[str]"
    assert signals["feature_nested"] == "Optional[MyType2@v1]"
    assert signals["feature_not_nested"] == "Optional[MyType1@v1]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {
            "schema_version": 2,
            "fields": {"aa": "int", "bb": "str"},
            "name": "MyType1@v1",
            "bases": [
                ("MyType1", "tests.unit.lib.test_signal_schema", "MyType1@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
        "MyType2@v1": {
            "schema_version": 2,
            "fields": {"name": "str", "deep": "MyType1@v1"},
            "name": "MyType2@v1",
            "bases": [
                ("MyType2", "tests.unit.lib.test_signal_schema", "MyType2@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
    }

    deserialized_schema = SignalSchema.deserialize(signals)
    assert deserialized_schema.values == schema


def test_feature_schema_serialize_complex():
    schema = {
        "name": str | None,
        "feature": MyTypeComplex | None,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Optional[str]"
    assert signals["feature"] == "Optional[MyTypeComplex@v1]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {
            "schema_version": 2,
            "fields": {"aa": "int", "bb": "str"},
            "name": "MyType1@v1",
            "bases": [
                ("MyType1", "tests.unit.lib.test_signal_schema", "MyType1@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
        "MyType2@v1": {
            "schema_version": 2,
            "fields": {"name": "str", "deep": "MyType1@v1"},
            "name": "MyType2@v1",
            "bases": [
                ("MyType2", "tests.unit.lib.test_signal_schema", "MyType2@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
        "MyTypeComplex@v1": {
            "schema_version": 2,
            "fields": {
                "name": "str",
                "items": "list[MyType1@v1]",
                "lookup": "dict[str, MyType2@v1]",
            },
            "name": "MyTypeComplex@v1",
            "bases": [
                (
                    "MyTypeComplex",
                    "tests.unit.lib.test_signal_schema",
                    "MyTypeComplex@v1",
                ),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
    }

    deserialized_schema = SignalSchema.deserialize(signals)
    assert deserialized_schema.values == schema


def test_feature_schema_serialize_complex_old():
    schema = {
        "name": str | None,
        "feature": MyTypeComplexOld | None,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Optional[str]"
    assert signals["feature"] == "Optional[MyTypeComplexOld@v1]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {
            "schema_version": 2,
            "fields": {"aa": "int", "bb": "str"},
            "name": "MyType1@v1",
            "bases": [
                ("MyType1", "tests.unit.lib.test_signal_schema", "MyType1@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
        "MyType2@v1": {
            "schema_version": 2,
            "fields": {"name": "str", "deep": "MyType1@v1"},
            "name": "MyType2@v1",
            "bases": [
                ("MyType2", "tests.unit.lib.test_signal_schema", "MyType2@v1"),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
        "MyTypeComplexOld@v1": {
            "schema_version": 2,
            "fields": {
                "name": "str",
                "items": "list[MyType1@v1]",
                "lookup": "dict[str, MyType2@v1]",
            },
            "name": "MyTypeComplexOld@v1",
            "bases": [
                (
                    "MyTypeComplexOld",
                    "tests.unit.lib.test_signal_schema",
                    "MyTypeComplexOld@v1",
                ),
                ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                ("BaseModel", "pydantic.main", None),
                ("object", "builtins", None),
            ],
            "hidden_fields": [],
        },
    }


def test_serialize_from_column():
    signals = SignalSchema.from_column_types({"age": Float, "name": String}).values

    assert len(signals) == 2
    assert signals["name"] is str
    assert signals["age"] is float


def test_serialize_from_column_error():
    with pytest.raises(SignalSchemaError):
        SignalSchema.from_column_types({"age": Float, "wrong_type": File})


def test_to_udf_spec():
    schema = SignalSchema(
        {"age": float, "address": str, "f": File, "init": str},
        {"init": lambda: 37},
    )

    spec = schema.to_udf_spec()

    assert len(spec) == 2 + len(File.model_fields)

    assert "age" in spec
    assert spec["age"] == Float

    assert "address" in spec
    assert spec["address"] == String

    assert "f__path" in spec
    assert spec["f__path"] == String

    assert "f__size" in spec
    assert spec["f__size"] == Int64


def test_resolve():
    schema = SignalSchema({"age": float, "address": str, "f": MyType1})

    new = schema.resolve("age", "f.aa", "f.bb")
    assert isinstance(new, SignalSchema)

    signals = new.values
    assert len(signals) == 3
    assert {"age", "f.aa", "f.bb"} == signals.keys()
    assert signals["age"] is float
    assert signals["f.aa"] is int
    assert signals["f.bb"] is str


@pytest.mark.parametrize(
    "names",
    [
        ["bar"],
        ["bar.age"],
        ["age.foo"],
        ["age", "f.aa", "f.bb", "f.cc"],
        ["age", "f.aa", "f.bb", 37],
    ],
)
def test_resolve_error(names):
    schema = SignalSchema({"age": float, "address": str, "f": MyType1})
    with pytest.raises(SignalResolvingError):
        schema.resolve(*names)


def test_clone_without_file_signals():
    schema = SignalSchema({**File.model_fields, "name": str, "age": float})

    new = schema.clone_without_file_signals()
    assert isinstance(new, SignalSchema)

    signals = new.values
    assert len(signals) == 2
    assert {"name", "age"} == signals.keys()
    assert signals["name"] is str
    assert signals["age"] is float


def test_clone_without_sys_signals():
    schema = SignalSchema({"sys": Sys, "name": str, "age": float})

    new = schema.clone_without_sys_signals()
    assert isinstance(new, SignalSchema)

    signals = new.values
    assert len(signals) == 2
    assert {"name", "age"} == signals.keys()
    assert signals["name"] is str
    assert signals["age"] is float


def test_merge():
    schema = SignalSchema({"name": str, "age": float})
    another_schema = SignalSchema({"age": int, "address": str, "f": MyType1})

    new = schema.merge(another_schema, "s2_")
    assert isinstance(new, SignalSchema)

    signals = new.values
    assert len(signals) == 5
    assert {"name", "age", "s2_age", "address", "f"} == signals.keys()
    assert signals["name"] is str
    assert signals["age"] is float
    assert signals["s2_age"] is int
    assert signals["address"] is str
    assert signals["f"] is MyType1


def test_merge_nested_key_without_collision():
    left = SignalSchema({"item.score": float})
    right = SignalSchema({"metadata.score": float})

    merged = left.merge(right, "right_")

    assert merged.values["item.score"] is float
    assert merged.values["metadata.score"] is float
    assert "right_metadata.score" not in merged.values


def test_merge_applies_suffix_when_prefixed_name_exists():
    left = SignalSchema(
        {
            "item.score": float,
            "right_item.score": float,
        }
    )
    right = SignalSchema(
        {
            "item.confidence": int,
            "item.score": int,
        }
    )

    merged = left.merge(right, "right_")

    assert merged.values["item.score"] is float
    assert merged.values["right_item.score"] is float
    assert merged.values["right_item_1.score"] is int
    assert merged.values["right_item_1.confidence"] is int


def test_merge_rename_collides_with_existing_column():
    left = SignalSchema(
        {
            "item.score": float,
        }
    )
    right = SignalSchema(
        {
            "item.score": int,
            "right_item.score": str,
        }
    )

    merged = left.merge(right, "right_")

    assert merged.values["item.score"] is float
    assert merged.values["right_item.score"] is str
    assert merged.values["right_item_1.score"] is int


def test_select_custom_type_backward_compatibility():
    schema = SignalSchema.deserialize(
        {
            "age": "float",
            "address": "str",
            "f": "ExternalCustomType1@v1",
            # Older custom types schema is supported
            # Can be removed a bit later
            "_custom_types": {"ExternalCustomType1@v1": {"aa": "int", "bb": "str"}},
        }
    )

    new = schema.resolve("age", "f.aa", "f.bb")
    assert isinstance(new, SignalSchema)

    signals = new.values
    assert len(signals) == 3
    assert {"age", "f.aa", "f.bb"} == signals.keys()
    assert signals["age"] is float
    assert signals["f.aa"] is int
    assert signals["f.bb"] is str


def test_select_custom_type():
    schema = SignalSchema.deserialize(
        {
            "age": "float",
            "address": "str",
            "f": "ExternalCustomType1@v1",
            "_custom_types": {
                "ExternalCustomType1@v1": {
                    "schema_version": 2,
                    "name": "ExternalCustomType1@v1",
                    "fields": {"aa": "int", "bb": "str"},
                    "bases": [],
                },
            },
        }
    )

    new = schema.resolve("age", "f.aa", "f.bb")
    assert isinstance(new, SignalSchema)

    signals = new.values
    assert len(signals) == 3
    assert {"age", "f.aa", "f.bb"} == signals.keys()
    assert signals["age"] is float
    assert signals["f.aa"] is int
    assert signals["f.bb"] is str


def test_select_except_signals():
    schema = SignalSchema({"age": float, "address": str, "f": MyType1})

    new = schema.select_except_signals("address")
    assert isinstance(new, SignalSchema)

    signals = new.values
    assert len(signals) == 2
    assert {"age", "f"} == signals.keys()
    assert signals["age"] is float
    assert signals["f"] is MyType1


def test_select_except_signals_nested_exclusion_creates_partial_model():
    schema = SignalSchema({"age": float, "address": str, "f": MyType1})

    new = schema.select_except_signals("address", "f.aa")
    assert set(new.values.keys()) == {"age", "f"}

    f_type = ModelStore.to_pydantic(new.values["f"])
    assert f_type is not None
    assert set(f_type.model_fields.keys()) == {"bb"}


def test_select_except_signals_empty_args_returns_self():
    schema = SignalSchema({"age": float, "address": str, "f": MyType1})
    assert schema.select_except_signals() is schema


def test_select_except_signals_error():
    schema = SignalSchema({"age": float, "address": str, "f": MyType1})

    # Still errors when there is no match for a provided signal.
    with pytest.raises(SignalRemoveError):
        schema.select_except_signals("address", "not_exist")

    with pytest.raises(SignalRemoveError):
        schema.select_except_signals("address", "f.cc")

    with pytest.raises(SignalResolvingTypeError):
        schema.select_except_signals("address", 37)


def test_deserialize_restores_known_base_type():
    schema = {"fr": MyType3}
    signals = SignalSchema(schema).serialize()
    ModelStore.remove(MyType3)

    # Since MyType3 is removed, deserialization restores it
    # from the meta information stored in the schema, including the base type
    # that is still known - MyType1
    deserialized_schema = SignalSchema.deserialize(signals)
    assert deserialized_schema.values["fr"].__name__ == "MyType3_v1"
    assert issubclass(deserialized_schema.values["fr"], MyType1)


def test_deserialize_custom_type_bad_schema():
    # No `bases` field
    with pytest.raises(SignalSchemaError):
        SignalSchema.deserialize(
            {
                "f": "ExternalCustomType1@v1",
                "_custom_types": {
                    "ExternalCustomType1@v1": {
                        "schema_version": 2,
                        "name": "ExternalCustomType1@v1",
                        "fields": {"aa": "int", "bb": "str"},
                    },
                },
            }
        )

    # Bad version
    with pytest.raises(SignalSchemaError):
        SignalSchema.deserialize(
            {
                "f": "ExternalCustomType1@v1",
                "_custom_types": {
                    "ExternalCustomType1@v1": {
                        "schema_version": 123,
                        "name": "ExternalCustomType1@v1",
                        "fields": {"aa": "int", "bb": "str"},
                        "bases": [],
                    },
                },
            }
        )


def test_select_nested_names():
    schema = SignalSchema.deserialize(
        {
            "address": "str",
            "fr": "MyType2@v1",
        }
    )

    fr_signals = schema.resolve("fr.deep").values
    assert "fr.deep" in fr_signals
    assert fr_signals["fr.deep"] == MyType1

    basic_signals = schema.resolve("fr.deep.aa", "fr.deep.bb").values
    assert "fr.deep.aa" in basic_signals
    assert "fr.deep.bb" in basic_signals
    assert basic_signals["fr.deep.aa"] is int
    assert basic_signals["fr.deep.bb"] is str


def test_select_nested_names_custom_types():
    schema = SignalSchema.deserialize(
        {
            "address": "str",
            "fr": "NestedType2@v1",
            "_custom_types": {
                "NestedType1@v1": {"aa": "int", "bb": "str"},
                "NestedType2@v1": {"deep": "NestedType1@v1", "name": "str"},
            },
        }
    )

    fr_signals = schema.resolve("fr.deep").values
    assert "fr.deep" in fr_signals
    # This is a dynamically restored model
    nested_type_1 = fr_signals["fr.deep"]
    assert issubclass(nested_type_1, DataModel)
    assert {n: fi.annotation for n, fi in nested_type_1.model_fields.items()} == {
        "aa": int,
        "bb": str,
    }

    basic_signals = schema.resolve("fr.deep.aa", "fr.deep.bb").values
    assert "fr.deep.aa" in basic_signals
    assert "fr.deep.bb" in basic_signals
    assert basic_signals["fr.deep.aa"] is int
    assert basic_signals["fr.deep.bb"] is str


def test_select_nested_errors():
    schema = SignalSchema.deserialize(
        {
            "address": "str",
            "fr": "MyType2@v1",
        }
    )

    schema = schema.resolve("fr.deep.aa", "fr.deep.bb")

    with pytest.raises(SignalResolvingError):
        schema.resolve("some_random")

    with pytest.raises(SignalResolvingError):
        schema.resolve("fr")

    with pytest.raises(SignalResolvingError):
        schema.resolve("fr.deep")

    with pytest.raises(SignalResolvingError):
        schema.resolve("fr.deep.not_exist")


def test_select_complex_names_custom_types():
    with pytest.warns(SignalSchemaWarning):
        schema = SignalSchema.deserialize(
            {
                "address": "str",
                "fr": "ComplexType@v1",
                "_custom_types": {
                    "NestedTypeComplex@v1": {
                        "aa": "float",
                        "bb": "bytes",
                        "items": "list[Union[dict[str, float], dict[str, int]]]",
                        "maybe_texts": "Union[list[Any], dict[str, Any], NoneType]",
                        "anything": "UnknownCustomType",
                    },
                    "ComplexType@v1": {"deep": "NestedTypeComplex@v1", "name": "str"},
                },
            }
        )

    fr_signals = schema.resolve("fr.deep").values
    assert "fr.deep" in fr_signals
    # This is a dynamically restored model
    nested_type_complex = fr_signals["fr.deep"]
    assert issubclass(nested_type_complex, DataModel)
    assert {n: fi.annotation for n, fi in nested_type_complex.model_fields.items()} == {
        "aa": float,
        "bb": bytes,
        "items": list[dict[str, float] | dict[str, int]],
        "maybe_texts": Union[list[Any], dict[str, Any], None],
        "anything": Any,
    }

    basic_signals = schema.resolve(
        "fr.deep.aa", "fr.deep.bb", "fr.deep.maybe_texts", "fr.deep.anything"
    ).values
    assert "fr.deep.aa" in basic_signals
    assert "fr.deep.bb" in basic_signals
    assert "fr.deep.maybe_texts" in basic_signals
    assert "fr.deep.anything" in basic_signals
    assert basic_signals["fr.deep.aa"] is float
    assert basic_signals["fr.deep.bb"] is bytes
    assert (
        basic_signals["fr.deep.maybe_texts"] is Union[list[Any], dict[str, Any], None]
    )
    assert basic_signals["fr.deep.anything"] is Any


def test_get_signals_basic():
    schema = {
        "name": str,
        "age": float,
        "f": File,
    }
    assert list(SignalSchema(schema).get_signals(File)) == ["f"]


def test_get_signals_no_signal():
    schema = {
        "name": str,
    }
    assert list(SignalSchema(schema).get_signals(File)) == []


def test_get_signals_nested(nested_file_schema):
    files = list(nested_file_schema.get_signals(File))
    assert files == ["f", "my_f", "my_f.nested_file"]


def test_get_features_nested(test_session, nested_file_schema):
    file = File(path="test")
    file_dict = file.model_dump()
    file_vals = list(file_dict.values())
    nested_file = nested_file_schema.values["my_f"](
        ref="str", nested_file=file, **file_dict
    )
    expected_features = ["str", 0.0, file, nested_file]
    row = ["str", 0.0, *file_vals, *file_vals, "str", *file_vals]
    actual_features = nested_file_schema.row_to_features(row, test_session.catalog)
    assert expected_features == actual_features
    assert actual_features[2]._catalog == test_session.catalog
    assert actual_features[3]._catalog == test_session.catalog
    assert actual_features[3].nested_file._catalog == test_session.catalog


def test_row_to_features_list_of_models(test_session):
    schema = SignalSchema({"items": list[MyType1]})
    row = ([{"aa": 1, "bb": "x"}, {"aa": 2, "bb": "y"}],)

    features = schema.row_to_features(row, test_session.catalog)

    assert len(features) == 1
    items = features[0]
    assert isinstance(items, list)
    assert [item.aa for item in items] == [1, 2]
    assert all(isinstance(item, MyType1) for item in items)


def test_row_to_features_dict_of_models(test_session):
    schema = SignalSchema({"lookup": dict[str, MyType2]})
    row = (
        {
            "first": {"name": "a", "deep": {"aa": 1, "bb": "x"}},
            "second": {"name": "b", "deep": {"aa": 2, "bb": "y"}},
        },
    )

    features = schema.row_to_features(row, test_session.catalog)

    assert len(features) == 1
    lookup = features[0]
    assert isinstance(lookup, dict)
    assert set(lookup) == {"first", "second"}
    assert isinstance(lookup["first"], MyType2)
    assert lookup["second"].deep.aa == 2


def test_row_to_features_optional_collection(test_session):
    schema = SignalSchema({"items": list[MyType1] | None})

    features_none = schema.row_to_features((None,), test_session.catalog)
    assert features_none == [None]

    row = ([{"aa": 3, "bb": "z"}],)
    features = schema.row_to_features(row, test_session.catalog)
    assert len(features) == 1
    items = features[0]
    assert isinstance(items, list)
    assert isinstance(items[0], MyType1)


@pytest.mark.parametrize(
    "union_type,union_name",
    [
        (Union[list[MyType1], None], "Union[list[MyType1], None]"),
        (list[MyType1] | None, "list[MyType1] | None"),
    ],
    ids=["old-style-union", "new-style-union"],
)
def test_row_to_features_union_types(test_session, union_type, union_name):
    """Test that both old-style Union[X, None] and new-style X | None work correctly."""
    schema = SignalSchema({"items": union_type})

    # Test None case
    features_none = schema.row_to_features((None,), test_session.catalog)
    assert features_none == [None], f"Failed for {union_name} with None value"

    # Test non-None case
    row = ([{"aa": 5, "bb": "test"}],)
    features = schema.row_to_features(row, test_session.catalog)
    assert len(features) == 1, f"Failed for {union_name}"
    items = features[0]
    assert isinstance(items, list), f"Expected list for {union_name}, got {type(items)}"
    assert len(items) == 1, f"Expected 1 item for {union_name}"
    assert isinstance(items[0], MyType1), f"Expected MyType1 instance for {union_name}"
    assert items[0].aa == 5, f"Wrong value for {union_name}"
    assert items[0].bb == "test", f"Wrong value for {union_name}"


def test_get_signals_subclass(nested_file_schema):
    class NewFile(File):
        pass

    schema = {
        "name": str,
        "age": float,
        "f": NewFile,
    }
    assert list(SignalSchema(schema).get_signals(File)) == ["f"]


def test_build_tree():
    spec = {"name": str, "age": float, "fr": MyType2}
    lst = list(SignalSchema(spec).get_flat_tree())

    assert lst == [
        (["name"], str, False, 0),
        (["age"], float, False, 0),
        (["fr"], MyType2, True, 0),
        (["fr", "name"], str, False, 1),
        (["fr", "deep"], MyType1, True, 1),
        (["fr", "deep", "aa"], int, False, 2),
        (["fr", "deep", "bb"], str, False, 2),
    ]


def test_resolve_types():
    mapping = {
        "int": int,
        "float": float,
        "NoneType": None,
        "MyType2@v1": MyType2,
        "Any": Any,
        "Literal": Any,
        "Final": Final,
        "Union[MyType2@v1, NoneType]": MyType2 | None,
        "Optional[MyType2@v1]": MyType2 | None,
        "Union[str, int]": str | int,
        "Union[str, int, bool]": str | int | bool,
        "Union[Optional[MyType2@v1]]": MyType2 | None,
        "list": list,
        "list[bool]": list[bool],
        "List[bool]": list[bool],
        "list[Union[bool, NoneType]]": list[bool | None],
        "List[Union[bool, NoneType]]": list[bool | None],
        "list[Optional[bool]]": list[bool | None],
        "List[Optional[bool]]": list[bool | None],
        "dict": dict,
        "dict[str, bool]": dict[str, bool],
        "Dict[str, bool]": dict[str, bool],
        "dict[str, Union[MyType1@v1, NoneType]]": dict[str, MyType1 | None],
        "Dict[str, Union[MyType1@v1, NoneType]]": dict[str, MyType1 | None],
        "dict[str, Optional[MyType1@v1]]": dict[str, MyType1 | None],
        "Dict[str, Optional[MyType1@v1]]": dict[str, MyType1 | None],
        "Union[str, list[str]]": str | list[str],
        "Union[str, List[str]]": str | list[str],
        "Union[Literal, NoneType]": Any | None,
        "Union[list[bytes], NoneType]": list[bytes] | None,
        "Union[List[bytes], NoneType]": list[bytes] | None,
        "tuple[int, float]": tuple[int, float],
        "Tuple[int, float]": tuple[int, float],
        "Optional[tuple[int, float]]": tuple[int, float] | None,
        "Optional[Tuple[int, float]]": tuple[int, float] | None,
        "tuple[int, ...]": tuple[int, ...],
        "Tuple[int, ...]": tuple[int, ...],
    }

    for s, t in mapping.items():
        assert SignalSchema._resolve_type(s, {}) == t

    # Test that unknown types are ignored, but raise a warning.
    mapping_warnings = {
        "BogusType": Any,
        "UnknownType": Any,
        "list[UnknownType]": list[Any],
        "List[UnknownType]": list[Any],
    }
    for s, t in mapping_warnings.items():
        with pytest.warns(SignalSchemaWarning):
            assert SignalSchema._resolve_type(s, {}) == t


def test_resolve_types_errors():
    bogus_types_messages = {
        "": r"cannot be empty",
        "[str]": r"cannot start with '\['",
        "Union[str": r"Unclosed square bracket",
        "Union]str[": r"Square brackets are out of order",
        "Union[]": r"Empty square brackets",
        "Union[str, int]]": r"Extra closing square bracket",
        "Union[str, Optional[int]": r"Unclosed square bracket",
    }

    for t, m in bogus_types_messages.items():
        with pytest.raises(ValueError, match=m):
            SignalSchema._resolve_type(t, {})


def test_db_signals():
    spec = {"name": str, "age": float, "fr": MyType2}
    lst = list(SignalSchema(spec).db_signals())

    assert lst == [
        "name",
        "age",
        "fr__name",
        "fr__deep__aa",
        "fr__deep__bb",
    ]


def test_db_signals_filtering_by_name():
    schema = SignalSchema({"name": str, "age": float, "fr": MyType2})

    assert list(schema.db_signals(name="fr")) == [
        "fr__name",
        "fr__deep__aa",
        "fr__deep__bb",
    ]
    assert list(schema.db_signals(name="fr.name")) == ["fr__name"]
    assert list(schema.db_signals(name="fr.deep")) == ["fr__deep__aa", "fr__deep__bb"]
    assert list(schema.db_signals(name="fr.deep.aa")) == ["fr__deep__aa"]
    assert list(schema.db_signals(name="name")) == ["name"]
    assert list(schema.db_signals(name="missing")) == []


def test_db_signals_as_columns():
    spec = {"name": str, "age": float, "fr": MyType2}
    lst = list(SignalSchema(spec).db_signals(as_columns=True))

    assert all(isinstance(s, Column) for s in lst)

    assert [(c.name, type(c.type)) for c in lst] == [
        ("name", String),
        ("age", Float),
        ("fr__name", String),
        ("fr__deep__aa", Int64),
        ("fr__deep__bb", String),
    ]


def test_row_to_objs():
    spec = {"name": str, "age": float, "fr": MyType2, "foo": int | None}
    schema = SignalSchema(spec)

    val = MyType2(name="Fred", deep=MyType1(aa=129, bb="qwe"))
    row = ("myname", 12.5, *flatten(val), None)

    res = schema.row_to_objs(row)

    assert res == ["myname", 12.5, val, None]


def test_row_to_objs_all_none_returns_none():
    schema = SignalSchema({"fr": MyType2})

    row = (None, None, None)

    res = schema.row_to_objs(row)

    assert res == [None]


def test_row_to_objs_some_none_values_raises():
    schema = SignalSchema({"fr": MyType2})

    row = ("name", None, None)

    with pytest.raises(ValidationError):
        schema.row_to_objs(row)


def test_row_to_objs_all_none_nested_collections():
    schema = SignalSchema({"id": int, "complex": MyTypeComplex, "label": str})

    row = (5, None, None, None, "tag")

    res = schema.row_to_objs(row)

    assert res == [5, None, "tag"]


def test_row_to_objs_nested_collections_some_values_missing_raises():
    schema = SignalSchema({"id": int, "complex": MyTypeComplex, "label": str})

    row = (5, "component", ["bad"], {"key": "value"}, "tag")

    with pytest.raises(ValidationError):
        schema.row_to_objs(row)


def test_row_to_objs_setup():
    spec = {"name": str, "age": float, "init_val": int, "fr": MyType2, "empty": dict}
    setup_value = 84635
    setup = {"init_val": lambda: setup_value, "empty": dict}
    schema = SignalSchema(spec, setup)

    val = MyType2(name="Fred", deep=MyType1(aa=129, bb="qwe"))
    row = ("myname", 12.5, *flatten(val))

    # run twice to check that setup_values are cached
    assert schema.setup_values is None
    schema.row_to_objs(row)
    assert schema.setup_values is not None
    res = schema.row_to_objs(row)
    assert schema.setup_values is not None

    assert res == ["myname", 12.5, setup_value, val, {}]


def test_setup_not_callable():
    with pytest.raises(SetupError):
        SignalSchema({"name": str}, {"init_val": "asdfd"})


def test_setup_error():
    schema = SignalSchema({"name": str, "value": int}, {"init": lambda: 1 / 0})
    with pytest.raises(SetupError):
        schema.row_to_objs(("myname", 37))


@pytest.mark.parametrize(
    "schema,hidden_fields",
    [
        ({"name": str, "value": int}, []),
        (
            {"file": File},
            [
                "file__source",
                "file__version",
                "file__etag",
                "file__is_latest",
                "file__last_modified",
                "file__location",
            ],
        ),
    ],
)
def test_get_flatten_hidden_fields(schema, hidden_fields):
    schema_serialized = SignalSchema(schema).serialize()
    assert SignalSchema.get_flatten_hidden_fields(schema_serialized) == hidden_fields


def test_slice():
    schema = {"name": str, "age": float, "address": str}
    setup_values = {"init": lambda: 37}
    keys = {"age": Any, "name": Any, "init": Any}
    sliced = SignalSchema(schema).slice(keys, setup_values)
    assert list(sliced.values.items()) == [("age", float), ("name", str), ("init", str)]


@pytest.mark.parametrize(
    "schema,keys,is_batch,result",
    [
        (
            {"name": str, "age": float, "address": str},
            {"name": Any, "age": float},
            False,
            [("name", str), ("age", float)],
        ),
        (
            {"name": str | None, "age": float, "address": str},
            {"name": Any, "age": Any},
            False,
            [("name", str | None), ("age", float)],
        ),
        (
            {"name": str | None, "age": float, "address": str},
            {"name": str, "age": Any},
            False,
            [("name", str), ("age", float)],
        ),
        (
            {"name": str | None, "age": float, "address": str},
            {"name": str | None, "age": Any},
            False,
            [("name", str), ("age", float)],
        ),
        (
            {"name": str, "age": float, "address": str},
            {"name": list[str], "age": Any},
            True,
            [("name", str), ("age", float)],
        ),
        (
            {"name": str, "age": float, "address": str},
            {"name": list, "age": Any},
            True,
            [("name", str), ("age", float)],
        ),
    ],
)
def test_slice_typed(schema, keys, is_batch, result):
    sliced = SignalSchema(schema).slice(keys, is_batch=is_batch)
    assert list(sliced.values.items()) == result


def test_slice_error():
    schema = {"name": str, "age": float, "address": str}
    keys = {"age": Any, "name": Any, "init": Any}
    with pytest.raises(SignalResolvingError):
        SignalSchema(schema).slice(keys)


@pytest.mark.parametrize(
    "keys,is_batch",
    [
        ({"name": str, "age": str}, False),
        ({"name": dict, "age": str}, True),
    ],
)
def test_slice_typed_error(keys, is_batch):
    schema = {"name": str, "age": float, "address": str}
    with pytest.raises(SignalResolvingError):
        SignalSchema(schema).slice(keys, is_batch=is_batch)


def test_slice_nested():
    schema = {
        "name": str,
        "feature": MyType1,
    }
    keys = {"feature.aa": Any}
    sliced = SignalSchema(schema).slice(keys)
    assert list(sliced.values.items()) == [("feature.aa", int)]


def test_mutate_rename():
    schema = SignalSchema({"name": str})
    schema = schema.mutate({"new_name": Column("name")})
    assert schema.values == {"new_name": str}


def test_mutate_rename_leaf(nested_file_schema):
    schema = nested_file_schema.mutate({"new_name": Column("my_f__nested_file")})
    assert schema.values == {**nested_file_schema.values, "new_name": File}


def test_mutate_new_signal():
    schema = SignalSchema({"name": str})
    with pytest.raises(
        SignalResolvingError, match="cannot resolve signal name 'age': is not found"
    ):
        schema.mutate({"age": Column("age", Float)})


def test_mutate_change_type():
    schema = SignalSchema({"name": str, "age": float, "f": File})
    schema = schema.mutate({"age": int, "f": TextFile})
    assert schema.values == {"name": str, "age": int, "f": TextFile}


def test_mutate_func():
    schema = SignalSchema({"name": str, "age": float, "f": File})
    schema = schema.mutate({"age": func.sum("age")})
    assert schema.values == {"name": str, "age": float, "f": File}


@pytest.mark.parametrize(
    "column_type,signal_type",
    [
        [String, str],
        [Boolean, bool],
        [Int, int],
        [Int32, int],
        [UInt32, int],
        [Int64, int],
        [UInt64, int],
        [Float, float],
        [Float32, float],
        [Float64, float],
        [Array(Int), list],
        [JSON, dict],
        [DateTime, datetime],
        [Binary, bytes],
    ],
)
def test_column_types(column_type, signal_type):
    signals = SignalSchema.from_column_types({"val": column_type}).values

    assert len(signals) == 1
    assert signals["val"] is signal_type


@pytest.mark.parametrize(
    "schema,_hash",
    [
        (
            {
                "name": str | None,
                "feature": MyType1 | None,
            },
            "a817682b89b3aea1f03d7467bfa56065ef379b49a80848adc549f63426ddddaa",
        ),
        (
            {"file": File},
            "26a08b3793e738814f199c89c4582f9bde052ff3dcba84c2020535063df4c36c",
        ),
        (
            {},
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        ),
    ],
)
def test_hash(schema, _hash):
    assert SignalSchema(schema).hash() == _hash
