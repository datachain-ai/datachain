from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal

import pytest
from pydantic import BaseModel, field_serializer, field_validator

from datachain import json
from datachain.lib.convert.column_codec import CODEC_VERSION, compile_codec
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Int64,
    String,
)


@pytest.mark.parametrize(
    "annotation,sql_cls,value",
    [
        (int, Int64, 1),
        (float, Float, 1.5),
        (str, String, "value"),
        (bool, Boolean, True),
        (bytes, Binary, b"value"),
        (datetime, DateTime, datetime(2024, 1, 2, 3, 4, 5)),
    ],
)
def test_scalar_codecs(annotation, sql_cls, value):
    codec = compile_codec(annotation)

    assert codec is not None
    assert isinstance(codec.sql_type, sql_cls)
    assert codec.decode(codec.encode(value)) == value


def test_codec_is_immutable():
    codec = compile_codec(int)
    assert codec is not None

    with pytest.raises(FrozenInstanceError):
        codec.annotation = str  # type: ignore[misc]


def test_annotated_optional_array_elements_are_structural_nulls():
    annotation = Annotated[list[int | None], "metadata"]
    codec = compile_codec(annotation)

    assert codec is not None
    assert isinstance(codec.sql_type, Array)
    assert codec.sql_type.dc_codec == CODEC_VERSION
    assert isinstance(codec.sql_type.item_type, Int64)
    assert codec.sql_type.item_type.dc_nullable
    assert codec.encode([None, 1, None]) == [None, 1, None]
    assert codec.decode([None, 1, None]) == [None, 1, None]


def test_variadic_and_fixed_tuples_restore_container_and_positions():
    variadic = compile_codec(tuple[int, ...])
    fixed = compile_codec(tuple[int, str, bytes])

    assert variadic is not None
    assert isinstance(variadic.sql_type, Array)
    assert isinstance(variadic.sql_type.item_type, Int64)
    assert variadic.encode((1, 2)) == [1, 2]
    assert variadic.decode([1, 2]) == (1, 2)

    assert fixed is not None
    assert isinstance(fixed.sql_type, Array)
    assert isinstance(fixed.sql_type.item_type, JSON)
    assert fixed.encode((1, "two", b"\x00\xff")) == [1, "two", "AP8="]
    assert fixed.decode([1, "two", "AP8="]) == (1, "two", b"\x00\xff")


def test_fixed_tuple_merges_scalar_nullability():
    codec = compile_codec(tuple[int, int | None])

    assert codec is not None
    assert isinstance(codec.sql_type, Array)
    assert isinstance(codec.sql_type.item_type, Int64)
    assert codec.sql_type.item_type.dc_nullable
    assert codec.decode(codec.encode((1, None))) == (1, None)


def test_optional_nested_array_uses_one_structural_json_item():
    codec = compile_codec(list[list[datetime] | None])
    value = [None, [datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)]]

    assert codec is not None
    assert isinstance(codec.sql_type, Array)
    assert isinstance(codec.sql_type.item_type, JSON)
    assert codec.sql_type.item_type.dc_nullable
    encoded = codec.encode(value)
    assert encoded == [None, ["2024-01-02T03:04:05+00:00"]]
    assert codec.decode(encoded) == value


def test_native_datetime_array_survives_sqlite_outer_json_conversion():
    from datachain.sql.sqlite.types import adapt_array, convert_array

    codec = compile_codec(list[datetime])
    value = [datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)]

    assert codec is not None
    encoded = codec.encode(value)
    assert encoded == value
    sqlite_value = convert_array(adapt_array(encoded))
    assert sqlite_value == ["2024-01-02T03:04:05Z"]
    assert codec.decode(sqlite_value) == value


def test_optional_top_level_array_is_a_nullable_json_document():
    codec = compile_codec(list[int] | None)

    assert codec is not None
    assert isinstance(codec.sql_type, JSON)
    assert codec.sql_type.dc_nullable
    assert codec.encode([1, 2]) == "[1,2]"
    assert codec.decode("[1,2]") == [1, 2]
    assert codec.encode(None) is None
    assert codec.decode(None) is None


class SnapshotChild(BaseModel):
    validator_calls: ClassVar[int] = 0

    number: int
    created: datetime
    payload: bytes

    @field_validator("number")
    @classmethod
    def increment(cls, value):
        cls.validator_calls += 1
        return value + 1

    @field_serializer("number")
    def multiply_for_export(self, value):
        return value * 10


class SnapshotParent(BaseModel):
    child: SnapshotChild | None
    timestamps: list[datetime]


def test_model_codec_snapshots_fields_without_serializers_or_revalidation():
    SnapshotChild.validator_calls = 0
    value = SnapshotChild(
        number=1,
        created=datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        payload=bytes(range(256)) * 8,
    )
    codec = compile_codec(SnapshotChild)

    assert codec is not None
    encoded = codec.encode(value)
    document = json.loads(encoded)
    assert document["number"] == 2
    assert document["number"] != value.model_dump()["number"]

    restored = codec.decode(encoded)
    assert restored == value
    assert type(restored) is SnapshotChild
    assert SnapshotChild.validator_calls == 1
    assert restored.payload == bytes(range(256)) * 8


def test_decode_fields_uses_native_leaves_and_constructs_nested_models():
    SnapshotChild.validator_calls = 0
    created = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    codec = compile_codec(SnapshotParent)

    assert codec is not None
    restored = codec.decode_fields(
        {
            "child": {"number": 7, "created": created, "payload": b"child"},
            "timestamps": [created],
        }
    )

    assert isinstance(restored, SnapshotParent)
    assert type(restored) is SnapshotParent
    assert isinstance(restored.child, SnapshotChild)
    assert type(restored.child) is SnapshotChild
    assert restored.child.number == 7
    assert restored.child.created is created
    assert restored.timestamps == [created]
    assert SnapshotChild.validator_calls == 0


def test_array_of_optional_models_uses_structural_json_items():
    codec = compile_codec(list[SnapshotChild | None])
    value = SnapshotChild(
        number=2,
        created=datetime(2024, 1, 2, 3, 4, 5),
        payload=b"payload",
    )

    assert codec is not None
    assert isinstance(codec.sql_type, Array)
    assert isinstance(codec.sql_type.item_type, JSON)
    assert codec.sql_type.item_type.dc_nullable
    encoded = codec.encode([None, value])
    assert encoded[0] is None
    assert isinstance(encoded[1], dict)
    assert encoded[1]["number"] == value.number
    assert codec.decode(encoded) == [None, value]


def test_model_mapping_is_validated_once_before_its_fields_are_snapshotted():
    SnapshotChild.validator_calls = 0
    codec = compile_codec(list[SnapshotChild | None])
    created = datetime(2024, 1, 2, 3, 4, 5)

    assert codec is not None
    encoded = codec.encode(
        [None, {"number": 2, "created": created, "payload": b"payload"}]
    )
    assert encoded[1]["number"] == 3
    restored = codec.decode(encoded)
    assert restored[1].number == 3
    assert SnapshotChild.validator_calls == 1


def test_typed_and_opaque_string_dicts():
    typed = compile_codec(dict[str, bytes | None])
    opaque = compile_codec(dict[str, Any])

    assert typed is not None
    encoded = typed.encode({"present": b"abc", "missing": None})
    assert isinstance(encoded, str)
    assert typed.decode(encoded) == {"present": b"abc", "missing": None}

    assert opaque is not None
    assert opaque.decode(opaque.encode({"nested": [1, None, "x"]})) == {
        "nested": [1, None, "x"]
    }
    with pytest.raises(TypeError, match="not a JSON-native value"):
        opaque.encode({"bytes": b"not self-describing"})


def test_numpy_runtime_values_keep_the_declared_codec():
    np = pytest.importorskip("numpy")
    scalar = compile_codec(int)
    array = compile_codec(list[int])

    assert scalar is not None
    assert array is not None
    assert scalar.encode(np.int64(3)) == 3
    assert array.encode(np.array([1, 2], dtype=np.int64)) == [1, 2]
    assert array.encode([np.int64(1), np.int64(2)]) == [1, 2]


class Kind(Enum):
    A = "a"


class RecursiveModel(BaseModel):
    child: "RecursiveModel | None" = None


@pytest.mark.parametrize(
    "annotation",
    [
        Any,
        Literal[1, 2],
        Kind,
        int | str,
        list[Any],
        list[bytes],
        list[bytes | None],
        list[list[bytes]],
        tuple[bytes, ...],
        tuple[bytes, bytes],
        dict[int, str],
        RecursiveModel,
    ],
)
def test_unsupported_annotations_defer_to_legacy(annotation):
    assert compile_codec(annotation) is None
