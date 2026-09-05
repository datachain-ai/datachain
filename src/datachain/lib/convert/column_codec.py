"""Declared-type codecs for values stored in a single database column.

The supported grammar is built-in scalars, ``Annotated``, one-arm optionals,
typed lists and tuples, string-key dictionaries, and acyclic ``BaseModel``
fields composed from the same grammar. General unions, Enum, Literal, Any
(except opaque ``dict[str, Any]``), unknown model fields, and native Binary
arrays return ``None`` so callers retain their legacy conversion path.

Supported models are persisted as snapshots of their declared fields: Pydantic
serializers are not an on-disk format, and validators are not rerun on read.
Opaque dictionaries are restricted to JSON-native values because their content
has no annotation from which non-JSON types could be restored.
"""

import base64
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, replace
from datetime import datetime
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel

from datachain import json
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Int64,
    SQLType,
    String,
    parse_datetime_text,
)

CODEC_VERSION = "typed-v1"

_SCALAR_SQL_TYPES: dict[type, type[SQLType]] = {
    bool: Boolean,
    int: Int64,
    float: Float,
    str: String,
    bytes: Binary,
    datetime: DateTime,
}


@dataclass(frozen=True, slots=True, eq=False)
class ColumnCodec:
    """A compiled description of one logical column.

    ``encode`` and ``decode`` operate at the database-column boundary. A JSON
    column is encoded as one JSON string; JSON items inside an Array stay as
    structural Python values so the enclosing array is encoded only once.
    """

    annotation: Any
    sql_type: SQLType
    _kind: str
    _children: tuple["ColumnCodec", ...] = ()
    _fields: tuple[tuple[str, "ColumnCodec"], ...] = ()
    _model: type[BaseModel] | None = None
    _scalar_type: type | None = None
    _allows_none: bool = False

    def encode(self, value: Any) -> Any:
        """Convert a logical value to a backend bind-ready value."""
        json_mode = isinstance(self.sql_type, JSON)
        encoded = self._encode_value(value, json_mode=json_mode)
        if encoded is None:
            return None
        return json.dumps(encoded, ensure_ascii=False) if json_mode else encoded

    def decode(self, value: Any) -> Any:
        """Restore a logical value after the SQL type's outer read conversion."""
        json_mode = isinstance(self.sql_type, JSON)
        if json_mode and isinstance(value, (str, bytes, bytearray)):
            value = json.loads(value)
        return self._decode_value(value, json_mode=json_mode)

    def decode_fields(self, value: Mapping[str, Any] | None) -> BaseModel | None:
        """Restore a flattened model snapshot whose leaves are already SQL-read."""
        self._check_none(value)
        if value is None:
            return None
        if self._kind == "optional":
            return self._children[0].decode_fields(value)
        if self._kind != "model":
            raise TypeError("decode_fields() requires a model codec")
        values = self._string_mapping(value)
        decoded = {
            name: child._decode_field_value(values[name])
            for name, child in self._fields
        }
        assert self._model is not None
        return self._model.model_construct(**decoded)

    def _check_none(self, value: Any) -> None:
        if value is None and not self._allows_none:
            raise TypeError(f"None is not valid for {self.annotation!r}")

    def _decode_field_value(self, value: Any) -> Any:
        if self._kind == "model" or (
            self._kind == "optional" and self._children[0]._kind == "model"
        ):
            return self.decode_fields(value)
        return self.decode(value)

    def _encode_value(self, value: Any, *, json_mode: bool) -> Any:  # noqa: PLR0911
        value = _normalize_numpy(value)
        self._check_none(value)
        if value is None:
            return None
        if self._kind == "optional":
            return self._children[0]._encode_value(value, json_mode=json_mode)
        if self._kind == "scalar":
            encoded = self._encode_scalar(value)
            if json_mode and self._scalar_type is datetime:
                return encoded.isoformat()
            if json_mode and self._scalar_type is bytes:
                return base64.b64encode(encoded).decode("ascii")
            return encoded
        if self._kind in ("list", "tuple", "variadic_tuple"):
            values = self._sequence(value)
            assert isinstance(self.sql_type, Array)
            item_json = json_mode or isinstance(self.sql_type.item_type, JSON)
            return [
                self._child_at(index)._encode_value(item, json_mode=item_json)
                for index, item in enumerate(values)
            ]
        if self._kind == "dict":
            child = self._children[0]
            return {
                key: child._encode_value(item, json_mode=True)
                for key, item in self._string_mapping(value).items()
            }
        if self._kind == "opaque_dict":
            return _opaque_json_value(value)
        if self._kind == "model":
            assert self._model is not None
            if isinstance(value, Mapping):
                value = self._model.model_validate(value)
            elif not isinstance(value, self._model):
                raise TypeError(f"Expected {self._model!r}, got {type(value).__name__}")
            return {
                name: child._encode_value(getattr(value, name), json_mode=True)
                for name, child in self._fields
            }
        raise AssertionError(f"Unknown codec kind {self._kind!r}")

    def _decode_value(self, value: Any, *, json_mode: bool) -> Any:  # noqa: PLR0911
        self._check_none(value)
        if value is None:
            return None
        if self._kind == "optional":
            return self._children[0]._decode_value(value, json_mode=json_mode)
        if self._kind == "scalar":
            if json_mode and self._scalar_type is datetime:
                if not isinstance(value, str):
                    raise TypeError("A JSON datetime must be an ISO string")
                return parse_datetime_text(value)
            if json_mode and self._scalar_type is bytes:
                if not isinstance(value, str):
                    raise TypeError("JSON bytes must be a base64 string")
                return base64.b64decode(value, validate=True)
            return self._decode_scalar(value)
        if self._kind in ("list", "tuple", "variadic_tuple"):
            values = self._sequence(value)
            assert isinstance(self.sql_type, Array)
            item_json = json_mode or isinstance(self.sql_type.item_type, JSON)
            result = [
                self._child_at(index)._decode_value(item, json_mode=item_json)
                for index, item in enumerate(values)
            ]
            return tuple(result) if "tuple" in self._kind else result
        if self._kind == "dict":
            child = self._children[0]
            return {
                key: child._decode_value(item, json_mode=True)
                for key, item in self._string_mapping(value).items()
            }
        if self._kind == "opaque_dict":
            return _opaque_json_value(value)
        if self._kind == "model":
            assert self._model is not None
            if isinstance(value, self._model):
                return value
            field_values = self._string_mapping(value)
            decoded = {
                name: child._decode_value(field_values[name], json_mode=True)
                for name, child in self._fields
            }
            return self._model.model_construct(**decoded)
        raise AssertionError(f"Unknown codec kind {self._kind!r}")

    def _child_at(self, index: int) -> "ColumnCodec":
        return self._children[index] if self._kind == "tuple" else self._children[0]

    def _encode_scalar(self, value: Any) -> Any:
        scalar = self._scalar_type
        if (
            scalar is float
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return float(value)
        if scalar is bytes and isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if scalar is int and type(value) is int:
            return value
        if scalar is bool and type(value) is bool:
            return value
        if scalar is str and isinstance(value, str):
            return value
        if scalar is datetime and isinstance(value, datetime):
            return value
        raise TypeError(f"Expected {scalar!r}, got {type(value).__name__}")

    def _decode_scalar(self, value: Any) -> Any:
        scalar = self._scalar_type
        if scalar is datetime and isinstance(value, str):
            return parse_datetime_text(value)
        if scalar is bytes:
            if isinstance(value, str):
                return value.encode()
            if isinstance(value, (bytes, bytearray, memoryview)):
                return bytes(value)
        if scalar is bool:
            if isinstance(value, bool):
                return value
            if type(value) is int and value in (0, 1):
                return bool(value)
        return self._encode_scalar(value)

    def _sequence(self, value: Any) -> list[Any] | tuple[Any, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected a list or tuple, got {type(value).__name__}")
        if self._kind == "tuple" and len(value) != len(self._children):
            raise ValueError(
                f"Expected {len(self._children)} tuple items, got {len(value)}"
            )
        return value

    @staticmethod
    def _string_mapping(value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected a mapping, got {type(value).__name__}")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Only string mapping keys are supported")
        return value


def compile_codec(annotation: Any) -> ColumnCodec | None:
    """Compile a codec for ``annotation``, or return ``None`` if unsupported."""
    return _compile_codec(annotation, frozenset())


def _compile_codec(  # noqa: PLR0911
    annotation: Any, model_stack: frozenset[type[BaseModel]]
) -> ColumnCodec | None:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        child = _compile_codec(args[0], model_stack)
        return replace(child, annotation=annotation) if child is not None else None

    if origin in (Union, UnionType):
        return _compile_optional(annotation, args, model_stack)

    if annotation in _SCALAR_SQL_TYPES:
        return ColumnCodec(
            annotation,
            _SCALAR_SQL_TYPES[annotation](),
            "scalar",
            _scalar_type=annotation,
        )

    if origin is list:
        return _compile_list(annotation, args, model_stack)

    if origin is tuple:
        return _compile_tuple(annotation, args, model_stack)

    if annotation is dict or origin is dict:
        return _compile_dict(annotation, args, model_stack)

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _compile_model(annotation, model_stack)

    return None


def _compile_optional(
    annotation: Any,
    args: tuple[Any, ...],
    model_stack: frozenset[type[BaseModel]],
) -> ColumnCodec | None:
    non_none = tuple(arg for arg in args if arg is not type(None))
    if len(args) != 2 or len(non_none) != 1:
        return None
    child = _compile_codec(non_none[0], model_stack)
    if child is None:
        return None
    sql_type = (
        SQLType.as_nullable(JSON())
        if isinstance(child.sql_type, Array)
        else SQLType.as_nullable(child.sql_type)
    )
    return ColumnCodec(
        annotation,
        sql_type,
        "optional",
        _children=(child,),
        _allows_none=True,
    )


def _compile_list(
    annotation: Any,
    args: tuple[Any, ...],
    model_stack: frozenset[type[BaseModel]],
) -> ColumnCodec | None:
    if len(args) != 1:
        return None
    child = _compile_codec(args[0], model_stack)
    if child is None or _contains_native_binary(child.sql_type):
        return None
    return ColumnCodec(
        annotation,
        _array(child.sql_type),
        "list",
        _children=(child,),
    )


def _compile_tuple(
    annotation: Any,
    args: tuple[Any, ...],
    model_stack: frozenset[type[BaseModel]],
) -> ColumnCodec | None:
    if len(args) == 2 and args[1] is Ellipsis:
        child = _compile_codec(args[0], model_stack)
        if child is None or _contains_native_binary(child.sql_type):
            return None
        return ColumnCodec(
            annotation,
            _array(child.sql_type),
            "variadic_tuple",
            _children=(child,),
        )
    if not args:
        return None
    children = tuple(_compile_codec(arg, model_stack) for arg in args)
    if any(child is None for child in children):
        return None
    resolved = tuple(child for child in children if child is not None)
    item_type = _common_item_type(resolved)
    if item_type is None:
        item_type = JSON()
        if any(child._allows_none for child in resolved):
            item_type = SQLType.as_nullable(item_type)
    elif _contains_native_binary(item_type):
        return None
    return ColumnCodec(
        annotation,
        _array(item_type),
        "tuple",
        _children=resolved,
    )


def _compile_dict(
    annotation: Any,
    args: tuple[Any, ...],
    model_stack: frozenset[type[BaseModel]],
) -> ColumnCodec | None:
    if not args or args == (str, Any):
        return ColumnCodec(annotation, JSON(), "opaque_dict")
    if len(args) != 2 or args[0] is not str:
        return None
    child = _compile_codec(args[1], model_stack)
    if child is None:
        return None
    return ColumnCodec(annotation, JSON(), "dict", _children=(child,))


def _compile_model(
    annotation: type[BaseModel], model_stack: frozenset[type[BaseModel]]
) -> ColumnCodec | None:
    if annotation in model_stack:
        return None
    fields: list[tuple[str, ColumnCodec]] = []
    nested_stack = model_stack | {annotation}
    for name, field in annotation.model_fields.items():
        child = _compile_codec(field.annotation, nested_stack)
        if child is None:
            return None
        fields.append((name, child))
    return ColumnCodec(
        annotation,
        JSON(),
        "model",
        _fields=tuple(fields),
        _model=annotation,
    )


def _array(item_type: SQLType) -> Array:
    result = Array(item_type)
    # Physical readers must leave typed items to the logical codec.
    result.dc_codec = CODEC_VERSION
    return result


def _common_item_type(children: tuple[ColumnCodec, ...]) -> SQLType | None:
    first = children[0].sql_type
    first_shape = _type_shape(first)
    if any(_type_shape(child.sql_type) != first_shape for child in children[1:]):
        return None
    result = copy(first)
    result.dc_nullable = any(child._allows_none for child in children)
    return result


def _type_shape(sql_type: SQLType) -> dict[str, Any]:
    shape = sql_type.to_dict()
    shape.pop("dc_nullable", None)
    shape.pop("dc_codec", None)
    return shape


def _contains_native_binary(sql_type: SQLType) -> bool:
    if isinstance(sql_type, Binary):
        return True
    return isinstance(sql_type, Array) and _contains_native_binary(sql_type.item_type)


def _normalize_numpy(value: Any) -> Any:
    """Retain the warehouse's numpy-to-Python normalization without importing it."""
    if type(value).__module__.partition(".")[0] != "numpy":
        return value
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        return value
    normalized = tolist()
    if normalized is value:
        raise TypeError(f"Cannot normalize numpy value {value!r}")
    return normalized


def _opaque_json_value(value: Any) -> Any:
    """Copy a JSON-native value, rejecting values whose type cannot be restored."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, list):
        return [_opaque_json_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Only string keys are supported in opaque JSON")
        return {key: _opaque_json_value(item) for key, item in value.items()}
    raise TypeError(f"{type(value).__name__} is not a JSON-native value")
