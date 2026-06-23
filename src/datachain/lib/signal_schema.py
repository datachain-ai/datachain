import copy
import hashlib
import logging
import math
import types
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from inspect import isclass
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Final,
    Optional,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, ValidationError, create_model
from sqlalchemy import Cast, cast
from sqlalchemy.sql.elements import BinaryExpression, Grouping, Label

from datachain import json
from datachain.func import literal
from datachain.func.func import Func
from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.convert.sql_to_python import sql_to_python
from datachain.lib.convert.unflatten import unflatten_to_json_pos
from datachain.lib.data_model import (
    DataModel,
    DataType,
    DataValue,
    compute_model_fingerprint,
)
from datachain.lib.file import File
from datachain.lib.model_store import ModelStore
from datachain.lib.utils import (
    DataChainColumnError,
    DataChainParamsError,
    type_to_str,
)
from datachain.query.schema import DEFAULT_DELIMITER, C, Column, ColumnExpr, ColumnMeta
from datachain.sql.types import SQLType

if TYPE_CHECKING:
    from datachain.catalog import Catalog


logger = logging.getLogger(__name__)

NAMES_TO_TYPES = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "bytes": bytes,
    "datetime": datetime,
    "Final": Final,
    "Union": Union,
    "Optional": Optional,
    "List": list,
    "Dict": dict,
    "Tuple": tuple,
    "Literal": Any,
    "Any": Any,
}


class SignalSchemaError(DataChainParamsError):
    pass


class SignalSchemaWarning(RuntimeWarning):
    pass


class SignalResolvingError(SignalSchemaError):
    def __init__(self, path: list[str] | None, msg: str):
        self._path = path
        self._msg = msg
        if path:
            name = " '" + ".".join(path) + "'"
            msg = f"cannot resolve signal name{name}: {msg}"
        super().__init__(msg)

    def __reduce__(self):
        return self.__class__, (self._path, self._msg)


class SetupError(SignalSchemaError):
    def __init__(self, name: str, msg: str):
        self._name = name
        self._msg = msg
        super().__init__(f"cannot setup value '{name}': {msg}")

    def __reduce__(self):
        return self.__class__, (self._name, self._msg)


def _collect_right_roots(
    right_names: Sequence[str],
    extract_root: Callable[[str], str],
) -> tuple[list[str], set[str]]:
    right_root_order: list[str] = []
    right_roots: set[str] = set()
    for name in right_names:
        root = extract_root(name)
        if root not in right_roots:
            right_roots.add(root)
            right_root_order.append(root)
    return right_root_order, right_roots


def _resolve_root_collision(
    root: str, used_roots: set[str], right_roots: set[str], prefix: str
) -> str:
    suffix = 0
    while True:
        base = prefix if root in prefix else f"{prefix}{root}"
        candidate = base if suffix == 0 else f"{base}_{suffix}"
        if candidate not in used_roots and candidate not in right_roots:
            return candidate
        suffix += 1


def generate_merge_root_mapping(
    left_names: Iterable[str],
    right_names: Sequence[str],
    *,
    extract_root: Callable[[str], str],
    prefix: str,
) -> dict[str, str]:
    existing_roots = {extract_root(name) for name in left_names}
    right_root_order, right_roots = _collect_right_roots(right_names, extract_root)

    used_roots = set(existing_roots)
    root_mapping: dict[str, str] = {}

    for root in right_root_order:
        if root not in used_roots:
            root_mapping[root] = root
            used_roots.add(root)
            continue
        candidate = _resolve_root_collision(root, used_roots, right_roots, prefix)
        root_mapping[root] = candidate
        used_roots.add(candidate)

    return root_mapping


class SignalResolvingTypeError(SignalResolvingError):
    def __init__(self, method: str, field: Any, supported_types: str = "`str` type"):
        self._method = method
        if isinstance(field, str):
            # Restoring from pickle; field is the pre-computed message
            msg = field
        else:
            msg = (
                f"{method} supports only {supported_types}"
                f" while '{field!r}' has type '{type(field).__name__}'"
            )
        super().__init__(None, msg)

    def __reduce__(self) -> tuple[type, tuple]:
        return self.__class__, (self._method, self._msg)


class SignalRemoveError(SignalSchemaError):
    def __init__(self, path: list[str] | None, msg: str):
        self._path = path
        self._msg = msg
        name = " '" + ".".join(path) + "'" if path else ""
        super().__init__(f"cannot remove signal name{name}: {msg}")

    def __reduce__(self):
        return self.__class__, (self._path, self._msg)


class CustomType(BaseModel):
    schema_version: int = Field(ge=1, le=2, strict=True)
    name: str
    fields: dict[str, str]
    bases: list[tuple[str, str, str | None]]
    hidden_fields: list[str] | None = None
    partial_fingerprint: str | None = None

    @classmethod
    def deserialize(cls, data: dict[str, Any], type_name: str) -> "CustomType":
        version = data.get("schema_version", 1)

        if version == 1:
            data = {
                "schema_version": 1,
                "name": type_name,
                "fields": data,
                "bases": [],
                "hidden_fields": [],
                "partial_fingerprint": None,
            }

        return cls(**data)


def _set_model_metadata(
    model: type[BaseModel],
    parsed_version: int,
    base_name: str,
    partial_fingerprint: str | None,
    hidden_fields: list[str] | None,
) -> None:
    model._version = parsed_version  # type: ignore[attr-defined]
    model._modelstore_base_name = base_name  # type: ignore[attr-defined]
    if partial_fingerprint is not None:
        model._partial_fingerprint = partial_fingerprint  # type: ignore[attr-defined]
    if hidden_fields is not None:
        model._hidden_fields = hidden_fields  # type: ignore[attr-defined]


def create_feature_model(
    name: str,
    fields: Mapping[str, Any],
    base: type | None = None,
    *,
    partial_fingerprint: str | None = None,
    hidden_fields: list[str] | None = None,
) -> type[BaseModel]:
    """
    Build and register a dynamic feature model so it can be resolved later by name.

    Used when the original definition is not available (e.g., Studio restores or
    cross-process dataset loads) and when deriving partial models in
    ``SignalSchema.to_partial``.

    Args:
        name: Logical model name. If it includes a version suffix like ``@v1``, the
            version is parsed into ``_version``.
        fields: Mapping of field definitions for the model body.
        base: Base class for the generated model (defaults to ``DataModel``).
        partial_fingerprint: If set, store ``_partial_fingerprint`` metadata.
        hidden_fields: If set, store ``_hidden_fields`` metadata.

    Notes:
        - The generated Python class name is versioned (e.g. ``MyType_v1``) to avoid
          collisions when multiple versions are loaded in one process.
        - ``_modelstore_base_name`` preserves the original/logical name (e.g.
          ``MyType``), and ``ModelStore.register()`` stores the model under both the
          logical name and the runtime class name for robust lookups.
    """
    base_name, parsed_version = ModelStore.parse_name_version(name)
    class_name = f"{base_name}_v{parsed_version}" if parsed_version > 0 else base_name
    model_name = class_name.replace("@", "_")
    model = create_model(
        model_name,
        __base__=base or DataModel,  # type: ignore[call-overload]
        # These are tuples for each field of: annotation, default (if any)
        **{
            field_name: anno if isinstance(anno, tuple) else (anno, None)
            for field_name, anno in fields.items()
        },  # type: ignore[arg-type]
    )

    _set_model_metadata(
        model, parsed_version, base_name, partial_fingerprint, hidden_fields
    )
    ModelStore.register(model)

    return model


@dataclass
class SignalSchema:
    values: dict[str, DataType]
    tree: dict[str, Any]
    setup_func: dict[str, Callable]
    setup_values: dict[str, Any] | None

    def __init__(
        self,
        values: dict[str, DataType],
        setup: dict[str, Callable] | None = None,
    ):
        self.values = values
        self.tree = self._build_tree(values)

        self.setup_func = setup or {}
        self.setup_values = None
        for key, func in self.setup_func.items():
            if not callable(func):
                raise SetupError(key, "value must be function or callable class")

    def _init_setup_values(self) -> None:
        if self.setup_values is not None:
            return

        res = {}
        for key, func in self.setup_func.items():
            try:
                res[key] = func()
            except Exception as ex:
                raise SetupError(key, f"error when call function: '{ex}'") from ex
        self.setup_values = res

    @staticmethod
    def from_column_types(col_types: dict[str, Any]) -> "SignalSchema":
        signals: dict[str, DataType] = {}
        for field, col_type in col_types.items():
            if isinstance(col_type, SQLType):
                signals[field] = col_type.python_type
            elif isclass(col_type) and issubclass(col_type, SQLType):
                signals[field] = col_type().python_type
            else:
                raise SignalSchemaError(
                    f"signal schema cannot be obtained for column '{field}':"
                    f" unsupported type '{col_type}'"
                )
        return SignalSchema(signals)

    @staticmethod
    def _get_bases(fr: type) -> list[tuple[str, str, str | None]]:
        bases: list[tuple[str, str, str | None]] = []
        for base in fr.__mro__:
            model_store_name = (
                ModelStore.get_name(base) if issubclass(base, DataModel) else None
            )
            base_name = getattr(base, "_modelstore_base_name", base.__name__)
            bases.append((base_name, base.__module__, model_store_name))
        return bases

    @staticmethod
    def _find_base_model(bases: list[tuple[str, str, str | None]]) -> type | None:
        for base in bases:
            _, _, model_store_name = base
            if model_store_name:
                base_model_name, base_version = ModelStore.parse_name_version(
                    model_store_name
                )
                base_model = ModelStore.get(base_model_name, base_version)
                if base_model:
                    return base_model
        return None

    @staticmethod
    def _serialize_custom_model(
        version_name: str, fr: type[BaseModel], custom_types: dict[str, Any]
    ) -> str:
        """This serializes any custom type information to the provided custom_types
        dict, and returns the name of the type serialized."""
        if version_name in custom_types:
            # This type is already stored in custom_types.
            return version_name
        fields = {}

        for field_name, info in fr.model_fields.items():
            field_type = info.annotation
            # All fields should be typed.
            assert field_type
            fields[field_name] = SignalSchema._serialize_type(field_type, custom_types)

        bases = SignalSchema._get_bases(fr)

        ct = CustomType(
            schema_version=2,
            name=version_name,
            fields=fields,
            bases=bases,
            hidden_fields=getattr(fr, "_hidden_fields", []),
            partial_fingerprint=getattr(fr, "_partial_fingerprint", None),
        )
        custom_types[version_name] = ct.model_dump(exclude_none=True)

        return version_name

    @staticmethod
    def _serialize_type(fr: type, custom_types: dict[str, Any]) -> str:
        """Serialize a given type to a string, including automatic ModelStore
        registration, and save this type and subtypes to custom_types as well."""
        subtypes: list[Any] = []
        type_name = SignalSchema._type_to_str(fr, subtypes)
        # Iterate over all subtypes (includes the input type).
        for st in subtypes:
            if st is None or not ModelStore.is_pydantic(st):
                continue
            # Register and save feature types.
            st_version_name = ModelStore.get_name(st)
            if st is fr:
                # If the main type is Pydantic, then use the ModelStore version name.
                type_name = st_version_name
            # Save this type to custom_types.
            SignalSchema._serialize_custom_model(st_version_name, st, custom_types)
        return type_name

    def serialize(self) -> dict[str, Any]:
        signals: dict[str, Any] = {}
        custom_types: dict[str, Any] = {}
        for name, fr_type in self.values.items():
            signals[name] = self._serialize_type(fr_type, custom_types)
        if custom_types:
            signals["_custom_types"] = custom_types
        return signals

    def hash(self) -> str:
        """Create SHA hash of this schema"""
        json_str = json.dumps(self.serialize(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _split_subtypes(type_name: str) -> list[str]:
        start = 0
        depth = 0
        subtypes = []
        for i, c in enumerate(type_name):
            if c == "[":
                depth += 1
            elif c == "]":
                SignalSchema._validate_bracket_depth(depth, type_name)
                depth -= 1
            elif c == "," and depth == 0:
                subtypes.append(type_name[start:i].strip())
                start = i + 1
        if depth > 0:
            raise ValueError("Unclosed square bracket when parsing subtype list")
        subtypes.append(type_name[start:].strip())
        return subtypes

    @staticmethod
    def _validate_bracket_depth(depth: int, type_name: str) -> None:
        if depth == 0:
            raise ValueError(
                "Extra closing square bracket when parsing subtype list"
            )

    @staticmethod
    def _validate_bracket_positions(
        bracket_idx: int, close_bracket_idx: int, type_name: str
    ) -> None:
        if bracket_idx == 0:
            raise ValueError("Type cannot start with '['")
        if close_bracket_idx == -1:
            raise ValueError("Unclosed square bracket when parsing type")
        if close_bracket_idx < bracket_idx:
            raise ValueError("Square brackets are out of order when parsing type")
        if close_bracket_idx == bracket_idx + 1:
            raise ValueError("Empty square brackets when parsing type")

    @staticmethod
    def _deserialize_custom_type(
        type_name: str, custom_types: dict[str, Any]
    ) -> type | None:
        """Given a type name like MyType@v1 gets a type from ModelStore or recreates
        it based on the information from the custom types dict that includes fields and
        bases."""
        model_name, target_version = ModelStore.parse_name_version(type_name)

        if type_name in custom_types:
            try:
                ct = CustomType.deserialize(custom_types[type_name], type_name)
            except ValidationError as exc:
                raise SignalSchemaError(
                    f"cannot deserialize custom type '{type_name}': {exc}"
                ) from exc

            if fr := ModelStore.get(model_name, target_version):
                return fr

            fields = {
                field_name: SignalSchema._resolve_type(field_type_str, custom_types)
                for field_name, field_type_str in ct.fields.items()
            }

            base_model = SignalSchema._find_base_model(ct.bases)

            return create_feature_model(
                type_name,
                fields,
                base=base_model,
                hidden_fields=ct.hidden_fields,
                partial_fingerprint=ct.partial_fingerprint,
            )

        return ModelStore.get(model_name, target_version)

    @staticmethod
    def _parse_type_string(type_name: str) -> tuple[str, list[str] | None]:
        bracket_idx = type_name.find("[")
        if bracket_idx == -1:
            return type_name, None

        close_bracket_idx = type_name.rfind("]")
        SignalSchema._validate_bracket_positions(
            bracket_idx, close_bracket_idx, type_name
        )

        base_name = type_name[:bracket_idx].strip()
        subtype_names = SignalSchema._split_subtypes(
            type_name[bracket_idx + 1 : close_bracket_idx]
        )
        return base_name, subtype_names

    @staticmethod
    def _construct_generic_type(base: Any, subtypes: tuple | None) -> Any:
        if subtypes is None:
            return base
        if len(subtypes) == 1:
            return base[subtypes[0]]
        if base is tuple and len(subtypes) == 2 and subtypes[1] is Ellipsis:
            return base[subtypes[0], ...]
        return base[subtypes]

    @staticmethod
    def _resolve_subtypes(
        subtype_names: list[str], custom_types: dict[str, Any]
    ) -> tuple:
        return tuple(
            Ellipsis
            if st == "..."
            else SignalSchema._resolve_type(st, custom_types)
            for st in subtype_names
        )

    @staticmethod
    def _resolve_known_type(
        base_name: str, subtypes: tuple | None
    ) -> object | None:
        fr = NAMES_TO_TYPES.get(base_name)
        if fr:
            return SignalSchema._construct_generic_type(fr, subtypes)
        return None

    @staticmethod
    def _validate_type_name(type_name: str) -> str:
        stripped = type_name.strip()
        if not stripped:
            raise ValueError("Type cannot be empty")
        return stripped

    @staticmethod
    def _resolve_base_type(
        base_name: str, custom_types: dict[str, Any], type_name: str
    ) -> object:
        result = SignalSchema._resolve_known_type(base_name, None)
        if result:
            return result

        fr = SignalSchema._deserialize_custom_type(base_name, custom_types)
        if fr:
            return fr

        warnings.warn(
            f"Could not resolve type: '{type_name}'.",
            SignalSchemaWarning,
            stacklevel=2,
        )
        return Any

    @staticmethod
    def _resolve_type(
        type_name: str, custom_types: dict[str, Any]
    ) -> object | None:
        type_name = SignalSchema._validate_type_name(type_name)
        if type_name == "NoneType":
            return None

        base_name, subtype_names = SignalSchema._parse_type_string(type_name)
        subtypes = (
            SignalSchema._resolve_subtypes(subtype_names, custom_types)
            if subtype_names is not None
            else None
        )

        if subtypes is not None:
            result = SignalSchema._resolve_known_type(base_name, subtypes)
            if result:
                return result

        return SignalSchema._resolve_base_type(base_name, custom_types, type_name)

    @staticmethod
    def _deserialize_signal(
        signal: str, type_name: str, custom_types: dict[str, Any]
    ) -> DataType | None:
        if not isinstance(type_name, str):
            raise SignalSchemaError(
                f"cannot deserialize '{type_name}': "
                "serialized types must be a string"
            )
        try:
            fr = SignalSchema._resolve_type(type_name, custom_types)
            if fr is Any:
                warnings.warn(
                    f"In signal '{signal}': "
                    f"unknown type '{type_name}'."
                    f" Try to add it with `ModelStore.register({type_name})`.",
                    SignalSchemaWarning,
                    stacklevel=2,
                )
                return None
            return fr  # type: ignore[return-value]
        except ValueError as err:
            raise SignalSchemaError(
                f"cannot deserialize '{signal}': {err}"
            ) from err

    @staticmethod
    def deserialize(schema: dict[str, Any]) -> "SignalSchema":
        if not isinstance(schema, dict):
            raise SignalSchemaError(f"cannot deserialize signal schema: {schema}")

        signals: dict[str, DataType] = {}
        custom_types: dict[str, Any] = schema.get("_custom_types", {})
        for signal, type_name in schema.items():
            if signal == "_custom_types":
                continue
            fr = SignalSchema._deserialize_signal(signal, type_name, custom_types)
            if fr is not None:
                signals[signal] = fr

        return SignalSchema(signals)

    @staticmethod
    def _traverse_hidden_fields(
        prefix: str,
        schema_info: dict,
        custom_types: dict,
        hidden_by_types: dict,
        hidden_fields: list,
    ) -> None:
        for field, field_type in schema_info.items():
            if field == "_custom_types":
                continue
            if field_type in custom_types:
                hidden_fields.extend(
                    f"{prefix}{field}__{f}" for f in hidden_by_types[field_type]
                )
                SignalSchema._traverse_hidden_fields(
                    prefix + field + "__",
                    custom_types[field_type].get("fields", {}),
                    custom_types,
                    hidden_by_types,
                    hidden_fields,
                )

    @staticmethod
    def get_flatten_hidden_fields(schema: dict):
        custom_types = schema.get("_custom_types", {})
        if not custom_types:
            return []

        hidden_by_types = {
            name: schema.get("hidden_fields", [])
            for name, schema in custom_types.items()
        }

        hidden_fields: list = []
        SignalSchema._traverse_hidden_fields(
            "", schema, custom_types, hidden_by_types, hidden_fields
        )
        return hidden_fields

    def to_udf_spec(self) -> dict[str, type]:
        res = {}
        for path, type_, has_subtree, _ in self.get_flat_tree():
            if path[0] in self.setup_func:
                continue
            if not has_subtree:
                db_name = DEFAULT_DELIMITER.join(path)
                res[db_name] = python_to_sql(type_)
        return res

    def _row_to_obj(
        self, name: str, fr_type: DataType, row: Sequence[Any], pos: int
    ) -> tuple[Any, int]:
        if self.setup_values and name in self.setup_values:
            return self.setup_values.get(name), pos
        if (fr := ModelStore.to_pydantic(fr_type)) is not None:
            j, pos = unflatten_to_json_pos(fr, row, pos)
            try:
                obj = fr(**j)
            except ValidationError as e:
                if self._all_values_none(j):
                    logger.debug("Failed to create input for %s: %s", name, e)
                    obj = None
                else:
                    raise
            return obj, pos
        return row[pos], pos + 1

    def row_to_objs(self, row: Sequence[Any]) -> list[Any]:
        self._init_setup_values()
        objs: list[Any] = []
        pos = 0
        for name, fr_type in self.values.items():
            obj, pos = self._row_to_obj(name, fr_type, row, pos)
            objs.append(obj)
        return objs

    @staticmethod
    def _all_values_none(value: Any) -> bool:
        if isinstance(value, dict):
            return all(SignalSchema._all_values_none(v) for v in value.values())
        if isinstance(value, (list, tuple, set)):
            return all(SignalSchema._all_values_none(v) for v in value)
        if isinstance(value, float):
            # NaN is used to represent NULL and NaN float values in datachain
            # Since SQLite does not have a separate NULL type, we need to check for NaN
            return math.isnan(value) or value is None
        return value is None

    def get_file_signal(self) -> str | None:
        for signal_name, signal_type in self.values.items():
            if (fr := ModelStore.to_pydantic(signal_type)) is not None and issubclass(
                fr, File
            ):
                return signal_name
        return None

    @staticmethod
    def _unwrap_optional(tp: DataType) -> DataType:
        origin = get_origin(tp)
        if origin in (Union, types.UnionType) and type(None) in get_args(tp):
            return get_args(tp)[0]
        return tp

    def _resolve_slice_param(
        self, param: str, param_type: DataType, is_batch: bool
    ) -> DataType:
        if param_type is Any:
            return self._find_in_tree(param.split("."))

        schema_type = self._find_in_tree(param.split("."))

        if is_batch:
            if param_type is list:
                return schema_type
            param_origin = get_origin(param_type)
            if param_origin is not list:
                raise SignalResolvingError(param.split("."), "is not a list")
            param_type = get_args(param_type)[0]

        schema_type = SignalSchema._unwrap_optional(schema_type)
        param_type = SignalSchema._unwrap_optional(param_type)

        if param_type == schema_type or (
            isclass(param_type)
            and isclass(schema_type)
            and issubclass(param_type, File)
            and issubclass(schema_type, File)
        ):
            return schema_type

        raise SignalResolvingError(
            param.split("."),
            f"types mismatch: {param_type} != {schema_type}",
        )

    def slice(
        self,
        params: dict[str, DataType | Any],
        setup: dict[str, Callable] | None = None,
        is_batch: bool = False,
    ) -> "SignalSchema":
        setup_params = setup.keys() if setup else []
        schema: dict[str, DataType] = {}

        for param, param_type in params.items():
            if param in setup_params:
                schema[param] = str
            else:
                schema[param] = self._resolve_slice_param(param, param_type, is_batch)

        return SignalSchema(schema, setup)

    def _row_to_feature(
        self, fr_cls: DataType, row: Sequence, pos: int, catalog: "Catalog", cache: bool
    ) -> tuple[Any, int]:
        if (fr := ModelStore.to_pydantic(fr_cls)) is None:
            value = row[pos]
            converted = self._convert_feature_value(fr_cls, value, catalog, cache)
            return converted, pos + 1
        json, pos = unflatten_to_json_pos(fr, row, pos)  # type: ignore[union-attr]
        try:
            obj = fr(**json)
            SignalSchema._set_file_stream(obj, catalog, cache)
        except ValidationError as e:
            if self._all_values_none(json):
                logger.debug("Failed to create feature for %s: %s", fr_cls, e)
                obj = None
            else:
                raise
        return obj, pos

    def row_to_features(
        self, row: Sequence, catalog: "Catalog", cache: bool = False
    ) -> list[DataValue]:
        res = []
        pos = 0
        for fr_cls in self.values.values():
            obj, pos = self._row_to_feature(fr_cls, row, pos, catalog, cache)
            res.append(obj)
        return res

    def _convert_union(
        self, annotation: DataType, value: Any
    ) -> tuple[DataType, Any] | None:
        non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0], value
        return None

    def _convert_pydantic(
        self, annotation: DataType, value: Any, catalog: "Catalog", cache: bool
    ) -> Any:
        if isinstance(value, annotation):
            obj = value
        elif isinstance(value, Mapping):
            obj = annotation(**value)
        else:
            return value
        assert isinstance(obj, BaseModel)
        SignalSchema._set_file_stream(obj, catalog, cache)
        return obj

    def _convert_list(
        self, annotation: DataType, value: Any, catalog: "Catalog", cache: bool
    ) -> Any:
        args = get_args(annotation)
        if args and isinstance(value, (list, tuple)):
            item_type = args[0]
            return [
                self._convert_feature_value(item_type, item, catalog, cache)
                if item is not None
                else None
                for item in value
            ]
        return value

    def _convert_dict(
        self, annotation: DataType, value: Any, catalog: "Catalog", cache: bool
    ) -> Any:
        args = get_args(annotation)
        if len(args) == 2 and isinstance(value, dict):
            key_type, val_type = args
            result = {}
            for key, val in value.items():
                if key_type is str:
                    converted_key = key
                else:
                    loaded_key = json.loads(key)
                    converted_key = self._convert_feature_value(
                        key_type, loaded_key, catalog, cache
                    )
                converted_val = (
                    self._convert_feature_value(val_type, val, catalog, cache)
                    if val_type is not Any
                    else val
                )
                result[converted_key] = converted_val
            return result
        return value

    def _convert_feature_value(
        self,
        annotation: DataType,
        value: Any,
        catalog: "Catalog",
        cache: bool,
    ) -> Any:
        if value is None:
            return None

        origin = get_origin(annotation)

        if origin in (Union, types.UnionType):
            unwrapped = self._convert_union(annotation, value)
            if unwrapped is None:
                return value
            annotation, value = unwrapped
            origin = get_origin(annotation)

        if ModelStore.is_pydantic(annotation):
            return self._convert_pydantic(annotation, value, catalog, cache)
        if origin is list:
            return self._convert_list(annotation, value, catalog, cache)
        if origin is dict:
            return self._convert_dict(annotation, value, catalog, cache)

        return value

    @staticmethod
    def _set_file_stream(
        obj: BaseModel, catalog: "Catalog", cache: bool = False
    ) -> None:
        if isinstance(obj, File):
            obj._set_stream(catalog, caching_enabled=cache)
        for field, finfo in type(obj).model_fields.items():
            if ModelStore.is_pydantic(finfo.annotation):
                SignalSchema._set_file_stream(getattr(obj, field), catalog, cache)

    def get_column_type(self, col_name: str, with_subtree: bool = False) -> DataType:
        """
        Returns column type by column name.

        If `with_subtree` is True, then it will return the type of the column
        even if it has a subtree (e.g. model with nested fields), otherwise it will
        return the type of the column (standard type field, not the model).

        If column is not found, raises `SignalResolvingError`.
        """
        for path, _type, has_subtree, _ in self.get_flat_tree():
            if (with_subtree or not has_subtree) and DEFAULT_DELIMITER.join(
                path
            ) == col_name:
                return _type
        raise SignalResolvingError([col_name], "is not found")

    @staticmethod
    def _filter_db_signals_by_name(
        signals: list, name: str
    ) -> list:
        if "." in name:
            name = ColumnMeta.to_db_name(name)
        return [
            s
            for s in signals
            if str(s) == name or str(s).startswith(f"{name}{DEFAULT_DELIMITER}")
        ]

    def db_signals(
        self, name: str | None = None, as_columns=False, include_hidden: bool = True
    ) -> list[str] | list[Column]:
        signals = [
            DEFAULT_DELIMITER.join(path)
            if not as_columns
            else Column(DEFAULT_DELIMITER.join(path), python_to_sql(_type))
            for path, _type, has_subtree, _ in self.get_flat_tree(
                include_hidden=include_hidden
            )
            if not has_subtree
        ]

        if name:
            signals = self._filter_db_signals_by_name(signals, name)

        return signals  # type: ignore[return-value]

    def user_signals(
        self,
        *,
        include_hidden: bool = True,
        include_sys: bool = False,
    ) -> list[str]:
        return [
            ".".join(path)
            for path, _, has_subtree, _ in self.get_flat_tree(
                include_hidden=include_hidden, include_sys=include_sys
            )
            if not has_subtree
        ]

    def compare_signals(
        self,
        other: "SignalSchema",
        *,
        include_hidden: bool = True,
        include_sys: bool = False,
    ) -> tuple[set[str], set[str]]:
        left = set(
            self.user_signals(include_hidden=include_hidden, include_sys=include_sys)
        )
        right = set(
            other.user_signals(include_hidden=include_hidden, include_sys=include_sys)
        )
        return left - right, right - left

    def resolve(self, *names: str) -> "SignalSchema":
        schema = {}
        for field in names:
            if not isinstance(field, str):
                raise SignalResolvingTypeError("select()", field)
            schema[field] = self._find_in_tree(field.split("."))

        return SignalSchema(schema)

    def _find_in_tree(self, path: list[str]) -> DataType:
        if val := self.tree.get(".".join(path)):
            # If the path is a single string, we can directly access it
            # without traversing the tree.
            return val[0]

        curr_tree = self.tree
        curr_type = None
        i = 0
        while curr_tree is not None and i < len(path):
            if val := curr_tree.get(path[i]):
                curr_type, curr_tree = val
            else:
                curr_type = None
                break
            i += 1

        if curr_type is None or i < len(path):
            # If we reached the end of the path and didn't find a type,
            # or if we didn't traverse the entire path, raise an error.
            raise SignalResolvingError(path, "is not found")

        return curr_type

    @staticmethod
    def _rebuild_expr_node(node: "ColumnExpr", typed_cols: dict) -> "ColumnExpr":
        if isinstance(node, Column):
            return typed_cols.get(node.name, node)
        if isinstance(node, Label):
            return (
                SignalSchema._rebuild_expr_node(node.element, typed_cols)
                .label(node.name)
            )
        if isinstance(node, Grouping):
            return Grouping(
                SignalSchema._rebuild_expr_node(node.element, typed_cols)
            )
        if isinstance(node, BinaryExpression):
            left = SignalSchema._rebuild_expr_node(node.left, typed_cols)
            right = SignalSchema._rebuild_expr_node(node.right, typed_cols)
            return left.operate(node.operator, right)
        if isinstance(node, Cast):
            return cast(
                SignalSchema._rebuild_expr_node(node.clause, typed_cols),
                node.typeclause.type,
            )
        return node

    def enrich_expr_types(self, expr: "ColumnExpr") -> "ColumnExpr":
        typed_cols = {
            c.name: c for c in self.db_signals(as_columns=True) if isinstance(c, Column)
        }
        return self._rebuild_expr_node(expr, typed_cols)

    def group_by(
        self, partition_by: Sequence[str], new_column: Sequence[Column]
    ) -> "SignalSchema":
        orig_schema = SignalSchema(copy.deepcopy(self.values))
        schema = orig_schema.to_partial(*partition_by)

        vals = {c.name: sql_to_python(c) for c in new_column}
        return SignalSchema(schema.values | vals)

    def _exclude_signal(self, signal: str, leaf_signals: set[str]) -> set[str]:
        if not isinstance(signal, str):
            raise SignalResolvingTypeError("select_except()", signal)
        matches = {
            s for s in leaf_signals if s == signal or s.startswith(f"{signal}.")
        }
        if not matches:
            raise SignalRemoveError(
                signal.split("."),
                "select_except() error - the signal does not exist",
            )
        return leaf_signals - matches

    def select_except_signals(self, *args: str) -> "SignalSchema":
        if not args:
            return self

        leaf_signals = set(self.user_signals(include_hidden=True, include_sys=True))
        keep = leaf_signals
        for signal in args:
            keep = self._exclude_signal(signal, keep)
        return self.to_partial(*sorted(keep)) if keep else SignalSchema({})

    def clone_without_file_signals(self) -> "SignalSchema":
        schema = copy.deepcopy(self.values)

        for signal in File._datachain_column_types:
            if signal in schema:
                del schema[signal]
        return SignalSchema(schema)

    def _resolve_mutation_value(self, value: Any) -> DataType:
        if isinstance(value, Column):
            return self.get_column_type(value.name, with_subtree=True)
        if isinstance(value, Func):
            return value.get_result_type(self)
        if isinstance(value, (bool, str, int, float)):
            val = literal(value)
            val.type = python_to_sql(type(value))()
            return sql_to_python(val)
        if isinstance(value, ColumnExpr):
            return sql_to_python(self.enrich_expr_types(value))
        return value

    def _get_nested_current_type(self, name: str) -> DataType | None:
        if not C.is_nested(name):
            return None
        try:
            return self.get_column_type(name)
        except SignalResolvingError as err:
            msg = f"Creating new nested columns directly is not allowed: {name}"
            raise ValueError(msg) from err

    def _validate_nested_type(
        self, name: str, current_type: Any, new_type: Any
    ) -> None:
        if not C.is_nested(name):
            return
        if current_type == new_type:
            return
        msg = (
            f"Altering nested column type is not allowed: {name}, "
            f"current type: {current_type}, new type: {new_type}"
        )
        raise ValueError(msg)

    def _apply_mutation(
        self, name: str, value: Any, new_values: dict
    ) -> None:
        current_type = self._get_nested_current_type(name)

        if isinstance(value, Column) and value.name in self.values:
            del new_values[value.name]
            new_values[name] = self.values[value.name]
        else:
            new_values[name] = self._resolve_mutation_value(value)

        self._validate_nested_type(name, current_type, new_values.get(name))
        if C.is_nested(name):
            del new_values[name]

    def mutate(self, args_map: dict) -> "SignalSchema":
        new_values = self.values.copy()
        for name, value in args_map.items():
            self._apply_mutation(name, value, new_values)
        return SignalSchema(new_values)

    def clone_without_sys_signals(self) -> "SignalSchema":
        schema = copy.deepcopy(self.values)
        schema.pop("sys", None)
        return SignalSchema(schema)

    def _merge_rename_key(
        self, key: str, type_: DataType, root_mapping: dict[str, str]
    ) -> tuple[str, DataType]:
        root = self._extract_root(key)
        tail = key.partition(".")[2]
        mapped_root = root_mapping[root]
        new_name = mapped_root if not tail else f"{mapped_root}.{tail}"
        return new_name, type_

    def merge(
        self,
        right_schema: "SignalSchema",
        rname: str,
    ) -> "SignalSchema":
        merged_values = dict(self.values)

        right_names = list(right_schema.values.keys())
        root_mapping = generate_merge_root_mapping(
            self.values.keys(),
            right_names,
            extract_root=self._extract_root,
            prefix=rname,
        )

        for key, type_ in right_schema.values.items():
            new_name, typ = self._merge_rename_key(key, type_, root_mapping)
            merged_values[new_name] = typ

        return SignalSchema(merged_values)

    @staticmethod
    def _extract_root(name: str) -> str:
        if "." in name:
            return name.split(".", 1)[0]
        return name

    def append(self, right: "SignalSchema") -> "SignalSchema":
        missing_schema = {
            key: right.values[key]
            for key in [k for k in right.values if k not in self.values]
        }
        return SignalSchema(self.values | missing_schema)

    def get_signals(self, target_type: type[DataModel]) -> Iterator[str]:
        for path, type_, has_subtree, _ in self.get_flat_tree():
            if has_subtree and issubclass(type_, target_type):
                yield ".".join(path)

    def create_model(self, name: str) -> type[DataModel]:
        fields = {key: (value, None) for key, value in self.values.items()}

        return create_model(
            name,
            __base__=(DataModel,),  # type: ignore[call-overload]
            **fields,  # type: ignore[arg-type]
        )

    @staticmethod
    def _build_tree(
        values: dict[str, DataType],
    ) -> dict[str, tuple[DataType, dict | None]]:
        return {
            name: (val, SignalSchema._build_tree_for_type(val))
            for name, val in values.items()
        }

    def get_flat_tree(
        self,
        include_hidden: bool = True,
        include_sys: bool = True,
    ) -> Iterator[tuple[list[str], DataType, bool, int]]:
        yield from self._get_flat_tree(self.tree, [], 0, include_hidden, include_sys)

    @staticmethod
    def _filter_hidden_subtree(
        type_: DataType, substree: dict | None, include_hidden: bool
    ) -> dict | None:
        if not include_hidden:
            hidden_fields = getattr(type_, "_hidden_fields", None)
            if hidden_fields and substree:
                return {
                    field: info
                    for field, info in substree.items()
                    if field not in hidden_fields
                }
        return substree

    def _get_flat_tree(
        self,
        tree: dict,
        prefix: list[str],
        depth: int,
        include_hidden: bool,
        include_sys: bool,
    ) -> Iterator[tuple[list[str], DataType, bool, int]]:
        for name, (type_, substree) in tree.items():
            suffix = name.split(".")
            new_prefix = prefix + suffix
            if not include_sys and new_prefix and new_prefix[0] == "sys":
                continue
            substree = self._filter_hidden_subtree(type_, substree, include_hidden)
            has_subtree = substree is not None
            yield new_prefix, type_, has_subtree, depth
            if substree is not None:
                yield from self._get_flat_tree(
                    substree, new_prefix, depth + 1, include_hidden, include_sys
                )

    def _print_list_subtree(
        self,
        type_: DataType,
        total_indent: int,
        indent: int,
        include_hidden: bool,
        file: IO | None,
    ) -> None:
        if get_origin(type_) is list:
            args = get_args(type_)
            if len(args) > 0 and ModelStore.is_pydantic(args[0]):
                sub_schema = SignalSchema({"* list of": args[0]})
                sub_schema.print_tree(
                    indent=indent,
                    start_at=total_indent + indent,
                    include_hidden=include_hidden,
                    file=file,
                )

    def print_tree(
        self,
        indent: int = 2,
        start_at: int = 0,
        file: IO | None = None,
        *,
        include_hidden: bool = True,
    ):
        for path, type_, _, depth in self.get_flat_tree(include_hidden=include_hidden):
            total_indent = start_at + depth * indent
            col_name = " " * total_indent + path[-1]
            col_type = SignalSchema._type_to_str(type_)
            print(col_name, col_type, sep=": ", file=file)
            self._print_list_subtree(type_, total_indent, indent, include_hidden, file)

    def get_headers_with_length(self, include_hidden: bool = True):
        paths = [
            path
            for path, _, has_subtree, _ in self.get_flat_tree(
                include_hidden=include_hidden
            )
            if not has_subtree
        ]
        max_length = max([len(path) for path in paths], default=0)
        return [
            path + [""] * (max_length - len(path)) if len(path) < max_length else path
            for path in paths
        ], max_length

    def _or_merge_signal(
        self, name: str, new_type: DataType, new_values: dict
    ) -> None:
        if name in new_values:
            current_type = new_values[name]
            if current_type != new_type:
                raise DataChainColumnError(
                    name,
                    "signal already exists with a different type",
                )
            return
        root = self._extract_root(name)
        if any(self._extract_root(existing) == root for existing in new_values):
            raise DataChainColumnError(
                name,
                "signal root already exists in schema",
            )
        new_values[name] = new_type

    def __or__(self, other):
        new_values = dict(self.values)
        for name, new_type in other.values.items():
            self._or_merge_signal(name, new_type, new_values)
        return self.__class__(new_values)

    def __contains__(self, name: str):
        return name in self.values

    @staticmethod
    def _type_to_str(
        type_: type | None | types.EllipsisType, subtypes: list | None = None
    ) -> str:
        """Convert a type to a string-based representation."""

        def _warn(msg: str) -> None:
            warnings.warn(msg, SignalSchemaWarning, stacklevel=2)

        return type_to_str(
            type_,
            subtypes,
            warn_with=_warn,
            register_pydantic=True,
        )

    @staticmethod
    def _build_tree_for_type(
        model: DataType,
    ) -> dict[str, tuple[DataType, dict | None]] | None:
        if (fr := ModelStore.to_pydantic(model)) is not None:
            return SignalSchema._build_tree_for_model(fr)
        return None

    @staticmethod
    def _build_field_node(
        anno: DataType,
    ) -> tuple[DataType, dict | None]:
        if (fr := ModelStore.to_pydantic(anno)) is not None:
            return anno, SignalSchema._build_tree_for_model(fr)
        return anno, None

    @staticmethod
    def _build_tree_for_model(
        model: type[BaseModel],
    ) -> dict[str, tuple[DataType, dict | None]] | None:
        return {
            name: SignalSchema._build_field_node(f_info.annotation)
            for name, f_info in model.model_fields.items()
        }

    def _validate_column_path(self, column: str) -> list[str]:
        parts = column.split(".")
        if parts[0] not in self.tree:
            raise SignalSchemaError(f"Column {column} not found in the schema")

        curr_type, curr_tree = self.tree[parts[0]]

        for part in parts[1:]:
            if curr_tree is None:
                raise SignalSchemaError(f"Column {column} not found in the schema")

            node = curr_tree.get(part)
            if node is None:
                parent_model = ModelStore.to_pydantic(curr_type)
                if parent_model is not None:
                    raise SignalSchemaError(
                        f"Field {part} not found in custom type "
                        f"{parent_model.__name__}"
                    )
                raise SignalSchemaError(f"Column {column} not found in the schema")

            curr_type, curr_tree = node

        return parts

    @staticmethod
    def _merge_column_selection(
        selections: dict[str, dict[str, Any] | None],
        parts: list[str],
    ) -> None:
        curr: dict[str, dict[str, Any] | None] = selections
        missing = object()
        for idx, part in enumerate(parts):
            is_last = idx == len(parts) - 1
            existing = curr.get(part, missing)

            if existing is None:
                return

            if is_last:
                curr[part] = None
                return

            if existing is missing:
                next_sel: dict[str, Any] = {}
                curr[part] = next_sel
                curr = next_sel
            else:
                curr = existing  # type: ignore[assignment]

    def _all_leaves_covered(
        self, path: list[str], leaf_signals: set[str], columns: tuple[str, ...]
    ) -> bool:
        prefix = ".".join(path)
        model_leaves = [s for s in leaf_signals if s.startswith(f"{prefix}.")]
        return all(leaf in columns for leaf in model_leaves)

    def _build_field_types(
        self,
        model: type[BaseModel],
        selection: dict[str, Any],
        path: list[str],
        leaf_signals: set[str],
        columns: tuple[str, ...],
    ) -> dict[str, Any]:
        field_types: dict[str, Any] = {}
        for field_name, sub_selection in selection.items():
            assert field_name in model.model_fields
            field_info = model.model_fields[field_name]
            field_type = field_info.annotation
            assert field_type is not None
            partial_type = self._build_partial_type(
                field_type, sub_selection, [*path, field_name], leaf_signals, columns
            )
            if field_info.is_required():
                field_types[field_name] = partial_type
            else:
                field_types[field_name] = (partial_type, field_info.default)
        return field_types

    @staticmethod
    def _resolve_or_create_partial(
        base_partial_name: str,
        fingerprint: str,
        field_types: dict,
        base_hidden_fields: list,
    ) -> type[BaseModel]:
        version = 1
        existing = ModelStore.get(base_partial_name, version)
        if existing is None:
            return create_feature_model(
                f"{base_partial_name}@v{version}",
                field_types,
                base=DataModel,
                partial_fingerprint=fingerprint,
                hidden_fields=[
                    fname for fname in base_hidden_fields if fname in field_types
                ],
            )
        if getattr(existing, "_partial_fingerprint", None) == fingerprint:
            return existing  # type: ignore[return-value]
        msg = (
            f"partial model name collision '{base_partial_name}@v{version}' "
            "with a different fingerprint"
        )
        raise SignalSchemaError(msg)

    def _resolve_partial_model(
        self, model: type[BaseModel], selection: dict, field_types: dict
    ) -> type[BaseModel]:
        fingerprint = compute_model_fingerprint(model, selection)
        base_name, _ = ModelStore.parse_name_version(ModelStore.get_name(model))
        base_partial_name = f"{base_name}Partial_{fingerprint[:10]}"
        base_hidden_fields = getattr(model, "_hidden_fields", [])
        return SignalSchema._resolve_or_create_partial(
            base_partial_name, fingerprint, field_types, base_hidden_fields
        )

    def _get_partial_model(
        self, base_type: Any, selection: dict[str, Any] | None, path: list[str]
    ) -> type[BaseModel] | None:
        if selection is None:
            return None
        if not selection:  # pragma: no cover
            raise RuntimeError(
                f"empty selection for '{'.'.join(path)}'"
            )
        model = ModelStore.to_pydantic(base_type)
        assert model is not None, "Expected complex type to be a Pydantic model"
        return model

    def _build_partial_type(
        self,
        base_type: Any,
        selection: dict[str, Any] | None,
        path: list[str],
        leaf_signals: set[str],
        columns: tuple[str, ...],
    ) -> Any:
        model = self._get_partial_model(base_type, selection, path)
        if model is None:
            return base_type

        if self._all_leaves_covered(path, leaf_signals, columns):
            return base_type

        field_types = self._build_field_types(
            model, selection, path, leaf_signals, columns
        )
        assert field_types, (
            f"Empty field set when building partial for {model.__name__}"
        )

        return self._resolve_partial_model(model, selection, field_types)

    def to_partial(self, *columns: str) -> "SignalSchema":
        if not columns:
            return SignalSchema({})

        selections: dict[str, dict[str, Any] | None] = {}

        for column in columns:
            if not isinstance(column, str):
                raise SignalResolvingTypeError("to_partial()", column)
            column_parts = self._validate_column_path(column)
            self._merge_column_selection(selections, column_parts)

        leaf_signals = set(self.user_signals(include_hidden=True, include_sys=True))

        new_values: dict[str, DataType] = {}
        for signal, selection in selections.items():
            base_type = self.values[signal]
            new_values[signal] = self._build_partial_type(
                base_type, selection, [signal], leaf_signals, columns
            )

        return SignalSchema(new_values)
