import hashlib
import inspect
import types
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar, Union, get_args, get_origin

from pydantic import AliasChoices, BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from datachain import json
from datachain.lib.model_store import ModelStore
from datachain.lib.utils import normalize_col_names, type_to_str

StandardType = (
    type[int]
    | type[str]
    | type[float]
    | type[bool]
    | type[list]
    | type[dict]
    | type[bytes]
    | type[datetime]
)
DataType = type[BaseModel] | StandardType
DataTypeNames = "BaseModel, int, str, float, bool, list, dict, bytes, datetime"
DataValue = BaseModel | int | str | float | bool | list | dict | bytes | datetime


class DataModel(BaseModel):
    """Pydantic model wrapper that registers model with `DataChain`."""

    _version: ClassVar[int] = 1
    _hidden_fields: ClassVar[list[str]] = []

    @classmethod
    def __pydantic_init_subclass__(cls):
        """It automatically registers every declared DataModel child class."""
        ModelStore.register(cls)

    @staticmethod
    def register(models: DataType | Sequence[DataType]):
        """For registering classes manually. It accepts a single class or a sequence of
        classes."""
        if not isinstance(models, Sequence):
            models = [models]
        for val in models:
            ModelStore.register(val)

    @classmethod
    def hidden_fields(cls) -> list[str]:
        """Returns a list of fields that should be hidden from the user."""
        return cls._hidden_fields


def compute_model_fingerprint(
    model: type[BaseModel], selection: dict[str, "dict[str, object] | None"]
) -> str:
    """
    Compute a deterministic fingerprint for a model given a selection subtree.

    Selection uses the same structure as SignalSchema.to_partial: a mapping from
    field name -> nested selection dict or None (leaf).
    """

    def _fingerprint_tree(
        model_type: type[BaseModel], sel: dict[str, "dict[str, object] | None"]
    ) -> dict[str, object]:
        tree: dict[str, object] = {}
        for field_name, sub_sel in sorted(sel.items()):
            if field_name not in model_type.model_fields:
                raise ValueError(
                    f"Field {field_name} not found in {model_type.__name__}"
                )

            finfo = model_type.model_fields[field_name]
            field_type = finfo.annotation
            required = finfo.is_required()
            entry: dict[str, object] = {
                "type": type_to_str(field_type, register_pydantic=False),
                "required": bool(required),
                "default": None if required else repr(finfo.default),
            }

            child_model = ModelStore.to_pydantic(field_type)
            if sub_sel is not None:
                if child_model is None:
                    raise ValueError(
                        f"Field {field_name} in {model_type.__name__} is not a model"
                    )
                entry["children"] = _fingerprint_tree(
                    child_model,
                    sub_sel,  # type: ignore[arg-type]
                )
            tree[field_name] = entry

        return tree

    payload = {
        "model": ModelStore.get_name(model),
        "selection": _fingerprint_tree(model, selection),
    }
    json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def is_chain_type(t: type) -> bool:
    """Return true if type is supported by `DataChain`."""
    if ModelStore.is_pydantic(t):
        return True
    if any(t is ft or t is get_args(ft)[0] for ft in get_args(StandardType)):
        return True

    orig = get_origin(t)
    args = get_args(t)
    if orig is list and len(args) == 1:
        return is_chain_type(get_args(t)[0])

    if orig is dict and len(args) == 2:
        return is_chain_type(args[0]) and is_chain_type(args[1])

    if orig in (Union, types.UnionType) and len(args) == 2 and (type(None) in args):
        return is_chain_type(args[0] if args[1] is type(None) else args[1])

    return False


def dict_to_data_model(
    name: str,
    data_dict: dict[str, DataType],
    original_names: list[str] | None = None,
) -> type[BaseModel]:
    if not original_names:
        # Gets a map of a normalized_name -> original_name
        columns = normalize_col_names(list(data_dict))
        data_dict = dict(zip(columns.keys(), data_dict.values(), strict=False))
        original_names = list(columns.values())

    fields = {
        name: (
            anno
            if inspect.isclass(anno) and issubclass(anno, BaseModel)
            else anno | None,
            Field(
                validation_alias=AliasChoices(name, original_names[idx] or name),
                default=None,
            ),
        )
        for idx, (name, anno) in enumerate(data_dict.items())
    }

    class _DataModelStrict(BaseModel, extra="forbid"):
        @classmethod
        def _model_fields_by_aliases(cls) -> dict[str, tuple[str, FieldInfo]]:
            """Returns a map of aliases to original field names and info."""
            field_info = {}
            for _name, field in cls.model_fields.items():
                assert isinstance(field.validation_alias, AliasChoices)
                # Add mapping for all aliases (both normalized and original names)
                for alias in field.validation_alias.choices:
                    field_info[str(alias)] = (_name, field)
            return field_info

    # Generate random unique name if not provided
    if not name:
        name = f"DataModel_{uuid.uuid4().hex[:8]}"

    return create_model(
        name,
        __base__=_DataModelStrict,
        **fields,
    )  # type: ignore[call-overload]
