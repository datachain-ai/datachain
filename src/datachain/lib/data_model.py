import hashlib
import inspect
import types
import uuid
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import lru_cache
from typing import Any, ClassVar, NamedTuple, Union, get_args, get_origin

from pydantic import AliasChoices, BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from datachain import json
from datachain.error import OutdatedDatasetFormatError
from datachain.lib.model_store import ModelStore
from datachain.lib.utils import (
    DataChainParamsError,
    normalize_col_names,
    type_to_str,
)

_skip_optional_promotion: ContextVar[bool] = ContextVar(
    "_skip_optional_promotion", default=False
)


@contextmanager
def skip_optional_promotion() -> Iterator[None]:
    """Disable ``default=None`` -> ``Optional`` promotion while building a model
    DataChain reconstructs from a stored schema (e.g. reading a dataset whose
    original model code isn't importable). Such fields already carry the exact
    annotation from the schema, so promoting them would corrupt the replayed type.
    """
    token = _skip_optional_promotion.set(True)
    try:
        yield
    finally:
        _skip_optional_promotion.reset(token)


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
        promote_default_none(cls)
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

            inner_type, _ = unwrap_optional(field_type)
            child_model = ModelStore.to_pydantic(inner_type)
            layout = union_layout(field_type)
            atomic_union = layout is not None and layout.use_slots
            if sub_sel is not None and not atomic_union:
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


def _origin_accepts(container: type, t: Any) -> bool:
    """Whether `container` would satisfy the origin of `t`.

    Note the direction: `issubclass(dict, origin)`, not `origin is dict`, so
    abstract spellings like `Mapping[str, X]` are recognised too. Some classes
    refuse `issubclass` outright -- TypedDicts and non-method protocols raise --
    and for those the answer is simply no.
    """
    orig = get_origin(t)
    if not inspect.isclass(orig):
        return False
    try:
        return issubclass(container, orig)
    except TypeError:
        return False


def is_mapping_annotation(t: Any) -> bool:
    """Whether `t` is a dict-like type, so `Mapping[str, X]` counts too.

    Origins loose enough to accept a list as well -- `Collection`, `Iterable` --
    are not mappings. Excluding them keeps this predicate mutually exclusive with
    `is_sequence_annotation`, so callers cannot disagree by testing in a different
    order, and matches `python_to_sql`, which maps those to `Array`.
    """
    return _origin_accepts(dict, t) and not _origin_accepts(list, t)


def is_sequence_annotation(t: Any) -> bool:
    """Whether `t` is a list- or tuple-like type; `Sequence[X]` counts too.

    `set` is excluded because it is not a `DataType` and `python_to_sql` has no
    column type for it.
    """
    return _origin_accepts(list, t) or _origin_accepts(tuple, t)


def is_tuple_annotation(t: Any) -> bool:
    """Whether `t` is genuinely a tuple, whose shape must be restored on read.

    Note the direction: `issubclass(orig, tuple)`, not the reversed form used by
    the container predicates. `Sequence[X]` is satisfiable by a tuple but is not
    one, and rebuilding an ordinary list as a tuple would be wrong.
    """
    orig = get_origin(t)
    return inspect.isclass(orig) and issubclass(orig, tuple)


def annotation_parts(t: Any) -> tuple[Any, ...]:
    """Components of a union or collection, `()` for a leaf.

    `list[X]` -> `(X,)`, `dict[K, V]` -> `(K, V)`, `X | None` -> `(X,)`.
    """
    if get_origin(t) in (Union, types.UnionType):
        return tuple(a for a in get_args(t) if a is not type(None))
    if is_mapping_annotation(t) or is_sequence_annotation(t):
        return tuple(a for a in get_args(t) if a is not Ellipsis)
    return ()


def key_needs_json_decode(t: Any) -> bool:
    """Whether mapping keys of type `t` are JSON-encoded in the DB.

    Keys are stored as strings, so `dict[int, ...]` reads back as `{"1": ...}`
    and needs decoding, but a key type that can be a plain string must not --
    `"foo"` is not valid JSON.
    """
    if t is str or t is Any:
        return False
    if get_origin(t) in (Union, types.UnionType):
        # one arm that can be a plain string makes decoding unsafe for all of them
        return all(key_needs_json_decode(p) for p in annotation_parts(t))
    # a collection key is never a plain string, whatever its components are
    return True


def unwrap_optional(t: Any) -> tuple[Any, bool]:
    """Unwrap a type that includes `None` to `(non_none, True)`.

    Handles `Optional[X]`, `Union[X, None]`, and PEP-604 `X | None`. Multi-arg
    unions like `Union[A, B, None]` return `(Union[A, B], True)`; non-Optional and
    None-free unions return `(t, False)`.
    """
    orig = get_origin(t)
    args = get_args(t)
    if orig in (Union, types.UnionType) and type(None) in args:
        non_none = tuple(a for a in args if a is not type(None))
        if len(non_none) == 1:
            return non_none[0], True
        return Union[non_none], True  # type: ignore[return-value]  # noqa: UP007
    return t, False


# Scalars whose Optional form maps to a nullable column, so None round-trips as a
# real NULL. SQLite stores NaN as NULL, so a stored NaN reads back as None there
# (the other backends keep them distinct); None itself is consistent everywhere.
NULLABLE_SCALARS: "tuple[type, ...]" = (int, float, str, bool, bytes, datetime)


def union_arms(anno: Any) -> tuple[list[Any], bool]:
    """``(non_none_arms, has_none)``, arms sorted by type string for a deterministic,
    order-independent column layout (``Union[int, str]`` == ``Union[str, int]``)."""
    if get_origin(anno) in (Union, types.UnionType):
        args = get_args(anno)
        non_none = sorted((a for a in args if a is not type(None)), key=type_to_str)
        return non_none, type(None) in args
    return [anno], False


# Scalar arm types a tagged union stores as its own column (DataModels handled too).
_TAGGABLE_SCALARS: "tuple[type, ...]" = NULLABLE_SCALARS


def _is_taggable_arm(arm: Any) -> bool:
    return ModelStore.is_pydantic(arm) or arm in _TAGGABLE_SCALARS


class UnionLayout(NamedTuple):
    """Physical layout of a tagged union. ``use_slots`` is True for multi-arm unions
    (each arm in a column named by its type, ``int``/``Pet``); False for
    ``Optional[Model]``, whose single model arm flattens directly under the signal."""

    arms: tuple[Any, ...]
    has_none: bool
    use_slots: bool


# hidden discriminator column of a tagged union
TYPE_TAG_FIELD = "_type_tag"


def union_layout(anno: Any) -> "UnionLayout | None":
    """Tagged-union layout, or None when no ``_type_tag`` is needed (plain leaf,
    ``Optional[basic]``, plain model, or a union with a non-taggable arm)."""
    return _union_layout(anno)


@lru_cache(maxsize=4096)
def _union_layout(anno: Any) -> "UnionLayout | None":
    arms_list, has_none = union_arms(anno)
    arms = tuple(arms_list)  # immutable: a cached layout must not be mutable
    if not all(_is_taggable_arm(arm) for arm in arms):
        return None
    if len(arms) >= 2:
        # arms sharing a type string collapse into one on reload (it is their
        # storage identity), so reject indistinguishable arms up front
        keys = [type_to_str(arm) for arm in arms]
        if len(set(keys)) != len(keys):
            dup = next(k for k in keys if keys.count(k) > 1)
            raise DataChainParamsError(
                f"Union has indistinguishable arms named {dup!r}; arms must have "
                "distinct type names (rename one of the models)"
            )
        # each arm is a column named by its type; those names must be unique and
        # must not shadow the discriminator column
        slots = [arm_selector(arm) for arm in arms]
        if len(set(slots)) != len(slots):
            dup = next(s for s in slots if slots.count(s) > 1)
            raise DataChainParamsError(
                f"Union arms map to the same column name {dup!r}; rename one arm."
            )
        if TYPE_TAG_FIELD in slots:
            raise DataChainParamsError(
                f"Union arm {TYPE_TAG_FIELD!r} is reserved; rename that model."
            )
        return UnionLayout(arms, has_none, use_slots=True)
    if len(arms) == 1 and has_none and ModelStore.is_pydantic(arms[0]):
        return UnionLayout(arms, has_none=True, use_slots=False)
    return None


def arm_selector(arm: Any) -> str:
    """Name of a union arm — its DB-column slot and its readable-path segment, which
    are the same. A model's stable logical name (reload-safe, survives reading a
    dataset without the model code) or a scalar type name."""
    if (fr := ModelStore.to_pydantic(arm)) is not None:
        return ModelStore._base_name(fr)
    return arm.__name__


def arm_for_tag(layout: "UnionLayout", tag: Any) -> Any:
    """Union arm a stored ``_type_tag`` selects; None when the value is absent."""
    for arm in layout.arms:
        if arm_selector(arm) == tag:
            return arm
    if tag is not None and not isinstance(tag, str):
        return _arm_for_index_tag(layout, tag)
    return None


# Index layout, read-only and deprecated (removal: #1949): only ever stored
# `Optional[DataModel]`, 0 for its single arm and anything else for None.
LEGACY_PRESENT_TAG = 0


def _arm_for_index_tag(layout: "UnionLayout", tag: Any) -> Any:
    if layout.use_slots:
        names = ", ".join(arm_selector(arm) for arm in layout.arms)
        raise OutdatedDatasetFormatError(f"unknown _type_tag {tag!r}: expected {names}")
    _warn_index_tag()
    return layout.arms[0] if tag == LEGACY_PRESENT_TAG else None


@lru_cache(maxsize=1)  # once per process: the read paths hit this per row
def _warn_index_tag() -> None:
    # FutureWarning: a DeprecationWarning is hidden by default outside __main__
    warnings.warn(
        "Legacy Optional[DataModel] _type_tag is deprecated; re-save the dataset.",
        FutureWarning,
        stacklevel=2,
    )


def promote_default_none(model: type[BaseModel]) -> None:
    """Auto-promote non-Optional fields with `default=None` to `Optional[...]`.

    `x: int = None` is treated as `x: Optional[int] = None`, so the column is
    nullable and `x=None` round-trips as `None`. Without this it would read back
    as the type default (`0`/`""`) on backends with non-nullable columns.

    Skipped under `skip_optional_promotion()` for models reconstructed from a
    stored schema, whose fields already carry their exact annotation (promoting
    `default=None` there breaks the partial-model tree walker).
    """
    if _skip_optional_promotion.get():
        return
    promoted = False
    for finfo in model.model_fields.values():
        if finfo.default is not None or finfo.is_required():
            continue
        anno = finfo.annotation
        if anno is None:
            continue
        _, is_optional = unwrap_optional(anno)
        if is_optional:
            continue
        finfo.annotation = anno | None  # type: ignore[assignment]
        promoted = True
    if promoted:
        model.model_rebuild(force=True)


def is_chain_type(t: type) -> bool:
    """Return true if type is supported by `DataChain`."""
    if ModelStore.is_pydantic(t):
        return True
    if any(t is ft or t is get_args(ft)[0] for ft in get_args(StandardType)):
        return True
    inner, is_optional = unwrap_optional(t)
    if is_optional:
        return is_chain_type(inner)
    return _is_chain_container_type(t)


def _is_chain_container_type(t: type) -> bool:
    """Whether a union / list / dict annotation is a supported DataChain type."""
    # Deliberately not using `annotation_parts` here. This is validation, not
    # traversal: it must see the raw args, since normalising away `Ellipsis` would
    # let `list[int, ...]` through to be serialized as `list[int]`. Only `list` and
    # `dict` at the exact arity `type_to_str` can write back out are accepted --
    # abstract origins serialize as a bare "Sequence"/"Mapping", and `python_to_sql`
    # mis-types tuples. Matching on the origin identity also avoids `issubclass`
    # against generics that reject it (TypedDicts, some protocols).
    arms, _ = union_arms(t)
    if len(arms) >= 2:
        # multi-arm union: supported only as a tagged union (scalar/DataModel arms)
        return union_layout(t) is not None and all(is_chain_type(arm) for arm in arms)

    orig, args = get_origin(t), get_args(t)
    if orig is list and len(args) == 1:
        _reject_list_of_model_union(args[0])
        return is_chain_type(args[0])
    if orig is dict and len(args) == 2:
        return is_chain_type(args[0]) and is_chain_type(args[1])
    return False


def _reject_list_of_model_union(elem: Any) -> None:
    """Reject a list of a union with a DataModel arm (model elements collapse to
    dicts on read; scalar arms round-trip via JSON)."""
    layout = union_layout(elem)
    if (
        layout is not None
        and layout.use_slots
        and any(ModelStore.is_pydantic(arm) for arm in layout.arms)
    ):
        raise DataChainParamsError(
            "list[Union[...]] with a DataModel arm is not supported: list elements "
            "lose their model type. Put the Union inside a DataModel field instead."
        )


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
