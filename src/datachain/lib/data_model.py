import functools
import hashlib
import inspect
import operator
import types
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    NewType,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import AliasChoices, BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from datachain import json
from datachain.lib.model_store import ModelStore
from datachain.lib.utils import normalize_col_names, type_to_str

_TYPE_ALIAS_TYPES: tuple[type, ...] = tuple(
    alias
    for alias in (
        getattr(types, "TypeAliasType", None),
        getattr(__import__("typing"), "TypeAliasType", None),
    )
    if isinstance(alias, type)
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


def _is_new_type(t: Any) -> bool:
    """Whether `t` is an actual `typing.NewType`, not merely shaped like one."""
    return isinstance(t, NewType)


def _is_type_alias(t: Any) -> bool:
    """Whether `t` is an actual PEP-695 alias, not merely shaped like one.

    Attribute-shaped duck typing would misread a user model that happens to
    define `__value__`, so the concrete type is what decides.
    """
    return bool(_TYPE_ALIAS_TYPES) and isinstance(t, _TYPE_ALIAS_TYPES)


def _effective_alias_args(alias: Any, supplied: tuple[Any, ...]) -> "tuple | None":
    """Arguments for `alias`, filling PEP-696 defaults, or `None` if incomplete."""
    params = getattr(alias, "__type_params__", ())
    if len(supplied) > len(params):
        return None
    args = list(supplied)
    for param in params[len(supplied) :]:
        if not getattr(param, "has_default", lambda: False)():
            return None
        args.append(param.__default__)
    return tuple(args)


def _substitute_type_vars(t: Any, mapping: "dict[Any, Any]") -> Any:
    """Replace type variables in `t` using `mapping`, as far as is reproducible."""
    if isinstance(t, TypeVar):
        return mapping.get(t, t)
    args = get_args(t)
    if not args:
        return t
    new_args = tuple(_substitute_type_vars(a, mapping) for a in args)
    if new_args == args:
        return t
    try:
        return _rebuild_with_args(t, new_args)
    except Exception:  # noqa: BLE001 - typing internals vary across versions
        return t


def _rebuild_with_args(t: Any, new_args: tuple[Any, ...]) -> Any:
    """Rebuild the generic `t` with `new_args` substituted in."""
    origin = get_origin(t)
    if origin is Annotated:
        return Annotated[new_args]
    if origin in (Union, types.UnionType):
        # runtime rebuild from a tuple; `X | Y` has no splat form
        return Union[new_args]  # noqa: UP007
    if (copy_with := getattr(t, "copy_with", None)) is not None:
        return copy_with(new_args)
    # class subscription, not the origin's own __getitem__
    return origin[new_args]


def _resolve_generic_alias(t: Any) -> Any:
    """Substitute a PEP-695 alias, e.g. `Key[str]` where `type Key[T] = T`.

    Handles the unsubscripted form too when every parameter carries a PEP-696
    default. Returns `t` unchanged when the parameters cannot be filled in.
    """
    alias = get_origin(t) if _is_type_alias(get_origin(t)) else t
    if not _is_type_alias(alias) or not getattr(alias, "__type_params__", ()):
        return t
    args = _effective_alias_args(alias, get_args(t))
    if args is None:
        return t
    return _substitute_type_vars(
        alias.__value__, dict(zip(alias.__type_params__, args, strict=True))
    )


def alias_cycle_key(t: Any) -> Any:
    """A stable key identifying an alias application, or `None`.

    Object identity is unusable here: intermediate alias objects are freed and
    their ids reused, and substitution mints a fresh object every pass, so a
    finite chain can look cyclic and a genuine cycle can look finite.
    """
    origin = get_origin(t)
    alias = origin if _is_type_alias(origin) else t
    if not _is_type_alias(alias):
        return None
    try:
        return (alias, get_args(t))
    except TypeError:
        return None


def unwrap_alias(t: Any) -> Any:
    """Strip `Annotated`, `NewType` and PEP-695 alias wrappers from `t`.

    Each is transparent for storage: the DB sees whatever the wrapped type
    produces, so every read-side decision must be made on the inner type.

    Alias parameters are substituted, including PEP-696 defaults, so `Key[str]`
    behaves as `str`. An alias body is used even when parameters remain free, so
    `type TupleKey[T] = tuple[T, ...]` keeps its shape. Recursive aliases stop at
    the repeat rather than spinning.
    """
    seen: set[Any] = set()
    while True:
        if (cycle_key := alias_cycle_key(t)) is not None:
            if cycle_key in seen:
                return t
            seen.add(cycle_key)
        if get_origin(t) is Annotated:
            t = get_args(t)[0]
        elif _is_new_type(t):
            t = t.__supertype__
        elif (resolved := _resolve_generic_alias(t)) is not t:
            t = resolved
        elif _is_type_alias(t):
            t = t.__value__
        else:
            return t


def literal_members(t: Any) -> tuple[Any, ...]:
    """Members of a `Literal`, gathered through aliases and unions, `()` if none.

    Members are returned as declared, so an `Enum` member stays a member rather
    than collapsing to its value.
    """
    t = unwrap_alias(t)
    if get_origin(t) is Literal:
        return get_args(t)
    if get_origin(t) in (Union, types.UnionType):
        return tuple(m for p in annotation_parts(t) for m in literal_members(p))
    return ()


def union_arms(t: Any) -> tuple[Any, ...]:
    """The arms of a union, normalised, or `(t,)` for a non-union."""
    t = unwrap_alias(t)
    if get_origin(t) in (Union, types.UnionType):
        return tuple(unwrap_alias(p) for p in annotation_parts(t))
    return (t,)


def resolve_literal_member(t: Any, value: Any) -> Any:
    """The declared `Literal` member equal to `value`, or `None` if there is none.

    Exact type wins over mere equality, in two steps. `Literal[Colour.RED, "red"]`
    resolves a stored `"red"` to the plain string member rather than the enum,
    which matches pydantic and makes the answer independent of declaration order.
    And a *non-Literal* arm whose type matches exactly keeps the value as it is:
    `True == 1 == 1.0` in Python, so `Literal[1] | bool` must not turn a stored
    `True` into `1`.
    """
    members = literal_members(t)
    if not members:
        return None
    for member in members:
        if type(member) is type(value) and member == value:
            return member
    for arm in union_arms(t):
        if get_origin(arm) is None and arm is type(value):
            return None
    for member in members:
        # `True == 1` and `1 == 1.0`: never let a bool cross-match a number
        if isinstance(member, bool) == isinstance(value, bool) and member == value:
            return member
    return None


def is_enum_annotation(t: Any) -> TypeGuard[type[Enum]]:
    """Whether `t` is an `Enum` subclass, whose members the DB stores as values."""
    return inspect.isclass(t) and issubclass(t, Enum)


def _enum_values_are_strings(t: Any) -> bool:
    return is_enum_annotation(t) and all(isinstance(member.value, str) for member in t)


def type_var_target(t: Any) -> Any:
    """The most specific type a type variable can stand for, or `None`.

    A PEP-696 default, a bound, or a constraint set narrows what may have been
    stored; an unconstrained variable narrows nothing.
    """
    if not isinstance(t, TypeVar):
        return None
    if getattr(t, "has_default", lambda: False)():
        return getattr(t, "__default__", None)  # PEP-696, Python 3.13+
    if t.__bound__ is not None:
        return t.__bound__
    if t.__constraints__:
        return functools.reduce(operator.or_, t.__constraints__)
    return None


def resolve_type_var(t: Any) -> Any:
    """`t` with a type variable replaced by what it can stand for."""
    return type_var_target(t) or t if isinstance(t, TypeVar) else t


def key_needs_json_decode(t: Any) -> bool:
    """Whether some key of type `t` may be JSON-encoded in the DB.

    Keys are stored as strings, so `dict[int, ...]` reads back as `{"1": ...}`
    and needs decoding. This is the annotation-level question used to decide
    whether a mapping needs converting at all; whether a *particular* key was
    stored verbatim is `key_is_stored_verbatim`.
    """
    t = unwrap_alias(t)
    if isinstance(t, TypeVar):
        target = type_var_target(t)
        # an unconstrained variable says nothing: do not risk decoding
        return target is not None and key_needs_json_decode(target)
    if t is str or t is Any or _is_type_alias(t):
        # still an alias after unwrapping means a cycle we could not resolve
        return False
    if get_origin(t) is Literal:
        return any(not isinstance(v, str) for v in get_args(t))
    if _enum_values_are_strings(t) or (inspect.isclass(t) and issubclass(t, str)):
        # a string-valued enum may be stored verbatim, so a key must be offered
        # to the converter rather than decoded blind
        return False
    if get_origin(t) in (Union, types.UnionType):
        # any arm that may be encoded is enough to warrant looking at the key
        return any(key_needs_json_decode(p) for p in annotation_parts(t))
    # a collection key is never a plain string, whatever its components are
    return True


def key_is_stored_verbatim(t: Any, key: str) -> bool:
    """Whether this exact stored `key` should be taken as a plain string.

    `Literal` needs the key itself to decide: `Literal["a", 1]` stores `"a"`
    verbatim but JSON-encodes `1`, and the same holds under a union or
    `Optional` wrapper. A type that admits *any* string -- `str` itself, a `str`
    subclass, a string-valued enum -- takes every key verbatim.
    """
    t = unwrap_alias(t)
    if isinstance(t, TypeVar) or t is str or t is Any or _is_type_alias(t):
        # a type variable defers to its target; anything still aliased after
        # unwrapping is a cycle we could not resolve, and neither may be decoded
        target = type_var_target(t) if isinstance(t, TypeVar) else None
        return target is None or key_is_stored_verbatim(target, key)
    if get_origin(t) is Literal:
        return any(isinstance(v, str) and v == key for v in get_args(t))
    if is_enum_annotation(t):
        # finite, so only an actual member value is stored verbatim; anything
        # else must stay available for the other union arms to decode
        return any(member.value == key for member in t)
    if inspect.isclass(t) and issubclass(t, str):
        return True
    if get_origin(t) in (Union, types.UnionType):
        return any(key_is_stored_verbatim(p, key) for p in annotation_parts(t))
    return False


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

# _type_tag discriminator for Optional[DataModel]: this value marks the present arm.
OPTIONAL_PRESENT_TAG = 0


def optional_tag_is_absent(tag: "Any") -> bool:
    """An Optional[DataModel] subtree is absent when its ``_type_tag`` is NULL
    (outer-join padding) or not the present-arm value."""
    return tag is None or tag != OPTIONAL_PRESENT_TAG


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

    # Deliberately not using `annotation_parts` here. This is validation, not
    # traversal: it must see the raw args, since normalising away `Ellipsis` would
    # let `list[int, ...]` through to be serialized as `list[int]`. Only `list` and
    # `dict` at the exact arity `type_to_str` can write back out are accepted --
    # abstract origins serialize as a bare "Sequence"/"Mapping", and `python_to_sql`
    # mis-types tuples. Matching on the origin identity also avoids `issubclass`
    # against generics that reject it (TypedDicts, some protocols).
    orig = get_origin(t)
    args = get_args(t)
    if orig is list:
        return len(args) == 1 and is_chain_type(args[0])
    if orig is dict:
        return len(args) == 2 and all(is_chain_type(arg) for arg in args)

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
