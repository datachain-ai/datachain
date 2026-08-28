from collections.abc import Generator, Iterator
from typing import Any, NamedTuple, get_args, get_origin

from pydantic import BaseModel

from datachain.lib.data_model import unwrap_optional
from datachain.lib.model_store import ModelStore


class FieldKind(NamedTuple):
    """Classification of a model field's annotation."""

    inner: Any
    is_optional: bool
    is_model: bool


def classify_field(annotation: Any) -> FieldKind:
    inner, is_optional = unwrap_optional(annotation)
    return FieldKind(inner, is_optional, ModelStore.is_pydantic(inner))


class FlatColumn(NamedTuple):
    """One column a model emits, in DB-column order. ``is_sentinel`` marks the
    ``_type_tag`` discriminator prepended for an ``Optional[DataModel]`` node; otherwise
    it is a scalar/list/dict leaf."""

    path: tuple[str, ...]
    is_sentinel: bool


def iter_flat_columns(
    model: type[BaseModel], _prefix: tuple[str, ...] = ()
) -> Iterator[FlatColumn]:
    """Yield the flat columns ``model`` emits, in order: each ``Optional[DataModel]``
    node contributes a leading sentinel, then its (recursively flattened) leaves.
    """
    for name, f_info in model.model_fields.items():
        kind = classify_field(f_info.annotation)
        path = (*_prefix, name)
        if kind.is_model:
            if kind.is_optional:
                yield FlatColumn(path, True)
            yield from iter_flat_columns(kind.inner, path)
        else:
            yield FlatColumn(path, False)


def flatten(obj: BaseModel) -> tuple:
    return tuple(_flatten_fields_values(type(obj).model_fields, obj))


def is_optional_model(anno: Any) -> bool:
    kind = classify_field(anno)
    return kind.is_optional and kind.is_model


def flatten_value(value: Any, anno: Any) -> tuple:
    """Flatten ``value`` for one column declared with annotation ``anno``.

    ``Optional[DataModel]`` emits a leading ``_type_tag`` before its leaves.
    ``Optional[basic]`` is a plain nullable column. Nulls inside collections
    (``list[Optional[T]]``) and bare ``Union[A, B]`` are not represented.
    """
    kind = classify_field(anno)
    if kind.is_model:
        if kind.is_optional:
            if value is None:
                return (1, *_emit_absent(kind.inner))
            return (0, *flatten(value))
        if value is None:
            # Non-Optional model None (outer-merge pad): per-leaf placeholders.
            return tuple(_emit_absent(kind.inner))
        return flatten(value)
    return (value,)


def flatten_list(obj_list: list[BaseModel]) -> tuple:
    return tuple(
        val
        for obj in obj_list
        for val in _flatten_fields_values(type(obj).model_fields, obj)
    )


_BARE_CONTAINERS = (dict, list, tuple, set, frozenset)


def _may_hold_model(annotation: Any) -> bool:
    """Whether a declared type can hold a model, erring towards yes when unsure.

    An erased annotation -- Any, object, a bare container -- says nothing about
    its contents, so it has to be walked.
    """
    if ModelStore.is_pydantic(annotation):
        return True
    if annotation is Any or annotation is object or annotation is None:
        return True
    args = get_args(annotation)
    if not args:
        return get_origin(annotation) is not None or annotation in _BARE_CONTAINERS
    return any(arg is not Ellipsis and _may_hold_model(arg) for arg in args)


def _flatten_sequence(value: Any, annotation: Any) -> Any:
    args = get_args(annotation)
    if (
        get_origin(annotation) is tuple
        and args
        and args[-1] is not Ellipsis
        and len(args) == len(value)
    ):
        if not any(_may_hold_model(arg) for arg in args):
            return value
        return [
            _flatten_nested(item, arg) for item, arg in zip(value, args, strict=True)
        ]
    item_type = args[0] if args else Any
    if not _may_hold_model(item_type):
        return value
    return [_flatten_nested(item, item_type) for item in value]


def _flatten_mapping(value: dict, annotation: Any) -> Any:
    args = get_args(annotation)
    value_type = args[1] if len(args) == 2 else Any
    if not _may_hold_model(value_type):
        return value
    return {key: _flatten_nested(item, value_type) for key, item in value.items()}


def _flatten_nested(value: Any, annotation: Any) -> Any:
    """Replace models with their JSON-mode dumps, leaving everything else alone.

    JSON mode is what the warehouse applies to a model it receives directly, so
    converting here keeps the two paths agreeing on dates, paths and serializers
    declared for JSON output. A value whose declared type cannot hold a model is
    returned as it is, so a vector of numbers is never copied.
    """
    if ModelStore.is_pydantic(type(value)):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return _flatten_sequence(value, annotation)
    if isinstance(value, dict):
        return _flatten_mapping(value, annotation)
    return value


def _leaf_count(model: type[BaseModel]) -> int:
    """Count of flat columns ``model`` emits (sentinels included)."""
    return sum(1 for _ in iter_flat_columns(model))


def _emit_absent(model: type[BaseModel]) -> Generator[int | None, None, None]:
    """Placeholder values shaped like ``model``'s flat columns, used when an
    ``Optional[DataModel]`` parent is None and the leaves still need a slot."""
    for col in iter_flat_columns(model):
        yield 1 if col.is_sentinel else None


def _flatten_fields_values(fields: dict, obj: BaseModel) -> Generator[Any, None, None]:
    for name, f_info in fields.items():
        kind = classify_field(f_info.annotation)
        # Direct attribute access skips Pydantic's model_dump().
        value = getattr(obj, name)
        if isinstance(value, (list, tuple, dict)):
            yield _flatten_nested(value, f_info.annotation)
        elif kind.is_model:
            if kind.is_optional:
                if value is None:
                    yield 1
                    yield from _emit_absent(kind.inner)
                else:
                    yield 0
                    yield from _flatten_fields_values(kind.inner.model_fields, value)
            else:
                yield from _flatten_fields_values(kind.inner.model_fields, value)
        else:
            yield value
