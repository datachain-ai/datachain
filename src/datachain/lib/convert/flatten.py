import inspect
from collections.abc import Generator, Iterator
from typing import Any, NamedTuple

from pydantic import BaseModel

from datachain.lib.data_model import (
    UnionLayout,
    union_layout,
    union_slot_key,
    unwrap_optional,
)
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
    """One column a model emits. ``is_sentinel`` marks a tagged-union ``_type_tag``
    discriminator; otherwise it is a leaf."""

    path: tuple[str, ...]
    is_sentinel: bool


def iter_flat_columns(
    model: type[BaseModel], _prefix: tuple[str, ...] = ()
) -> Iterator[FlatColumn]:
    """Flat columns ``model`` emits, in storage order; a tagged-union node yields a
    leading _type_tag then its arms."""
    for name, f_info in model.model_fields.items():
        yield from _iter_field_columns(f_info.annotation, (*_prefix, name))


def _iter_field_columns(anno: Any, path: tuple[str, ...]) -> Iterator[FlatColumn]:
    if (layout := union_layout(anno)) is not None:
        yield FlatColumn(path, True)  # _type_tag
        for i, arm in enumerate(layout.arms):
            arm_path = (*path, union_slot_key(i)) if layout.use_slots else path
            if (fr := ModelStore.to_pydantic(arm)) is not None:
                yield from iter_flat_columns(fr, arm_path)
            else:
                yield FlatColumn(arm_path, False)
        return
    inner, _ = unwrap_optional(anno)
    if (fr := ModelStore.to_pydantic(inner)) is not None:
        yield from iter_flat_columns(fr, path)
    else:
        yield FlatColumn(path, False)


def flatten(obj: BaseModel) -> tuple:
    return tuple(_flatten_fields_values(type(obj).model_fields, obj))


def flatten_value(value, anno) -> tuple:
    """Flatten ``value`` for a column of type ``anno``. A tagged union emits its
    ``_type_tag`` then every arm's columns, only the active arm populated."""
    if (layout := union_layout(anno)) is not None:
        return tuple(_flatten_union(value, layout))
    if isinstance(value, (list, dict)):
        return (_flatten_scalar(value),)
    kind = classify_field(anno)
    if kind.is_model:
        if value is None:  # outer-merge pad
            return tuple(_emit_absent(kind.inner))
        return flatten(value)
    return (value,)


def _flatten_union(value, layout: UnionLayout) -> Generator:
    active = _match_union_arm(value, layout)
    yield active  # _type_tag: arm index, or NULL for None
    for i, arm in enumerate(layout.arms):
        yield from _flatten_arm(value if i == active else None, arm)


def _flatten_arm(value, arm) -> Generator:
    if (fr := ModelStore.to_pydantic(arm)) is not None:
        if value is None:
            yield from _emit_absent(fr)
        else:
            # flatten by the declared fields, not the (maybe subclass) instance's
            yield from _flatten_fields_values(fr.model_fields, value)
    else:
        yield _flatten_scalar(value)


def _match_union_arm(value, layout: UnionLayout) -> int | None:
    """Arm index for ``value`` (None for the None arm). Exact-type match beats
    ``isinstance`` so ``bool`` isn't swallowed by an ``int`` arm."""
    if value is None:
        if not layout.has_none:
            raise TypeError(f"value None does not match any arm of union {layout.arms}")
        return None
    for exact in (True, False):
        for i, arm in enumerate(layout.arms):
            if _arm_matches(value, arm, exact=exact):
                return i
    raise TypeError(f"value {value!r} does not match any arm of union {layout.arms}")


def _arm_matches(value, arm, *, exact: bool) -> bool:
    # exact match (incl. models) so a subclass isn't swallowed by a base-class arm
    if not inspect.isclass(arm):
        return False
    return type(value) is arm if exact else isinstance(value, arm)


def _flatten_scalar(value):
    if isinstance(value, list):
        return _flatten_list_field(value)
    if isinstance(value, dict):
        return {
            key: val.model_dump() if ModelStore.is_pydantic(type(val)) else val
            for key, val in value.items()
        }
    return value


def flatten_list(obj_list: list[BaseModel]) -> tuple:
    return tuple(
        val
        for obj in obj_list
        for val in _flatten_fields_values(type(obj).model_fields, obj)
    )


def _flatten_list_field(value: list) -> list:
    assert isinstance(value, list)
    if value and ModelStore.is_pydantic(type(value[0])):
        return [val.model_dump() for val in value]
    if value and isinstance(value[0], list):
        return [_flatten_list_field(v) for v in value]
    return value


def _leaf_count(model: type[BaseModel]) -> int:
    """Count of flat columns ``model`` emits (sentinels included)."""
    return sum(1 for _ in iter_flat_columns(model))


def _emit_absent(model: type[BaseModel]) -> Generator:
    """NULL for each of ``model``'s flat columns (an absent/inactive arm's slots)."""
    for _ in iter_flat_columns(model):
        yield None


def _flatten_fields_values(fields: dict, obj: BaseModel) -> Generator[Any, None, None]:
    for name, f_info in fields.items():
        yield from flatten_value(getattr(obj, name), f_info.annotation)
