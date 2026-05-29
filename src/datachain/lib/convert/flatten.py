from collections.abc import Generator

from pydantic import BaseModel

from datachain.lib.data_model import unwrap_optional
from datachain.lib.model_store import ModelStore


def flatten(obj: BaseModel) -> tuple:
    return tuple(_flatten_fields_values(type(obj).model_fields, obj))


def is_optional_model(anno) -> bool:
    """True when ``anno`` is ``Optional[DataModel]`` — the only shape that needs
    an ``is_null`` sentinel emitted alongside its flattened leaves."""
    inner, is_optional = unwrap_optional(anno)
    return is_optional and ModelStore.is_pydantic(inner)


def flatten_value(value, anno) -> tuple:
    """Flatten ``value`` for one column declared with annotation ``anno``.

    Optional shapes (see also ``unwrap_optional`` in ``data_model.py``):

    - ``Optional[basic]`` (int/str/float/bool/bytes/datetime): native nullable
      column.
    - ``Optional[DataModel]``: leading ``is_null`` sentinel + leaf columns.
    - ``Optional[list[T]]`` / ``Optional[dict[K, V]]``: stored as NULL where the
      backend has a nullable array/map column; on backends that reject a nullable
      array/map type, use ``list[T] = []`` / ``dict = {{}}`` instead of an
      Optional collection until that is supported.
    - ``list[Optional[T]]`` / ``dict[K, Optional[V]]``: not supported — no
      per-element sentinels are emitted.
    - True multi-arg ``Union[A, B]`` without None: falls back to JSON, no
      schema-level discrimination.

    """
    inner, is_optional = unwrap_optional(anno)
    if ModelStore.is_pydantic(inner):
        if is_optional:
            if value is None:
                return (True, *_emit_absent(inner))
            return (False, *flatten(value))
        if value is None:
            # Non-Optional model but the value is None (e.g. an unmatched side of
            # a full outer merge). There is no sentinel column, so emit absent
            # placeholders for every leaf — SQL NULL, or the column type's default
            # on backends with non-nullable leaves — matching the schema width.
            return tuple(_emit_absent(inner))
        return flatten(value)
    return (value,)


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
    """Count of flat columns ``_flatten_fields_values`` emits for ``model``."""
    count = 0
    for f_info in model.model_fields.values():
        inner, is_optional = unwrap_optional(f_info.annotation)
        if ModelStore.is_pydantic(inner):
            if is_optional:
                count += 1
            count += _leaf_count(inner)
        else:
            count += 1
    return count


def _emit_absent(model: type[BaseModel]) -> Generator:
    """Placeholder values shaped like ``model``'s flat columns, used when an
    ``Optional[DataModel]`` parent is None and the leaves still need a slot."""
    for f_info in model.model_fields.values():
        inner, is_optional = unwrap_optional(f_info.annotation)
        if ModelStore.is_pydantic(inner):
            if is_optional:
                yield True
            yield from _emit_absent(inner)
        else:
            yield None


def _flatten_fields_values(fields: dict, obj: BaseModel) -> Generator:
    for name, f_info in fields.items():
        inner, is_optional = unwrap_optional(f_info.annotation)
        # Direct attribute access skips Pydantic's model_dump().
        value = getattr(obj, name)
        if isinstance(value, list):
            yield _flatten_list_field(value)
        elif isinstance(value, dict):
            yield {
                key: val.model_dump() if ModelStore.is_pydantic(type(val)) else val
                for key, val in value.items()
            }
        elif ModelStore.is_pydantic(inner):
            if is_optional:
                if value is None:
                    yield True
                    yield from _emit_absent(inner)
                else:
                    yield False
                    yield from _flatten_fields_values(inner.model_fields, value)
            else:
                yield from _flatten_fields_values(inner.model_fields, value)
        else:
            yield value


def _flatten(obj: BaseModel) -> tuple:
    return tuple(_flatten_fields_values(type(obj).model_fields, obj))
