import copy
import inspect
import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from datachain.lib.convert.flatten import _leaf_count
from datachain.lib.data_model import (
    UnionLayout,
    arm_selector,
    union_layout,
    unwrap_optional,
)
from datachain.lib.model_store import ModelStore
from datachain.query.schema import DEFAULT_DELIMITER

# Hydrates one model arm of a union: (arm_model, row, pos) -> (value, next_pos).
ModelReader = Callable[[type[BaseModel], Sequence[Any], int], tuple[Any, int]]


def unflatten_to_json(model: type[BaseModel], row: Sequence[Any], pos: int = 0) -> dict:
    return unflatten_to_json_pos(model, row, pos)[0]


def _arm_width(arm: Any) -> int:
    if (fr := ModelStore.to_pydantic(arm)) is not None:
        return _leaf_count(fr)
    return 1


def read_union(
    layout: UnionLayout,
    row: Sequence[Any],
    pos: int,
    read_model: ModelReader,
) -> tuple[Any, int]:
    """Read a tagged-union value: the ``_type_tag`` then every arm's columns,
    hydrating only the active arm. A NULL ``_type_tag`` reads back as None."""
    tag = row[pos]
    pos += 1
    result: Any = None
    for arm in layout.arms:
        if arm_selector(arm) == tag:
            if (fr := ModelStore.to_pydantic(arm)) is not None:
                result, pos = read_model(fr, row, pos)
            else:
                result, pos = row[pos], pos + 1
        else:
            pos += _arm_width(arm)
    return result, pos


def unflatten_to_json_pos(
    model: type[BaseModel], row: Sequence[Any], pos: int = 0
) -> tuple[dict, int]:
    res: dict[str, Any] = {}
    for name, f_info in model.model_fields.items():
        anno = f_info.annotation
        if (layout := union_layout(anno)) is not None:
            res[name], pos = read_union(layout, row, pos, unflatten_to_json_pos)
            continue
        inner, _ = unwrap_optional(anno)
        if ModelStore.is_pydantic(inner):
            res[name], pos = unflatten_to_json_pos(inner, row, pos)
        else:
            res[name] = row[pos]
            pos += 1
    return res, pos


def _normalize(name: str) -> str:
    if DEFAULT_DELIMITER in name:
        raise RuntimeError(
            f"variable '{name}' cannot be used because it contains {DEFAULT_DELIMITER}"
        )
    return _to_snake_case(name)


def _to_snake_case(name: str) -> str:
    """Convert a CamelCase name to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _unflatten_with_path(
    model: type[BaseModel], dump: dict[str, Any], name_path: list[str]
) -> BaseModel:
    res = {}
    for name, f_info in model.model_fields.items():
        anno = f_info.annotation
        name_norm = _normalize(name)
        lst = copy.copy(name_path)

        if inspect.isclass(anno) and issubclass(anno, BaseModel):
            lst.append(name_norm)
            val = _unflatten_with_path(anno, dump, lst)
            res[name] = val
        else:
            lst.append(name_norm)
            curr_path = DEFAULT_DELIMITER.join(lst)
            res[name] = dump[curr_path]
    return model(**res)


def unflatten(model: type[BaseModel], dump: dict[str, Any]) -> BaseModel:
    return _unflatten_with_path(model, dump, [])
