import copy
import inspect
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from datachain.lib.convert.flatten import _leaf_count, classify_field
from datachain.lib.data_model import optional_tag_is_absent
from datachain.query.schema import DEFAULT_DELIMITER


def unflatten_to_json(model: type[BaseModel], row: Sequence[Any], pos: int = 0) -> dict:
    return unflatten_to_json_pos(model, row, pos)[0]


def read_optional_sentinel(
    inner: type[BaseModel], row: Sequence[Any], pos: int
) -> tuple[bool, int]:
    """Consume the leading ``_type_tag`` discriminator of an ``Optional[DataModel]``
    subtree, returning ``(absent, next_pos)``. A NULL sentinel (e.g. outer-join
    padding) counts as absent.
    """
    sentinel = row[pos]
    pos += 1
    if optional_tag_is_absent(sentinel):
        return True, pos + _leaf_count(inner)
    return False, pos


def unflatten_to_json_pos(
    model: type[BaseModel], row: Sequence[Any], pos: int = 0
) -> tuple[dict, int]:
    res: dict[str, Any] = {}
    for name, f_info in model.model_fields.items():
        kind = classify_field(f_info.annotation)
        if kind.is_model:
            if kind.is_optional:
                absent, pos = read_optional_sentinel(kind.inner, row, pos)
                if absent:
                    res[name] = None
                    continue
            res[name], pos = unflatten_to_json_pos(kind.inner, row, pos)
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
