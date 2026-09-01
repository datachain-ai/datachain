import inspect
from datetime import datetime
from enum import Enum
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from typing_extensions import Literal as LiteralEx

from datachain.lib.data_model import (
    NULLABLE_SCALARS,
    is_mapping_annotation,
    is_sequence_annotation,
)
from datachain.lib.model_store import ModelStore
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
)

PYTHON_TO_SQL = {
    int: Int64,
    str: String,
    Literal: String,
    LiteralEx: String,
    Enum: String,
    float: Float,
    bool: Boolean,
    datetime: DateTime,  # Note, list of datetime is not supported yet
    bytes: Binary,  # Note, list of bytes is not supported yet
    list: Array,
    dict: JSON,
}


def python_to_sql(typ):  # noqa: PLR0911
    if inspect.isclass(typ):
        if issubclass(typ, SQLType):
            return typ
        if issubclass(typ, Enum):
            return str

    res = PYTHON_TO_SQL.get(typ)
    if res:
        return res

    orig = get_origin(typ)

    if orig in (Literal, LiteralEx):
        return String

    args = get_args(typ)
    if is_sequence_annotation(typ):
        return _list_to_array(typ, args)

    if orig is Annotated:
        # Ignoring annotations
        return python_to_sql(args[0])

    if is_mapping_annotation(typ):
        return JSON

    if orig in (Union, UnionType):
        if len(args) == 2 and (type(None) in args):
            non_none_arg = args[0] if args[0] is not type(None) else args[1]
            return python_to_sql(non_none_arg)

        if all(arg is str or get_origin(arg) in (Literal, LiteralEx) for arg in args):
            return String

        if _is_json_inside_union(orig, args):
            return JSON

    raise TypeError(f"Cannot recognize type {typ}")


def _list_to_array(typ, args):
    if args is None:
        raise TypeError(f"Cannot resolve type '{typ}' for flattening features")
    args0 = args[0]
    if ModelStore.is_pydantic(args0):
        return Array(JSON())

    # Resolve what the wrappers hold, not the wrappers: composed ones --
    # Annotated[str, ...] | Literal[None] -- are recognized by neither the scalar
    # table nor the union handling on their own.
    list_type = list_of_args_to_type(tuple(_unwrapped_for_lookup(a) for a in args))

    # Optional[scalar] elements map to a nullable Array element so None survives
    # (ClickHouse: Array(Nullable(T))). A fixed tuple keeps every slot in the one
    # column, so any slot admitting a None decides it, not just the first.
    admits_none = any(_peel_optional(arg)[1] for arg in args if arg is not Ellipsis)
    if admits_none and _takes_null(list_type):
        list_type = SQLType.as_nullable(list_type)
    return Array(list_type)


def _unwrapped_for_lookup(annotation: Any) -> Any:
    """What to resolve an element annotation as.

    Peeling is only useful here when what is left resolves to something. It does
    not for a Literal holding nothing but None, for Ellipsis marking a variadic
    tuple, or for a bare model that only the union around it made resolvable, and
    the annotation as written is what those already mapped to.
    """
    if annotation is Ellipsis:
        return annotation

    peeled, _ = _peel_optional(annotation)
    if peeled is annotation:
        return annotation

    try:
        python_to_sql(peeled)
    except TypeError:
        return annotation
    return peeled


def _peel_optional(annotation: Any) -> tuple[Any, bool]:
    """Strip the wrappers around an element type, reporting whether it admits None.

    They nest either way round -- ``Optional[Annotated[int, ...]]`` and
    ``Annotated[int | None, ...]`` -- and the None can be spelled as a union arm,
    as one of a Literal's values, or as ``Literal[None]`` inside a union.
    """
    is_optional = False
    while True:
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
            continue

        if get_origin(annotation) in (Literal, LiteralEx):
            values = get_args(annotation)
            remaining = tuple(v for v in values if v is not None)
            if not remaining:
                return type(None), True
            if len(remaining) != len(values):
                is_optional = True
                annotation = Literal[remaining]
                continue
            return annotation, is_optional

        if get_origin(annotation) in (Union, UnionType):
            arms = []
            for arm in get_args(annotation):
                peeled, arm_optional = _peel_optional(arm)
                if arm_optional or peeled is type(None):
                    is_optional = True
                if peeled is not type(None):
                    arms.append(peeled)
            if not arms:
                return type(None), True
            if len(arms) == 1:
                annotation = arms[0]
                continue
            return Union[tuple(arms)], is_optional  # noqa: UP007

        return annotation, is_optional


def _takes_null(sql_type: Any) -> bool:
    """Whether a NULL can sit in this column type.

    Compared by subclass: a project or a user may have subclassed the scalar.
    """
    sql_cls = sql_type if isinstance(sql_type, type) else type(sql_type)
    for scalar in NULLABLE_SCALARS:
        nullable = python_to_sql(scalar)
        nullable_cls = nullable if isinstance(nullable, type) else type(nullable)
        if issubclass(sql_cls, nullable_cls):
            return True
    return False


def list_of_args_to_type(args) -> SQLType:
    first_type = python_to_sql(args[0])
    for next_arg in args[1:]:
        try:
            next_type = python_to_sql(next_arg)
            if next_type != first_type:
                return JSON()
        except TypeError:
            return JSON()
    return first_type


def _is_json_inside_union(orig, args) -> bool:
    if orig in (Union, UnionType) and len(args) >= 2:
        # List in JSON: Union[dict, list[dict]]
        args_no_nones = [arg for arg in args if arg != type(None)]  # noqa: E721
        if len(args_no_nones) == 2:
            args_no_dicts = [arg for arg in args_no_nones if arg is not dict]
            if len(args_no_dicts) == 1 and get_origin(args_no_dicts[0]) is list:
                arg = get_args(args_no_dicts[0])
                if len(arg) == 1 and arg[0] is dict:
                    return True

        # List of objects: Union[MyClass, OtherClass]
        if any(inspect.isclass(arg) and issubclass(arg, BaseModel) for arg in args):
            return True
    return False
