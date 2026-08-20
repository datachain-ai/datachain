import inspect
from datetime import datetime
from enum import Enum
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from typing_extensions import Literal as LiteralEx

from datachain.lib.data_model import (
    NULLABLE_SCALARS,
    is_mapping_annotation,
    is_sequence_annotation,
    unwrap_optional,
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
            # __members__, not iteration: EnumMeta.__iter__ skips zero-valued
            # and composite flag members on 3.11+, which would make the column
            # type depend on the Python version
            value_types = {type(m.value) for m in typ.__members__.values()}
            if not value_types:
                return String

            sql_type = (
                PYTHON_TO_SQL.get(value_types.pop()) if len(value_types) == 1 else None
            )
            if sql_type is None or sql_type in (Array, JSON):
                raise TypeError(
                    f"Cannot recognize type {typ}: enum member values"
                    " must all be of one supported scalar type"
                )

            return sql_type

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

    list_type = list_of_args_to_type(args)
    # Optional[scalar] elements map to a nullable Array element so None survives
    # (ClickHouse: Array(Nullable(T))).
    inner, is_optional = unwrap_optional(args0)
    if is_optional and (
        inner in NULLABLE_SCALARS
        or (inspect.isclass(inner) and issubclass(inner, Enum))
    ):
        list_type = SQLType.as_nullable(list_type)
    return Array(list_type)


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
