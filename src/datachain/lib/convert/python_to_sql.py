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


class UnstorableTypeError(TypeError):
    """No column type holds these values faithfully.

    Distinct from a type simply not resolving: a heterogeneous tuple falls back
    to JSON, but a refusal must not, or the values it refuses would be stored
    lossily by the fallback instead.
    """


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
            return _enum_to_sql(typ)

    res = PYTHON_TO_SQL.get(typ)
    if res:
        return res

    orig = get_origin(typ)

    if orig in (Literal, LiteralEx):
        return _values_to_sql(get_args(typ))

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

        if all(get_origin(arg) in (Literal, LiteralEx) for arg in args):
            # Literal[1] | Literal[2] holds the same values as Literal[1, 2] and
            # has to reach the same column type.
            return _values_to_sql(value for arg in args for value in get_args(arg))

        if all(arg is str or get_origin(arg) in (Literal, LiteralEx) for arg in args):
            # A str arm contributes str; a Literal arm contributes its values.
            return _values_to_sql(
                "" if arg is str else value
                for arg in args
                for value in ((arg,) if arg is str else get_args(arg))
            )

        if _is_json_inside_union(orig, args):
            return JSON

    raise TypeError(f"Cannot recognize type {typ}")


def _enum_to_sql(typ: Any) -> Any:
    """The column type an enum stores as.

    Read from the primitive it mixes in rather than from iterating its members,
    which a zero-only IntFlag has none of. A plain enum's member is not its value
    and nothing converts it through .value here, so it has no column type at all;
    raising is what lets an array fall back to JSON.
    """
    for primitive in (bool, int, float, str):
        if issubclass(typ, primitive):
            return PYTHON_TO_SQL[primitive]
    raise UnstorableTypeError(f"Cannot store enum {typ!r}: its members are not values")


def _values_to_sql(values) -> Any:
    """The column type for a set of literal or enum values.

    By the exact type of each value, so a bool is not taken for an int, and
    ignoring None -- whether the column is nullable is decided separately.
    Values of more than one storage category have no single column type between
    them and are carried as JSON.
    """
    kinds: set[type] = set()
    member_values: set = set()
    raw_values: set = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, Enum):
            if not isinstance(value, (bool, int, float, str)):
                # A plain enum member is not its value and nothing converts it
                # through .value on the way to the column.
                raise UnstorableTypeError(f"Cannot store enum member {value!r}")
            member_values.add(value.value)
            kinds.add(type(value.value))
        else:
            raw_values.add(value)
            kinds.add(type(value))

    collisions = sorted(map(repr, member_values & raw_values))

    if collisions:
        # Stored, IntKind.ONE and 1 are the same value; reading cannot tell which
        # was written, so the member would come back as the raw one. Members
        # whose values nothing else claims are safe.
        raise UnstorableTypeError(
            f"Cannot store enum members beside the same raw values: {collisions}"
        )
    if not kinds:
        # Nothing but None, so the column only ever holds NULL and any type
        # would do; String is what it mapped to before.
        return String
    if len(kinds) != 1:
        # No column type holds both faithfully, and JSON does not either: it
        # cannot tell a stored "1" from the number, so refuse rather than
        # corrupt one of the arms.
        raise UnstorableTypeError(
            f"Cannot resolve values {sorted(map(str, kinds))} to one type"
        )
    sql_type = PYTHON_TO_SQL.get(kinds.pop())
    if sql_type is None:
        raise UnstorableTypeError("Cannot resolve these values to a column type")
    return sql_type


def _list_to_array(typ, args):
    if args is None:
        raise TypeError(f"Cannot resolve type '{typ}' for flattening features")
    if get_origin(typ) is tuple and len(args) == 2 and args[1] is Ellipsis:
        # tuple[T, ...] is a homogeneous tuple: the Ellipsis marks variable
        # length rather than naming a second element type. Anywhere else it is
        # not valid, and stays unresolvable.
        args = args[:1]
    if not args:
        raise TypeError(f"Cannot resolve type '{typ}' for flattening features")
    args0 = args[0]
    if ModelStore.is_pydantic(args0):
        return Array(JSON())

    list_type = list_of_args_to_type(args)
    # Optional[scalar] elements map to a nullable Array element so None survives
    # (ClickHouse: Array(Nullable(T))).
    inner, is_optional = unwrap_optional(args0)
    if is_optional and inner in NULLABLE_SCALARS:
        list_type = SQLType.as_nullable(list_type)
    return Array(list_type)


def list_of_args_to_type(args) -> SQLType:
    first_type = python_to_sql(args[0])
    heterogeneous = False
    for next_arg in args[1:]:
        # Every slot is resolved even once the answer is known to be JSON: a
        # later one may be refused outright, and returning early would store it
        # lossily through the fallback instead.
        try:
            if python_to_sql(next_arg) != first_type:
                heterogeneous = True
        except UnstorableTypeError:
            raise
        except TypeError:
            heterogeneous = True
    return JSON() if heterogeneous else first_type


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
