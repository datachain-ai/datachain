from decimal import Decimal
from typing import Any

from sqlalchemy import JSON

from datachain.query.schema import ColumnExpr


def sql_to_python(sql_exp: ColumnExpr) -> Any:
    type_ = _python_type(sql_exp.type)
    if type_ is list and hasattr(sql_exp.type, "item_type"):
        return list[_python_type(sql_exp.type.item_type)]  # type: ignore[misc]
    return type_


def _python_type(sql_type: Any) -> Any:
    if isinstance(sql_type, JSON):
        return dict
    try:
        type_ = sql_type.python_type
    except NotImplementedError:
        return str
    if type_ == Decimal:
        return float
    if type_ is object:
        return str
    return type_
