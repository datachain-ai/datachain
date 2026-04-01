from decimal import Decimal
from typing import Any

from datachain.query.schema import ColumnExpr


def sql_to_python(sql_exp: ColumnExpr) -> Any:
    try:
        type_ = sql_exp.type.python_type
        if type_ == Decimal:
            type_ = float
        elif type_ is list:
            if hasattr(sql_exp.type, "item_type") and hasattr(
                sql_exp.type.item_type, "python_type"
            ):
                item_type = getattr(sql_exp.type.item_type, "python_type", Any)
                type_ = list[item_type]  # type: ignore[valid-type]
            else:
                type_ = list
    except NotImplementedError:
        type_ = str
    return type_
