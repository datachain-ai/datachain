import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

from sqlalchemy import BindParameter, Case, ColumnElement, Integer, cast, desc
from sqlalchemy.sql import func as sa_func

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.convert.sql_to_python import sql_to_python
from datachain.lib.model_store import ModelStore
from datachain.lib.utils import DataChainColumnError, DataChainParamsError
from datachain.query.schema import Column, ColumnMeta
from datachain.sql.functions import numeric

from .base import Function

if TYPE_CHECKING:
    from sqlalchemy import TableClause

    from datachain import DataType
    from datachain.lib.signal_schema import SignalSchema

    from .window import Window


ColT = Union[str, tuple, Column, ColumnElement, "Func"]


class Func(Function):  # noqa: PLW1641
    """Represents a function to be applied to a column in a SQL query."""

    cols: Sequence[ColT]
    args: Sequence[Any]

    def __init__(
        self,
        name: str,
        inner: Callable,
        cols: Sequence[ColT] | None = None,
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        result_type: "DataType | None" = None,
        type_from_args: Callable[..., "DataType"] | None = None,
        is_array: bool = False,
        from_array: bool = False,
        is_window: bool = False,
        window: "Window | None" = None,
        label: str | None = None,
    ) -> None:
        self.name = name
        self.inner = inner
        self.cols = cols or []
        self.args = args or []
        self.kwargs = kwargs or {}
        self.result_type = result_type
        self.type_from_args = type_from_args
        self.is_array = is_array
        self.from_array = from_array
        self.is_window = is_window
        self.window = window
        self.col_label = label

    def __str__(self) -> str:
        return self.name + "()"

    def over(self, window: "Window") -> "Func":
        if not self.is_window:
            raise DataChainParamsError(f"{self} doesn't support window (over())")

        return Func(
            "over",
            self.inner,
            self.cols,
            self.args,
            self.kwargs,
            self.result_type,
            self.type_from_args,
            self.is_array,
            self.from_array,
            self.is_window,
            window,
            self.col_label,
        )

    @property
    def _db_cols(self) -> Sequence[ColT]:
        from sqlalchemy.ext.hybrid import Comparator

        return (
            [
                col
                if isinstance(col, (Func, BindParameter, Case, Comparator, tuple))
                else ColumnMeta.to_db_name(
                    col.name if isinstance(col, ColumnElement) else col
                )
                for col in self.cols
            ]
            if self.cols
            else []
        )

    def _db_col_type(self, signals_schema: "SignalSchema") -> "DataType | None":
        if not self._db_cols:
            return None

        col_type: type = get_db_col_type(signals_schema, self._db_cols[0])
        for col in self._db_cols[1:]:
            if get_db_col_type(signals_schema, col) != col_type:
                raise DataChainColumnError(
                    str(self),
                    "Columns must have the same type to infer result type",
                )

        if self.from_array:
            if get_origin(col_type) is not list:
                raise DataChainColumnError(
                    str(self),
                    "Array column must be of type list",
                )
            if self.is_array:
                return col_type
            col_args = get_args(col_type)
            if len(col_args) != 1:
                raise DataChainColumnError(
                    str(self),
                    "Array column must have a single type argument",
                )
            return col_args[0]

        return list[col_type] if self.is_array else col_type  # type: ignore[valid-type]

    def __add__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("add", lambda a: a + other, [self])
        return Func("add", lambda a1, a2: a1 + a2, [self, other])

    def __radd__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("add", lambda a: other + a, [self])
        return Func("add", lambda a1, a2: a1 + a2, [other, self])

    def __sub__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("sub", lambda a: a - other, [self])
        return Func("sub", lambda a1, a2: a1 - a2, [self, other])

    def __rsub__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("sub", lambda a: other - a, [self])
        return Func("sub", lambda a1, a2: a1 - a2, [other, self])

    def __mul__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("mul", lambda a: a * other, [self])
        return Func("mul", lambda a1, a2: a1 * a2, [self, other])

    def __rmul__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("mul", lambda a: other * a, [self])
        return Func("mul", lambda a1, a2: a1 * a2, [other, self])

    def __truediv__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("div", lambda a: _truediv(a, other), [self], result_type=float)
        return Func(
            "div", lambda a1, a2: _truediv(a1, a2), [self, other], result_type=float
        )

    def __rtruediv__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("div", lambda a: _truediv(other, a), [self], result_type=float)
        return Func(
            "div", lambda a1, a2: _truediv(a1, a2), [other, self], result_type=float
        )

    def __floordiv__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "floordiv", lambda a: _floordiv(a, other), [self], result_type=int
            )
        return Func(
            "floordiv", lambda a1, a2: _floordiv(a1, a2), [self, other], result_type=int
        )

    def __rfloordiv__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "floordiv", lambda a: _floordiv(other, a), [self], result_type=int
            )
        return Func(
            "floordiv", lambda a1, a2: _floordiv(a1, a2), [other, self], result_type=int
        )

    def __mod__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("mod", lambda a: a % other, [self], result_type=int)
        return Func("mod", lambda a1, a2: a1 % a2, [self, other], result_type=int)

    def __rmod__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("mod", lambda a: other % a, [self], result_type=int)
        return Func("mod", lambda a1, a2: a1 % a2, [other, self], result_type=int)

    def __and__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "and", lambda a: numeric.bit_and(a, other), [self], result_type=int
            )
        return Func(
            "and",
            lambda a1, a2: numeric.bit_and(a1, a2),
            [self, other],
            result_type=int,
        )

    def __rand__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "and", lambda a: numeric.bit_and(other, a), [self], result_type=int
            )
        return Func(
            "and",
            lambda a1, a2: numeric.bit_and(a1, a2),
            [other, self],
            result_type=int,
        )

    def __or__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "or", lambda a: numeric.bit_or(a, other), [self], result_type=int
            )
        return Func(
            "or", lambda a1, a2: numeric.bit_or(a1, a2), [self, other], result_type=int
        )

    def __ror__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "or", lambda a: numeric.bit_or(other, a), [self], result_type=int
            )
        return Func(
            "or", lambda a1, a2: numeric.bit_or(a1, a2), [other, self], result_type=int
        )

    def __xor__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "xor", lambda a: numeric.bit_xor(a, other), [self], result_type=int
            )
        return Func(
            "xor",
            lambda a1, a2: numeric.bit_xor(a1, a2),
            [self, other],
            result_type=int,
        )

    def __rxor__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "xor", lambda a: numeric.bit_xor(other, a), [self], result_type=int
            )
        return Func(
            "xor",
            lambda a1, a2: numeric.bit_xor(a1, a2),
            [other, self],
            result_type=int,
        )

    def __rshift__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "rshift",
                lambda a: numeric.bit_rshift(a, other),
                [self],
                result_type=int,
            )
        return Func(
            "rshift",
            lambda a1, a2: numeric.bit_rshift(a1, a2),
            [self, other],
            result_type=int,
        )

    def __rrshift__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "rshift",
                lambda a: numeric.bit_rshift(other, a),
                [self],
                result_type=int,
            )
        return Func(
            "rshift",
            lambda a1, a2: numeric.bit_rshift(a1, a2),
            [other, self],
            result_type=int,
        )

    def __lshift__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "lshift",
                lambda a: numeric.bit_lshift(a, other),
                [self],
                result_type=int,
            )
        return Func(
            "lshift",
            lambda a1, a2: numeric.bit_lshift(a1, a2),
            [self, other],
            result_type=int,
        )

    def __rlshift__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "lshift",
                lambda a: numeric.bit_lshift(other, a),
                [self],
                result_type=int,
            )
        return Func(
            "lshift",
            lambda a1, a2: numeric.bit_lshift(a1, a2),
            [other, self],
            result_type=int,
        )

    def __lt__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("lt", lambda a: a < other, [self], result_type=bool)
        return Func("lt", lambda a1, a2: a1 < a2, [self, other], result_type=bool)

    def __le__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("le", lambda a: a <= other, [self], result_type=bool)
        return Func("le", lambda a1, a2: a1 <= a2, [self, other], result_type=bool)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return Func("eq", lambda a: a == other, [self], result_type=bool)
        return Func("eq", lambda a1, a2: a1 == a2, [self, other], result_type=bool)

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            return Func("ne", lambda a: a != other, [self], result_type=bool)
        return Func("ne", lambda a1, a2: a1 != a2, [self, other], result_type=bool)

    def __gt__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("gt", lambda a: a > other, [self], result_type=bool)
        return Func("gt", lambda a1, a2: a1 > a2, [self, other], result_type=bool)

    def __ge__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("ge", lambda a: a >= other, [self], result_type=bool)
        return Func("ge", lambda a1, a2: a1 >= a2, [self, other], result_type=bool)

    def label(self, label: str) -> "Func":
        return Func(
            self.name,
            self.inner,
            self.cols,
            self.args,
            self.kwargs,
            self.result_type,
            self.type_from_args,
            self.is_array,
            self.from_array,
            self.is_window,
            self.window,
            label,
        )

    def get_col_name(self, label: str | None = None) -> str:
        if label:
            return label
        if self.col_label:
            return self.col_label
        if (db_cols := self._db_cols) and len(db_cols) == 1:
            if isinstance(db_cols[0], str):
                return db_cols[0]
            if isinstance(db_cols[0], Column):
                return db_cols[0].name
            if isinstance(db_cols[0], Func):
                return db_cols[0].get_col_name()
        return self.name

    def get_result_type(
        self, signals_schema: "SignalSchema | None" = None
    ) -> "DataType":
        if self.result_type:
            return self.result_type

        if signals_schema and (col_type := self._db_col_type(signals_schema)):
            return col_type

        if (
            self.type_from_args
            and (self.cols is None or self.cols == [])
            and self.args is not None
            and len(self.args) > 0
            and (result_type := self.type_from_args(*self.args)) is not None
        ):
            return result_type

        raise DataChainColumnError(
            str(self),
            "Column name is required to infer result type",
        )

    def get_column(
        self,
        signals_schema: "SignalSchema | None" = None,
        label: str | None = None,
        table: "TableClause | None" = None,
    ) -> Column:
        # Guard against using complex (pydantic) object columns in SQL funcs
        if signals_schema and self._db_cols:
            for arg in self._db_cols:
                # _db_cols normalizes known columns to strings; skip non-string args
                if not isinstance(arg, str):
                    continue
                t_with_sub = signals_schema.get_column_type(arg, with_subtree=True)
                if ModelStore.is_pydantic(t_with_sub):
                    raise DataChainParamsError(
                        f"Function {self.name} doesn't support complex object "
                        f"columns like '{arg}'. Use a leaf field (e.g., "
                        f"'{arg}.path') or use UDFs to operate on complex objects."
                    )

        col_type = self.get_result_type(signals_schema)
        sql_type = python_to_sql(col_type)

        def get_col(col: ColT, string_as_literal=False) -> ColT:
            # string_as_literal is used only for conditionals like `case()` where
            # literals are nested inside ColT as we have tuples of condition - values
            # and if user wants to set some case value as column, explicit `C("col")`
            # syntax must be used to distinguish from literals
            if isinstance(col, tuple):
                return tuple(get_col(x, string_as_literal=True) for x in col)
            if isinstance(col, Func):
                return col.get_column(signals_schema, table=table)
            if isinstance(col, str) and not string_as_literal:
                column = Column(col, sql_type)
                column.table = table
                return column
            return col

        cols = [get_col(col) for col in self._db_cols]

        kwargs = {k: get_col(v, string_as_literal=True) for k, v in self.kwargs.items()}
        func_col = self.inner(*cols, *self.args, **kwargs)

        if self.is_window:
            if not self.window:
                raise DataChainParamsError(
                    f"Window function {self} requires over() clause with a window spec",
                )
            func_col = func_col.over(
                partition_by=self.window.partition_by,
                order_by=(
                    desc(self.window.order_by)
                    if self.window.desc
                    else self.window.order_by
                ),
            )

        func_col.type = sql_type() if inspect.isclass(sql_type) else sql_type

        if col_name := self.get_col_name(label):
            func_col = func_col.label(col_name)

        return func_col


def get_db_col_type(signals_schema: "SignalSchema", col: ColT) -> "DataType":
    if isinstance(col, tuple):
        # we can only get tuple from case statement where the first tuple item
        # is condition, and second one is value which type is important
        col = col[1]
    if isinstance(col, Func):
        return col.get_result_type(signals_schema)

    if isinstance(col, ColumnElement) and not hasattr(col, "name"):
        return sql_to_python(col)

    name = col.name if isinstance(col, ColumnElement) else col  # type: ignore[assignment]
    return signals_schema.get_column_type(name)  # type: ignore[arg-type]


def _truediv(a, b):
    # Using sqlalchemy.sql.func.divide here instead of / operator
    # because of a bug in ClickHouse SQLAlchemy dialect
    # See https://github.com/xzkostyan/clickhouse-sqlalchemy/issues/335
    return sa_func.divide(a, b)


def _floordiv(a, b):
    return cast(_truediv(a, b), Integer)
