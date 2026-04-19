import inspect
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

from sqlalchemy import Integer, desc
from sqlalchemy import cast as sa_cast
from sqlalchemy.sql import func as sa_func
from sqlalchemy.sql.elements import ColumnElement

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.convert.sql_to_python import sql_to_python
from datachain.lib.model_store import ModelStore
from datachain.lib.utils import DataChainColumnError, DataChainParamsError
from datachain.query.schema import Column, ColumnExpr, ColumnMeta
from datachain.sql.functions import numeric
from datachain.sql.functions.conversion import datetime_to_string

from .base import Function

if TYPE_CHECKING:
    from sqlalchemy import TableClause

    from datachain import DataType
    from datachain.lib.signal_schema import SignalSchema

    from .window import Window


ColT = Union[str, tuple, Column, ColumnExpr, "Func"]


class Func(Function):  # noqa: PLW1641
    """A built-in function applied to dataset columns, created by calling functions
    from the `func` module.

    There are three kinds of functions:

    - **Row-level** — transform each row independently, used in
      [`mutate`][datachain.lib.dc.DataChain.mutate],
      [`filter`][datachain.lib.dc.DataChain.filter], and
      [`merge`][datachain.lib.dc.DataChain.merge]:
      `func.path.file_stem(C("file.path"))`, `func.string.length(C("name"))`
    - **Aggregate** — collapse rows into a single value, used in
      [`group_by`][datachain.lib.dc.DataChain.group_by]:
      `func.count()`, `func.sum("file.size")`, `func.avg("score")`
    - **Window** — compute over a partition of rows, require `.over()`:
      `func.row_number().over(window)`, `func.rank().over(window)`
    """

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
        db_cols: list[ColT] = []
        for col in self.cols:
            if isinstance(col, Column):
                db_cols.append(ColumnMeta.to_db_name(col.name))
            elif isinstance(col, str):
                db_cols.append(ColumnMeta.to_db_name(col))
            else:
                db_cols.append(col)
        return db_cols

    def _result_type_from_db_cols(
        self, signals_schema: "SignalSchema"
    ) -> "DataType | None":
        if not self._db_cols:
            return None

        col_type = infer_col_type(signals_schema, self._db_cols[0])
        for col in self._db_cols[1:]:
            if infer_col_type(signals_schema, col) != col_type:
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
        return Func("div", _truediv, [self, other], result_type=float)

    def __rtruediv__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func("div", lambda a: _truediv(other, a), [self], result_type=float)
        return Func("div", _truediv, [other, self], result_type=float)

    def __floordiv__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "floordiv", lambda a: _floordiv(a, other), [self], result_type=int
            )
        return Func("floordiv", _floordiv, [self, other], result_type=int)

    def __rfloordiv__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "floordiv", lambda a: _floordiv(other, a), [self], result_type=int
            )
        return Func("floordiv", _floordiv, [other, self], result_type=int)

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
            numeric.bit_and,
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
            numeric.bit_and,
            [other, self],
            result_type=int,
        )

    def __or__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "or", lambda a: numeric.bit_or(a, other), [self], result_type=int
            )
        return Func("or", numeric.bit_or, [self, other], result_type=int)

    def __ror__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "or", lambda a: numeric.bit_or(other, a), [self], result_type=int
            )
        return Func("or", numeric.bit_or, [other, self], result_type=int)

    def __xor__(self, other: ColT | float) -> "Func":
        if isinstance(other, (int, float)):
            return Func(
                "xor", lambda a: numeric.bit_xor(a, other), [self], result_type=int
            )
        return Func(
            "xor",
            numeric.bit_xor,
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
            numeric.bit_xor,
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
            numeric.bit_rshift,
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
            numeric.bit_rshift,
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
            numeric.bit_lshift,
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
            numeric.bit_lshift,
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
            first_col = db_cols[0]
            if isinstance(first_col, str):
                return first_col
            if isinstance(first_col, Func):
                return first_col.get_col_name()
        return self.name

    def get_result_type(
        self, signals_schema: "SignalSchema | None" = None
    ) -> "DataType":
        if self.result_type:
            return self.result_type

        if (
            signals_schema is not None
            and (result_type := self._result_type_from_db_cols(signals_schema))
            is not None
        ):
            return result_type

        if self._db_cols:
            raise DataChainColumnError(
                str(self),
                "A dataset context is required to infer result type",
            )

        if self.type_from_args and not self.cols and self.args:
            inferred_result_type = self.type_from_args(*self.args)
            if inferred_result_type is not None:
                return inferred_result_type

        raise DataChainColumnError(
            str(self),
            "Column name is required to infer result type",
        )

    def _validate_sql_func_columns(
        self,
        signals_schema: "SignalSchema | None",
    ) -> None:
        if not signals_schema:
            return

        for arg in self._db_cols:
            if not isinstance(arg, str):
                continue
            t_with_sub = signals_schema.get_column_type(arg, with_subtree=True)
            if ModelStore.is_pydantic(t_with_sub):
                raise DataChainParamsError(
                    f"Function {self.name} doesn't support complex object "
                    f"columns like '{arg}'. Use a leaf field (e.g., "
                    f"'{arg}.path') or use UDFs to operate on complex objects."
                )

    def _resolve_col(
        self,
        col: ColT,
        sql_type: Any,
        signals_schema: "SignalSchema | None",
        table: "TableClause | None",
        *,
        string_as_literal: bool = False,
    ) -> ColT:
        # string_as_literal is used only for conditionals like `case()` where
        # literals are nested inside ColT as tuples of condition/value pairs.
        # If a user wants to reference a column in that position, they must use
        # explicit `C("col")` syntax so it isn't treated as a literal.
        if isinstance(col, tuple):
            return tuple(
                self._resolve_col(
                    x,
                    sql_type,
                    signals_schema,
                    table,
                    string_as_literal=True,
                )
                for x in col
            )

        if isinstance(col, Func):
            return col.get_column(signals_schema, table=table)

        if isinstance(col, str) and not string_as_literal:
            column_sql_type = (
                python_to_sql(signals_schema.get_column_type(col))
                if signals_schema
                else sql_type
            )
            if inspect.isclass(column_sql_type):
                column_sql_type = column_sql_type()
            column = Column(col, column_sql_type)
            column.table = table
            return column

        return col

    def _finalize_column(
        self,
        func_col: Any,
        sql_type: Any,
        label: str | None,
    ) -> Any:
        result: Any = func_col

        if self.is_window:
            if not self.window:
                raise DataChainParamsError(
                    f"Window function {self} requires over() clause with a window spec",
                )
            result = result.over(
                partition_by=self.window.partition_by,
                order_by=(
                    desc(self.window.order_by)
                    if self.window.desc
                    else self.window.order_by
                ),
            )

        result.type = sql_type() if inspect.isclass(sql_type) else sql_type

        if col_name := self.get_col_name(label):
            result = result.label(col_name)

        return result

    def get_column(
        self,
        signals_schema: "SignalSchema | None" = None,
        label: str | None = None,
        table: "TableClause | None" = None,
    ) -> Column:
        self._validate_sql_func_columns(signals_schema)

        col_type = self.get_result_type(signals_schema)
        sql_type = python_to_sql(col_type)

        cols = [
            self._resolve_col(col, sql_type, signals_schema, table)
            for col in self._db_cols
        ]

        kwargs = {
            k: self._resolve_col(
                v,
                sql_type,
                signals_schema,
                table,
                string_as_literal=True,
            )
            for k, v in self.kwargs.items()
        }
        func_col = self.inner(*cols, *self.args, **kwargs)
        return self._finalize_column(func_col, sql_type, label)


class CastFunc(Func):
    def __init__(
        self,
        col: str | ColumnExpr | Func,
        result_type: "DataType",
        sql_type: Any,
    ) -> None:
        self.cast_col: str | ColumnExpr | Func = col
        super().__init__(
            "cast",
            inner=sa_cast,
            cols=[col],
            args=[sql_type],
            result_type=result_type,
        )

    def get_column(
        self,
        signals_schema: "SignalSchema | None" = None,
        label: str | None = None,
        table: "TableClause | None" = None,
    ) -> Column:
        self._validate_sql_func_columns(signals_schema)

        sql_type = self.args[0]
        value = self._resolve_col(
            self._db_cols[0],
            sql_type,
            signals_schema,
            table,
        )
        source_type = infer_col_type(signals_schema, self.cast_col)
        func_col: ColumnElement[Any]

        if self.result_type is str and source_type is datetime:
            func_col = datetime_to_string(value)
        else:
            func_col = sa_cast(value, sql_type())

        return self._finalize_column(func_col, sql_type, label)


def infer_col_type(
    signals_schema: "SignalSchema | None",
    col: ColT | ColumnElement,
) -> "DataType":
    if isinstance(col, tuple):
        # we can only get tuple from case statement where the first tuple item
        # is condition, and second one is value which type is important
        col = col[1]

    if isinstance(col, Func):
        result_type = col.get_result_type(signals_schema)
    elif signals_schema is None:
        raise DataChainColumnError(
            str(col),
            "A dataset context is required to infer column type",
        )
    elif isinstance(col, str):
        result_type = signals_schema.get_column_type(col)
    elif isinstance(col, Column):
        result_type = signals_schema.get_column_type(col.name)
    elif isinstance(col, ColumnElement):
        result_type = sql_to_python(signals_schema.enrich_expr_types(col))
    else:
        raise DataChainColumnError(
            str(col),
            "Unsupported value type to infer column type",
        )

    return result_type


def _truediv(a, b):
    # Using sqlalchemy.sql.func.divide here instead of / operator
    # because of a bug in ClickHouse SQLAlchemy dialect
    # See https://github.com/xzkostyan/clickhouse-sqlalchemy/issues/335
    return sa_func.divide(a, b)


def _floordiv(a, b):
    return sa_cast(_truediv(a, b), Integer)


def cast(col: str | ColumnExpr | Func, type_: Any) -> Func:
    """
    Cast a column or expression to a target DataChain type.

    Args:
        col (str | ColumnExpr | Func): Column, column expression, or DataChain
            expression to cast.
            If a string is provided, it is treated as a column name.
        type_ (type): Supported target types are int, float, str, bool, bytes,
            and datetime.

    Returns:
        Func: A `Func` object representing the cast expression.

    Example:
        ```py
        from datachain import func

        chain = dc.read_values(id_str=["1", "2", "3"])
        chain = chain.mutate(id_int=func.cast("id_str", int))
        ```

    Notes:
        - This is a regular DataChain expression and can be used anywhere other
          functions are accepted, including `mutate()`, `filter()`, `order_by()`,
          and `merge()`.
        - Strings are interpreted as column names. To cast a string literal,
          wrap it with `func.literal(...)` first.
        - Casting datetimes to strings materializes to a canonical DataChain
          datetime string in Python: `YYYY-MM-DD HH:MM:SS[.ffffff]`.
    """
    if not isinstance(col, (str, ColumnElement, Func)):
        raise DataChainParamsError(
            "func.cast() expects its first argument to be a dataset column "
            "or expression, for example 'file.path', C('file.path'), or "
            "another DataChain expression. To cast a literal value, wrap "
            "it with func.literal(...)."
        )

    supported_scalar_types = {int, float, str, bool, bytes, datetime}
    if type_ not in supported_scalar_types:
        raise TypeError(
            "func.cast() supports target types int, float, str, bool, bytes, "
            "and datetime."
        )

    return CastFunc(col, type_, python_to_sql(type_))
