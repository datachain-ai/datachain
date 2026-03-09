from sqlalchemy import ColumnElement
from sqlalchemy import and_ as sql_and
from sqlalchemy import case as sql_case
from sqlalchemy import not_ as sql_not
from sqlalchemy import or_ as sql_or

from datachain.lib.utils import DataChainParamsError
from datachain.query.schema import Column
from datachain.sql.functions import conditional

from .func import Func

CaseT = int | float | complex | bool | str | Func | ColumnElement


def greatest(*args: str | Column | Func | float) -> Func:
    """
    Returns the greatest (largest) value from the given input values.

    Args:
        args (str | Column | Func | int | float): The values to compare.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a value.
            If an int or float is provided, it is assumed to be a literal.

    Returns:
        Func: A `Func` object that represents the greatest function.

    Example:
        ```py
        dc.mutate(
            greatest=func.greatest(dc.C("signal.value"), "signal.value2", 0.5, 1.0),
        )
        ```

    Notes:
        - The result column will always be of the same type as the input columns.
    """
    cols, func_args = [], []

    for arg in args:
        if isinstance(arg, (str, Column, Func)):
            cols.append(arg)
        else:
            func_args.append(arg)

    return Func(
        "greatest",
        inner=conditional.greatest,
        cols=cols,
        args=func_args,
        result_type=int,
    )


def least(*args: str | Column | Func | float) -> Func:
    """
    Returns the least (smallest) value from the given input values.

    Args:
        args (str | Column | Func | int | float): The values to compare.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a value.
            If an int or float is provided, it is assumed to be a literal.

    Returns:
        Func: A `Func` object that represents the least function.

    Example:
        ```py
        dc.mutate(
            least=func.least(dc.C("signal.value"), "signal.value2", -1.0, 0),
        )
        ```

    Notes:
        - The result column will always be of the same type as the input columns.
    """
    cols, func_args = [], []

    for arg in args:
        if isinstance(arg, (str, Column, Func)):
            cols.append(arg)
        else:
            func_args.append(arg)

    return Func(
        "least", inner=conditional.least, cols=cols, args=func_args, result_type=int
    )


def case(
    *args: tuple[ColumnElement | Func | bool, CaseT], else_: CaseT | None = None
) -> Func:
    """
    Returns a case expression that evaluates a list of conditions and returns
    corresponding results. Results can be Python primitives (string, numbers, booleans),
    nested functions (including case function), or columns.

    Args:
        args (tuple[ColumnElement | Func | bool, CaseT]): Tuples of (condition, value)
            pairs. Each condition is evaluated in order, and the corresponding value
            is returned for the first condition that evaluates to True.
        else_ (CaseT, optional): Value to return if no conditions are satisfied.
            If omitted and no conditions are satisfied, the result will be None
            (NULL in DB).

    Returns:
        Func: A `Func` object that represents the case function.

    Example:
        ```py
        dc.mutate(
            res=func.case((dc.C("num") > 0, "P"), (dc.C("num") < 0, "N"), else_="Z"),
        )
        ```

    Notes:
        - The result type is inferred from the values provided in the case statements.
    """
    supported_types = [int, float, complex, str, bool]

    def _get_type(val):
        from enum import Enum

        if isinstance(val, Func):
            # nested functions
            return val.result_type
        if isinstance(val, Column):
            # at this point we cannot know what is the type of a column
            return None
        if isinstance(val, Enum):
            return type(val.value)
        return type(val)

    if not args:
        raise DataChainParamsError("Missing statements")

    type_ = _get_type(else_) if else_ is not None else None

    for arg in args:
        arg_type = _get_type(arg[1])
        if arg_type is None:
            # we couldn't figure out the type of case value
            continue
        if type_ and arg_type != type_:
            raise DataChainParamsError(
                f"Statement values must be of the same type, got {type_} and {arg_type}"
            )
        type_ = arg_type

    if type_ is not None and type_ not in supported_types:
        raise DataChainParamsError(
            f"Only python literals ({supported_types}) are supported for values"
        )

    kwargs = {"else_": else_}

    return Func("case", inner=sql_case, cols=args, kwargs=kwargs, result_type=type_)


def ifelse(condition: ColumnElement | Func, if_val: CaseT, else_val: CaseT) -> Func:
    """
    Returns an if-else expression that evaluates a condition and returns one
    of two values based on the result. Values can be Python primitives
    (string, numbers, booleans), nested functions, or columns.

    Args:
        condition (ColumnElement | Func): Condition to evaluate.
        if_val (ColumnElement | Func | literal): Value to return if condition is True.
        else_val (ColumnElement | Func | literal): Value to return if condition
            is False.

    Returns:
        Func: A `Func` object that represents the ifelse function.

    Example:
        ```py
        dc.mutate(
            res=func.ifelse(isnone("col"), "EMPTY", "NOT_EMPTY")
        )
        ```

    Notes:
        - The result type is inferred from the values provided in the ifelse statement.
    """
    return case((condition, if_val), else_=else_val)


def isnone(col: str | ColumnElement) -> Func:
    """
    Returns a function that checks if the column value is `None` (NULL in DB).

    Args:
        col (str | Column): Column to check if it's None or not.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.

    Returns:
        Func: A `Func` object that represents the isnone function.
            Returns True if column value is None, otherwise False.

    Example:
        ```py
        dc.mutate(test=ifelse(isnone("col"), "EMPTY", "NOT_EMPTY"))
        ```

    Notes:
        - The result column will always be of type bool.
    """
    if isinstance(col, str):
        # if string is provided, it is assumed to be the name of the column
        col = Column(col)

    return case((col.is_(None) if col is not None else True, True), else_=False)


def or_(*args: ColumnElement | Func) -> Func:
    """
    Returns the function that produces conjunction of expressions joined by OR
    logical operator.

    Args:
        args (ColumnElement | Func): The expressions for OR statement.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a value.

    Returns:
        Func: A `Func` object that represents the OR function.

    Example:
        ```py
        dc.mutate(
            test=ifelse(or_(isnone("name"), dc.C("name") == ''), "Empty", "Not Empty")
        )
        ```

    Notes:
        - The result column will always be of type bool.
    """
    cols, func_args = [], []

    for arg in args:
        if isinstance(arg, (str, Column, Func)):
            cols.append(arg)
        else:
            func_args.append(arg)

    return Func("or", inner=sql_or, cols=cols, args=func_args, result_type=bool)


def and_(*args: ColumnElement | Func) -> Func:
    """
    Returns the function that produces conjunction of expressions joined by AND
    logical operator.

    Args:
        args (ColumnElement | Func): The expressions for AND statement.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a value.

    Returns:
        Func: A `Func` object that represents the AND function.

    Example:
        ```py
        dc.mutate(
            test=ifelse(and_(isnone("name"), isnone("surname")), "Empty", "Not Empty")
        )
        ```

    Notes:
        - The result column will always be of type bool.
    """
    cols, func_args = [], []

    for arg in args:
        if isinstance(arg, (str, Func)):
            cols.append(arg)
        else:
            func_args.append(arg)

    return Func("and", inner=sql_and, cols=cols, args=func_args, result_type=bool)


def not_(arg: ColumnElement | Func) -> Func:
    """
    Returns the function that produces NOT of the given expressions.

    Args:
        arg (ColumnElement | Func): The expression for NOT statement.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a value.

    Returns:
        Func: A `Func` object that represents the NOT function.

    Example:
        ```py
        dc.mutate(
            test=not_(C("value") == 5)
        )
        ```

    Notes:
        - The result column will always be of type bool.
    """
    cols, func_args = [], []

    if isinstance(arg, (str, Func)):
        cols.append(arg)
    else:
        func_args.append(arg)

    return Func("not", inner=sql_not, cols=cols, args=func_args, result_type=bool)
