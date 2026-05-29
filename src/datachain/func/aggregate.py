from typing import TYPE_CHECKING

from sqlalchemy import Integer
from sqlalchemy import case as sa_case
from sqlalchemy import cast as sa_cast
from sqlalchemy import func as sa_func

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.data_model import unwrap_optional
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalResolvingError
from datachain.query.schema import Column, ColumnExpr
from datachain.sql.functions import aggregate

from .func import ColT, Func

if TYPE_CHECKING:
    from sqlalchemy import TableClause

    from datachain.lib.signal_schema import SignalSchema


AggColT = str | Column | ColumnExpr | Func


def _optional_parent_sentinel_db_path(
    db_col: str, signals_schema: "SignalSchema | None"
) -> str | None:
    """Walk ``db_col``'s ancestors and return the DB path of the closest
    ``Optional[DataModel]`` sentinel — or ``None`` if no such ancestor exists.

    Used so aggregates on a leaf below an Optional[DataModel] can mask absent
    rows: on ClickHouse leaves are coerced to type-defaults rather than NULL,
    and a naive SUM/AVG/MIN/MAX would otherwise count those defaults. Both
    ``db_col`` and the returned path use the DB separator (``__``).
    """
    from datachain.lib.signal_schema import DEFAULT_DELIMITER
    from datachain.lib.signal_schema import SignalSchema as _Schema

    if signals_schema is None:
        return None
    parts = db_col.split(DEFAULT_DELIMITER)
    for i in range(len(parts), 0, -1):
        prefix = DEFAULT_DELIMITER.join(parts[:i])
        try:
            anno = signals_schema.get_column_type(prefix, with_subtree=True)
        except SignalResolvingError:
            continue
        inner, is_optional = unwrap_optional(anno)
        if is_optional and ModelStore.is_pydantic(inner):
            return f"{prefix}{DEFAULT_DELIMITER}{_Schema._OPTIONAL_SENTINEL_FIELD}"
    return None


class _SentinelAwareAggFunc(Func):
    """Aggregate that skips rows under an absent ``Optional[DataModel]`` parent.

    On ClickHouse the leaf columns of an absent parent hold the type-default
    (``0``/``""``/``[]``) instead of SQL NULL, so a naive ``SUM(leaf)`` /
    ``AVG(leaf)`` returns different numbers than on SQLite. Wrapping the
    leaf in ``CASE WHEN {parent}__is_null = 0 THEN leaf END`` makes both
    backends agree (the aggregate then skips the absent rows on both).
    """

    def get_column(
        self,
        signals_schema: "SignalSchema | None" = None,
        label: str | None = None,
        table: "TableClause | None" = None,
    ) -> Column:
        col = self._db_cols[0] if self._db_cols else None
        if isinstance(col, str):
            sentinel_path = _optional_parent_sentinel_db_path(col, signals_schema)
            if sentinel_path is not None:
                return self._wrapped_column(
                    col, sentinel_path, signals_schema, label, table
                )
        return super().get_column(signals_schema, label, table)

    def _wrapped_column(
        self,
        col: str,
        sentinel_path: str,
        signals_schema: "SignalSchema | None",
        label: str | None,
        table: "TableClause | None",
    ) -> Column:
        sentinel = Column(sentinel_path)
        sentinel.table = table
        leaf = self._resolve_col(col, None, signals_schema, table)
        masked = sa_case((sentinel == 0, leaf), else_=None)
        sql_type = python_to_sql(self.get_result_type(signals_schema))
        func_col = self.inner(masked)
        return self._finalize_column(func_col, sql_type, label)


class _CountFunc(_SentinelAwareAggFunc):
    """``count(col)`` with two sentinel-aware paths:

    * ``count(<Optional[DataModel]>)`` → ``SUM(1 - {prefix}__is_null)``: the
      column doesn't exist on disk, but "rows where the parent is present"
      is what users mean.
    * ``count(<leaf-under-Optional[DataModel]>)`` → inherits the CASE-wrap
      from the base class so absent rows aren't counted as present.
    """

    def get_column(
        self,
        signals_schema: "SignalSchema | None" = None,
        label: str | None = None,
        table: "TableClause | None" = None,
    ) -> Column:
        # Case 1: column IS an Optional[DataModel]
        if (parent_sentinel := self._self_sentinel_path(signals_schema)) is not None:
            sentinel = Column(parent_sentinel)
            sentinel.table = table
            func_col = sa_func.sum(1 - sa_cast(sentinel, Integer))
            return self._finalize_column(func_col, Integer, label)
        # Case 2: column is a leaf under an Optional[DataModel]; let the base
        # class wrap it. Case 3: plain column or no col: delegate to Func.
        return super().get_column(signals_schema, label, table)

    def _self_sentinel_path(self, signals_schema: "SignalSchema | None") -> str | None:
        from datachain.lib.signal_schema import DEFAULT_DELIMITER
        from datachain.lib.signal_schema import SignalSchema as _Schema

        if signals_schema is None or not self._db_cols:
            return None
        col = self._db_cols[0]
        if not isinstance(col, str):
            return None
        try:
            anno = signals_schema.get_column_type(col, with_subtree=True)
        except SignalResolvingError:
            return None
        inner, is_optional = unwrap_optional(anno)
        if not (is_optional and ModelStore.is_pydantic(inner)):
            return None
        return f"{col}{DEFAULT_DELIMITER}{_Schema._OPTIONAL_SENTINEL_FIELD}"


def count(col: AggColT | None = None) -> Func:
    """
    Returns a COUNT aggregate SQL function for the specified column.

    The COUNT function returns the number of rows. If a column or expression is
    provided, it counts the rows where that input evaluates to a non-NULL value.

    For an ``Optional[DataModel]`` column, counts rows where the parent is
    present.

    Args:
        col (str | Column | ColumnExpr | Func, optional): The column,
            column expression, or DataChain expression to count.
            If omitted, counts all rows.

    Returns:
        Func: A `Func` object representing the COUNT aggregate function.

    Example:
        ```py
        dc.group_by(
            total_signals=func.count(),
            signals_with_id=func.count("signal.id"),
            rows_with_both_scores=func.count(
                dc.C("signal.left") + dc.C("signal.right")
            ),
            partition_by="signal.category",
        )
        ```

    Notes:
        - The result column will always have an integer type.
    """
    return _CountFunc(
        "count",
        inner=sa_func.count,
        cols=[col] if col is not None else None,
        result_type=int,
    )


def sum(col: AggColT) -> Func:
    """
    Returns the SUM aggregate SQL function for the specified column.

    The SUM function returns the total sum of a numeric column or expression in a
    table. It sums up all the values for the specified input.

    Args:
        col (str | Column | ColumnExpr | Func): The numeric column, column
            expression, or DataChain expression for which to calculate the sum.

    Returns:
        Func: A `Func` object that represents the SUM aggregate function.

    Example:
        ```py
        dc.group_by(
            files_size=func.sum("file.size"),
            total_size=func.sum(dc.C("size")),
            adjusted_size=func.sum(dc.C("size") + 1),
            partition_by="signal.category",
        )
        ```

    Notes:
        - The `sum` function should be used on numeric columns or expressions.
        - The result column type will be inferred from the input expression type.
        - Skips rows where the value's ``Optional[DataModel]`` parent is absent.
    """
    return _SentinelAwareAggFunc("sum", inner=sa_func.sum, cols=[col])


def avg(col: AggColT) -> Func:
    """
    Returns the AVG aggregate SQL function for the specified column.

    The AVG function returns the average of a numeric column or expression in a
    table. It calculates the mean of all values in the specified input.

    Args:
        col (str | Column | ColumnExpr | Func): The numeric column, column
            expression, or DataChain expression for which to calculate the average.

    Returns:
        Func: A Func object that represents the AVG aggregate function.

    Example:
        ```py
        dc.group_by(
            avg_file_size=func.avg("file.size"),
            avg_signal_value=func.avg(dc.C("signal.value")),
            avg_adjusted_value=func.avg(
                (dc.C("signal.left") + dc.C("signal.right")) / 2
            ),
            partition_by="signal.category",
        )
        ```

    Notes:
        - The `avg` function should be used on numeric columns or expressions.
        - The result column will always be of type float.
        - Skips rows where the value's ``Optional[DataModel]`` parent is absent.
    """
    return _SentinelAwareAggFunc(
        "avg", inner=aggregate.avg, cols=[col], result_type=float
    )


def min(col: AggColT) -> Func:
    """
    Returns the MIN aggregate SQL function for the specified column.

    The MIN function returns the smallest value in the specified column or
    expression. It can be used on both numeric and non-numeric inputs.

    Args:
        col (str | Column | ColumnExpr | Func): The column, column expression,
            or DataChain expression for which to find the minimum value.

    Returns:
        Func: A Func object that represents the MIN aggregate function.

    Example:
        ```py
        dc.group_by(
            smallest_file=func.min("file.size"),
            min_signal=func.min(dc.C("signal")),
            smallest_stem=func.min(func.path.file_stem("file.path")),
            partition_by="signal.category",
        )
        ```

    Notes:
        - The `min` function can be used with numeric, date, and string inputs.
        - The result column will have the same type as the input expression.
        - Skips rows where the value's ``Optional[DataModel]`` parent is absent.
    """
    return _SentinelAwareAggFunc("min", inner=sa_func.min, cols=[col])


def max(col: AggColT) -> Func:
    """
    Returns the MAX aggregate SQL function for the given column or expression.

    The MAX function returns the largest value in the specified column or
    expression. It can be used on both numeric and non-numeric inputs.

    Args:
        col (str | Column | ColumnExpr | Func): The column, column expression,
            or DataChain expression for which to find the maximum value.

    Returns:
        Func: A Func object that represents the MAX aggregate function.

    Example:
        ```py
        dc.group_by(
            largest_file=func.max("file.size"),
            max_signal=func.max(dc.C("signal")),
            max_total=func.max(dc.C("signal.left") + dc.C("signal.right")),
            partition_by="signal.category",
        )
        ```

    Notes:
        - The `max` function can be used with numeric, date, and string inputs.
        - The result column will have the same type as the input expression.
        - Skips rows where the value's ``Optional[DataModel]`` parent is absent.
    """
    return _SentinelAwareAggFunc("max", inner=sa_func.max, cols=[col])


def any_value(col: AggColT) -> Func:
    """
    Returns the ANY_VALUE aggregate SQL function for the given column or
    expression.

    The ANY_VALUE function returns an arbitrary value from the specified column
    or expression. It is useful when you do not care which particular value is
    returned, as long as it comes from one of the rows in the group.

    Args:
        col (str | Column | ColumnExpr | Func): The column, column expression,
            or DataChain expression from which to return an arbitrary value.

    Returns:
        Func: A Func object that represents the ANY_VALUE aggregate function.

    Example:
        ```py
        dc.group_by(
            file_example=func.any_value("file.path"),
            signal_example=func.any_value(dc.C("signal.value")),
            stem_example=func.any_value(func.path.file_stem("file.path")),
            partition_by="signal.category",
        )
        ```

    Notes:
        - The `any_value` function can be used with any type of input.
        - The result column will have the same type as the input expression.
        - The result of `any_value` is non-deterministic,
          meaning it may return different values for different executions.
        - Skips rows where the value's ``Optional[DataModel]`` parent is absent.
    """
    return _SentinelAwareAggFunc("any_value", inner=aggregate.any_value, cols=[col])


def collect(col: AggColT) -> Func:
    """
    Returns the COLLECT aggregate SQL function for the given column or expression.

    The COLLECT function gathers all values from the specified column or
    expression into an array or similar structure. It is useful for combining
    values into a collection, often for further processing or aggregation.

    Args:
        col (str | Column | ColumnExpr | Func): The column, column expression,
            or DataChain expression from which to collect values.

    Returns:
        Func: A Func object that represents the COLLECT aggregate function.

    Example:
        ```py
        dc.group_by(
            signals=func.collect("signal"),
            file_paths=func.collect(dc.C("file.path")),
            stems=func.collect(func.path.file_stem("file.path")),
            partition_by="signal.category",
        )
        ```

    Notes:
        - The `collect` function can be used with any input whose values can be
          collected.
        - The result column will have an array type derived from the input
          expression type.
    """
    return Func("collect", inner=aggregate.collect, cols=[col], is_array=True)


def concat(col: AggColT, separator="") -> Func:
    """
    Returns the CONCAT aggregate SQL function for the given column or expression.

    The CONCAT function concatenates values from the specified column or
    expression into a single string. It is useful for merging text values from
    multiple rows into a single combined value.

    Args:
        col (str | Column | ColumnExpr | Func): The string-valued column,
            column expression, or DataChain expression from which to concatenate
            values.
        separator (str, optional): The separator to use between concatenated values.
            Defaults to an empty string.

    Returns:
        Func: A Func object that represents the CONCAT aggregate function.

    Example:
        ```py
        dc.group_by(
            files=func.concat("file.path", separator=", "),
            For example, `func.count(dc.C("signal.left") + dc.C("signal.right"))`
            counts rows where both values are present.
            partition_by="signal.category",
        )
        ```

    Notes:
        - The `concat` function should be used with string-valued inputs.
        - The result column will have a string type.
        - Skips rows where the value's ``Optional[DataModel]`` parent is absent.
    """

    def inner(arg):
        return aggregate.group_concat(arg, separator)

    return _SentinelAwareAggFunc("concat", inner=inner, cols=[col], result_type=str)


def xor_agg(col: str | Column | ColT) -> Func:
    """
    Returns the XOR aggregate SQL function for the specified column.

    Computes the bitwise XOR of all values. Order-independent, so the
    result is the same regardless of row ordering. Useful for computing
    content fingerprints when combined with ``func.string.string_hash``.

    Args:
        col: The column (typically an integer hash) to XOR-aggregate.

    Returns:
        Func: A ``Func`` object that represents the XOR aggregate function.

    Example:
        ```py
        dc.group_by(
            fingerprint=func.xor_agg(func.string.string_hash("file.path", "file.etag")),
        )
        ```

    Notes:
        - Skips rows where the value's ``Optional[DataModel]`` parent is absent.
    """
    return _SentinelAwareAggFunc(
        "xor_agg", inner=aggregate.xor_agg, cols=[col], result_type=int
    )


def row_number() -> Func:
    """
    Returns the ROW_NUMBER window function for SQL queries.

    The ROW_NUMBER function assigns a unique sequential integer to rows
    within a partition of a result set, starting from 1 for the first row
    in each partition. It is commonly used to generate row numbers within
    partitions or ordered results.

    Returns:
        Func: A Func object that represents the ROW_NUMBER window function.

    Example:
        ```py
        window = func.window(partition_by="signal.category", order_by="created_at")
        dc.mutate(
            row_number=func.row_number().over(window),
        )
        ```

    Note:
        - The result column will always be of type int.
    """
    return Func("row_number", inner=sa_func.row_number, result_type=int, is_window=True)


def rank() -> Func:
    """
    Returns the RANK window function for SQL queries.

    The RANK function assigns a rank to each row within a partition of a result set,
    with gaps in the ranking for ties. Rows with equal values receive the same rank,
    and the next rank is skipped (i.e., if two rows are ranked 1,
    the next row is ranked 3).

    Returns:
        Func: A Func object that represents the RANK window function.

    Example:
        ```py
        window = func.window(partition_by="signal.category", order_by="created_at")
        dc.mutate(
            rank=func.rank().over(window),
        )
        ```

    Notes:
        - The result column will always be of type int.
        - The RANK function differs from ROW_NUMBER in that rows with the same value
          in the ordering column(s) receive the same rank.
    """
    return Func("rank", inner=sa_func.rank, result_type=int, is_window=True)


def dense_rank() -> Func:
    """
    Returns the DENSE_RANK window function for SQL queries.

    The DENSE_RANK function assigns a rank to each row within a partition
    of a result set, without gaps in the ranking for ties. Rows with equal values
    receive the same rank, but the next rank is assigned consecutively
    (i.e., if two rows are ranked 1, the next row will be ranked 2).

    Returns:
        Func: A Func object that represents the DENSE_RANK window function.

    Example:
        ```py
        window = func.window(partition_by="signal.category", order_by="created_at")
        dc.mutate(
            dense_rank=func.dense_rank().over(window),
        )
        ```

    Notes:
        - The result column will always be of type int.
        - The DENSE_RANK function differs from RANK in that it does not leave gaps
          in the ranking for tied values.
    """
    return Func("dense_rank", inner=sa_func.dense_rank, result_type=int, is_window=True)


def first(col: AggColT) -> Func:
    """
    Returns the FIRST_VALUE window function for SQL queries.

    The FIRST_VALUE function returns the first value in an ordered set of values
    within a partition. The first value is determined by the specified order and
    can be useful for retrieving the leading value of a column or expression in a
    group of rows.

    Args:
        col (str | Column | ColumnExpr | Func): The column, column expression,
            or DataChain expression from which to retrieve the first value.

    Returns:
        Func: A Func object that represents the FIRST_VALUE window function.

    Example:
        ```py
        window = func.window(partition_by="signal.category", order_by="created_at")
        dc.mutate(
            first_file=func.first("file.path").over(window),
            first_signal=func.first(dc.C("signal.value")).over(window),
            first_stem=func.first(func.path.file_stem("file.path")).over(window),
        )
        ```

    Note:
        - The result of `first_value` will always reflect the value of the first row
          in the specified order.
        - The result column will have the same type as the input expression.
    """
    return Func("first", inner=sa_func.first_value, cols=[col], is_window=True)
