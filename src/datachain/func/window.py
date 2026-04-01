from dataclasses import dataclass

from datachain.query.schema import ColumnExpr, ColumnMeta


@dataclass
class Window:
    """Represents a window specification for SQL window functions."""

    partition_by: str | ColumnExpr
    order_by: str | ColumnExpr
    desc: bool = False


def window(
    partition_by: str | ColumnExpr,
    order_by: str | ColumnExpr,
    desc: bool = False,
) -> Window:
    """
    Defines a window specification for SQL window functions.

    The `window` function specifies how to partition and order the result set
    for the associated window function. It is used to define the scope of the rows
    that the window function will operate on.

    Args:
        partition_by (str | ColumnExpr): The column name or expression by which
            to partition the result set. Rows with the same value in the partition
            column will be grouped together for the window function.
        order_by (str | ColumnExpr): The column name or expression by which to
            order the rows within each partition. This determines the sequence in
            which the window function is applied.
        desc (bool, optional): If True, the rows will be ordered in descending order.
            Defaults to False, which orders the rows in ascending order.

    Returns:
        Window: A `Window` object representing the window specification.

    Example:
        ```py
        window = func.window(partition_by="signal.category", order_by="created_at")
        dc.mutate(
            row_number=func.row_number().over(window),
        )
        ```
    """
    return Window(
        partition_by
        if isinstance(partition_by, ColumnExpr)
        else ColumnMeta.to_db_name(partition_by),
        order_by
        if isinstance(order_by, ColumnExpr)
        else ColumnMeta.to_db_name(order_by),
        desc,
    )
