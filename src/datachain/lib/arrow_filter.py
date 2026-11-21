import operator
from collections.abc import Iterable, Iterator, Sequence
from functools import reduce
from operator import and_, or_
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from sqlalchemy.sql import operators as sa_op
from sqlalchemy.sql.elements import (
    BinaryExpression,
    BindParameter,
    BooleanClauseList,
    ColumnClause,
    False_,
    Grouping,
    Null,
    True_,
    UnaryExpression,
)
from sqlalchemy.sql.visitors import iterate

from datachain.hash_utils import hash_column_elements
from datachain.lib.data_model import unwrap_optional
from datachain.lib.utils import normalize_col_names

_NOT = (sa_op.inv, sa_op.is_false, sa_op.is_true)
_COMPARISONS = (
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
)


def _filter_fields(
    schema: pa.Schema, output: dict[str, Any]
) -> dict[str, tuple[str, type]]:
    """Return the read columns of a basic type, as (raw_name, python_type)."""
    fields: dict[str, tuple[str, type]] = {}
    for name, raw in normalize_col_names(schema.names).items():
        if name not in output:
            continue
        field = schema.field(raw)
        col_type = field.type
        anno, optional = unwrap_optional(output[name])
        if (
            not isinstance(col_type, pa.ExtensionType)
            and anno in (int, str, bool, bytes)
            and (optional or not field.nullable)
            and not pa.types.is_null(col_type)
            and not pa.types.is_uint64(col_type)
        ):
            fields[name] = (raw, anno)
    return fields


def _literal(node: Any) -> Any:
    if isinstance(node, BindParameter):
        return node.value
    if isinstance(node, (True_, False_)):
        return isinstance(node, True_)
    return None


def _filter_field(
    expr: Any, fields: dict[str, tuple[str, type]]
) -> tuple[str, type] | None:
    if isinstance(expr, ColumnClause):
        return fields.get(expr.name)
    return None


def _compare(
    col: pc.Expression, op: Any, right: Any, kind: type
) -> pc.Expression | None:
    value = _literal(right)
    if op in (sa_op.is_, sa_op.is_not) and isinstance(right, Null):
        return col.is_null() if op is sa_op.is_ else col.is_valid()
    if op in (sa_op.is_, sa_op.is_not) and type(value) is kind is bool:
        return (
            (col.is_valid() & (col == value))
            if op is sa_op.is_
            else (col.is_null() | (col != value))
        )
    try:
        if op in _COMPARISONS and type(value) is kind:
            return op(col, value)  # type: ignore[arg-type,return-value]
        if (
            op in (sa_op.in_op, sa_op.not_in_op)
            and value
            and all(type(v) is kind for v in value)
        ):
            return col.isin(value) if op is sa_op.in_op else ~col.isin(value)
        if op in (sa_op.between_op, sa_op.not_between_op):
            lo, hi = (_literal(b) for b in right.clauses)
            if type(lo) is kind and type(hi) is kind:
                between = (col >= lo) & (col <= hi)
                return between if op is sa_op.between_op else ~between
    except OverflowError:
        pass
    return None


def _arrow_predicate(
    expr: Any, fields: dict[str, tuple[str, type]]
) -> pc.Expression | None:
    """Translate a SQL predicate over `fields` to Arrow, or None if it can't be."""
    while isinstance(expr, Grouping):
        expr = expr.element
    if isinstance(expr, BooleanClauseList) and expr.operator in (and_, or_):
        parts = [_arrow_predicate(c, fields) for c in expr.clauses]
        return (
            None
            if not parts or any(p is None for p in parts)
            else reduce(expr.operator, parts)  # type: ignore[arg-type,return-value]
        )
    if isinstance(expr, UnaryExpression) and expr.operator in _NOT:
        inner = _arrow_predicate(expr.element, fields)
        return (
            None
            if inner is None
            else (inner if expr.operator is sa_op.is_true else ~inner)
        )
    if (field := _filter_field(expr, fields)) is not None and field[1] is bool:
        return pc.field(field[0])
    if (
        isinstance(expr, BinaryExpression)
        and (field := _filter_field(expr.left, fields)) is not None
    ):
        return _compare(pc.field(field[0]), expr.operator, expr.right, field[1])
    return None


def _conjuncts(exprs: Iterable[Any]) -> Iterator[Any]:
    for expr in exprs:
        if isinstance(expr, BooleanClauseList) and expr.operator is and_:
            yield from _conjuncts(expr.clauses)
        else:
            yield expr


def to_arrow_filter(
    filters: Sequence[Any], schema: pa.Schema, output: dict[str, Any]
) -> tuple[pc.Expression, tuple[str, ...], str] | None:
    """Translate the pushable conjuncts of `filters` to an Arrow predicate, the
    input column names it reads, and a hash of the SQL clauses translated."""
    fields = _filter_fields(schema, output)
    pushed = [
        (clause, expr)
        for clause in _conjuncts(filters)
        if (expr := _arrow_predicate(clause, fields)) is not None
    ]
    if not pushed:
        return None
    clauses, exprs = zip(*pushed, strict=True)
    used = {c.name for e in clauses for c in iterate(e) if isinstance(c, ColumnClause)}
    columns = tuple(raw for n, (raw, _) in fields.items() if n in used)
    return reduce(and_, exprs), columns, hash_column_elements(list(clauses))
