import operator
from collections.abc import Iterable, Iterator
from operator import and_
from typing import Any

from sqlalchemy.sql.elements import (
    BindParameter,
    BooleanClauseList,
    False_,
    True_,
    _anonymous_label,
)

COMPARISONS = (
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
)


def literal(node: Any) -> Any:
    """The value of a static literal, else None."""
    if isinstance(node, BindParameter):
        # a callable's or a named parameter's value is known only at execution
        named = not isinstance(node.key, _anonymous_label)
        return None if node.callable or named else node.value
    if isinstance(node, (True_, False_)):
        return isinstance(node, True_)
    return None


def conjuncts(exprs: Iterable[Any]) -> Iterator[Any]:
    for expr in exprs:
        if isinstance(expr, BooleanClauseList) and expr.operator is and_:
            yield from conjuncts(expr.clauses)
        else:
            yield expr
