# ruff: noqa: TC004

from typing import TYPE_CHECKING

from datachain.lazy import lazy_exports

if TYPE_CHECKING:
    from .dataset import DatasetQuery
    from .params import param
    from .schema import C
    from .session import Session

_LAZY = {
    "C": "datachain.query.schema:C",
    "DatasetQuery": "datachain.query.dataset:DatasetQuery",
    "Session": "datachain.query.session:Session",
    "param": "datachain.query.params:param",
}

__all__ = [
    "C",
    "DatasetQuery",
    "Session",
    "param",
]


lazy_exports(globals(), _LAZY)
