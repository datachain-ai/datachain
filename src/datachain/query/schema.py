from typing import TypeAlias

import sqlalchemy as sa
from sqlalchemy import Float, Integer, cast
from sqlalchemy.sql import func as sa_func

DEFAULT_DELIMITER = "__"


class ColumnMeta(type):
    @staticmethod
    def to_db_name(name: str) -> str:
        return name.replace(".", DEFAULT_DELIMITER)

    def __getattr__(cls, name: str):
        return cls(ColumnMeta.to_db_name(name))

    @staticmethod
    def is_nested(name: str) -> bool:
        return DEFAULT_DELIMITER in name


class Column(sa.ColumnClause, metaclass=ColumnMeta):
    inherit_cache: bool | None = True

    def __init__(self, text, type_=None, is_literal=False, _selectable=None):
        """Dataset column."""
        self.name = ColumnMeta.to_db_name(text)
        super().__init__(
            self.name, type_=type_, is_literal=is_literal, _selectable=_selectable
        )

    def __getattr__(self, name: str):
        return Column(self.name + DEFAULT_DELIMITER + name)

    def __truediv__(self, other):
        return sa_func.divide(self, other, type_=Float)

    def __rtruediv__(self, other):
        return sa_func.divide(other, self, type_=Float)

    def __floordiv__(self, other):
        return cast(sa_func.divide(self, other), Integer)

    def __rfloordiv__(self, other):
        return cast(sa_func.divide(other, self), Integer)

    def glob(self, glob_str):
        """Search for matches using glob pattern matching."""
        return self.op("GLOB", is_comparison=True)(glob_str)

    def regexp(self, regexp_str):
        """Search for matches using regexp pattern matching."""
        return self.op("REGEXP", is_comparison=True)(regexp_str)


C = Column

ColumnExpr: TypeAlias = sa.ColumnElement
