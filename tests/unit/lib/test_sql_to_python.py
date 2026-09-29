import pytest
from sqlalchemy import ARRAY, JSON
from sqlalchemy.sql.sqltypes import NullType

from datachain import Column
from datachain.lib.convert.sql_to_python import sql_to_python
from datachain.sql.types import Array, Float, Int64, String


@pytest.mark.parametrize(
    "sql_column, expected",
    [
        (Column("name", String), str),
        (Column("age", Int64), int),
        (Column("score", Float), float),
        # SQL expression
        (Column("age", Int64) - 2, int),
        # Default type
        (Column("null", NullType), str),
        (Column("meta", JSON), dict),
        (Column("metas", ARRAY(JSON)), list[dict]),
        # List type
        (Column("tags", Array(Int64)), list[int]),
    ],
)
def test_sql_columns_to_python_types(sql_column, expected):
    assert sql_to_python(sql_column) == expected
