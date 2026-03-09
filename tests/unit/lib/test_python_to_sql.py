from collections.abc import Mapping
from typing import Literal

import pytest

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.sql.types import JSON, Array, Float, String
from tests.unit.lib.test_utils import MyModel


@pytest.mark.parametrize(
    "typ,expected",
    (
        (str, String),
        (String, String),
        (Literal["text"], String),
        (dict[str, int], JSON),
        (Mapping[str, int], JSON),
        (str | None, String),
        (dict | list[dict], JSON),
    ),
)
def test_convert_type_to_datachain(typ, expected):
    assert python_to_sql(typ) == expected


def test_list_of_tuples_matching_types():
    assert (
        python_to_sql(list[tuple[float, float]]).to_dict()
        == Array(Array(Float)).to_dict()
    )


def test_list_of_tuples_not_matching_types():
    assert (
        python_to_sql(list[tuple[float, String]]).to_dict()
        == Array(Array(JSON)).to_dict()
    )


def test_list_of_tuples_object():
    assert (
        python_to_sql(list[tuple[float, MyModel]]).to_dict()
        == Array(Array(JSON)).to_dict()
    )


def test_pep_604_union_syntax():
    from datachain.sql.types import Int64

    str_or_none = str | None
    int_or_none = int | None
    dict_or_list_dict = dict | list[dict]

    assert python_to_sql(str_or_none) == String
    assert python_to_sql(int_or_none) == Int64
    assert python_to_sql(dict_or_list_dict) == JSON

    str_literal_union = Literal["a", "b"]
    assert python_to_sql(str_literal_union) == String
