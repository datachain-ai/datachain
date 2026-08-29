from collections.abc import Mapping
from typing import Dict, Literal  # noqa: UP035

import pytest

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.sql.types import JSON, Array, Float, Int64, String
from tests.unit.lib.test_utils import MyModel


@pytest.mark.parametrize(
    "typ,expected",
    (
        (str, String),
        (String, String),
        (Literal["text"], String),
        (dict[str, int], JSON),
        (Mapping[str, int], JSON),
        # a zero-argument mapping alias still needs a column
        (Dict, JSON),  # noqa: UP006
        (str | None, String),
        (dict | list[dict], JSON),
    ),
    ids=[
        "str",
        "String",
        "Literal",
        "dict",
        "Mapping",
        "bare-typing-Dict",
        "optional-str",
        "dict-or-list-of-dict",
    ],
)
def test_python_to_sql_conversions(typ, expected):
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


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (tuple[int, ...], Array(Int64)),
        (tuple[str, ...], Array(String)),
        (tuple[int, int], Array(Int64)),
        (tuple[int, str], Array(JSON)),
        (tuple[tuple[int, ...], ...], Array(Array(Int64))),
        (list[int], Array(Int64)),
    ],
    ids=[
        "variadic-int",
        "variadic-str",
        "fixed-same",
        "fixed-mixed",
        "nested-variadic",
        "list-unchanged",
    ],
)
def test_variadic_tuple_keeps_its_element_type(annotation, expected):
    assert python_to_sql(annotation).to_dict() == expected.to_dict()
