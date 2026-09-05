import enum
from collections.abc import Mapping
from typing import Dict, Literal, Union  # noqa: UP035

import pytest

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.sql.types import JSON, Array, Boolean, Float, Int64, String
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


def test_a_tuple_with_no_element_type_is_refused():
    # Ellipsis is stripped before the element type is read, so a tuple that
    # names no type at all has to be refused rather than indexed into.
    with pytest.raises(TypeError, match="Cannot resolve type"):
        python_to_sql(tuple[()])


class IntKind(enum.IntEnum):
    ONE = 1


class StrKind(str, enum.Enum):
    A = "a"


class PlainKind(enum.Enum):
    A = "a"


class ZeroFlag(enum.IntFlag):
    NONE = 0


@pytest.mark.parametrize(
    "annotation,expected",
    [
        pytest.param(IntKind, Int64, id="int-enum"),
        pytest.param(StrKind, String, id="str-mixin-enum"),
        pytest.param(Literal[1, 2], Int64, id="int-literal"),
        pytest.param(Literal["a", "b"], String, id="str-literal"),
        pytest.param(Literal[True, False], Boolean, id="bool-literal"),
        pytest.param(Literal[IntKind.ONE], Int64, id="literal-of-int-enum"),
        pytest.param(Literal[StrKind.A], String, id="literal-of-str-enum"),
        pytest.param(ZeroFlag, Int64, id="zero-only-int-flag"),
        pytest.param(Literal[IntKind.ONE], Int64, id="literal-of-mixin-member"),
        pytest.param(Literal[IntKind.ONE, 2], Int64, id="member-beside-another-value"),
        pytest.param(Literal[StrKind.A, "b"], String, id="str-member-beside-another"),
        pytest.param(
            Literal["a", None],  # noqa: PYI061
            String,
            id="literal-ignores-none",
        ),
    ],
)
def test_values_decide_the_column_type(annotation, expected):
    assert python_to_sql(annotation) is expected


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(PlainKind, id="plain-enum"),
        pytest.param(Literal[1, True], id="bool-is-not-int"),
        pytest.param(Literal[1, "a"], id="mixed-categories"),
        pytest.param(str | Literal[1], id="str-beside-an-int-literal"),
        pytest.param(Literal[IntKind.ONE, 1], id="member-collides-with-raw-value"),
        pytest.param(Literal[PlainKind.A], id="literal-of-plain-enum-member"),
    ],
)
def test_values_with_no_single_column_type_are_refused(annotation):
    # A plain enum's members are not their values, and values of more than one
    # storage category have no type that holds both faithfully -- JSON included,
    # since it cannot tell a stored "1" from the number.
    with pytest.raises(TypeError):
        python_to_sql(annotation)


def test_a_literal_holding_only_none_still_maps():
    # It has no value to categorize, but the column only ever holds NULL.
    assert python_to_sql(Literal[None]) is String  # noqa: PYI061


def test_a_union_of_literals_matches_the_same_values_written_as_one():
    # Built through a variable so no rewrite can fold the two spellings into
    # one. Naming the type matters too: comparing the two results to each other
    # would hold even if both were wrong.
    split = Union[Literal[1], Literal[2]]  # noqa: UP007, PYI030
    together = Literal[1, 2]

    assert python_to_sql(split) is Int64
    assert python_to_sql(together) is Int64


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(tuple[IntKind, ...], id="int-enum"),
        pytest.param(tuple[Literal[1, 2], ...], id="int-literal"),
    ],
)
def test_a_variadic_tuple_keeps_a_value_typed_element(annotation):
    assert python_to_sql(annotation).to_dict() == {
        "type": "Array",
        "item_type": {"type": "Int64"},
    }


def test_ellipsis_is_only_read_as_a_variadic_tuple():
    # list[int, ...] is not a valid annotation; stripping the Ellipsis anywhere
    # it appears would quietly accept it as list[int].
    assert python_to_sql(list[int, ...]).to_dict() == {
        "type": "Array",
        "item_type": {"type": "JSON"},
    }


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(tuple[Literal[1, "1"], int], id="refused-slot-first"),
        pytest.param(tuple[int, Literal[1, "1"]], id="refused-slot-second"),
    ],
)
def test_a_refused_slot_is_not_swallowed_by_the_json_fallback(annotation):
    # A heterogeneous tuple falls back to JSON; a slot refused because JSON
    # would store it lossily must not reach that fallback, wherever it sits.
    with pytest.raises(TypeError):
        python_to_sql(annotation)


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(tuple[int, float, Literal[1, "1"]], id="refused-last-slot"),
        pytest.param(tuple[PlainKind, int], id="plain-enum-first"),
        pytest.param(tuple[int, PlainKind], id="plain-enum-second"),
    ],
)
def test_a_later_refused_slot_is_still_reached(annotation):
    # The fallback used to answer at the first mismatch, so a slot refused
    # further along was never resolved and its values were stored lossily.
    with pytest.raises(TypeError):
        python_to_sql(annotation)
