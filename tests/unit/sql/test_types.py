import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import types as sa_types

from datachain.sql.types import TypeConverter, TypeReadConverter


class EchoItemType:
    def on_read_convert(self, value, dialect):
        return (dialect, value)


@pytest.mark.parametrize(
    "method_name",
    ["string", "int", "int32", "uint32", "int64", "uint64"],
)
def test_type_read_converter_passthrough_scalar_methods(method_name):
    converter = TypeReadConverter()
    value = object()

    assert getattr(converter, method_name)(value) is value


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("TRUE", True),
        ("t", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("F", False),
        ("no", False),
        ("0", False),
    ],
)
def test_type_read_converter_boolean_normalizes_known_values(value, expected):
    converter = TypeReadConverter()
    assert converter.boolean(value) is expected


def test_type_read_converter_boolean_passthrough_for_unknown_strings():
    converter = TypeReadConverter()
    value = "maybe"
    assert converter.boolean(value) == value


def test_type_read_converter_boolean_passthrough_for_unhandled_values():
    converter = TypeReadConverter()
    value = 1.5

    assert converter.boolean(value) == value


@pytest.mark.parametrize("method_name", ["float", "float32", "float64"])
def test_type_read_converter_float_methods_return_nan_for_none(method_name):
    converter = TypeReadConverter()

    result = getattr(converter, method_name)(None)

    assert math.isnan(result)


@pytest.mark.parametrize("method_name", ["float", "float32", "float64"])
@pytest.mark.parametrize("value", ["NaN", "nan"])
def test_type_read_converter_float_methods_normalize_nan_strings(method_name, value):
    converter = TypeReadConverter()

    result = getattr(converter, method_name)(value)

    assert math.isnan(result)


@pytest.mark.parametrize("method_name", ["float", "float32", "float64"])
def test_type_read_converter_float_methods_passthrough_other_values(method_name):
    converter = TypeReadConverter()
    value = "3.14"

    assert getattr(converter, method_name)(value) == value


def test_type_read_converter_array_passthrough_for_none():
    converter = TypeReadConverter()

    assert converter.array(None, EchoItemType(), dialect="sqlite") is None


def test_type_read_converter_array_passthrough_for_missing_item_type():
    converter = TypeReadConverter()
    value = [1, 2, 3]

    assert converter.array(value, None, dialect="sqlite") is value


def test_type_read_converter_array_converts_each_item():
    converter = TypeReadConverter()

    assert converter.array([1, 2], EchoItemType(), dialect="sqlite") == [
        ("sqlite", 1),
        ("sqlite", 2),
    ]


def test_type_read_converter_json_loads_strings():
    converter = TypeReadConverter()

    assert converter.json('{"a": 1}') == {"a": 1}


def test_type_read_converter_json_returns_empty_dict_for_empty_string():
    converter = TypeReadConverter()

    assert converter.json("") == {}


def test_type_read_converter_json_passthrough_for_objects():
    converter = TypeReadConverter()
    value = {"a": 1}

    assert converter.json(value) is value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            "2024-01-02",
            datetime(2024, 1, 2, 0, 0, 0),
            id="date_only",
        ),
        pytest.param(
            "2024-01-02 03:04:05",
            datetime(2024, 1, 2, 3, 4, 5),
            id="naive_datetime",
        ),
        pytest.param(
            "2024-01-02T03:04:05Z",
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            id="z_suffix",
        ),
        pytest.param(
            "2024-01-02T03:04:05-05:00",
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=-5))),
            id="offset",
        ),
        pytest.param(
            "2024-01-02 03:04:05.123456789",
            datetime(2024, 1, 2, 3, 4, 5, 123456),
            id="extra_fraction",
        ),
    ],
)
def test_type_read_converter_datetime_parses_valid_strings(value, expected):
    converter = TypeReadConverter()

    assert converter.datetime(value) == expected


def test_type_read_converter_datetime_passthrough_for_datetime_values():
    converter = TypeReadConverter()
    value = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    assert converter.datetime(value) is value


def test_type_read_converter_datetime_passthrough_for_none():
    converter = TypeReadConverter()

    assert converter.datetime(None) is None


def test_type_read_converter_datetime_raises_for_invalid_strings():
    converter = TypeReadConverter()

    with pytest.raises(ValueError, match="Invalid isoformat string"):
        converter.datetime("not-a-datetime")


def test_type_read_converter_datetime_raises_for_unexpected_values():
    converter = TypeReadConverter()

    with pytest.raises(TypeError, match="expected str, datetime, or None"):
        converter.datetime(123)


def test_type_read_converter_binary_encodes_strings():
    converter = TypeReadConverter()

    assert converter.binary("abc") == b"abc"


def test_type_read_converter_binary_passthrough_for_bytes():
    converter = TypeReadConverter()
    value = b"abc"

    assert converter.binary(value) is value


@pytest.mark.parametrize(
    ("method_name", "expected_type"),
    [
        ("string", sa_types.String),
        ("boolean", sa_types.Boolean),
        ("int", sa_types.Integer),
        ("int32", sa_types.Integer),
        ("uint32", sa_types.Integer),
        ("int64", sa_types.Integer),
        ("uint64", sa_types.Integer),
        ("float", sa_types.Float),
        ("float32", sa_types.Float),
        ("float64", sa_types.Float),
        ("json", sa_types.JSON),
        ("datetime", sa_types.DATETIME),
        ("binary", sa_types.BINARY),
    ],
)
def test_type_converter_returns_expected_sqlalchemy_types(method_name, expected_type):
    converter = TypeConverter()

    assert isinstance(getattr(converter, method_name)(), expected_type)


def test_type_converter_array_returns_sqlalchemy_array():
    converter = TypeConverter()
    item_type = sa_types.Integer()

    result = converter.array(item_type)

    assert isinstance(result, sa_types.ARRAY)
    assert isinstance(result.item_type, sa_types.Integer)
