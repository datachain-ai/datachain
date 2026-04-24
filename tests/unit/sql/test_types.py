from datetime import date, datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import CompileError

from datachain.sql.types import TypeReadConverter, validate_datetime_cast_input_type


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
        pytest.param(
            "2024-01-02 03:04:05.123456789+00:00",
            datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
            id="extra_fraction_with_offset",
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


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        pytest.param(123, "int", id="int"),
        pytest.param(1.5, "float", id="float"),
        pytest.param(True, "bool", id="bool"),
    ],
)
def test_validate_datetime_cast_input_type_rejects_unsupported_types(value, type_name):
    with pytest.raises(
        CompileError,
        match=(
            r"func\.cast\(\.\.\., datetime\) only supports string, bytes, "
            f"date, or datetime inputs; got {type_name}"
        ),
    ):
        validate_datetime_cast_input_type(sa.literal(value).type)


def test_validate_datetime_cast_input_type_allows_date():
    validate_datetime_cast_input_type(sa.literal(date(2024, 1, 2)).type)


def test_validate_datetime_cast_input_type_allows_unknown_types():
    validate_datetime_cast_input_type(sa.null().type)
