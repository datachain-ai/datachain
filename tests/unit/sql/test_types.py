import pytest

from datachain.sql.types import TypeReadConverter


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
