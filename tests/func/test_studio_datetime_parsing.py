from datetime import datetime

import pytest

from datachain.error import DataChainError
from datachain.studio import parse_start_time


def test_parse_start_time_none():
    """Test that None input returns None."""
    assert parse_start_time(None) is None


def test_parse_start_time_empty_string():
    """Test that empty string input returns None."""
    assert parse_start_time("") is None


def test_parse_start_time_iso_format():
    """Test parsing ISO format datetime strings."""
    # Test ISO format with timezone
    result = parse_start_time("2024-01-15T14:30:00Z")
    assert result == "2024-01-15T14:30:00+00:00"

    # Test ISO format without timezone
    result = parse_start_time("2024-01-15T14:30:00")
    assert result.startswith("2024-01-15T14:30:00")


def test_parse_start_time_standard_format():
    """Test parsing standard datetime format."""
    result = parse_start_time("2024-01-15 14:30:00")
    assert result.startswith("2024-01-15T14:30:00")


def test_parse_start_time_natural_language():
    """Test parsing natural language datetime strings."""
    # Test natural language formats that dateparser supports
    result = parse_start_time("tomorrow 3pm")
    assert result is not None
    assert isinstance(result, str)

    result = parse_start_time("monday 9am")
    assert result is not None
    assert isinstance(result, str)

    result = parse_start_time("in 2 hours")
    assert result is not None
    assert isinstance(result, str)

    result = parse_start_time("next week")
    assert result is not None
    assert isinstance(result, str)


def test_parse_start_time_various_formats():
    """Test parsing various datetime formats."""
    test_cases = [
        "2024-01-15 14:30:00",
        "2024-01-15T14:30:00Z",
        "2024-01-15T14:30:00+00:00",
        "Jan 15, 2024 2:30 PM",
        "15/01/2024 14:30",
        "2024-01-15",
        "tomorrow",
        "next week",
        "in 2 hours",
        "monday 9am",
        "tomorrow 3pm",
    ]

    for test_case in test_cases:
        result = parse_start_time(test_case)
        assert result is not None
        assert isinstance(result, str)
        # Verify it's a valid ISO format
        try:
            datetime.fromisoformat(result)
        except ValueError:
            pytest.fail(f"Failed to parse result '{result}' as ISO format")


def test_parse_start_time_invalid_format():
    """Test that invalid datetime formats raise DataChainError."""
    invalid_formats = [
        "not a date",
        "invalid datetime string",
        "2024-13-45 25:70:99",  # Invalid date/time values
    ]

    for invalid_format in invalid_formats:
        with pytest.raises(DataChainError) as exc_info:
            parse_start_time(invalid_format)

        assert "Could not parse datetime string" in str(exc_info.value)
        assert invalid_format in str(exc_info.value)


def test_parse_start_time_timezone_handling():
    """Test timezone handling in datetime parsing."""
    # Test with explicit timezone
    result = parse_start_time("2024-01-15 14:30:00 UTC")
    assert result is not None

    # Test with local timezone (should be preserved)
    result = parse_start_time("2024-01-15 14:30:00")
    assert result is not None
