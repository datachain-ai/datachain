from datetime import MAXYEAR, MINYEAR, datetime, timedelta, timezone, tzinfo

import numpy as np
import pytest
import ujson as json

from datachain.sql.sqlite import base as sqlite_base
from datachain.sql.sqlite.base import (
    adapt_datetime,
    convert_datetime,
    sqlite_datetime_cast,
)
from datachain.sql.sqlite.types import (
    SQLiteTypeReadConverter,
    adapt_np_array,
)


class ValueErrorTZ(tzinfo):
    def utcoffset(self, dt):
        raise ValueError("boom")

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "BAD"


@pytest.mark.parametrize(
    "dtype,arr,expected",
    (
        (float, [], "[]"),
        (float, [0.5, 0.6], "[0.5,0.6]"),
        (float, [[0.5, 0.6], [0.7, 0.8]], "[[0.5,0.6],[0.7,0.8]]"),
        (np.dtypes.ObjectDType, [], "[]"),
        (np.dtypes.ObjectDType, [0.5, 0.6], "[0.5,0.6]"),
        (np.dtypes.ObjectDType, [[0.5, 0.6], [0.7, 0.8]], "[[0.5,0.6],[0.7,0.8]]"),
    ),
)
def test_adapt_np_array(dtype, arr, expected):
    assert adapt_np_array(np.array(arr, dtype=dtype)) == expected


def test_adapt_np_array_nan_inf():
    arr_with_nan = np.array([1.0, np.nan, 3.0])
    result = adapt_np_array(arr_with_nan)
    assert result == "[1.0,NaN,3.0]"

    arr_with_inf = np.array([1.0, np.inf, -np.inf])
    result = adapt_np_array(arr_with_inf)
    assert result == "[1.0,Infinity,-Infinity]"

    arr_2d = np.array([[np.nan, 1.0], [2.0, np.inf]])
    result = adapt_np_array(arr_2d)
    assert result == "[[NaN,1.0],[2.0,Infinity]]"

    parsed = json.loads(result)
    assert np.isnan(parsed[0][0])
    assert parsed[0][1] == 1.0
    assert parsed[1][0] == 2.0
    assert np.isinf(parsed[1][1])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param(
            datetime(2024, 1, 2, 3, 4, 5),
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            id="naive_datetime",
        ),
        pytest.param(
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=-5))),
            datetime(2024, 1, 2, 8, 4, 5, tzinfo=timezone.utc),
            id="aware_datetime",
        ),
        pytest.param(
            "2024-01-02",
            datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            id="date_only_string",
        ),
        pytest.param(
            "2024-01-02 03:04:05",
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            id="naive_string",
        ),
        pytest.param(
            "2024-01-02T03:04:05Z",
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            id="z_suffix_string",
        ),
        pytest.param(
            "2024-01-02T03:04:05-05:00",
            datetime(2024, 1, 2, 8, 4, 5, tzinfo=timezone.utc),
            id="offset_string",
        ),
        pytest.param(
            "2024-01-02 03:04:05.123456789",
            datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
            id="extra_fraction_string",
        ),
    ],
)
def test_sqlite_type_read_converter_datetime_normalizes_values(value, expected):
    converter = SQLiteTypeReadConverter()

    assert converter.datetime(value) == expected


def test_sqlite_type_read_converter_datetime_raises_for_invalid_strings():
    converter = SQLiteTypeReadConverter()

    with pytest.raises(ValueError, match="Invalid isoformat string"):
        converter.datetime("not-a-datetime")


def test_sqlite_type_read_converter_datetime_raises_for_unexpected_values():
    converter = SQLiteTypeReadConverter()

    with pytest.raises(TypeError, match="expected str, datetime, or None"):
        converter.datetime(123)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            datetime(2024, 1, 2, 3, 4, 5, 123456),
            "2024-01-02 03:04:05.123456",
            id="naive_datetime",
        ),
        pytest.param(
            datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone(timedelta(hours=-5))),
            "2024-01-02 08:04:05.123456",
            id="aware_datetime",
        ),
    ],
)
def test_adapt_datetime_serializes_values(value, expected):
    assert adapt_datetime(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            datetime(
                MAXYEAR,
                12,
                31,
                23,
                59,
                59,
                999999,
                tzinfo=timezone(-timedelta(hours=1)),
            ),
            datetime.max.isoformat(" "),
            id="max_year",
        ),
        pytest.param(
            datetime(
                MINYEAR,
                1,
                1,
                0,
                0,
                0,
                0,
                tzinfo=timezone(timedelta(hours=1)),
            ),
            datetime.min.isoformat(" "),
            id="min_year",
        ),
    ],
)
def test_adapt_datetime_falls_back_for_boundary_years(value, expected):
    assert adapt_datetime(value) == expected


def test_adapt_datetime_reraises_other_errors():
    value = datetime(2024, 1, 2, 3, 4, 5, tzinfo=ValueErrorTZ())

    with pytest.raises(ValueError, match="boom"):
        adapt_datetime(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            b"2024-01-02",
            datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            id="date_only",
        ),
        pytest.param(
            b"2024-01-02 03:04:05.123456",
            datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
            id="naive_datetime",
        ),
        pytest.param(
            b"2024-01-02T03:04:05Z",
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            id="z_suffix",
        ),
        pytest.param(
            b"2024-01-02 03:04:05.123456+00:00",
            datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
            id="utc_offset",
        ),
        pytest.param(
            b"2024-01-02 03:04:05.123456789",
            datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
            id="extra_fraction",
        ),
    ],
)
def test_convert_datetime_normalizes_values(value, expected):
    assert convert_datetime(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            b"9999-12-31 23:59:59.999999-01:00",
            datetime.max.replace(tzinfo=timezone.utc),
            id="max_year",
        ),
        pytest.param(
            b"0001-01-01 00:00:00+01:00",
            datetime.min.replace(tzinfo=timezone.utc),
            id="min_year",
        ),
    ],
)
def test_convert_datetime_falls_back_for_boundary_years(value, expected):
    assert convert_datetime(value) == expected


def test_convert_datetime_reraises_other_errors(monkeypatch):
    monkeypatch.setattr(
        sqlite_base,
        "parse_datetime_text",
        lambda _value: datetime(2024, 1, 2, 3, 4, 5, tzinfo=ValueErrorTZ()),
    )

    with pytest.raises(ValueError, match="boom"):
        convert_datetime(b"ignored")


def test_sqlite_datetime_cast_returns_none_for_none():
    assert sqlite_datetime_cast(None) is None


def test_sqlite_datetime_cast_decodes_bytes():
    assert sqlite_datetime_cast(b"2024-01-02T03:04:05-05:00") == "2024-01-02 08:04:05"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            datetime(2024, 1, 2, 3, 4, 5, 123456),
            "2024-01-02 03:04:05.123456",
            id="naive_datetime",
        ),
        pytest.param(
            datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone(timedelta(hours=-5))),
            "2024-01-02 08:04:05.123456",
            id="aware_datetime",
        ),
    ],
)
def test_sqlite_datetime_cast_serializes_datetime_inputs(value, expected):
    assert sqlite_datetime_cast(value) == expected


def test_sqlite_datetime_cast_passthrough_non_string_values():
    value = 123

    assert sqlite_datetime_cast(value) == value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("2024-01-02", "2024-01-02 00:00:00", id="date_only"),
        pytest.param(
            "2024-01-02 03:04:05.123456",
            "2024-01-02 03:04:05.123456",
            id="naive_datetime",
        ),
        pytest.param(
            "2024-01-02T03:04:05Z",
            "2024-01-02 03:04:05",
            id="z_suffix",
        ),
        pytest.param(
            "2024-01-02T03:04:05+00:00",
            "2024-01-02 03:04:05",
            id="utc_offset",
        ),
        pytest.param(
            "2024-01-02T03:04:05-05:00",
            "2024-01-02 08:04:05",
            id="negative_offset",
        ),
        pytest.param(
            "2024-01-02 03:04:05.123456789",
            "2024-01-02 03:04:05.123456",
            id="extra_fraction",
        ),
    ],
)
def test_sqlite_datetime_cast_parses_strings(value, expected):
    assert sqlite_datetime_cast(value) == expected


def test_sqlite_datetime_cast_rejects_invalid_strings():
    with pytest.raises(ValueError, match="Invalid isoformat string"):
        sqlite_datetime_cast("not-a-datetime")
