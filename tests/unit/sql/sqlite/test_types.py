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
from datachain.sql.sqlite.types import adapt_np_array


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


def test_adapt_datetime_serializes_naive_as_utc():
    value = datetime(2024, 1, 2, 3, 4, 5, 123456)

    assert adapt_datetime(value) == "2024-01-02 03:04:05.123456"


def test_adapt_datetime_normalizes_aware_to_utc():
    eastern = timezone(timedelta(hours=-5))
    value = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=eastern)

    assert adapt_datetime(value) == "2024-01-02 08:04:05.123456"


def test_convert_datetime_returns_utc_aware_for_naive_rows():
    value = convert_datetime(b"2024-01-02 03:04:05.123456")

    assert value == datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)


def test_convert_datetime_normalizes_offset_rows_to_utc():
    value = convert_datetime(b"2024-01-02 03:04:05.123456+00:00")

    assert value == datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)


def test_sqlite_datetime_cast_returns_none_for_none():
    assert sqlite_datetime_cast(None) is None


def test_sqlite_datetime_cast_decodes_bytes():
    assert sqlite_datetime_cast(b"2024-01-02T03:04:05-05:00") == ("2024-01-02 08:04:05")


def test_sqlite_datetime_cast_serializes_datetime_inputs():
    eastern = timezone(timedelta(hours=-5))
    value = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=eastern)

    assert sqlite_datetime_cast(value) == "2024-01-02 08:04:05.123456"


def test_sqlite_datetime_cast_passthrough_non_string_values():
    value = 123

    assert sqlite_datetime_cast(value) == value


def test_sqlite_datetime_cast_serializes_naive_strings_as_utc():
    assert sqlite_datetime_cast("2024-01-02 03:04:05.123456") == (
        "2024-01-02 03:04:05.123456"
    )


def test_sqlite_datetime_cast_normalizes_aware_strings_to_utc():
    assert sqlite_datetime_cast("2024-01-02T03:04:05-05:00") == "2024-01-02 08:04:05"


def test_sqlite_datetime_cast_truncates_extra_fractional_precision():
    assert sqlite_datetime_cast("2024-01-02 03:04:05.123456789") == (
        "2024-01-02 03:04:05.123456"
    )


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
