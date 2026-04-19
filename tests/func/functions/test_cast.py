from datetime import datetime, timezone

import pytest

import datachain as dc
from datachain import C, func


def test_cast_in_mutate_filter_and_order_by(test_session):
    rows = (
        dc.read_values(
            id_str=["10", "2", "1"],
            session=test_session,
        )
        .mutate(id_int=func.cast("id_str", int))
        .filter(func.cast("id_str", int) >= 2)
        .order_by(func.cast("id_str", int))
        .to_list("id_str", "id_int")
    )

    assert rows == [("2", 2), ("10", 10)]


def test_cast_column_expression_in_mutate(test_session):
    rows = (
        dc.read_values(
            num=[1, 2, 3],
            session=test_session,
        )
        .mutate(num_str=func.cast(C("num") + 1, str))
        .order_by("num")
        .to_values("num_str")
    )

    assert rows == ["2", "3", "4"]


def test_cast_in_merge(test_session):
    left = dc.read_values(
        id_str=["1", "2", "4"],
        left_value=["left-1", "left-2", "left-4"],
        session=test_session,
    )
    right = dc.read_values(
        id_int=[1, 2, 3],
        right_value=["right-1", "right-2", "right-3"],
        session=test_session,
    )

    rows = (
        left.merge(
            right,
            on=func.cast(left.c("id_str"), int),
            right_on=right.c("id_int"),
            inner=True,
        )
        .order_by("id_int")
        .to_list("id_str", "left_value", "id_int", "right_value")
    )

    assert rows == [
        ("1", "left-1", 1, "right-1"),
        ("2", "left-2", 2, "right-2"),
    ]


def test_cast_multiple_scalar_types_in_mutate(test_session):
    rows = (
        dc.read_values(
            int_str=["10", "2", "1"],
            float_str=["1.5", "2.25", "3.0"],
            raw_int=[2, 10, 1],
            flag_int=[1, 0, 2],
            session=test_session,
        )
        .mutate(
            as_int=func.cast("int_str", int),
            as_float=func.cast("float_str", float),
            as_str=func.cast("raw_int", str),
            as_bool=func.cast("flag_int", bool),
        )
        .order_by("as_int")
        .to_list("as_int", "as_float", "as_str", "as_bool")
    )

    assert rows == [
        (1, 3.0, "1", True),
        (2, 2.25, "10", False),
        (10, 1.5, "2", True),
    ]


def test_cast_datetime_in_mutate_and_order_by(test_session):
    rows = (
        dc.read_values(
            ts_str=["2024-01-02 03:04:05", "2024-01-01 00:00:00"],
            session=test_session,
        )
        .mutate(ts=func.cast("ts_str", datetime))
        .order_by("ts")
        .to_list("ts_str", "ts")
    )

    assert rows == [
        (
            "2024-01-01 00:00:00",
            datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ),
        (
            "2024-01-02 03:04:05",
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        ),
    ]


def test_cast_datetime_to_string_and_round_trip(test_session):
    rows = (
        dc.read_values(
            row_id=[1, 2, 3],
            ts=[
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 2, 3, 4, 5, 123000),
                datetime(2024, 1, 2, 3, 4, 5, 123456),
            ],
            session=test_session,
        )
        .mutate(
            ts_str=func.cast("ts", str),
            ts_round_trip=func.cast(func.cast("ts", str), datetime),
        )
        .order_by("row_id")
        .to_list("ts", "ts_str", "ts_round_trip")
    )

    assert rows == [
        (
            datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "2024-01-01 00:00:00",
            datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ),
        (
            datetime(2024, 1, 2, 3, 4, 5, 123000, tzinfo=timezone.utc),
            "2024-01-02 03:04:05.123000",
            datetime(2024, 1, 2, 3, 4, 5, 123000, tzinfo=timezone.utc),
        ),
        (
            datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
            "2024-01-02 03:04:05.123456",
            datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
        ),
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "2024-01-02 03:04:05.123000",
            datetime(2024, 1, 2, 3, 4, 5, 123000),
        ),
        (
            "2024-01-02 03:04:05.123456",
            datetime(2024, 1, 2, 3, 4, 5, 123456),
        ),
        (
            "2024-01-02 03:04:05.123456789",
            datetime(2024, 1, 2, 3, 4, 5, 123456),
        ),
    ],
)
def test_cast_datetime_from_fractional_string(test_session, source, expected):
    rows = (
        dc.read_values(row_id=[1], ts_str=[source], session=test_session)
        .mutate(ts=func.cast("ts_str", datetime))
        .to_values("ts")
    )

    assert rows == [expected.replace(tzinfo=timezone.utc)]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "2024-01-02T03:04:05",
            datetime(2024, 1, 2, 3, 4, 5),
        ),
        ("2024-01-02", datetime(2024, 1, 2, 0, 0, 0)),
    ],
)
def test_cast_datetime_from_simple_iso_string(test_session, source, expected):
    rows = (
        dc.read_values(row_id=[1], ts_str=[source], session=test_session)
        .mutate(ts=func.cast("ts_str", datetime))
        .to_values("ts")
    )

    assert rows == [expected.replace(tzinfo=timezone.utc)]


def test_filter_cast_datetime_against_python_datetime_literal(test_session):
    predicate = func.cast("ts_str", datetime) >= datetime(2024, 1, 2, 0, 0, 0)  # type: ignore[operator]

    rows = (
        dc.read_values(
            row_id=[1, 2, 3],
            ts_str=[
                "2024-01-01 00:00:00",
                "2024-01-02 03:04:05.123456",
                "2024-01-03 00:00:00",
            ],
            session=test_session,
        )
        .filter(predicate)
        .order_by("row_id")
        .to_values("row_id")
    )

    assert rows == [2, 3]


def test_filter_regular_datetime_equals_casted_datetime(test_session):
    rows = (
        dc.read_values(
            row_id=[1, 2, 3],
            ts=[
                datetime(2024, 1, 2, 3, 4, 5, 123456),
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 5, 0, 0, 0),
            ],
            ts_str=[
                "2024-01-02 03:04:05.123456",
                "2024-01-03 00:00:00",
                "2024-01-05 00:00:00",
            ],
            session=test_session,
        )
        .filter(C("ts") == func.cast("ts_str", datetime))
        .order_by("row_id")
        .to_values("row_id")
    )

    assert rows == [1, 3]


def test_merge_regular_datetime_with_casted_datetime(test_session):
    left = dc.read_values(
        left_id=[1, 2, 3],
        ts=[
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 2, 3, 4, 5, 123456),
            datetime(2024, 1, 4, 0, 0, 0),
        ],
        session=test_session,
    )
    right = dc.read_values(
        rid=[10, 20, 30],
        ts_str=[
            "2024-01-02 03:04:05.123456",
            "2024-01-01 00:00:00",
            "2024-01-03 00:00:00",
        ],
        session=test_session,
    )

    rows = (
        left.merge(
            right,
            on=left.c("ts"),
            right_on=func.cast(right.c("ts_str"), datetime),
            inner=True,
        )
        .order_by("left_id")
        .to_list("left_id", "rid")
    )

    assert rows == [(1, 20), (2, 10)]


@pytest.mark.parametrize(
    ("source_values", "target_type", "expected_row_ids"),
    [
        (["10", "2", "1"], int, [3, 2, 1]),
        (["1.5", "10.25", "2.0"], float, [1, 3, 2]),
        ([2, 10, 1], str, [3, 2, 1]),
    ],
)
def test_cast_scalar_types_in_order_by(
    test_session, source_values, target_type, expected_row_ids
):
    rows = (
        dc.read_values(row_id=[1, 2, 3], value=source_values, session=test_session)
        .order_by(func.cast("value", target_type))
        .to_values("row_id")
    )

    assert rows == expected_row_ids
