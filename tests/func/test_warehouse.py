from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import GeneratorType
from unittest.mock import patch

import sqlalchemy as sa

import datachain as dc
from datachain import func
from tests.utils import skip_if_not_sqlite


def test_dataset_stats_no_table(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.warehouse.drop_dataset_rows_table(dogs_dataset, version="1.0.0")
    num_objects, size = catalog.warehouse.dataset_stats(dogs_dataset, version="1.0.0")
    assert num_objects is None
    assert size is None


def test_dataset_select_paginated_dataset_larger_than_batch_size(test_session):
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    chain = dc.read_values(value=list(range(10_000)), session=test_session).save(
        "large"
    )
    table = warehouse.get_table(
        warehouse.dataset_table_name(chain.dataset, chain.dataset.latest_version)
    )
    db_values = chain.to_values("value")

    rows = warehouse.dataset_select_paginated(
        sa.select(table.c.value).order_by(table.c.value), page_size=1000
    )
    assert isinstance(rows, GeneratorType)

    rows = list(rows)
    assert len(rows) == 10_000
    (values,) = zip(*rows, strict=False)
    assert set(values) == set(db_values)


def test_dataset_insert_batch_size(test_session, warehouse, ignore_checkpoints):
    def udf_map(value: int) -> int:
        return value + 100

    def udf_gen(value: int) -> Iterator[int]:
        yield value
        yield value + 100

    with patch.object(
        warehouse.db,
        attribute="executemany",
        wraps=warehouse.db.executemany,
    ) as mock_executemany:
        dc.read_values(value=list(range(100)), session=test_session).save("values")
        # 1 for read_values gen() output, 1 for save
        # Note: processed_table no longer exists (sys__input_id is in output table now)
        assert mock_executemany.call_count == 2
        mock_executemany.reset_mock()

        # Mapper

        dc.read_dataset("values", session=test_session).map(x2=udf_map).save("large")
        assert mock_executemany.call_count == 1
        mock_executemany.reset_mock()

        chain = (
            dc.read_dataset("values", session=test_session)
            .settings(batch_size=10)
            .map(x2=udf_map)
            .save("large")
        )
        assert mock_executemany.call_count == 10
        mock_executemany.reset_mock()
        assert set(chain.to_values("x2")) == set(range(100, 200))

        # Generator

        dc.read_dataset("values", session=test_session).gen(x2=udf_gen).save("large")
        # Only 1 call for gen() output (processed_table no longer exists)
        assert mock_executemany.call_count == 1
        mock_executemany.reset_mock()

        chain = (
            dc.read_dataset("values", session=test_session)
            .settings(batch_size=10)
            .gen(x2=udf_gen)
            .save("large")
        )
        # Only 20 for outputs (processed_table no longer exists)
        assert mock_executemany.call_count == 20
        mock_executemany.reset_mock()
        assert set(chain.to_values("x2")) == set(range(200))


def test_warehouse_keeps_utc_datetime_unchanged(test_session):
    timestamp = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)

    dc.read_values(ts=[timestamp], session=test_session).save(
        "warehouse_utc_datetime_round_trip"
    )

    rows = dc.read_dataset(
        "warehouse_utc_datetime_round_trip", session=test_session
    ).to_values("ts")

    assert rows == [timestamp]


def test_warehouse_round_trips_datetime_cast_from_string(test_session):
    dc.read_values(
        row_id=[1, 2],
        ts_str=["2024-01-01 00:00:00", "2024-01-02 03:04:05.123456"],
        session=test_session,
    ).mutate(ts=func.cast("ts_str", datetime)).save(
        "warehouse_cast_datetime_from_string"
    )

    rows = (
        dc.read_dataset("warehouse_cast_datetime_from_string", session=test_session)
        .order_by("row_id")
        .to_list("row_id", "ts")
    )

    assert rows == [
        (1, datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        (2, datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)),
    ]


def test_warehouse_round_trips_datetime_to_string_cast(test_session):
    dc.read_values(
        row_id=[1, 2, 3],
        ts=[
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 2, 3, 4, 5, 123000),
            datetime(2024, 1, 2, 3, 4, 5, 123456),
        ],
        session=test_session,
    ).mutate(ts_str=func.cast("ts", str)).save("warehouse_cast_datetime_to_string")

    rows = (
        dc.read_dataset("warehouse_cast_datetime_to_string", session=test_session)
        .order_by("row_id")
        .to_list("row_id", "ts_str")
    )

    assert rows == [
        (1, "2024-01-01 00:00:00"),
        (2, "2024-01-02 03:04:05.123000"),
        (3, "2024-01-02 03:04:05.123456"),
    ]


@skip_if_not_sqlite
def test_sqlite_warehouse_converts_other_timezones_to_utc(test_session):
    eastern = timezone(timedelta(hours=-5))
    timestamp = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=eastern)

    dc.read_values(ts=[timestamp], session=test_session).save(
        "sqlite_warehouse_timezone_to_utc"
    )

    rows = dc.read_dataset(
        "sqlite_warehouse_timezone_to_utc", session=test_session
    ).to_values("ts")

    assert rows == [timestamp.astimezone(timezone.utc)]
