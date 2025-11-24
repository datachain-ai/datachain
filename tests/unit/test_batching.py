import math
from types import GeneratorType

import pytest
import sqlalchemy as sa

from datachain.data_storage.schema import PARTITION_COLUMN_ID, partition_columns
from datachain.query.batch import Batch, NoBatching, Partition
from tests.conftest import PRIMES_UP_TO_73


def to_str(val: bytes | str) -> str:
    """Convert bytes to string if necessary."""
    return val.decode("utf-8") if isinstance(val, bytes) else val


def test_no_batching_full_row(warehouse, numbers_table):
    cols = (numbers_table.c.sys__id, numbers_table.c.number)
    db_ids, db_nums = zip(*warehouse.db.execute(sa.select(*cols)), strict=False)

    batching = NoBatching()
    rows = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols).order_by(numbers_table.c.sys__id),
    )
    assert isinstance(rows, GeneratorType)

    rows = list(rows)
    assert len(rows) == 73
    assert all(len(row) == 2 for row in rows)

    ids, nums = zip(*rows, strict=False)
    assert set(ids) == set(db_ids)
    assert set(nums) == set(db_nums)


def test_no_batching_ids_only(warehouse, numbers_table):
    cols = (numbers_table.c.sys__id, numbers_table.c.number)
    db_ids = [r[0] for r in warehouse.db.execute(sa.select(numbers_table.c.sys__id))]

    batching = NoBatching()
    rows = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols).order_by(numbers_table.c.sys__id),
        id_col=numbers_table.c.sys__id,
    )
    assert isinstance(rows, GeneratorType)

    rows = list(rows)
    assert len(rows) == 73
    assert set(rows) == set(db_ids)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 7, 10])
def test_batching_full_row(batch_size, warehouse, numbers_table):
    cols = (numbers_table.c.sys__id, numbers_table.c.number)
    db_values = set(warehouse.db.execute(sa.select(*cols)))

    batching = Batch(batch_size)
    batches = batching(
        warehouse.dataset_select_paginated,
        sa.select(numbers_table.c.sys__id, numbers_table.c.number).order_by(
            numbers_table.c.sys__id
        ),
    )
    assert isinstance(batches, GeneratorType)

    batches = list(batches)
    assert len(batches) == math.ceil(73 / batch_size)
    for i, batch in enumerate(batches):
        assert isinstance(batch, list)
        if i < len(batches) - 1:
            assert len(batch) == batch_size
        else:
            assert 0 < len(batch) <= batch_size

        for row in batch:
            row = (row[0], to_str(row[1]))
            assert row in db_values
            db_values.remove(row)

    assert not db_values


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 7, 10])
def test_batching_ids_only(batch_size, warehouse, numbers_table):
    cols = (numbers_table.c.sys__id, numbers_table.c.number)
    db_ids = {r[0] for r in warehouse.db.execute(sa.select(numbers_table.c.sys__id))}

    batching = Batch(count=batch_size)
    batches = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols).order_by(numbers_table.c.sys__id),
        id_col=numbers_table.c.sys__id,
    )
    assert isinstance(batches, GeneratorType)

    batches = list(batches)
    assert len(batches) == math.ceil(73 / batch_size)
    for i, batch in enumerate(batches):
        assert isinstance(batch, list)
        if i < len(batches) - 1:
            assert len(batch) == batch_size
        else:
            assert 0 < len(batch) <= batch_size

        for row_id in batch:
            assert row_id in db_ids
            db_ids.remove(row_id)

    assert not db_ids


@pytest.fixture
def numbers_partitioned(warehouse, numbers_table):
    partition_by = [numbers_table.c.primality]

    # create table with partitions
    partition_tbl = warehouse.create_udf_table(partition_columns())

    # fill table with partitions
    cols = [
        numbers_table.c.sys__id,
        sa.func.dense_rank().over(order_by=partition_by).label(PARTITION_COLUMN_ID),
    ]
    warehouse.db.execute(partition_tbl.insert().from_select(cols, sa.select(*cols)))

    yield (
        sa.select(*numbers_table.columns)
        .outerjoin(partition_tbl, partition_tbl.c.sys__id == numbers_table.c.sys__id)
        .add_columns(*partition_columns())
    )

    warehouse.db.drop_table(partition_tbl, if_exists=True)


def test_partition_full_row(warehouse, numbers_partitioned):
    query = numbers_partitioned.subquery()
    cols = (query.c.sys__id, query.c.number)

    db_ids, db_files = zip(*warehouse.db.execute(sa.select(*cols)), strict=False)
    db_ids, db_files = set(db_ids), set(db_files)

    partitions = {
        PRIMES_UP_TO_73,
        tuple(sorted(set(range(1, 74)) - set(PRIMES_UP_TO_73))),
    }

    batching = Partition()
    batches = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols, query.c.partition_id).order_by(query.c.sys__id),
    )
    assert isinstance(batches, GeneratorType)

    batches = list(batches)
    assert len(batches) == 2

    for batch in batches:
        assert isinstance(batch, list)

        ids = {row[0] for row in batch}
        assert ids.issubset(db_ids)
        db_ids = db_ids - ids

        files = {to_str(row[1]) for row in batch}
        assert files.issubset(db_files)
        db_files = db_files - files

        numbers = tuple(sorted(files))
        assert numbers in partitions
        partitions.remove(numbers)

    assert not db_ids
    assert not db_files
    assert not partitions


def test_partition_ids_only(warehouse, numbers_partitioned):
    query = numbers_partitioned.subquery()
    cols = (query.c.sys__id, query.c.number)

    db_rows = {num: id_ for id_, num in warehouse.db.execute(sa.select(*cols))}
    partitions = {
        PRIMES_UP_TO_73,
        tuple(sorted(set(range(1, 74)) - set(PRIMES_UP_TO_73))),
    }
    partition_ids = {
        tuple(sorted(db_rows[num] for num in numbers)) for numbers in partitions
    }
    db_ids = set(db_rows.values())

    batching = Partition()
    batches = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols, query.c.partition_id).order_by(query.c.sys__id),
        id_col=query.c.sys__id,
    )
    assert isinstance(batches, GeneratorType)

    batches = list(batches)
    assert len(batches) == 2

    for batch in batches:
        assert isinstance(batch, list)

        ids = set(batch)
        assert ids.issubset(db_ids)
        db_ids = db_ids - ids

        ids = tuple(sorted(ids))
        assert ids in partition_ids
        partition_ids.remove(ids)

    assert not db_ids
    assert not partition_ids


def test_partition_missing_column_raises(warehouse, numbers_table):
    # Build a query without partition_id to ensure Partition raises
    query = sa.select(
        numbers_table.c.sys__id,
        numbers_table.c.number,
    ).order_by(numbers_table.c.sys__id)

    batching = Partition()
    with pytest.raises(RuntimeError, match="partition column not found in query"):
        # exhaust the generator to trigger execution
        list(batching(warehouse.dataset_select_paginated, query))
