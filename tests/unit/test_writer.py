import sqlite3

import pytest
import sqlalchemy as sa

from datachain.data_storage.writer import (
    ID_BITS,
    MAX_WRITER_NO,
    committed_part_keys,
    parts_columns,
    trim_to_ledger,
)
from datachain.sql.types import Int
from tests.utils import skip_if_not_sqlite


@pytest.fixture
def tables(warehouse):
    table = warehouse.create_dataset_rows_table(
        "out", columns=[sa.Column("x", Int)], if_not_exists=False
    )
    parts = warehouse.create_udf_table(parts_columns(), name="out_parts")
    yield table, parts
    warehouse.db.drop_table(table, if_exists=True)
    warehouse.db.drop_table(parts, if_exists=True)


def rows(warehouse, table):
    query = sa.select(table.c.sys__id, table.c.x).order_by(table.c.sys__id)
    return list(warehouse.db.execute(query))


def ledger(warehouse, parts):
    columns = [parts.c[c] for c in ("part_key", "writer_no", "lo", "hi")]
    query = sa.select(*columns).order_by(parts.c.lo, parts.c.hi)
    return list(warehouse.db.execute(query))


def test_writer_zero_writes_ids_from_one(warehouse, tables):
    table, parts = tables
    with warehouse.writer(table, ["x"], parts=parts, batch_size=2) as w:
        w.extend([(10,), (11,), (12,)])
        w.commit(7)
        w.commit(8)  # a part with no rows
        w.extend([(13,)])
        w.commit(9)
    assert rows(warehouse, table) == [(1, 10), (2, 11), (3, 12), (4, 13)]
    assert ledger(warehouse, parts) == [(7, 0, 1, 3), (8, 0, 4, 3), (9, 0, 4, 4)]


def test_writer_number_is_the_high_bits(warehouse, tables):
    table, parts = tables
    with warehouse.writer(table, ["x"], writer_no=3, parts=parts) as w:
        w.extend([(1,), (2,)])
        w.commit(1)
    base = 3 << ID_BITS
    assert [i for i, _ in rows(warehouse, table)] == [base + 1, base + 2]


def test_rows_with_ids_are_their_own_record(warehouse, tables):
    table, _ = tables
    with warehouse.writer(table, ["sys__id", "x"]) as w:
        w.extend([(5, 50), (9, 90)])
    assert rows(warehouse, table) == [(5, 50), (9, 90)]


def test_abort_drops_the_unfinished_part(warehouse, tables):
    table, parts = tables
    with (
        pytest.raises(RuntimeError),
        warehouse.writer(table, ["x"], parts=parts, batch_size=100) as w,
    ):
        w.extend([(1,), (2,)])
        w.commit(1)
        w.extend([(3,)])
        raise RuntimeError
    assert rows(warehouse, table) == [(1, 1), (2, 2)]
    assert ledger(warehouse, parts) == [(1, 0, 1, 2)]


def test_claims_are_unique_and_durable(warehouse, tables):
    _, parts = tables
    assert [warehouse.claim_writer_no(parts) for _ in range(3)] == [0, 1, 2]
    assert ledger(warehouse, parts) == [
        (None, w, (w << ID_BITS) + 1, w << ID_BITS) for w in range(3)
    ]
    assert list(warehouse.db.execute(committed_part_keys(parts))) == []


def test_writer_number_range(warehouse, tables):
    table, _ = tables
    warehouse.writer(table, ["x"], writer_no=MAX_WRITER_NO)
    with pytest.raises(ValueError, match="out of range"):
        warehouse.writer(table, ["x"], writer_no=MAX_WRITER_NO + 1)


def test_trim_after_a_crash(warehouse, tables):
    table, parts = tables
    # writer 0 commits part 1, then crashes after flushing half of part 2
    w = warehouse.writer(
        table, ["x"], warehouse.claim_writer_no(parts), parts, batch_size=2
    )
    w.extend([(1,), (2,), (3,)])
    w.commit(1)
    w.extend([(4,)])  # the block (3, 4) is flushed with part 1's ledger row
    # writer 1 flushes rows but commits nothing before the crash
    w1 = warehouse.writer(
        table, ["x"], warehouse.claim_writer_no(parts), parts, batch_size=1
    )
    w1.extend([(100,)])
    # writer 2 commits one part
    with warehouse.writer(table, ["x"], warehouse.claim_writer_no(parts), parts) as w2:
        w2.extend([(200,)])
        w2.commit(5)

    trim_to_ledger(warehouse, table, parts)
    assert rows(warehouse, table) == [
        (1, 1),
        (2, 2),
        (3, 3),
        ((2 << ID_BITS) + 1, 200),
    ]
    # the next attempt never reuses a number, committed or not
    assert warehouse.claim_writer_no(parts) == 3


class _BusyOnce:
    """The sqlite3 connection, except that the first executemany consumes one
    parameter row and then fails as a lock error, as SQLITE_BUSY does."""

    def __init__(self, db):
        self._db = db
        self.failed = False

    def executemany(self, sql, params):
        if not self.failed:
            self.failed = True
            next(iter(params))
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = 5  # SQLITE_BUSY
            raise error
        return self._db.executemany(sql, params)

    def __getattr__(self, name):
        return getattr(self._db, name)

    def __enter__(self):
        return self._db.__enter__()

    def __exit__(self, *exc):
        return self._db.__exit__(*exc)


@skip_if_not_sqlite  # retries SQLite's "database is locked"
def test_lock_retry_writes_every_row(warehouse, tables, monkeypatch):
    table, parts = tables
    monkeypatch.setattr(warehouse.db, "db", _BusyOnce(warehouse.db.db))
    with warehouse.writer(table, ["x"], parts=parts) as w:
        w.extend([(10,), (20,), (30,)])
        w.commit(42)
    assert warehouse.db.db.failed
    assert rows(warehouse, table) == [(1, 10), (2, 20), (3, 30)]
    assert ledger(warehouse, parts) == [(42, 0, 1, 3)]


def test_trim_after_many_claims(warehouse, tables):
    table, parts = tables
    for _ in range(1000):
        last = warehouse.claim_writer_no(parts)
    with warehouse.writer(table, ["x"], last, parts) as w:
        w.extend([(1,), (2,)])
        w.commit(7)
        w.extend([(3,)])  # unfinished part
    trim_to_ledger(warehouse, table, parts)
    base = last << ID_BITS
    assert rows(warehouse, table) == [(base + 1, 1), (base + 2, 2)]
