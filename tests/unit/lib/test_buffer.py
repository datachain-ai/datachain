from types import SimpleNamespace

import pytest

from datachain.data_storage.buffer import InsertBuffer


@pytest.fixture
def mock_db():
    db = {
        "tables": {},
        "flush_calls": [],
    }

    def insert(table, columns, entries, final=False, cursor=None):
        assert columns == ["id"]
        if table not in db["tables"]:
            db["tables"][table] = []
        db["tables"][table].extend(entries)
        db["flush_calls"].append(
            {
                "table": table,
                "count": len(entries),
                "final": final,
            }
        )

    db["insert"] = insert
    return db


@pytest.fixture
def buffer(mock_db):
    return InsertBuffer(
        table="test_table",
        columns=["id"],
        execute_callback=mock_db["insert"],
        buffer_size=10,
    )


def test_insert_single_entry(buffer, mock_db):
    buffer.insert((1,))

    assert len(buffer) == 1
    assert len(mock_db["flush_calls"]) == 0


def test_insert_fills_buffer_triggers_flush(buffer, mock_db):
    for i in range(10):
        buffer.insert((i,))

    assert len(buffer) == 0
    assert len(mock_db["flush_calls"]) == 1
    assert mock_db["flush_calls"][0]["count"] == 10
    assert len(mock_db["tables"]["test_table"]) == 10


def test_insert_multiple_batches(buffer, mock_db):
    for i in range(25):
        buffer.insert((i,))

    assert len(buffer) == 5
    assert len(mock_db["flush_calls"]) == 2
    assert len(mock_db["tables"]["test_table"]) == 20


def test_flush_remaining_entries(buffer, mock_db):
    for i in range(5):
        buffer.insert((i,))

    assert len(mock_db["flush_calls"]) == 0

    buffer.flush()

    assert len(buffer) == 0
    assert len(mock_db["flush_calls"]) == 1
    assert mock_db["flush_calls"][0]["final"] is True
    assert len(mock_db["tables"]["test_table"]) == 5


def test_flush_empty_buffer(buffer, mock_db):
    buffer.flush()

    assert len(mock_db["flush_calls"]) == 0


def test_insert_many_basic(buffer, mock_db):
    entries = [(i,) for i in range(15)]
    buffer.insert_many(entries)

    assert len(buffer) == 5
    assert len(mock_db["flush_calls"]) == 1
    assert len(mock_db["tables"]["test_table"]) == 10


def test_insert_many_lazy_iteration(mock_db):
    yielded_count = 0
    flush_yielded_counts = []

    def tracking_insert(table, columns, entries, final=False, cursor=None):
        flush_yielded_counts.append(yielded_count)
        mock_db["insert"](table, columns, entries, final, cursor)

    buffer = InsertBuffer(
        table="test_table",
        columns=["id"],
        execute_callback=tracking_insert,
        buffer_size=10,
    )

    def lazy_generator():
        nonlocal yielded_count
        for i in range(35):
            yielded_count += 1
            yield (i,)

    buffer.insert_many(lazy_generator())

    assert len(flush_yielded_counts) == 3
    assert flush_yielded_counts[0] == 10
    assert flush_yielded_counts[1] == 20
    assert flush_yielded_counts[2] == 30


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        "datachain.data_storage.buffer.time",
        SimpleNamespace(monotonic=lambda: clock.now),
    )
    return clock


@pytest.fixture(params=["insert", "insert_many"])
def add(request):
    if request.param == "insert":
        return lambda buffer, row: buffer.insert(row)
    return lambda buffer, row: buffer.insert_many([row])


def test_time_based_flush(mock_db, clock, add):
    buffer = InsertBuffer(
        "test_table", ["id"], mock_db["insert"], buffer_size=100, flush_interval=2.0
    )
    for now, row in [(0.0, 1), (1.9, 2), (2.0, 3), (3.9, 4), (4.0, 5)]:
        clock.now = now
        add(buffer, (row,))
    assert [call["count"] for call in mock_db["flush_calls"]] == [3, 2]


def test_no_time_flush_when_interval_none(mock_db, clock, add):
    buffer = InsertBuffer(
        "test_table", ["id"], mock_db["insert"], buffer_size=100, flush_interval=None
    )
    add(buffer, (1,))
    clock.now = 10.0
    add(buffer, (2,))
    assert not mock_db["flush_calls"]
    assert len(buffer) == 2


def test_close_with_cursor(mock_db):
    class MockCursor:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    cursor = MockCursor()
    buffer = InsertBuffer(
        table="test_table",
        columns=["id"],
        execute_callback=mock_db["insert"],
        buffer_size=10,
        cursor=cursor,
    )

    buffer.close()
    assert cursor.closed is True


def test_len(buffer):
    assert len(buffer) == 0

    buffer.insert((1,))
    assert len(buffer) == 1

    buffer.insert((2,))
    assert len(buffer) == 2


def test_insert_many_flushes_slow_iterator_before_exhaustion(mock_db, clock):
    buffer = InsertBuffer(
        "test_table", ["id"], mock_db["insert"], buffer_size=100, flush_interval=2.0
    )

    def rows():
        for i in range(10):
            clock.now += 0.5
            yield (i,)
            assert len(mock_db["tables"].get("test_table", [])) == ((i + 1) // 4) * 4

    buffer.insert_many(rows())
    assert [call["count"] for call in mock_db["flush_calls"]] == [4, 4]
    assert len(buffer) == 2
    buffer.flush()
    assert mock_db["tables"]["test_table"] == [(i,) for i in range(10)]
    assert mock_db["flush_calls"][-1]["final"] is True


def test_insert_many_time_flush_retains_rows_on_insert_error(mock_db, clock):
    attempts = 0

    def insert(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("insert failed")
        mock_db["insert"](*args, **kwargs)

    buffer = InsertBuffer(
        "test_table", ["id"], insert, buffer_size=100, flush_interval=2.0
    )
    buffer.insert_many([(1,)])
    clock.now = 2.0
    with pytest.raises(RuntimeError, match="insert failed"):
        buffer.insert_many([(2,)])
    assert len(buffer) == 2
    buffer.flush()
    assert mock_db["tables"]["test_table"] == [(1,), (2,)]
