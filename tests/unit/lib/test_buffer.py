import time

import pytest

from datachain.data_storage.buffer import InsertBuffer


@pytest.fixture
def mock_db():
    db = {
        "tables": {},
        "flush_calls": [],
    }

    def insert(table, entries, final=False, cursor=None):
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
        execute_callback=mock_db["insert"],
        buffer_size=10,
    )


def test_insert_single_entry(buffer, mock_db):
    buffer.insert({"id": 1})

    assert len(buffer) == 1
    assert len(mock_db["flush_calls"]) == 0


def test_insert_fills_buffer_triggers_flush(buffer, mock_db):
    for i in range(10):
        buffer.insert({"id": i})

    assert len(buffer) == 0
    assert len(mock_db["flush_calls"]) == 1
    assert mock_db["flush_calls"][0]["count"] == 10
    assert len(mock_db["tables"]["test_table"]) == 10


def test_insert_multiple_batches(buffer, mock_db):
    for i in range(25):
        buffer.insert({"id": i})

    assert len(buffer) == 5
    assert len(mock_db["flush_calls"]) == 2
    assert len(mock_db["tables"]["test_table"]) == 20


def test_flush_remaining_entries(buffer, mock_db):
    for i in range(5):
        buffer.insert({"id": i})

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
    entries = [{"id": i} for i in range(15)]
    buffer.insert_many(entries)

    assert len(buffer) == 5
    assert len(mock_db["flush_calls"]) == 1
    assert len(mock_db["tables"]["test_table"]) == 10


def test_insert_many_lazy_iteration(mock_db):
    yielded_count = 0
    flush_yielded_counts = []

    def tracking_insert(table, entries, final=False, cursor=None):
        flush_yielded_counts.append(yielded_count)
        mock_db["insert"](table, entries, final, cursor)

    buffer = InsertBuffer(
        table="test_table",
        execute_callback=tracking_insert,
        buffer_size=10,
    )

    def lazy_generator():
        nonlocal yielded_count
        for i in range(35):
            yielded_count += 1
            yield {"id": i}

    buffer.insert_many(lazy_generator())

    assert len(flush_yielded_counts) == 3
    assert flush_yielded_counts[0] == 10
    assert flush_yielded_counts[1] == 20
    assert flush_yielded_counts[2] == 30


def test_time_based_flush(mock_db):
    buffer = InsertBuffer(
        table="test_table",
        execute_callback=mock_db["insert"],
        buffer_size=100,
        flush_interval=0.1,
    )

    buffer.insert({"id": 1})
    assert len(mock_db["flush_calls"]) == 0

    time.sleep(0.15)

    buffer.insert({"id": 2})
    assert len(mock_db["flush_calls"]) == 1
    assert mock_db["flush_calls"][0]["count"] == 2


def test_time_based_flush_resets_timer(mock_db):
    buffer = InsertBuffer(
        table="test_table",
        execute_callback=mock_db["insert"],
        buffer_size=100,
        flush_interval=0.1,
    )

    buffer.insert({"id": 1})
    time.sleep(0.15)
    buffer.insert({"id": 2})

    assert len(mock_db["flush_calls"]) == 1

    buffer.insert({"id": 3})
    assert len(mock_db["flush_calls"]) == 1

    time.sleep(0.15)
    buffer.insert({"id": 4})
    assert len(mock_db["flush_calls"]) == 2


def test_no_time_flush_when_interval_none(mock_db):
    buffer = InsertBuffer(
        table="test_table",
        execute_callback=mock_db["insert"],
        buffer_size=100,
        flush_interval=None,
    )

    buffer.insert({"id": 1})
    time.sleep(0.1)
    buffer.insert({"id": 2})

    assert len(mock_db["flush_calls"]) == 0
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
        execute_callback=mock_db["insert"],
        buffer_size=10,
        cursor=cursor,
    )

    buffer.close()
    assert cursor.closed is True


def test_len(buffer):
    assert len(buffer) == 0

    buffer.insert({"id": 1})
    assert len(buffer) == 1

    buffer.insert({"id": 2})
    assert len(buffer) == 2
