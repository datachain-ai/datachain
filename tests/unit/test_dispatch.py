from contextlib import nullcontext
from queue import Empty, Full, Queue

import pytest

from datachain.query.dispatch import (
    FINISHED_STATUS,
    NOTIFY_STATUS,
    OK_STATUS,
    STATS_INTERVAL_SEC,
    STOP_SIGNAL,
    DownloadCallback,
    ProcessedCallback,
    UDFWorker,
    WorkerStats,
    get_from_queue,
    put_into_queue,
)


class MockQueue:
    def __init__(self) -> None:
        self.return_empty_once = True
        self.return_full_once = True
        self.put_signal: str | None = None
        self.put_count: int = 0

    def put_nowait(self, task: str) -> None:
        if self.return_full_once:
            self.return_full_once = False
            raise Full
        assert task == STOP_SIGNAL
        self.put_signal = task
        self.put_count += 1

    def get_nowait(self) -> str:
        if self.return_empty_once:
            self.return_empty_once = False
            raise Empty
        return STOP_SIGNAL


def test_get_from_queue():
    mock_queue = MockQueue()

    assert get_from_queue(mock_queue) == STOP_SIGNAL


def test_put_into_queue():
    mock_queue = MockQueue()

    assert put_into_queue(mock_queue, STOP_SIGNAL) is None
    assert mock_queue.put_signal == STOP_SIGNAL


@pytest.fixture
def udf_worker(mocker):
    catalog = mocker.Mock()
    udf = mocker.Mock(prefetch=0)
    udf.run.return_value = iter(())
    return UDFWorker(catalog, udf, Queue(), Queue(), None, None, False, False, None, [])


def test_udf_worker_acknowledges_each_input_task(udf_worker):
    worker = udf_worker
    warehouse = worker.catalog.warehouse.clone.return_value
    warehouse.dataset_rows_select_from_ids.side_effect = (
        lambda query, ids, is_batching: iter(ids)
    )
    for task in ([1, 2], [], [3], STOP_SIGNAL):
        worker.task_queue.put(task)
    inputs = worker.get_inputs()
    assert next(inputs) == 1
    assert next(inputs) == 2
    assert worker.done_queue.empty()
    assert next(inputs) == 3
    assert list(inputs) == []
    assert [worker.done_queue.get_nowait() for _ in range(3)] == [
        {"status": OK_STATUS}
    ] * 3
    assert worker.done_queue.empty()


@pytest.mark.parametrize("fail", [False, True])
def test_udf_worker_flushes_progress_on_exit(mocker, udf_worker, fail):
    worker = udf_worker
    catalog = worker.catalog
    queue = worker.done_queue
    mocker.patch("datachain.query.dispatch._get_cache", return_value=nullcontext())
    mocker.patch(
        "datachain.query.dispatch.clone_catalog_with_cache", return_value=catalog
    )

    def process(*args, **kwargs):
        worker.processed_cb.relative_update(3)
        kwargs["cb"].relative_update(7)
        if fail:
            raise ValueError("failed")

    mocker.patch("datachain.query.dispatch.process_udf_outputs", side_effect=process)
    if fail:
        with pytest.raises(ValueError, match="failed"):
            worker.run()
    else:
        worker.run()
    assert queue.get_nowait() == {
        "status": NOTIFY_STATUS,
        "processed": 3,
        "generated": 7,
    }
    if not fail:
        assert queue.get_nowait() == {"status": FINISHED_STATUS}
    assert queue.empty()


def test_worker_sends_progress_once_per_interval(monkeypatch):
    now = [0]
    monkeypatch.setattr("datachain.query.dispatch.monotonic", lambda: now[0])
    queue = Queue()
    stats = WorkerStats(queue)
    processed = ProcessedCallback("processed", stats)
    download = DownloadCallback(stats)

    for _ in range(300):
        now[0] += 1
        processed.relative_update(1)
        download.increment_file_count(1)
        download.relative_update(10)
    stats.send(FINISHED_STATUS)

    messages = list(queue.queue)
    assert len(messages) == 300 // STATS_INTERVAL_SEC + 1
    assert sum(m.get("processed", 0) for m in messages) == 300
    assert sum(m.get("downloaded", 0) for m in messages) == 300
    assert sum(m.get("bytes_downloaded", 0) for m in messages) == 3000


def test_worker_stats_flush_is_idempotent():
    queue = Queue()
    stats = WorkerStats(queue)
    stats.add("processed", 2)
    stats.flush()
    stats.flush()
    assert queue.get_nowait() == {"status": NOTIFY_STATUS, "processed": 2}
    assert queue.empty()
