import os
from contextlib import nullcontext
from queue import Empty, Full, Queue

import pytest

from datachain.query.batch import NoBatching
from datachain.query.dispatch import (
    FINISHED_STATUS,
    NOTIFY_STATUS,
    OK_STATUS,
    STATS_INTERVAL_SEC,
    STOP_SIGNAL,
    DownloadCallback,
    ProcessedCallback,
    UDFDispatcher,
    UDFWorker,
    WorkerStats,
    get_from_queue,
    guided_tasks,
    put_into_queue,
)
from datachain.query.udf import UdfInfo


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


@pytest.mark.parametrize(
    "items,total,workers,max_size",
    [
        (255, 255, 8, 31),
        (1, 1, 4, 1),
        (16, 16, 8, 2),
        (100_000, 100_000, 4, 10_000),
        (7, 7, 16, 1),
        (10, 3, 2, 5),
    ],
)
def test_guided_tasks_cover_items_once_and_end_with_single_items(
    items, total, workers, max_size
):
    tasks = list(guided_tasks(range(items), total, workers, max_size))
    sizes = [len(t) for t in tasks]

    assert [i for t in tasks for i in t] == list(range(items))
    assert max(sizes) <= max_size
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-min(workers, len(sizes)) :] == [1] * min(workers, len(sizes))


@pytest.mark.parametrize("workers,max_size", [(0, 1), (-1, 1), (1, 0)])
def test_guided_tasks_require_positive_limits(workers, max_size):
    with pytest.raises(ValueError, match="must be positive"):
        list(guided_tasks(range(3), 3, workers, max_size))


def test_automatic_single_worker_applies_and_restores_thread_budget(
    monkeypatch, mocker
):
    monkeypatch.setattr("datachain.query.dispatch.usable_cpu_count", lambda: 1)
    monkeypatch.setattr("datachain.utils.usable_cpu_count", lambda: 1)
    monkeypatch.setenv("OMP_NUM_THREADS", "1000")
    monkeypatch.delenv("DATACHAIN_WORKER_THREADS", raising=False)
    dispatcher = UDFDispatcher(
        UdfInfo(
            udf_data=b"",
            catalog_init={},
            metastore_clone_params=(object, [], {}),
            warehouse_clone_params=(object, [], {}),
            query=mocker.sentinel.query,
            table=mocker.sentinel.table,
            udf_fields=[],
            batching=NoBatching(),
            processes=True,
            is_generator=False,
            cache=False,
            rows_total=8,
            batch_size=None,
        )
    )

    def run_single(*args):
        assert os.environ["DATACHAIN_WORKER_THREADS"] == "1"
        assert os.environ["OMP_NUM_THREADS"] == "1"

    single = mocker.patch.object(dispatcher, "run_udf_single", side_effect=run_single)
    mocker.patch.object(dispatcher, "run_udf_parallel", side_effect=AssertionError)
    before = dict(os.environ)

    dispatcher.run_udf([])

    single.assert_called_once()
    assert dict(os.environ) == before
