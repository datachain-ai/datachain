import pickle
from queue import Empty, Full

import pytest

from datachain.query.dispatch import STOP_SIGNAL, get_from_queue, put_into_queue


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


def test_processed_callback_batches_and_flushes(mocker):
    from queue import Queue

    from datachain.query.dispatch import NOTIFY_STATUS, ProcessedCallback

    now = mocker.patch("datachain.query.dispatch.monotonic", return_value=0)
    queue = Queue()
    callback = ProcessedCallback("processed", queue)
    for _ in range(999):
        callback.relative_update()
    assert queue.empty()
    callback.relative_update()
    assert queue.get_nowait() == {"status": NOTIFY_STATUS, "processed": 1000}
    callback.relative_update(7)
    now.return_value = 0.2
    callback.relative_update(3)
    assert queue.get_nowait() == {"status": NOTIFY_STATUS, "processed": 10}
    callback.relative_update(2)
    callback.flush()
    callback.flush()
    assert queue.get_nowait() == {"status": NOTIFY_STATUS, "processed": 2}
    assert queue.empty()


def test_udf_worker_acknowledges_each_input_task(mocker):
    from queue import Queue

    from datachain.query.dispatch import OK_STATUS, UDFWorker

    worker = UDFWorker.__new__(UDFWorker)
    worker.task_queue = Queue()
    worker.done_queue = Queue()
    worker.query = mocker.sentinel.query
    worker.is_batching = False
    worker.catalog = mocker.Mock()
    warehouse = worker.catalog.warehouse.clone.return_value
    warehouse.dataset_rows_select_from_ids.side_effect = (
        lambda query, ids, is_batching: iter(ids)
    )
    for task in ([1, 2], [], [3], STOP_SIGNAL):
        worker.task_queue.put(pickle.dumps(task))
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
def test_udf_worker_flushes_progress_on_exit(mocker, fail):
    from contextlib import nullcontext
    from queue import Queue

    from datachain.query.dispatch import FINISHED_STATUS, NOTIFY_STATUS, UDFWorker

    catalog = mocker.Mock()
    udf = mocker.Mock(prefetch=0)
    udf.run.return_value = iter(())
    queue = Queue()
    worker = UDFWorker(catalog, udf, Queue(), queue, None, None, False, False, None, [])
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
    assert queue.get_nowait() == {"status": NOTIFY_STATUS, "processed": 3}
    assert queue.get_nowait() == {"status": NOTIFY_STATUS, "generated": 7}
    if not fail:
        assert queue.get_nowait() == {"status": FINISHED_STATUS}
    assert queue.empty()
