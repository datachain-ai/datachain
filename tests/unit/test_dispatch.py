import os
from queue import Empty, Full

import pytest

from datachain.query.batch import NoBatching
from datachain.query.dispatch import (
    STOP_SIGNAL,
    UDFDispatcher,
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
