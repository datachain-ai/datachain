from queue import Empty, Full

import pytest

from datachain.query.dispatch import (
    STOP_SIGNAL,
    get_from_queue,
    guided_tasks,
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
