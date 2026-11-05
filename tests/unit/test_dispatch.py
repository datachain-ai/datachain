from queue import Empty, Full

import pytest

from datachain.query.dispatch import (
    STOP_SIGNAL,
    get_from_queue,
    get_n_workers_from_arg,
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
    "cpus,requested,expected", [(8, None, 8), (1, None, 1), (8, 12, 12)]
)
def test_worker_count_respects_cpu_quota_and_explicit_override(
    monkeypatch, cpus, requested, expected
):
    monkeypatch.setattr("datachain.query.dispatch.usable_cpu_count", lambda: cpus)

    assert get_n_workers_from_arg(requested) == expected
