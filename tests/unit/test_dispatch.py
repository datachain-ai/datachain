import os
from queue import Empty, Full

import pytest

from datachain.query.batch import NoBatching
from datachain.query.dispatch import (
    FINISHED_STATUS,
    STOP_SIGNAL,
    UDFDispatcher,
    get_from_queue,
    get_n_workers_from_arg,
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


@pytest.fixture
def dispatcher(mocker):
    return UDFDispatcher(
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


@pytest.mark.parametrize(
    "requested,rows,workers,share",
    [
        (True, 8, 8, 1),
        (True, 2, 2, 4),
        (8, 2, 2, 4),
        (8, 1, 1, 8),
        (8, 0, 1, 8),
        (False, 8, 1, 8),
        (16, 32, 16, 1),
    ],
)
@pytest.mark.parametrize("fail", [False, True])
def test_dispatch_workers_apply_and_restore_cpu_share(
    dispatcher, monkeypatch, mocker, requested, rows, workers, share, fail
):
    dispatcher.processes = requested
    dispatcher.rows_total = rows
    monkeypatch.setattr("datachain.query.dispatch.usable_cpu_count", lambda: 8)
    monkeypatch.setattr("datachain.utils.usable_cpu_count", lambda: 8)
    monkeypatch.setenv("OMP_NUM_THREADS", "1000")
    monkeypatch.delenv("DATACHAIN_WORKER_THREADS", raising=False)

    def run(*args):
        assert os.environ["DATACHAIN_WORKER_THREADS"] == str(share)
        assert os.environ["OMP_NUM_THREADS"] == str(share)
        if workers > 1:
            assert args[0] == workers
        if fail:
            raise RuntimeError("UDF failed")

    selected = "run_udf_single" if workers == 1 else "run_udf_parallel"
    other = "run_udf_parallel" if workers == 1 else "run_udf_single"
    execution = mocker.patch.object(dispatcher, selected, side_effect=run)
    mocker.patch.object(dispatcher, other, side_effect=AssertionError)
    before = dict(os.environ)

    if fail:
        with pytest.raises(RuntimeError, match="UDF failed"):
            dispatcher.run_udf([])
    else:
        dispatcher.run_udf([])

    execution.assert_called_once()
    assert dict(os.environ) == before


def test_worker_start_failure_cleans_up_and_restores_environment(
    dispatcher, mocker, monkeypatch
):
    monkeypatch.setattr("datachain.query.dispatch.usable_cpu_count", lambda: 2)
    monkeypatch.setattr("datachain.utils.usable_cpu_count", lambda: 2)
    monkeypatch.setenv("OMP_NUM_THREADS", "1000")
    before = dict(os.environ)
    first, second = mocker.Mock(), mocker.Mock()
    first.is_alive.side_effect = [True, False]
    second.is_alive.return_value = False

    def start():
        assert os.environ["OMP_NUM_THREADS"] == "1"

    first.start.side_effect = start
    second.start.side_effect = OSError("worker could not start")
    mocker.patch.object(dispatcher.ctx, "Process", side_effect=[first, second])

    with pytest.raises(OSError, match="worker could not start"):
        dispatcher.run_udf([])

    first.terminate.assert_called_once()
    for queue in (dispatcher.task_queue, dispatcher.done_queue):
        with pytest.raises(ValueError, match="closed"):
            queue.put_nowait(1)
    assert dict(os.environ) == before


@pytest.mark.parametrize("units", [0, 1, 2, 12])
def test_batched_workers_share_cpus_between_actual_work_units(
    dispatcher, mocker, monkeypatch, units
):
    dispatcher.is_batching = True
    dispatcher.rows_total = 2000
    monkeypatch.setattr("datachain.query.dispatch.usable_cpu_count", lambda: 8)
    monkeypatch.setattr("datachain.utils.usable_cpu_count", lambda: 8)
    monkeypatch.delenv("DATACHAIN_WORKER_THREADS", raising=False)
    batches = [list(range(i * 100, (i + 1) * 100)) for i in range(units)]
    workers = max(1, min(units, 8))

    def run(*args):
        if workers > 1:
            assert args[0] == workers
        inputs = args[1] if workers > 1 else args[0]
        assert list(inputs) == batches
        assert os.environ["DATACHAIN_WORKER_THREADS"] == str(8 // workers)

    selected = "run_udf_single" if workers == 1 else "run_udf_parallel"
    other = "run_udf_parallel" if workers == 1 else "run_udf_single"
    execution = mocker.patch.object(dispatcher, selected, side_effect=run)
    mocker.patch.object(dispatcher, other, side_effect=AssertionError)
    dispatcher.run_udf(iter(batches))
    execution.assert_called_once()


@pytest.mark.parametrize(
    "cpus,requested,expected", [(8, None, 8), (1, None, 1), (8, 12, 12)]
)
def test_worker_count_respects_cpu_quota_and_explicit_override(
    monkeypatch, cpus, requested, expected
):
    monkeypatch.setattr("datachain.query.dispatch.usable_cpu_count", lambda: cpus)

    assert get_n_workers_from_arg(requested) == expected


@pytest.mark.parametrize("sizes", [[100] * 100, [1, 1, 1, 1000]])
def test_batched_udf_groups_tasks_without_splitting_work_units(
    dispatcher, mocker, sizes
):
    dispatcher.is_batching = True
    dispatcher.rows_total = sum(sizes)
    batches = [list(range(i * 1000, i * 1000 + size)) for i, size in enumerate(sizes)]
    task_queue, done_queue = mocker.Mock(), mocker.Mock()
    done_queue.get_nowait.side_effect = [{"status": FINISHED_STATUS}] * 4
    mocker.patch.object(dispatcher.ctx, "Queue", side_effect=[task_queue, done_queue])
    mocker.patch.object(dispatcher.ctx, "Process")
    mocker.patch.object(dispatcher, "_shutdown_workers")

    dispatcher.run_udf_parallel(4, batches, None, None, None)

    queued = [
        call.args[0]
        for call in task_queue.put_nowait.call_args_list
        if call.args[0] != STOP_SIGNAL
    ]
    assert [batch for task in queued for batch in task] == batches
    assert queued[:4] == [(batch,) for batch in batches[:4]]
    if len(batches) > 4:
        assert 4 < len(queued) < len(batches)
    else:
        assert len(queued) == 4


@pytest.mark.parametrize("estimate", [0, 1000])
def test_guided_tasks_keep_query_count_bounded_with_an_underestimate(estimate):
    tasks = list(guided_tasks(range(100_000), estimate, 8, 10_000))
    assert [item for task in tasks for item in task] == list(range(100_000))
    assert len(tasks) < 1000
    assert max(map(len, tasks)) <= 10_000


@pytest.mark.parametrize("estimate", [0, 20_000])
def test_small_partitions_do_not_require_a_query_per_partition(estimate):
    partitions = [list(range(i, i + 2)) for i in range(0, 20_000, 2)]
    tasks = list(guided_tasks(partitions, estimate, 4, 10_000, weight=len))
    assert [partition for task in tasks for partition in task] == partitions
    assert 4 < len(tasks) < 200
    assert max(sum(map(len, task)) for task in tasks) <= 10_000
