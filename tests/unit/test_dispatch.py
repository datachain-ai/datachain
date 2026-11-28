import os
import queue as queue_mod
import sqlite3
from io import BytesIO
from queue import Empty, Full
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import datachain as dc
from datachain.lib.file import File
from datachain.query import dispatch
from datachain.query.dispatch import (
    FINISHED_STATUS,
    STOP_SIGNAL,
    DownloadCallback,
    ProcessedCallback,
    UDFWorker,
    WorkerStats,
    get_from_queue,
    get_n_workers_from_arg,
    put_into_queue,
    udf_entrypoint,
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


def test_worker_aggregates_progress_until_flushed():
    queue = queue_mod.Queue()
    stats = WorkerStats(queue)
    processed = ProcessedCallback("processed", stats)
    download = DownloadCallback(stats)
    for _ in range(300):
        processed.relative_update(1)
        download.increment_file_count(1)
        download.relative_update(10)
    stats.send(FINISHED_STATUS)
    assert queue.get_nowait() == {
        "status": FINISHED_STATUS,
        "processed": 300,
        "downloaded": 300,
        "bytes_downloaded": 3000,
    }
    assert queue.empty()


def test_worker_reports_pending_progress_without_another_update(monkeypatch):
    monkeypatch.setattr("datachain.query.dispatch.STATS_INTERVAL_SEC", 0.01)
    queue = queue_mod.Queue()
    stats = WorkerStats(queue)
    with stats.track():
        stats.add("processed", 1)
        assert queue.get(timeout=1) == {"status": "NOTIFY", "processed": 1}
    assert queue.empty()


def test_worker_acknowledges_tasks_before_fetching_rows():
    tasks = queue_mod.Queue()
    done = queue_mod.Queue()
    tasks.put([[1], [2]])
    tasks.put(STOP_SIGNAL)
    catalog = MagicMock()
    fetch = catalog.warehouse.clone().dataset_rows_select_from_ids
    fetch.return_value = iter(["first", "second"])
    worker = UDFWorker(
        catalog,
        MagicMock(),
        tasks,
        done,
        MagicMock(),
        MagicMock(),
        cache=False,
        is_batching=False,
        batch_size=None,
        udf_fields=[],
    )
    rows = iter(worker.get_inputs())
    assert next(rows) == "first"
    assert done.get_nowait() == {"status": "OK"}
    assert list(rows) == ["second"]
    assert done.empty()
    fetch.assert_called_once()


def test_worker_flushes_progress_when_processing_fails(monkeypatch):
    catalog = MagicMock()
    done = queue_mod.Queue()
    udf = MagicMock(prefetch=0)
    worker = UDFWorker(
        catalog,
        udf,
        queue_mod.Queue(),
        done,
        MagicMock(),
        MagicMock(),
        cache=False,
        is_batching=False,
        batch_size=None,
        udf_fields=[],
    )

    def fail(*args, **kwargs):
        worker.processed_cb.relative_update(3)
        worker.download_cb.relative_update(1024)
        raise RuntimeError("processing failed")

    monkeypatch.setattr("datachain.query.dispatch.process_udf_outputs", fail)
    with pytest.raises(RuntimeError, match="processing failed"):
        worker.run()
    assert done.get_nowait() == {
        "status": "NOTIFY",
        "processed": 3,
        "bytes_downloaded": 1024,
    }


@pytest.mark.parametrize(
    "cpus,requested,expected", [(8, None, 8), (1, None, 1), (8, 12, 12)]
)
def test_worker_count_respects_cpu_quota_and_explicit_override(
    monkeypatch, cpus, requested, expected
):
    monkeypatch.setattr("datachain.query.dispatch.usable_cpu_count", lambda: cpus)

    assert get_n_workers_from_arg(requested) == expected


@pytest.mark.parametrize("injected", [False, True])
def test_udf_entrypoint_restores_only_injected_safe_path(monkeypatch, mocker, injected):
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    if injected:
        monkeypatch.setenv("_DATACHAIN_UDF_SAFE_PATH", "1")
    else:
        monkeypatch.delenv("_DATACHAIN_UDF_SAFE_PATH", raising=False)
    mocker.patch("datachain.query.dispatch.load", side_effect=RuntimeError("stop"))
    with pytest.raises(RuntimeError, match="stop"):
        udf_entrypoint()
    assert ("PYTHONSAFEPATH" in os.environ) is not injected
    assert "_DATACHAIN_UDF_SAFE_PATH" not in os.environ


def test_one_process_dispatcher_prefetches_into_a_temporary_cache(
    test_session_tmpfile, tmp_path, monkeypatch
):
    files = []
    for name in "ab":
        (tmp_path / name).write_bytes(name.encode() * 2)
        files.append(File(source=tmp_path.as_uri(), path=name, size=2))

    class Popen:
        """Runs the dispatcher of a parallel UDF here, instead of in a child."""

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def communicate(self, data):
            opened = []
            connect = sqlite3.connect

            def track(*args, **kwargs):
                # closable here, though the input reader opens one in its thread
                opened.append(connect(*args, **kwargs, check_same_thread=False))
                return opened[-1]

            with monkeypatch.context() as m:
                m.setattr(dispatch, "stdin", SimpleNamespace(buffer=BytesIO(data)))
                m.setattr(sqlite3, "connect", track)
                try:
                    udf_entrypoint()
                finally:
                    for db in opened:  # as a child process does on exit
                        db.close()

        def poll(self):
            return 0

    monkeypatch.setattr("datachain.query.dataset.subprocess.Popen", Popen)
    monkeypatch.setattr(dispatch, "get_n_workers_from_arg", lambda n: 1)
    rows = (
        dc.read_values(file=files, session=test_session_tmpfile)
        .settings(prefetch=1, cache=False, parallel=2)
        .map(
            lambda file: (file.read(), file.get_local_path()),
            output={"data": bytes, "path": str},
        )
        .order_by("sys.id")
        .to_list("data", "path")
    )

    cache = test_session_tmpfile.catalog.cache
    assert [data for data, _ in rows] == [b"aa", b"bb"]
    for _, path in rows:
        assert path.startswith(os.path.join(cache.tmp_dir, "prefetch-"))
        assert not os.path.exists(path)
    assert not any(cache.contains(file) for file in files)
