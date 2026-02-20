import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import datachain as dc
from datachain.catalog import Catalog
from datachain.query.session import Session
from tests.utils import reset_session_job_state


def clone_session(session: Session) -> Session:
    """
    Create a new session with cloned metastore and warehouse for thread-safe access.

    This is needed for tests that run DataChain operations in threads, as SQLite
    connections cannot be shared across threads. For other databases (PostgreSQL,
    Clickhouse), cloning ensures each thread has its own connection.

    Args:
        session: The session to clone catalog from.

    Returns:
        Session: A new session with cloned catalog components.
    """
    catalog = session.catalog
    thread_metastore = catalog.metastore.clone()
    thread_warehouse = catalog.warehouse.clone()
    thread_catalog = Catalog(metastore=thread_metastore, warehouse=thread_warehouse)
    return Session("TestSession", catalog=thread_catalog)


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


def test_threading_disables_checkpoints(test_session_tmpfile, caplog):
    test_session = test_session_tmpfile
    metastore = test_session.catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN (main thread) -------------------
    reset_session_job_state()

    dc.read_dataset("nums", session=test_session).save("result1")

    job1 = test_session.get_or_create_job()
    checkpoints_main = list(metastore.list_checkpoints(job1.id))
    assert len(checkpoints_main) > 0, "Checkpoint should be created in main thread"

    # -------------- SECOND RUN (in thread) -------------------
    reset_session_job_state()

    thread_ran = {"value": False}

    def run_datachain_in_thread():
        """Run DataChain operation in a thread - checkpoint should NOT be created."""
        thread_session = clone_session(test_session)
        try:
            thread_ran["value"] = True
            dc.read_dataset("nums", session=thread_session).save("result2")
        finally:
            thread_session.catalog.close()

    thread = threading.Thread(target=run_datachain_in_thread)
    thread.start()
    thread.join()

    assert thread_ran["value"] is True

    assert any(
        "Concurrent thread detected" in record.message for record in caplog.records
    ), "Warning about concurrent thread should be logged"

    # Verify no checkpoint was created in thread
    job2 = test_session.get_or_create_job()
    checkpoints_thread = list(metastore.list_checkpoints(job2.id))
    assert len(checkpoints_thread) == 0, "No checkpoints should be created in thread"


def test_threading_with_executor(test_session_tmpfile, caplog):
    test_session = test_session_tmpfile
    metastore = test_session.catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN (main thread) -------------------
    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).save("before_threading")

    job1 = test_session.get_or_create_job()
    checkpoints_before = len(list(metastore.list_checkpoints(job1.id)))
    assert checkpoints_before > 0, "Checkpoint should be created before threading"

    # -------------- SECOND RUN (in thread pool) -------------------
    reset_session_job_state()

    def worker(i):
        """Worker function that runs DataChain operations in thread pool."""
        thread_session = clone_session(test_session)
        try:
            dc.read_dataset("nums", session=thread_session).save(f"result_{i}")
        finally:
            thread_session.catalog.close()

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(worker, range(3)))

    assert any(
        "Concurrent thread detected" in record.message for record in caplog.records
    ), "Warning should be logged when using thread pool"

    job2 = test_session.get_or_create_job()
    checkpoints_after = len(list(metastore.list_checkpoints(job2.id)))
    assert checkpoints_after == 0, "No checkpoints should be created in thread pool"


def test_multiprocessing_disables_checkpoints(test_session, monkeypatch):
    catalog = test_session.catalog
    metastore = catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN (main process) -------------------
    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).save("main_result")

    job1 = test_session.get_or_create_job()
    checkpoints_main = list(metastore.list_checkpoints(job1.id))
    assert len(checkpoints_main) > 0, "Checkpoint should be created in main process"

    # -------------- SECOND RUN (simulated subprocess) -------------------
    reset_session_job_state()

    # Simulate being in a subprocess by setting DATACHAIN_MAIN_PROCESS_PID
    # to a different PID than the current one
    monkeypatch.setenv("DATACHAIN_MAIN_PROCESS_PID", str(os.getpid() + 1000))

    # Run DataChain operation - checkpoint should NOT be created
    dc.read_dataset("nums", session=test_session).save("subprocess_result")

    job2 = test_session.get_or_create_job()
    checkpoints_subprocess = list(metastore.list_checkpoints(job2.id))
    assert len(checkpoints_subprocess) == 0, (
        "No checkpoints should be created in subprocess"
    )


def test_checkpoint_reuse_after_threading(test_session_tmpfile):
    test_session = test_session_tmpfile
    metastore = test_session.catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN (creates checkpoints) -------------------
    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).save("result1")
    dc.read_dataset("nums", session=test_session).save("result2")

    job1 = test_session.get_or_create_job()
    checkpoints_initial = len(list(metastore.list_checkpoints(job1.id)))
    assert checkpoints_initial > 0, "Checkpoints should be created initially"

    # Run something in a thread (disables checkpoints globally)
    def thread_work():
        thread_session = clone_session(test_session)
        try:
            dc.read_dataset("nums", session=thread_session).save("thread_result")
        finally:
            thread_session.catalog.close()

    thread = threading.Thread(target=thread_work)
    thread.start()
    thread.join()

    # No new checkpoints should have been created in thread
    assert len(list(metastore.list_checkpoints(job1.id))) == checkpoints_initial

    # -------------- SECOND RUN (new job, after threading) -------------------
    reset_session_job_state()
    dc.read_dataset("nums", session=test_session).save("new_result")

    job2 = test_session.get_or_create_job()
    checkpoints_new_job = list(metastore.list_checkpoints(job2.id))
    assert len(checkpoints_new_job) > 0, "New job should create checkpoints normally"


def test_warning_shown_once(test_session_tmpfile, caplog):
    test_session = test_session_tmpfile

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    reset_session_job_state()

    def run_multiple_operations():
        """Run multiple DataChain operations in a thread."""
        thread_session = clone_session(test_session)
        try:
            # Each operation would check checkpoints_enabled()
            dc.read_dataset("nums", session=thread_session).save("result1")
            dc.read_dataset("nums", session=thread_session).save("result2")
            dc.read_dataset("nums", session=thread_session).save("result3")
        finally:
            thread_session.catalog.close()

    thread = threading.Thread(target=run_multiple_operations)
    thread.start()
    thread.join()

    warning_count = sum(
        1 for record in caplog.records if "Concurrent thread detected" in record.message
    )

    assert warning_count == 1, "Warning should be shown only once per process"
