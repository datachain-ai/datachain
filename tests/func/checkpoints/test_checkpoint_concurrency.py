import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import datachain as dc
from datachain.catalog import Catalog
from datachain.query.session import Session
from tests.utils import reset_session_job_state


def clone_session(session: Session) -> Session:
    """Create a new session with cloned metastore and warehouse.

    Needed because SQLite connections aren't shareable across threads;
    PostgreSQL / ClickHouse also benefit from per-thread connections.
    """
    catalog = session.catalog
    thread_metastore = catalog.metastore.clone(use_new_connection=True)
    thread_warehouse = catalog.warehouse.clone(use_new_connection=True)
    thread_catalog = Catalog(metastore=thread_metastore, warehouse=thread_warehouse)
    return Session("TestSession", catalog=thread_catalog)


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


def test_checkpoints_created_in_thread(test_session_tmpfile):
    """Per-chain hashing is deterministic, so threads can create checkpoints."""
    test_session = test_session_tmpfile
    metastore = test_session.catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    reset_session_job_state()

    def thread_work():
        thread_session = clone_session(test_session)
        try:
            dc.read_dataset("nums", session=thread_session).save("result")
        finally:
            thread_session.catalog.close()

    thread = threading.Thread(target=thread_work)
    thread.start()
    thread.join()

    job = test_session.get_or_create_job()
    assert len(list(metastore.list_checkpoints([job.id]))) > 0


def test_checkpoints_created_in_thread_pool(test_session_tmpfile):
    test_session = test_session_tmpfile
    metastore = test_session.catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    reset_session_job_state()

    def worker(i):
        thread_session = clone_session(test_session)
        try:
            dc.read_dataset("nums", session=thread_session).save(f"result_{i}")
        finally:
            thread_session.catalog.close()

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(worker, range(3)))

    job = test_session.get_or_create_job()
    assert len(list(metastore.list_checkpoints([job.id]))) > 0
