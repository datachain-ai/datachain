"""Tests for checkpoint behavior with threading and multiprocessing.

This module tests that checkpoints are properly disabled when Python threading
or multiprocessing is detected, preventing race conditions and non-deterministic
hash calculations.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import datachain as dc
from tests.utils import reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


def test_threading_disables_checkpoints(test_session, caplog):
    catalog = test_session.catalog
    metastore = catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    reset_session_job_state()
    job = test_session.get_or_create_job()

    assert len(list(metastore.list_checkpoints(job.id))) == 0

    # Create a checkpoint in the main thread (should work)
    checkpoint1 = metastore.get_or_create_checkpoint(job.id, "hash1", partial=False)
    assert checkpoint1 is not None
    assert checkpoint1.hash == "hash1"

    assert len(list(metastore.list_checkpoints(job.id))) == 1

    thread_ran = {"value": False}
    checkpoint_in_thread = {"value": None}

    def create_checkpoint_in_thread():
        thread_ran["value"] = True
        checkpoint_in_thread["value"] = metastore.get_or_create_checkpoint(
            job.id, "hash2", partial=False
        )

    thread = threading.Thread(target=create_checkpoint_in_thread)
    thread.start()
    thread.join()

    # Verify thread ran
    assert thread_ran["value"] is True

    # Verify checkpoint creation returned None in thread
    assert checkpoint_in_thread["value"] is None

    # Verify warning was logged
    assert any(
        "Concurrent execution detected" in record.message for record in caplog.records
    )

    # Verify no new checkpoint was created (still just 1)
    assert len(list(metastore.list_checkpoints(job.id))) == 1

    found = metastore.find_checkpoint(job.id, "hash1", partial=False)
    assert found is None  # Should be disabled now


def test_threading_with_executor(test_session, caplog):
    """Test checkpoint disabling with ThreadPoolExecutor."""
    catalog = test_session.catalog
    metastore = catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    reset_session_job_state()
    job = test_session.get_or_create_job()

    checkpoint1 = metastore.get_or_create_checkpoint(
        job.id, "hash_before", partial=False
    )
    assert checkpoint1 is not None

    def worker(i):
        return metastore.get_or_create_checkpoint(job.id, f"hash_{i}", partial=False)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(worker, range(3)))

    # All checkpoint creations in threads should return None
    assert all(r is None for r in results)

    assert any(
        "Concurrent execution detected" in record.message for record in caplog.records
    )

    assert len(list(metastore.list_checkpoints(job.id))) == 1


def test_multiprocessing_disables_checkpoints(test_session, monkeypatch):
    catalog = test_session.catalog
    metastore = catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    reset_session_job_state()
    job = test_session.get_or_create_job()

    # Create checkpoint in main process (should work)
    checkpoint1 = metastore.get_or_create_checkpoint(job.id, "hash_main", partial=False)
    assert checkpoint1 is not None

    # Simulate being in a subprocess by mocking current_process().name
    class MockProcess:
        name = "SpawnProcess-1"  # Not "MainProcess"

    monkeypatch.setattr(
        "datachain.utils.multiprocessing.current_process",
        lambda: MockProcess(),
    )

    # Try to create checkpoint - should return None because we're "in a subprocess"
    checkpoint2 = metastore.get_or_create_checkpoint(
        job.id, "hash_subprocess", partial=False
    )
    assert checkpoint2 is None

    # Verify only the main process checkpoint exists
    assert len(list(metastore.list_checkpoints(job.id))) == 1


def test_checkpoint_reuse_after_threading(test_session):
    catalog = test_session.catalog
    metastore = catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    job1 = test_session.get_or_create_job()

    checkpoint1 = metastore.get_or_create_checkpoint(job1.id, "hash_A", partial=False)
    checkpoint2 = metastore.get_or_create_checkpoint(job1.id, "hash_B", partial=False)
    assert checkpoint1 is not None
    assert checkpoint2 is not None

    assert len(list(metastore.list_checkpoints(job1.id))) == 2

    def thread_work():
        return metastore.get_or_create_checkpoint(job1.id, "hash_C", partial=False)

    thread = threading.Thread(target=thread_work)
    thread.start()
    thread.join()

    assert len(list(metastore.list_checkpoints(job1.id))) == 2

    # -------------- SECOND RUN (new job) -------------------
    reset_session_job_state()
    job2 = test_session.get_or_create_job()

    checkpoint_new = metastore.get_or_create_checkpoint(
        job2.id, "hash_D", partial=False
    )
    assert checkpoint_new is not None

    assert len(list(metastore.list_checkpoints(job2.id))) == 1


def test_warning_shown_once(test_session, caplog):
    catalog = test_session.catalog
    metastore = catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    reset_session_job_state()
    job = test_session.get_or_create_job()

    def create_checkpoints():
        metastore.get_or_create_checkpoint(job.id, "h1", partial=False)
        metastore.get_or_create_checkpoint(job.id, "h2", partial=False)
        metastore.find_checkpoint(job.id, "h3", partial=False)

    thread = threading.Thread(target=create_checkpoints)
    thread.start()
    thread.join()

    warning_count = sum(
        1
        for record in caplog.records
        if "Concurrent execution detected" in record.message
    )

    assert warning_count == 1
