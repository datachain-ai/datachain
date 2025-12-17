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
    """Test that checkpoints are disabled when threading is detected."""
    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create initial dataset
    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    reset_session_job_state()
    job = test_session.get_or_create_job()

    # Initially, no checkpoints should exist
    assert len(list(metastore.list_checkpoints(job.id))) == 0

    # Create a checkpoint in the main thread (should work)
    checkpoint1 = metastore.get_or_create_checkpoint(job.id, "hash1", partial=False)
    assert checkpoint1 is not None
    assert checkpoint1.hash == "hash1"

    # Verify checkpoint was created
    assert len(list(metastore.list_checkpoints(job.id))) == 1

    # Track whether thread ran
    thread_ran = {"value": False}
    checkpoint_in_thread = {"value": None}

    def create_checkpoint_in_thread():
        """Try to create checkpoint from a thread."""
        thread_ran["value"] = True
        # This should return None because threading is detected
        checkpoint_in_thread["value"] = metastore.get_or_create_checkpoint(
            job.id, "hash2", partial=False
        )

    # Create a thread and run checkpoint creation
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

    # Verify find_checkpoint also returns None after threading detected
    found = metastore.find_checkpoint(job.id, "hash1", partial=False)
    assert found is None  # Should be disabled now


def test_threading_with_executor(test_session, caplog):
    """Test checkpoint disabling with ThreadPoolExecutor."""
    catalog = test_session.catalog
    metastore = catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    reset_session_job_state()
    job = test_session.get_or_create_job()

    # Create checkpoint before threading
    checkpoint1 = metastore.get_or_create_checkpoint(
        job.id, "hash_before", partial=False
    )
    assert checkpoint1 is not None

    def worker(i):
        """Worker function that tries to create checkpoints."""
        return metastore.get_or_create_checkpoint(job.id, f"hash_{i}", partial=False)

    # Use ThreadPoolExecutor to create multiple threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(worker, range(3)))

    # All checkpoint creations in threads should return None
    assert all(r is None for r in results)

    # Verify warning was logged
    assert any(
        "Concurrent execution detected" in record.message for record in caplog.records
    )

    # Verify only the first checkpoint exists
    assert len(list(metastore.list_checkpoints(job.id))) == 1


def test_multiprocessing_disables_checkpoints(test_session, monkeypatch):
    """Test that checkpoints are disabled when multiprocessing is detected."""
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
        "datachain.query.session.multiprocessing.current_process",
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
    """Test that checkpoints created before threading can be reused in next run."""
    catalog = test_session.catalog
    metastore = catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    job1 = test_session.get_or_create_job()

    # Create some checkpoints before threading
    checkpoint1 = metastore.get_or_create_checkpoint(job1.id, "hash_A", partial=False)
    checkpoint2 = metastore.get_or_create_checkpoint(job1.id, "hash_B", partial=False)
    assert checkpoint1 is not None
    assert checkpoint2 is not None

    # Verify both checkpoints exist
    assert len(list(metastore.list_checkpoints(job1.id))) == 2

    # Now use threading - should disable checkpoints from this point
    def thread_work():
        # Try to create another checkpoint
        return metastore.get_or_create_checkpoint(job1.id, "hash_C", partial=False)

    thread = threading.Thread(target=thread_work)
    thread.start()
    thread.join()

    # Still only 2 checkpoints (hash_C was not created)
    assert len(list(metastore.list_checkpoints(job1.id))) == 2

    # -------------- SECOND RUN (new job) -------------------
    reset_session_job_state()
    job2 = test_session.get_or_create_job()

    # In new run, should be able to create checkpoints again
    checkpoint_new = metastore.get_or_create_checkpoint(
        job2.id, "hash_D", partial=False
    )
    assert checkpoint_new is not None

    # Verify new checkpoint was created in new job
    assert len(list(metastore.list_checkpoints(job2.id))) == 1


def test_warning_shown_once(test_session, caplog):
    """Test that threading warning is only shown once."""
    catalog = test_session.catalog
    metastore = catalog.metastore

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")
    reset_session_job_state()
    job = test_session.get_or_create_job()

    def create_checkpoints():
        """Try to create multiple checkpoints."""
        metastore.get_or_create_checkpoint(job.id, "h1", partial=False)
        metastore.get_or_create_checkpoint(job.id, "h2", partial=False)
        metastore.find_checkpoint(job.id, "h3", partial=False)

    # Run in thread
    thread = threading.Thread(target=create_checkpoints)
    thread.start()
    thread.join()

    # Count how many times the warning appeared
    warning_count = sum(
        1
        for record in caplog.records
        if "Concurrent execution detected" in record.message
    )

    # Should only appear once
    assert warning_count == 1
