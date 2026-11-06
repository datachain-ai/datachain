import asyncio
import threading
from collections import Counter
from threading import Lock
from unittest.mock import MagicMock

import pytest

from datachain.lib.file import File, FileError
from datachain.lib.udf import _prefetch_input, _remove_prefetched


def _make_catalog():
    catalog = MagicMock()
    catalog.cache = MagicMock()
    return catalog


def _make_file(catalog, file_hash: str) -> MagicMock:
    f = MagicMock(spec=File)
    f._catalog = catalog
    f.get_hash.return_value = file_hash
    f.path = "bucket/key.bin"
    f.size = 100
    return f


def _make_prefetchable_file(catalog, file_hash: str) -> MagicMock:
    f = _make_file(catalog, file_hash)
    f.path = "bucket/key.bin"
    f.size = 128
    f.source = "s3://bucket"
    return f


# Basic refcount behaviour


def test_remove_prefetched_refcount_delays_removal():
    catalog = _make_catalog()
    h = "abc123"
    file1 = _make_file(catalog, h)
    file2 = _make_file(catalog, h)

    refcounts: Counter[str] = Counter({h: 2})
    lock = Lock()

    _remove_prefetched((file1,), refcounts, lock)
    catalog.cache.remove.assert_not_called()
    assert refcounts[h] == 1

    _remove_prefetched((file2,), refcounts, lock)
    catalog.cache.remove.assert_called_once_with(file2)
    assert h not in refcounts


def test_remove_prefetched_no_refcounts_removes_immediately():
    catalog = _make_catalog()
    file1 = _make_file(catalog, "any")

    _remove_prefetched((file1,), refcounts=None, refcounts_lock=None)
    catalog.cache.remove.assert_called_once_with(file1)


def test_remove_prefetched_distinct_files_cleaned_up_independently():
    catalog = _make_catalog()
    hash_a, hash_b = "hash_a", "hash_b"
    file_a = _make_file(catalog, hash_a)
    file_b = _make_file(catalog, hash_b)

    refcounts: Counter[str] = Counter({hash_a: 1, hash_b: 1})
    lock = Lock()

    _remove_prefetched((file_a, file_b), refcounts, lock)

    assert catalog.cache.remove.call_count == 2
    assert hash_a not in refcounts
    assert hash_b not in refcounts


# Concurrency: deterministic lock-correctness tests


def test_n_concurrent_removals_call_cache_remove_exactly_once():
    N = 8

    for _trial in range(5):
        catalog = _make_catalog()
        h = "shared"
        files = [_make_file(catalog, h) for _ in range(N)]
        refcounts: Counter[str] = Counter({h: N})
        lock = Lock()
        barrier = threading.Barrier(N)
        errors: list[str] = []

        def do_remove(f):
            try:
                barrier.wait(timeout=5.0)
                _remove_prefetched((f,), refcounts, lock)
            except Exception as e:
                errors.append(f"trial {_trial}: {e}")

        threads = [threading.Thread(target=do_remove, args=(f,)) for f in files]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        for i, t in enumerate(threads):
            assert not t.is_alive(), f"Thread {i} did not terminate (trial {_trial})"
        assert not errors, f"Thread errors: {errors}"
        assert catalog.cache.remove.call_count == 1, (
            f"Expected 1 cache.remove call, got {catalog.cache.remove.call_count}"
        )
        assert h not in refcounts


def test_new_consumer_added_while_existing_consumer_finishes():
    TRIALS = 10

    for _trial in range(TRIALS):
        catalog = _make_catalog()
        h = "shared"
        file_a = _make_file(catalog, h)
        refcounts: Counter[str] = Counter({h: 2})
        lock = Lock()
        barrier = threading.Barrier(2)
        errors: list[str] = []

        def do_remove():
            try:
                barrier.wait(timeout=5.0)
                _remove_prefetched((file_a,), refcounts, lock)
            except Exception as e:
                errors.append(f"do_remove trial {_trial}: {e}")

        def do_increment():
            try:
                barrier.wait(timeout=5.0)
                with lock:
                    refcounts[h] += 1
            except Exception as e:
                errors.append(f"do_increment trial {_trial}: {e}")

        t1 = threading.Thread(target=do_remove)
        t2 = threading.Thread(target=do_increment)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not t1.is_alive(), f"t1 did not terminate (trial {_trial})"
        assert not t2.is_alive(), f"t2 did not terminate (trial {_trial})"
        assert not errors, f"Thread errors: {errors}"
        assert refcounts[h] == 2, f"Trial {_trial}: expected 2, got {refcounts[h]}"
        catalog.cache.remove.assert_not_called()


# Pre-increment: refcount raised BEFORE _prefetch() is awaited


def test_refcount_incremented_before_prefetch_is_awaited():
    catalog = _make_catalog()
    h = "shared"
    refcounts: Counter[str] = Counter()
    lock = Lock()
    seen: list[int] = []

    f = _make_file(catalog, h)
    f.path = "some/path.txt"
    f.size = 100

    async def spy_prefetch(download_cb=None):
        seen.append(refcounts[h])
        return True

    f._prefetch = spy_prefetch

    asyncio.run(_prefetch_input((f,), refcounts=refcounts, refcounts_lock=lock))

    assert seen == [1], f"Expected refcount=1 at prefetch call, got {seen}"


# Failure-path accounting


def test_file_error_row_still_returned_refcount_balanced():
    catalog = _make_catalog()
    h = "shared"
    f = _make_prefetchable_file(catalog, h)
    refcounts: Counter[str] = Counter()
    lock = Lock()

    async def prefetch_raises(download_cb=None):
        raise FileError("timeout", f.source, f.path)

    f._prefetch = prefetch_raises

    row = asyncio.run(
        _prefetch_input((f,), download_cb=None, refcounts=refcounts, refcounts_lock=lock)
    )

    assert row == (f,)
    assert refcounts[h] == 1

    _remove_prefetched(row, refcounts, lock)

    assert h not in refcounts
    # cache.remove called even on failed prefetch; it's a no-op when the file wasn't downloaded.
    catalog.cache.remove.assert_called_once_with(f)


def test_file_error_shared_file_not_deleted_for_other_consumer():
    catalog = _make_catalog()
    h = "shared"
    file_m = _make_prefetchable_file(catalog, h)
    file_n = _make_prefetchable_file(catalog, h)
    refcounts: Counter[str] = Counter()
    lock = Lock()

    async def ok(download_cb=None):
        return True

    async def fails(download_cb=None):
        raise FileError("timeout", file_n.source, file_n.path)

    file_m._prefetch = ok
    file_n._prefetch = fails

    asyncio.run(_prefetch_input((file_m,), refcounts=refcounts, refcounts_lock=lock))
    asyncio.run(_prefetch_input((file_n,), refcounts=refcounts, refcounts_lock=lock))
    assert refcounts[h] == 2

    _remove_prefetched((file_m,), refcounts, lock)
    catalog.cache.remove.assert_not_called()
    assert refcounts[h] == 1

    _remove_prefetched((file_n,), refcounts, lock)
    catalog.cache.remove.assert_called_once()
    assert h not in refcounts


def test_non_file_error_propagates_with_pre_increment():
    catalog = _make_catalog()
    h = "shared"
    f = _make_prefetchable_file(catalog, h)
    refcounts: Counter[str] = Counter()
    lock = Lock()

    async def crashes(download_cb=None):
        raise RuntimeError("unexpected storage error")

    f._prefetch = crashes

    with pytest.raises(RuntimeError, match="unexpected storage error"):
        asyncio.run(
            _prefetch_input(
                (f,), download_cb=None, refcounts=refcounts, refcounts_lock=lock
            )
        )

    assert refcounts[h] == 1


# Post-zero-decrement race: new increment serialized with cache.remove


def test_cache_remove_held_inside_lock():
    """cache.remove must execute while refcounts_lock is held.

    Verified properties:
    1. lock is held when cache.remove executes (worker's non-blocking acquire fails).
    2. removal completes before the consumer can register an increment.
    3. consumer successfully registers after removal (refcount == 1).
    """
    catalog = _make_catalog()
    h = "shared"
    f = _make_file(catalog, h)
    refcounts: Counter[str] = Counter({h: 1})
    lock = Lock()

    remove_entered = threading.Event()
    worker_checked = threading.Event()

    lock_was_held: list[bool] = []
    order: list[str] = []
    timeout_failures: list[str] = []

    def spy_remove(obj):
        remove_entered.set()
        if not worker_checked.wait(timeout=5.0):
            timeout_failures.append("worker_checked timed out in spy_remove")
        order.append("remove")

    catalog.cache.remove.side_effect = spy_remove

    def new_worker():
        if not remove_entered.wait(timeout=5.0):
            timeout_failures.append("remove_entered timed out in new_worker")
            return
        acquired = lock.acquire(blocking=False)
        lock_was_held.append(not acquired)
        if acquired:
            lock.release()
        worker_checked.set()
        with lock:
            refcounts[h] += 1
        order.append("increment")

    worker = threading.Thread(target=new_worker)
    worker.start()
    _remove_prefetched((f,), refcounts, lock)
    worker.join(timeout=5.0)

    assert not worker.is_alive(), "worker thread did not complete within 5 s"
    assert not timeout_failures, f"event synchronization timed out: {timeout_failures}"
    assert lock_was_held == [True], (
        "lock must be held during cache.remove to prevent concurrent increment on a "
        "file being deleted"
    )
    assert order == ["remove", "increment"], (
        f"removal must complete before increment registers: got {order}"
    )
    assert refcounts[h] == 1, "consumer must successfully register after removal"
