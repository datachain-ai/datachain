import asyncio
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from unittest.mock import AsyncMock, MagicMock

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
    f.source = "s3://bucket"
    f.path = "bucket/key.bin"
    f.size = 100
    f._prefetch = AsyncMock(return_value=True)
    return f


def _register(row, refcounts: Counter[str], lock: Lock) -> None:
    asyncio.run(_prefetch_input(row, refcounts=refcounts, refcounts_lock=lock))


def _run_concurrently(*fns: Callable[[], object]) -> None:
    barrier = threading.Barrier(len(fns))

    def run(fn: Callable[[], object]) -> None:
        barrier.wait(timeout=5)
        fn()

    pool = ThreadPoolExecutor(len(fns))
    try:
        for future in [pool.submit(run, fn) for fn in fns]:
            future.result(timeout=5)
    finally:
        pool.shutdown(wait=False)


def test_remove_prefetched_refcount_delays_removal():
    catalog = _make_catalog()
    h = "abc123"
    file1 = _make_file(catalog, h)
    file2 = _make_file(catalog, h)

    refcounts: Counter[str] = Counter({h: 2})
    lock = Lock()

    _remove_prefetched((file1,), refcounts=refcounts, refcounts_lock=lock)
    catalog.cache.remove.assert_not_called()
    assert refcounts[h] == 1

    _remove_prefetched((file2,), refcounts=refcounts, refcounts_lock=lock)
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

    _remove_prefetched((file_a, file_b), refcounts=refcounts, refcounts_lock=lock)

    assert catalog.cache.remove.call_count == 2
    assert hash_a not in refcounts
    assert hash_b not in refcounts


def test_n_concurrent_removals_call_cache_remove_exactly_once():
    n = 8
    for _ in range(5):
        catalog = _make_catalog()
        h = "shared"
        refcounts: Counter[str] = Counter({h: n})
        lock = Lock()
        files = [_make_file(catalog, h) for _ in range(n)]

        _run_concurrently(
            *(
                partial(
                    _remove_prefetched, (f,), refcounts=refcounts, refcounts_lock=lock
                )
                for f in files
            )
        )

        catalog.cache.remove.assert_called_once()
        assert h not in refcounts


def test_new_consumer_added_while_existing_consumer_finishes():
    for _ in range(10):
        catalog = _make_catalog()
        h = "shared"
        finishing = _make_file(catalog, h)
        arriving = _make_file(catalog, h)
        refcounts: Counter[str] = Counter({h: 2})
        lock = Lock()

        _run_concurrently(
            partial(
                _remove_prefetched,
                (finishing,),
                refcounts=refcounts,
                refcounts_lock=lock,
            ),
            partial(_register, (arriving,), refcounts, lock),
        )

        assert refcounts[h] == 2
        catalog.cache.remove.assert_not_called()


def test_refcount_incremented_before_prefetch_is_awaited():
    catalog = _make_catalog()
    h = "shared"
    refcounts: Counter[str] = Counter()
    lock = Lock()
    seen: list[int] = []

    f = _make_file(catalog, h)

    async def spy_prefetch(download_cb=None):
        seen.append(refcounts[h])
        return True

    f._prefetch = spy_prefetch

    _register((f,), refcounts, lock)

    assert seen == [1]


def test_file_error_row_still_returned_refcount_balanced():
    catalog = _make_catalog()
    h = "shared"
    f = _make_file(catalog, h)
    f._prefetch = AsyncMock(side_effect=FileError("timeout", f.source, f.path))
    refcounts: Counter[str] = Counter()
    lock = Lock()

    row = asyncio.run(_prefetch_input((f,), refcounts=refcounts, refcounts_lock=lock))

    assert row == (f,)
    assert refcounts[h] == 1

    _remove_prefetched(row, refcounts=refcounts, refcounts_lock=lock)

    assert h not in refcounts
    # The reference was taken before the failed fetch, so its release still removes.
    catalog.cache.remove.assert_called_once_with(f)


def test_file_error_shared_file_not_deleted_for_other_consumer():
    catalog = _make_catalog()
    h = "shared"
    file_m = _make_file(catalog, h)
    file_n = _make_file(catalog, h)
    file_n._prefetch = AsyncMock(
        side_effect=FileError("timeout", file_n.source, file_n.path)
    )
    refcounts: Counter[str] = Counter()
    lock = Lock()

    _register((file_m,), refcounts, lock)
    _register((file_n,), refcounts, lock)
    assert refcounts[h] == 2

    _remove_prefetched((file_m,), refcounts=refcounts, refcounts_lock=lock)
    catalog.cache.remove.assert_not_called()
    assert refcounts[h] == 1

    _remove_prefetched((file_n,), refcounts=refcounts, refcounts_lock=lock)
    catalog.cache.remove.assert_called_once()
    assert h not in refcounts


def test_non_file_error_propagates_with_pre_increment():
    catalog = _make_catalog()
    h = "shared"
    f = _make_file(catalog, h)
    f._prefetch = AsyncMock(side_effect=RuntimeError("unexpected storage error"))
    refcounts: Counter[str] = Counter()
    lock = Lock()

    with pytest.raises(RuntimeError, match="unexpected storage error"):
        _register((f,), refcounts, lock)

    assert refcounts[h] == 1


def test_cache_remove_held_inside_lock():
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
        # _remove_prefetched swallows exceptions, so failures are recorded instead.
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
    _remove_prefetched((f,), refcounts=refcounts, refcounts_lock=lock)
    worker.join(timeout=5.0)

    assert not worker.is_alive(), "worker thread did not complete within 5 s"
    assert not timeout_failures, f"event synchronization timed out: {timeout_failures}"
    assert lock_was_held == [True], "cache.remove must run while the lock is held"
    assert order == ["remove", "increment"]
    assert refcounts[h] == 1


def test_batch_column_files_counted_per_occurrence():
    catalog = _make_catalog()
    h = "shared"
    batch = [_make_file(catalog, h), _make_file(catalog, h)]
    other_row = (_make_file(catalog, h),)
    refcounts: Counter[str] = Counter()
    lock = Lock()

    asyncio.run(
        _prefetch_input(
            (batch,), batch_columns={0}, refcounts=refcounts, refcounts_lock=lock
        )
    )
    _register(other_row, refcounts, lock)
    assert refcounts[h] == 3

    _remove_prefetched((batch,), {0}, refcounts=refcounts, refcounts_lock=lock)
    catalog.cache.remove.assert_not_called()
    assert refcounts[h] == 1

    _remove_prefetched(other_row, refcounts=refcounts, refcounts_lock=lock)
    catalog.cache.remove.assert_called_once()
    assert h not in refcounts


@pytest.mark.parametrize(
    "path,size",
    [("bucket/key.bin", 0), ("bucket/key.bin", 200), ("", 30)],
    ids=["zero-size", "oversized", "no-path"],
)
def test_ineligible_file_does_not_release_prefetched_shared_hash(path, size):
    catalog = _make_catalog()
    h = "shared"
    max_size = 50
    eligible = _make_file(catalog, h)
    eligible.size = 30
    ineligible = _make_file(catalog, h)
    ineligible.path = path
    ineligible.size = size
    refcounts: Counter[str] = Counter()
    lock = Lock()

    for f in (eligible, ineligible):
        asyncio.run(
            _prefetch_input(
                (f,), max_size=max_size, refcounts=refcounts, refcounts_lock=lock
            )
        )
    ineligible._prefetch.assert_not_called()
    assert refcounts[h] == 1

    _remove_prefetched(
        (ineligible,), refcounts=refcounts, refcounts_lock=lock, max_size=max_size
    )
    catalog.cache.remove.assert_not_called()
    assert refcounts[h] == 1

    _remove_prefetched(
        (eligible,), refcounts=refcounts, refcounts_lock=lock, max_size=max_size
    )
    catalog.cache.remove.assert_called_once_with(eligible)
    assert h not in refcounts
