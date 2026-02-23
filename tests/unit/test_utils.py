import os
import threading
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from datachain.fs.utils import is_subpath
from datachain.utils import (
    _CheckpointState,
    batched,
    batched_it,
    checkpoints_enabled,
    datachain_paths_join,
    determine_processes,
    determine_workers,
    interprocess_file_lock,
    nested_dict_path_set,
    retry_with_backoff,
    row_to_nested_dict,
    sizeof_fmt,
    sql_escape_like,
    suffix_to_number,
    uses_glob,
    with_last_flag,
)

DATACHAIN_TEST_PATHS = ["/file1", "file2", "/dir/file3", "dir/file4"]
DATACHAIN_EX_ROOT = ["/file1", "/file2", "/dir/file3", "/dir/file4"]
DATACHAIN_EX_SUBDIR = [
    "subdir/file1",
    "subdir/file2",
    "subdir/dir/file3",
    "subdir/dir/file4",
]
DATACHAIN_EX_DOUBLE_SUBDIR = [
    "subdir/double/file1",
    "subdir/double/file2",
    "subdir/double/dir/file3",
    "subdir/double/dir/file4",
]


@pytest.mark.parametrize(
    "src,paths,expected",
    (
        ("", DATACHAIN_TEST_PATHS, DATACHAIN_EX_ROOT),
        ("/", DATACHAIN_TEST_PATHS, DATACHAIN_EX_ROOT),
        ("/*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_ROOT),
        ("/file*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_ROOT),
        ("subdir", DATACHAIN_TEST_PATHS, DATACHAIN_EX_SUBDIR),
        ("subdir/", DATACHAIN_TEST_PATHS, DATACHAIN_EX_SUBDIR),
        ("subdir/*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_SUBDIR),
        ("subdir/file*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_SUBDIR),
        ("subdir/double", DATACHAIN_TEST_PATHS, DATACHAIN_EX_DOUBLE_SUBDIR),
        ("subdir/double/", DATACHAIN_TEST_PATHS, DATACHAIN_EX_DOUBLE_SUBDIR),
        ("subdir/double/*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_DOUBLE_SUBDIR),
        ("subdir/double/file*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_DOUBLE_SUBDIR),
    ),
)
def test_datachain_paths_join(src, paths, expected):
    assert list(datachain_paths_join(src, paths)) == expected


@pytest.mark.parametrize(
    "num,suffix,si,expected",
    (
        (1, "", False, "   1"),
        (536, "", False, " 536"),
        (1000, "", False, "1000"),
        (1000, "", True, "1.0K"),
        (1000, " tests", False, "1000 tests"),
        (1000, " tests", True, "1.0K tests"),
        (100000, "", False, "97.7K"),
        (100000, "", True, "100.0K"),
        (1000000, "", True, "1.0M"),
        (1000000000, "", True, "1.0G"),
        (1000000000000, "", True, "1.0T"),
        (1000000000000000, "", True, "1.0P"),
        (1000000000000000000, "", True, "1.0E"),
        (1000000000000000000000, "", True, "1.0Z"),
        (1000000000000000000000000, "", True, "1.0Y"),
        (1000000000000000000000000000, "", True, "1.0R"),
        (1000000000000000000000000000000, "", True, "1.0Q"),
    ),
)
def test_sizeof_fmt(num, suffix, si, expected):
    assert sizeof_fmt(num, suffix, si) == expected


@pytest.mark.parametrize(
    "text,expected",
    (
        ("1", 1),
        ("50", 50),
        ("1K", 1024),
        ("1k", 1024),
        ("2M", 1024 * 1024 * 2),
    ),
)
def test_suffix_to_number(text, expected):
    assert suffix_to_number(text) == expected


@pytest.mark.parametrize(
    "text",
    (
        "",
        "Bogus",
        "50H",
    ),
)
def test_suffix_to_number_invalid(text):
    with pytest.raises(ValueError):
        suffix_to_number(text)


@pytest.mark.parametrize(
    "text,expected",
    (
        ("test like", "test like"),
        ("Can%t \\escape_this", "Can\\%t \\\\escape\\_this"),
    ),
)
def test_sql_escape_like(text, expected):
    assert sql_escape_like(text) == expected


def test_retry_with_backoff():
    called = 0
    retries = 2

    @retry_with_backoff(retries=retries, backoff_sec=0.05)
    def func_with_exception():
        nonlocal called
        called += 1
        raise RuntimeError("Error")

    with pytest.raises(RuntimeError):
        func_with_exception()
    assert called == retries + 1

    called = 0  # resetting called

    @retry_with_backoff(retries=retries, backoff_sec=0.05)
    def func_ok():
        nonlocal called
        called += 1

    func_ok()
    assert called == 1


@pytest.mark.parametrize(
    "workers,rows_total,settings,expected",
    (
        (False, None, None, False),
        (False, None, "-1", False),
        (False, None, "0", False),
        (False, None, "5", 5),
        (-1, None, "5", False),
        (0, None, "5", False),
        (10, None, "5", 10),
        (True, None, None, True),
        (True, None, "5", True),
        (10, 0, None, False),
        (10, 1, None, False),
        (10, 2, None, 10),
        (True, 0, None, False),
        (True, 1, None, False),
        (True, 2, None, True),
    ),
)
def test_determine_workers(workers, rows_total, settings, expected, monkeypatch):
    if settings is not None:
        monkeypatch.setenv("DATACHAIN_DISTRIBUTED", "some_defined_value")
        monkeypatch.setenv("DATACHAIN_SETTINGS_WORKERS", settings)
    assert determine_workers(workers, rows_total=rows_total) == expected


@pytest.mark.parametrize(
    "parallel,rows_total,settings,expected",
    (
        (None, None, None, False),
        (None, None, "-1", True),
        (None, None, "0", False),
        (None, None, "5", 5),
        (-1, None, "5", True),
        (0, None, "5", False),
        (10, None, "5", 10),
        (True, None, None, True),
        (True, None, "5", True),
        (10, 0, None, False),
        (10, 1, None, False),
        (10, 2, None, 10),
        (True, 0, None, False),
        (True, 1, None, False),
        (True, 2, None, True),
    ),
)
def test_determine_processes(parallel, rows_total, settings, expected, monkeypatch):
    if settings is not None:
        monkeypatch.setenv("DATACHAIN_SETTINGS_PARALLEL", settings)
    assert determine_processes(parallel, rows_total=rows_total) == expected


@pytest.mark.parametrize(
    "path,expected",
    (
        ("/dogs", False),
        ("/dogs/", False),
        ("/dogs/*", True),
        ("/home/user/bucket/animals/", False),
        ("/home/user/bucket/animals/*", True),
        ("", False),
        ("*", True),
    ),
)
def test_uses_glob(path, expected):
    assert uses_glob(path) is expected


@pytest.mark.parametrize(
    "data,path,value,expected",
    (
        ({}, ["test"], True, {"test": True}),
        ({"extra": False}, ["test"], True, {"extra": False, "test": True}),
        (
            {"extra": False},
            ["test", "nested"],
            True,
            {"extra": False, "test": {"nested": True}},
        ),
        (
            {"extra": False},
            ["test", "nested", "deep"],
            True,
            {"extra": False, "test": {"nested": {"deep": True}}},
        ),
        (
            {"extra": False, "test": {"test2": 5, "nested": {}}},
            ["test", "nested", "deep"],
            True,
            {"extra": False, "test": {"test2": 5, "nested": {"deep": True}}},
        ),
    ),
)
def test_nested_dict_path_set(data, path, value, expected):
    assert nested_dict_path_set(data, path, value) == expected


@pytest.mark.parametrize(
    "headers,row,expected",
    (
        ([["a"], ["b"]], (3, 7), {"a": 3, "b": 7}),
        ([["a"], ["b", "c"]], (3, 7), {"a": 3, "b": {"c": 7}}),
        (
            [["a", "b"], ["a", "c"], ["d"], ["a", "e", "f"]],
            (1, 5, "test", 11),
            {"a": {"b": 1, "c": 5, "e": {"f": 11}}, "d": "test"},
        ),
    ),
)
def test_row_to_nested_dict(headers, row, expected):
    assert row_to_nested_dict(headers, row) == expected


def test_batched_basic():
    """Test basic batching functionality."""
    data = list(range(10))
    batches = list(batched(data, 3))
    assert batches == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]


def test_batched_row_limit():
    """Test dynamic batching with row count limit."""
    data = list(range(15))
    batches = list(batched(data, batch_size=4))
    assert len(batches) == 4  # 15 items / 4 max = 4 batches
    assert batches[0] == (0, 1, 2, 3)
    assert batches[1] == (4, 5, 6, 7)
    assert batches[2] == (8, 9, 10, 11)
    assert batches[3] == (12, 13, 14)


@pytest.mark.parametrize(
    "num_rows,batch_size",
    (
        (300_000, 500),
        (100_000, 500),
        (100_000, 100_000),
        (1, 1),
    ),
)
def test_batched_it(num_rows, batch_size):
    rows = iter(list(range(num_rows)))
    batches = batched_it(iter(rows), batch_size)
    num_batches = 0
    uniq_data = set()
    for batch in batches:
        num_batches += 1
        batch_l = list(batch)
        assert len(batch_l) == batch_size
        uniq_data.update(batch_l)

    assert num_batches == num_rows / batch_size
    assert len(uniq_data) == num_rows


def test_interprocess_file_lock_writes_pid_and_releases(tmp_path):
    lock_path = tmp_path / "pull.lock"
    pid_path = Path(f"{lock_path.as_posix()}.pid")

    with interprocess_file_lock(lock_path.as_posix()):
        pid_in_lock = int(pid_path.read_text(encoding="utf-8").strip())
        assert pid_in_lock == os.getpid()

    # After release, a new acquire should succeed.
    with interprocess_file_lock(lock_path.as_posix(), timeout=0):
        pass


def test_interprocess_file_lock_cleans_up_pid_file(tmp_path):
    lock_path = tmp_path / "pull.lock"
    pid_path = Path(f"{lock_path.as_posix()}.pid")

    with interprocess_file_lock(lock_path.as_posix()):
        assert lock_path.exists()
        assert pid_path.exists()

    assert not pid_path.exists()


def test_interprocess_file_lock_timeout_without_wait_message(tmp_path):
    lock_path = tmp_path / "pull.lock"

    with interprocess_file_lock(lock_path.as_posix()):
        with pytest.raises(Timeout):
            with interprocess_file_lock(lock_path.as_posix(), timeout=0):
                pass


def test_interprocess_file_lock_prints_pid_when_blocked(tmp_path, capsys):
    lock_path = tmp_path / "pull.lock"
    pid_path = Path(f"{lock_path.as_posix()}.pid")
    wait_message = "Another pull is already in progress."

    with interprocess_file_lock(lock_path.as_posix()):
        with pytest.raises(Timeout):
            with interprocess_file_lock(
                lock_path.as_posix(),
                wait_message=wait_message,
                timeout=0,
            ):
                pass

    captured = capsys.readouterr().out
    assert wait_message in captured
    assert f"pid={os.getpid()}" in captured
    assert f"delete: {lock_path.as_posix()} (and {pid_path.as_posix()})" in captured
    assert f"ps -p {os.getpid()}" in captured


def test_interprocess_file_lock_wait_hint_without_pid(tmp_path, capsys):
    lock_path = tmp_path / "pull.lock"
    pid_path = Path(f"{lock_path.as_posix()}.pid")
    wait_message = "Another pull is already in progress."

    # Acquire the raw file lock without writing a PID sidecar.
    raw_lock = FileLock(lock_path.as_posix())
    raw_lock.acquire()
    try:
        with pytest.raises(Timeout):
            with interprocess_file_lock(
                lock_path.as_posix(),
                wait_message=wait_message,
                timeout=0,
            ):
                pass
    finally:
        raw_lock.release()

    captured = capsys.readouterr().out
    assert wait_message in captured
    assert "ps -p" not in captured
    assert f"delete: {lock_path.as_posix()} (and {pid_path.as_posix()})" in captured


def test_interprocess_file_lock_timeout_preserves_other_pid(tmp_path):
    lock_path = tmp_path / "pull.lock"
    pid_path = Path(f"{lock_path.as_posix()}.pid")

    # Simulate another process holding the lock with its own PID file.
    raw_lock = FileLock(lock_path.as_posix())
    raw_lock.acquire()
    holder_pid = "99999"
    pid_path.write_text(holder_pid)
    try:
        with pytest.raises(Timeout):
            with interprocess_file_lock(lock_path.as_posix(), timeout=0):
                pass

        # The holder's PID file must still exist and be untouched.
        assert pid_path.exists()
        assert pid_path.read_text() == holder_pid
    finally:
        raw_lock.release()


def test_interprocess_file_lock_no_dir_in_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bare = "bare_test.lock"
    with interprocess_file_lock(bare, timeout=0):
        assert (tmp_path / bare).exists()


def gen3():
    yield from range(3)


@pytest.mark.parametrize(
    "input_data, expected",
    [
        (
            [10, 20, 30],
            [(10, False), (20, False), (30, True)],
        ),
        (
            [42],
            [(42, True)],
        ),
        (
            [],
            [],
        ),
        (
            gen3(),  # generator input
            [(0, False), (1, False), (2, True)],
        ),
    ],
)
def test_with_last_flag(input_data, expected):
    assert list(with_last_flag(input_data)) == expected


@pytest.fixture(autouse=True)
def reset_checkpoint_state(monkeypatch):
    """Reset checkpoint state before each test."""
    _CheckpointState.disabled = False
    _CheckpointState.warning_shown = False
    _CheckpointState.owner_thread = None
    # Clear any existing env vars
    monkeypatch.delenv("DATACHAIN_MAIN_PROCESS_PID", raising=False)
    monkeypatch.delenv("DATACHAIN_SUBPROCESS", raising=False)
    yield
    # Reset after test as well
    _CheckpointState.disabled = False
    _CheckpointState.warning_shown = False
    _CheckpointState.owner_thread = None


def test_checkpoints_enabled_main_thread():
    """Test that checkpoints are enabled in main thread and main process."""
    assert checkpoints_enabled() is True
    assert _CheckpointState.disabled is False
    assert _CheckpointState.warning_shown is False


def test_checkpoints_enabled_non_main_thread(monkeypatch):
    # First call establishes ownership with the real current thread
    assert checkpoints_enabled() is True
    owner_ident = _CheckpointState.owner_thread
    assert owner_ident == threading.current_thread().ident

    # Now simulate a different thread by mocking the ident
    class MockThread:
        ident = owner_ident + 1
        name = "Thread-1"

    monkeypatch.setattr("datachain.utils.threading.current_thread", MockThread)

    assert checkpoints_enabled() is False
    assert _CheckpointState.disabled is True


def test_checkpoints_enabled_non_main_process(monkeypatch):
    # Simulate a subprocess by setting DATACHAIN_MAIN_PROCESS_PID to a different PID
    monkeypatch.setenv("DATACHAIN_MAIN_PROCESS_PID", str(os.getpid() + 1000))

    assert checkpoints_enabled() is False
    # Note: disabled flag is not set for subprocess detection, just returns False
    assert _CheckpointState.warning_shown is True


def test_checkpoints_enabled_warning_shown_once(monkeypatch, caplog):
    """Test that the warning is only shown once even when called multiple times."""
    # First call establishes ownership
    assert checkpoints_enabled() is True
    owner_ident = _CheckpointState.owner_thread

    class MockThread:
        ident = owner_ident + 1  # Different thread ident
        name = "Thread-1"

    monkeypatch.setattr("datachain.utils.threading.current_thread", MockThread)

    # Call multiple times from non-owner thread
    assert checkpoints_enabled() is False
    assert checkpoints_enabled() is False
    assert checkpoints_enabled() is False

    # Verify warning was logged only once
    warning_count = sum(
        1 for record in caplog.records if "Concurrent thread detected" in record.message
    )
    assert warning_count == 1
    assert _CheckpointState.warning_shown is True


def test_checkpoints_enabled_stays_disabled(monkeypatch):
    """Test that once disabled, checkpoints stay disabled even in owner thread."""
    # First call establishes ownership
    assert checkpoints_enabled() is True
    owner_ident = _CheckpointState.owner_thread

    class MockThread:
        ident = owner_ident + 1  # Different thread ident
        name = "Thread-1"

    class MockOwnerThread:
        ident = owner_ident  # Same as owner
        name = "MainThread"

    # Call from non-owner thread disables checkpoints
    monkeypatch.setattr("datachain.utils.threading.current_thread", MockThread)
    assert checkpoints_enabled() is False
    assert _CheckpointState.disabled is True

    # Even if we go back to owner thread, it should stay disabled
    monkeypatch.setattr("datachain.utils.threading.current_thread", MockOwnerThread)
    assert checkpoints_enabled() is False
    assert _CheckpointState.disabled is True


def test_checkpoints_enabled_datachain_subprocess(monkeypatch):
    """Test that DATACHAIN_SUBPROCESS env var enables checkpoints regardless of PID."""
    # Set the main PID to something different (simulating we're in a subprocess)
    monkeypatch.setenv("DATACHAIN_MAIN_PROCESS_PID", str(os.getpid() + 1000))

    # Without DATACHAIN_SUBPROCESS, checkpoints should be disabled
    assert checkpoints_enabled() is False

    # Reset state for next test
    _CheckpointState.warning_shown = False

    # With DATACHAIN_SUBPROCESS, checkpoints should be enabled
    monkeypatch.setenv("DATACHAIN_SUBPROCESS", "1")
    assert checkpoints_enabled() is True


def test_is_subpath_rejects_equal(tmp_path):
    output = (tmp_path / "out").as_posix()
    assert not is_subpath(output, output)


def test_is_subpath_rejects_outside(tmp_path):
    output = (tmp_path / "out").as_posix()
    dst = (tmp_path / "escape.txt").as_posix()
    assert not is_subpath(output, dst)


def test_is_subpath_rejects_prefix_overlap(tmp_path):
    output = (tmp_path / "out").as_posix()
    dst = (tmp_path / "out2" / "file.txt").as_posix()
    assert not is_subpath(output, dst)


def test_is_subpath_accepts_contained(tmp_path):
    output = (tmp_path / "out").as_posix()
    dst = (tmp_path / "out" / "sub" / "file.txt").as_posix()
    assert is_subpath(output, dst)


def test_is_subpath_asserts_parent_absolute():
    with pytest.raises(AssertionError, match="parent must be absolute"):
        is_subpath("relative/dir", os.path.abspath("child"))


def test_is_subpath_asserts_child_absolute():
    with pytest.raises(AssertionError, match="child must be absolute"):
        is_subpath(os.path.abspath("parent"), "relative/child")


@pytest.mark.skipif(os.name == "nt", reason="case-sensitive filesystem")
def test_is_subpath_case_sensitive_on_unix(tmp_path):
    parent = (tmp_path / "Out").as_posix()
    child = (tmp_path / "out" / "file.txt").as_posix()
    assert not is_subpath(parent, child)


@pytest.mark.skipif(os.name != "nt", reason="Windows-only case folding")
def test_is_subpath_case_insensitive_on_windows(tmp_path):
    parent = (tmp_path / "Out").as_posix()
    child = (tmp_path / "out" / "file.txt").as_posix()
    assert is_subpath(parent, child)


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator normalization")
def test_is_subpath_win_backslash_parent():
    assert is_subpath("C:\\Users\\out", "C:\\Users\\out\\file.txt")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator normalization")
def test_is_subpath_win_forward_slash_parent():
    assert is_subpath("C:/Users/out", "C:/Users/out/file.txt")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator normalization")
def test_is_subpath_win_mixed_separators():
    # normcase normalizes both to backslash, so mixed forms must match
    assert is_subpath("C:\\Users\\out", "C:/Users/out/file.txt")
    assert is_subpath("C:/Users/out", "C:\\Users\\out\\file.txt")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator normalization")
def test_is_subpath_win_mixed_rejects_outside():
    assert not is_subpath("C:\\Users\\out", "C:/Users/out2/file.txt")
    assert not is_subpath("C:/Users/out", "C:\\Users\\escape.txt")


def test_is_subpath_rejects_dotdot_traversal(tmp_path):
    """Defence in depth: ``..`` in an absolute child must not bypass the check."""
    output = str(tmp_path / "output")
    # Craft a child that starts with the parent prefix but escapes via '..'
    child = str(tmp_path / "output" / ".." / "escaped.txt")
    assert not is_subpath(output, child)


def test_is_subpath_rejects_deep_dotdot_traversal(tmp_path):
    """Multiple ``..`` segments escaping several levels up."""
    output = str(tmp_path / "a" / "b" / "output")
    child = str(tmp_path / "a" / "b" / "output" / ".." / ".." / ".." / "etc" / "passwd")
    assert not is_subpath(output, child)


def test_is_subpath_accepts_dotdot_that_stays_inside(tmp_path):
    """``..`` that resolves back inside the parent should be accepted."""
    output = str(tmp_path / "output")
    # output/sub/../file.txt  →  output/file.txt  (still inside output)
    child = str(tmp_path / "output" / "sub" / ".." / "file.txt")
    assert is_subpath(output, child)


def test_is_subpath_rejects_dotdot_to_parent_itself(tmp_path):
    """``..`` resolving to exactly the parent (not strictly inside) → False."""
    output = str(tmp_path / "output")
    # output/sub/..  →  output  (equal, not strictly inside)
    child = str(tmp_path / "output" / "sub" / "..")
    assert not is_subpath(output, child)


def test_is_subpath_normalises_dot_segments(tmp_path):
    """Single ``.`` segments don't fool the check."""
    output = str(tmp_path / "output")
    child = str(tmp_path / "output" / "." / "file.txt")
    assert is_subpath(output, child)


def test_is_subpath_parent_with_dotdot(tmp_path):
    """``..`` in the parent path is also normalised."""
    parent = str(tmp_path / "a" / ".." / "output")
    child = str(tmp_path / "output" / "file.txt")
    assert is_subpath(parent, child)
