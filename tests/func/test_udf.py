import functools
import os
import pickle
import posixpath
import sys
import time
from collections.abc import Iterator

import multiprocess as mp
import pytest

import datachain as dc
from datachain.func import path as pathfunc
from datachain.lib.file import AudioFile, AudioFragment, File
from datachain.lib.udf import Mapper, UdfRunError
from datachain.lib.utils import DataChainColumnError
from tests.utils import LARGE_TREE, NUM_TREE


def test_udf_none_nested_datamodel_after_outer_merge(test_session):
    """
    Test that UDFs can handle None values for nested DataModel objects
    """
    # Get warehouse default values for proper NULL handling checks
    catalog = test_session.catalog
    from datachain.sql.types import Int, String

    int_default = Int.default_value(catalog.warehouse.db.dialect)
    str_default = String.default_value(catalog.warehouse.db.dialect)

    # Create sample data with AudioFragment (which has nested AudioFile)
    left = dc.read_values(
        id=[1, 2],
        audio_fragment=[
            AudioFragment(
                audio=AudioFile(path="audio1.wav", source="file://"),
                start=0.0,
                end=1.0,
            ),
            AudioFragment(
                audio=AudioFile(path="audio2.wav", source="file://"),
                start=0.0,
                end=1.0,
            ),
        ],
        session=test_session,
    )

    right = dc.read_values(
        id=[2, 3],
        audio_info=[
            AudioFragment(
                audio=AudioFile(path="audio2_right.wav", source="file://"),
                start=1.0,
                end=2.0,
            ),
            AudioFragment(
                audio=AudioFile(path="audio3_right.wav", source="file://"),
                start=0.0,
                end=1.0,
            ),
        ],
        session=test_session,
    )

    # Full outer merge creates None (or default values in CH) for unmatched sides
    merged = left.merge(right, on="id", full=True)
    assert merged.count() == 3

    def extract_paths(
        id: int, right_id: int, audio_fragment: AudioFragment, audio_info: AudioFragment
    ) -> tuple:
        left_path = audio_fragment.audio.path if audio_fragment else None
        right_path = audio_info.audio.path if audio_info else None
        extracted_file = audio_fragment.audio if audio_fragment else None
        return id, right_id, left_path, right_path, extracted_file

    result = merged.settings(prefetch=False).map(
        extract_paths,
        params=["id", "right_id", "audio_fragment", "audio_info"],
        output={
            "merged_id": int,
            "merged_right_id": int,
            "left_path": str,
            "right_path": str,
            "extracted_file": AudioFile,
        },
    )
    rows = sorted(
        result.select(
            "merged_id",
            "merged_right_id",
            "left_path",
            "right_path",
            "extracted_file",
        ).to_iter(),
        key=lambda r: (
            r[0] if r[0] is not None else -1,
            r[1] if r[1] is not None else -1,
        ),
    )

    assert len(rows) == 3

    # Row with right-only data: left id=int_default (NULL), right_id=3
    assert rows[0][0] == int_default  # merged_id (NULL → int_default)
    assert rows[0][1] == 3  # merged_right_id
    assert rows[0][2] == str_default  # left_path (NULL → str_default)
    assert rows[0][3] == "audio3_right.wav"  # right_path
    # extracted_file: NULL AudioFile → None on SQLite, default AudioFile on ClickHouse
    if rows[0][4] is None:
        # SQLite: NULL object becomes None
        assert rows[0][4] is None
    else:
        # ClickHouse: NULL object becomes object with default values
        assert isinstance(rows[0][4], AudioFile)
        assert rows[0][4].path == str_default

    # Row with left-only data: left id=1, right_id=0 (NULL)
    assert rows[1][0] == 1
    assert rows[1][1] == int_default
    assert rows[1][2] == "audio1.wav"
    assert rows[1][3] == str_default
    assert isinstance(rows[1][4], AudioFile)
    assert rows[1][4].path == "audio1.wav"
    assert rows[1][4].source == "file://"

    # Row with matched data: id=2, right_id=2 (both sides present)
    assert rows[2][0] == 2
    assert rows[2][1] == 2
    assert rows[2][2] == "audio2.wav"
    assert rows[2][3] == "audio2_right.wav"
    assert isinstance(rows[2][4], AudioFile)
    assert rows[2][4].path == "audio2.wav"
    assert rows[2][4].source == "file://"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_udf(cloud_test_catalog):
    session = cloud_test_catalog.session

    def name_len(path):
        return (len(posixpath.basename(path)),)

    chain = (
        dc.read_storage(cloud_test_catalog.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .map(name_len, params=["file.path"], output={"name_len": int})
    )

    result1 = chain.to_list("file.path", "name_len")
    # ensure that we're able to run with same query multiple times

    result2 = chain.to_list("file.path", "name_len")
    count = chain.count()
    assert len(result1) == 3
    assert len(result2) == 3
    assert count == 3

    for r1, r2 in zip(result1, result2, strict=False):
        # Check that the UDF ran successfully
        assert len(posixpath.basename(r1[0])) == r1[1]
        assert len(posixpath.basename(r2[0])) == r2[1]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    def name_len(name):
        return (len(name),)

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .settings(parallel=True)
        .map(name_len, params=["file.path"], output={"name_len": int})
    )

    # Check that the UDF ran successfully
    count = 0
    for r in chain.to_iter("file.path", "name_len"):
        count += 1
        assert len(r[0]) == r[1]
    assert count == 7


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_class_udf(cloud_test_catalog):
    session = cloud_test_catalog.session

    class MyUDF(Mapper):
        def __init__(self, constant, multiplier=1):
            self.constant = constant
            self.multiplier = multiplier

        def process(self, size):
            return (self.constant + size * self.multiplier,)

    chain = (
        dc.read_storage(cloud_test_catalog.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .map(
            MyUDF(5, multiplier=2),
            output={"total": int},
            params=["file.size"],
        )
        .select("file.size", "total")
        .order_by("file.size")
    )

    assert chain.to_list("file.size", "total") == [
        (3, 11),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_class_udf_parallel(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    class MyUDF(Mapper):
        def __init__(self, constant, multiplier=1):
            self.constant = constant
            self.multiplier = multiplier

        def process(self, size):
            return (self.constant + size * self.multiplier,)

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .settings(parallel=2)
        .map(
            MyUDF(5, multiplier=2),
            output={"total": int},
            params=["file.size"],
        )
        .select("file.size", "total")
        .order_by("file.size")
    )

    assert chain.to_list("file.size", "total") == [
        (3, 11),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, NUM_TREE), ("file", False, NUM_TREE)],
    indirect=True,
)
def test_udf_after_limit(cloud_test_catalog):
    ctc = cloud_test_catalog

    def name_int(name: str) -> int:
        try:
            return int(name)
        except ValueError:
            return 0

    def get_result(chain):
        res = chain.limit(100).map(name_int=name_int).order_by("name")
        return res.to_list("name", "name_int")

    expected = [(f"{i:06d}", i) for i in range(100)]
    chain = (
        dc.read_storage(ctc.src_uri, session=ctc.session)
        .mutate(name=pathfunc.name("file.path"))
        .persist()
    )
    # We test a few different orderings here, because we've had strange
    # bugs in the past where calling add_signals() after limit() gave us
    # incorrect results on clickhouse cloud.
    # See https://github.com/iterative/dvcx/issues/940
    assert get_result(chain.order_by("name")) == expected
    assert len(get_result(chain.order_by("sys.rand"))) == 100
    assert len(get_result(chain)) == 100


def test_udf_different_types(cloud_test_catalog):
    obj = {"name": "John", "age": 30}

    def test_types():
        return (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            {"a": 1},
            pickle.dumps(obj),
        )

    chain = (
        dc.read_storage(cloud_test_catalog.src_uri, session=cloud_test_catalog.session)
        .filter(dc.C("file.path").glob("*cat1"))
        .map(
            test_types,
            params=[],
            output={
                "int_col": int,
                "int_col_32": int,
                "int_col_64": int,
                "float_col": float,
                "float_col_32": float,
                "float_col_64": float,
                "array_col": list[float],
                "array_col_nested": list[list[float]],
                "array_col_32": list[float],
                "array_col_64": list[float],
                "string_col": str,
                "bool_col": bool,
                "json_col": dict,
                "binary_col": bytes,
            },
        )
    )

    results = chain.to_records()
    col_values = [
        (
            r["int_col"],
            r["int_col_32"],
            r["int_col_64"],
            r["float_col"],
            r["float_col_32"],
            r["float_col_64"],
            r["array_col"],
            r["array_col_nested"],
            r["array_col_32"],
            r["array_col_64"],
            r["string_col"],
            r["bool_col"],
            r["json_col"],
            pickle.loads(r["binary_col"]),  # noqa: S301
        )
        for r in results
    ]

    assert col_values == [
        (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            {"a": 1},
            obj,
        )
    ]


def test_udf_rejects_root_override(test_session):
    class X(dc.DataModel):
        x: int

    chain = dc.read_values(x=[X(x=0), X(x=1)], session=test_session)

    with pytest.raises(
        DataChainColumnError,
        match="Error for column x: signal already exists with a different type",
    ):
        chain.map(
            lambda x: x.model_dump(),
            params=["x"],
            output={"x": dict},
        )


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("prefetch", [0, 2])
def test_map_file(cloud_test_catalog, use_cache, prefetch, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    ctc = cloud_test_catalog
    ctc.catalog.cache.clear()

    def is_prefetched(file: File) -> bool:
        return file._catalog.cache.contains(file) and bool(file.get_local_path())

    def verify_cache_used(file):
        catalog = file._catalog
        if use_cache or not prefetch:
            assert catalog.cache == cloud_test_catalog.catalog.cache
            return
        head, tail = os.path.split(catalog.cache.cache_dir)
        assert head == catalog.cache.tmp_dir
        assert tail.startswith("prefetch-")

    def with_checks(func, seen=[]):  # noqa: B006
        @functools.wraps(func)
        def wrapped(file, *args, **kwargs):
            # previously prefetched files should be removed if `cache` is disabled.
            for f in seen:
                assert f._catalog.cache.contains(f) == use_cache
            seen.append(file)

            assert is_prefetched(file) == (prefetch > 0)
            verify_cache_used(file)
            return func(file, *args, **kwargs)

        return wrapped

    def new_signal(file: File) -> str:
        with file.open() as f:
            return file.name + " -> " + f.read().decode("utf-8")

    chain = (
        dc.read_storage(ctc.src_uri, session=ctc.session)
        .settings(cache=use_cache, prefetch=prefetch)
        .map(signal=with_checks(new_signal))
        .persist()
    )

    expected = {
        "description -> Cats and Dogs",
        "cat1 -> meow",
        "cat2 -> mrow",
        "dog1 -> woof",
        "dog2 -> arf",
        "dog3 -> bark",
        "dog4 -> ruff",
    }
    assert set(chain.to_values("signal")) == expected
    for file in chain.to_values("file"):
        assert bool(file.get_local_path()) is use_cache
    assert not os.listdir(ctc.catalog.cache.tmp_dir)


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("prefetch", [0, 2])
def test_gen_file(cloud_test_catalog, use_cache, prefetch, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    ctc = cloud_test_catalog
    ctc.catalog.cache.clear()

    def is_prefetched(file: File) -> bool:
        return file._catalog.cache.contains(file) and bool(file.get_local_path())

    def verify_cache_used(file):
        catalog = file._catalog
        if use_cache or not prefetch:
            assert catalog.cache == cloud_test_catalog.catalog.cache
            return
        head, tail = os.path.split(catalog.cache.cache_dir)
        assert head == catalog.cache.tmp_dir
        assert tail.startswith("prefetch-")

    def with_checks(func, seen=[]):  # noqa: B006
        @functools.wraps(func)
        def wrapped(file, *args, **kwargs):
            # previously prefetched files should be removed if `cache` is disabled.
            for f in seen:
                assert f._catalog.cache.contains(f) == use_cache
            seen.append(file)

            assert is_prefetched(file) == (prefetch > 0)
            verify_cache_used(file)
            return func(file, *args, **kwargs)

        return wrapped

    def new_signal(file: File) -> list[str]:
        with file.open("rb") as f:
            return [file.name, f.read().decode("utf-8")]

    chain = (
        dc.read_storage(ctc.src_uri, session=ctc.session)
        .settings(cache=use_cache, prefetch=prefetch)
        .gen(signal=with_checks(new_signal), output=str)
        .persist()
    )
    expected = {
        "Cats and Dogs",
        "arf",
        "bark",
        "cat1",
        "cat2",
        "description",
        "dog1",
        "dog2",
        "dog3",
        "dog4",
        "meow",
        "mrow",
        "ruff",
        "woof",
    }
    assert set(chain.to_values("signal")) == expected
    assert not os.listdir(ctc.catalog.cache.tmp_dir)


def test_batch_for_map(test_session):
    # Create a chain with batch settings
    chain = dc.read_values(x=list(range(100)), session=test_session)
    chain_with_settings = chain.settings(batch_size=15)

    def add_one(x):
        return x + 1

    result = chain_with_settings.map(add_one, output={"result": int})

    results = [
        r[0] for r in result.to_iter("result")
    ]  # Access the first element of each tuple

    assert len(results) == 100
    assert set(results) == set(range(1, 101))


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel_exec_error(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .settings(parallel=True)
        .map(name_len_error, params=["file.path"], output={"name_len": int})
    )

    with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
        chain.show()


@pytest.mark.parametrize(
    "failure_mode,expected_exit_code,error_marker",
    [
        ("exception", 1, "Worker 1 failure!"),
        ("keyboard_interrupt", -2, "KeyboardInterrupt"),
        ("os_exit", 1, None),  # os._exit - immediate termination
    ],
)
def test_udf_parallel_worker_failure_exits_peers(
    test_session_tmpfile,
    tmp_path,
    capfd,
    failure_mode,
    expected_exit_code,
    error_marker,
):
    """
    Test that when one worker fails, all other workers exit immediately.

    Tests different failure modes:
    - exception: Worker raises RuntimeError (normal exception)
    - keyboard_interrupt: Worker raises KeyboardInterrupt (simulates Ctrl+C)
    - sys_exit: Worker calls sys.exit() (clean Python exit)
    - os_exit: Worker calls os._exit() (immediate process termination)
    """
    import platform

    # Windows uses different exit codes for KeyboardInterrupt
    # 3221225786 (0xC000013A) is STATUS_CONTROL_C_EXIT on Windows
    # while POSIX systems use -2 (SIGINT)
    if platform.system() == "Windows" and failure_mode == "keyboard_interrupt":
        expected_exit_code = 3221225786

    vals = list(range(100))

    barrier_dir = tmp_path / "udf_workers_barrier"
    barrier_dir_str = str(barrier_dir)
    os.makedirs(barrier_dir_str, exist_ok=True)
    expected_workers = 3

    def slow_process(val: int) -> int:
        proc_name = mp.current_process().name
        with open(os.path.join(barrier_dir_str, f"{proc_name}.started"), "w") as f:
            f.write(str(time.time()))

        # Wait until all expected workers have written their markers
        deadline = time.time() + 1.0
        while time.time() < deadline:
            try:
                count = len(
                    [n for n in os.listdir(barrier_dir_str) if n.endswith(".started")]
                )
            except FileNotFoundError:
                count = 0
            if count >= expected_workers:
                break
            time.sleep(0.01)

        if proc_name == "Worker-UDF-1":
            if failure_mode == "exception":
                raise RuntimeError("Worker 1 failure!")
            if failure_mode == "keyboard_interrupt":
                raise KeyboardInterrupt("Worker interrupted")
            if failure_mode == "sys_exit":
                sys.exit(1)
            if failure_mode == "os_exit":
                os._exit(1)
        time.sleep(5)
        return val * 2

    chain = (
        dc.read_values(val=vals, session=test_session_tmpfile)
        .settings(parallel=3)
        .map(slow_process, output={"result": int})
    )

    start = time.time()
    with pytest.raises(RuntimeError, match="UDF Execution Failed!") as exc_info:
        list(chain.to_iter("result"))
    elapsed = time.time() - start

    # Verify timing: should exit immediately when worker fails
    assert elapsed < 10, f"took {elapsed:.1f}s, should exit immediately"

    # Verify multiple workers were started via barrier markers
    try:
        started_files = [
            n for n in os.listdir(barrier_dir_str) if n.endswith(".started")
        ]
    except FileNotFoundError:
        started_files = []
    assert len(started_files) == 3, (
        f"Expected all 3 workers to start, but saw markers for: {started_files}"
    )

    captured = capfd.readouterr()

    # Verify the RuntimeError has a meaningful message with exit code
    error_message = str(exc_info.value)
    assert f"UDF Execution Failed! Exit code: {expected_exit_code}" in error_message, (
        f"Expected exit code {expected_exit_code}, got: {error_message}"
    )

    if error_marker:
        assert error_marker in captured.err, (
            f"Expected '{error_marker}' in stderr for {failure_mode} mode. "
            f"stderr output: {captured.err[:500]}"
        )

    assert "semaphore" not in captured.err


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_reuse_on_error(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    error_state = {"error": True}

    def name_len_maybe_error(path):
        if error_state["error"]:
            # A udf that raises an exception
            raise RuntimeError("Test Error!")
        return (len(path),)

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .map(name_len_maybe_error, params=["file.path"], output={"path_len": int})
    )

    with pytest.raises(RuntimeError, match="Test Error!"):
        chain.show()

    # Simulate fixing the error
    error_state["error"] = False

    # Retry Query
    count = 0
    for r in chain.to_iter("file.path", "path_len"):
        # Check that the UDF ran successfully
        count += 1
        assert len(r[0]) == r[1]

    assert count == 3


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel_interrupt(cloud_test_catalog_tmpfile, capfd):
    session = cloud_test_catalog_tmpfile.session

    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .settings(parallel=True)
        .map(name_len_interrupt, params=["file.path"], output={"name_len": int})
    )
    with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
        chain.show()
    captured = capfd.readouterr()
    assert "semaphore" not in captured.err


@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel_boostrap(test_session_tmpfile):
    vals = ["a", "b", "c", "d", "e", "f"]

    class MyMapper(Mapper):
        DEFAULT_VALUE = 84
        BOOTSTRAP_VALUE = 1452
        TEARDOWN_VALUE = 98763

        def __init__(self):
            super().__init__()
            self.value = MyMapper.DEFAULT_VALUE
            self._had_teardown = False

        def process(self, key) -> int:
            return self.value

        def setup(self):
            self.value = MyMapper.BOOTSTRAP_VALUE

        def teardown(self):
            self.value = MyMapper.TEARDOWN_VALUE

    chain = dc.read_values(key=vals, session=test_session_tmpfile)

    res = chain.settings(parallel=4).map(res=MyMapper()).to_values("res")

    assert res == [MyMapper.BOOTSTRAP_VALUE] * len(vals)


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("parallel", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed(
    cloud_test_catalog_tmpfile, workers, parallel, tree, run_datachain_worker
):
    session = cloud_test_catalog_tmpfile.session

    def name_len(name: str) -> int:
        return len(name)

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .settings(parallel=parallel, workers=workers)
        .map(name_len, params=["file.path"], output={"name_len": int})
    )

    # Check that the UDF ran successfully
    count = 0
    for r in chain.to_iter("file.path", "name_len"):
        count += 1
        assert len(r[0]) == r[1]
    assert count == 225


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("parallel", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed_exec_error(
    cloud_test_catalog_tmpfile, workers, parallel, tree, run_datachain_worker
):
    session = cloud_test_catalog_tmpfile.session

    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .settings(parallel=parallel, workers=workers)
        .map(name_len_error, params=["file.path"], output={"name_len": int})
    )
    with pytest.raises(UdfRunError, match="Test Error!"):
        chain.show()


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("parallel", (1, 2))
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed_interrupt(
    cloud_test_catalog_tmpfile, capfd, tree, workers, parallel, run_datachain_worker
):
    session = cloud_test_catalog_tmpfile.session

    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(dc.C("file.size") < 13)
        .filter(dc.C("file.path").glob("cats*") | (dc.C("file.size") < 4))
        .settings(parallel=parallel, workers=workers)
        .map(name_len_interrupt, params=["file.path"], output={"name_len": int})
    )
    with pytest.raises(Exception, match="UDF task failed with exit code"):
        chain.show()
    captured = capfd.readouterr()
    assert "semaphore" not in captured.err


def test_gen_works_after_union(test_session_tmpfile, monkeypatch):
    """
    Union drops sys columns, we test that UDF generates them correctly after that.
    """
    monkeypatch.setattr("datachain.query.dispatch.DEFAULT_BATCH_SIZE", 5, raising=False)
    n = 30

    x_ids = list(range(n))
    y_ids = list(range(n, 2 * n))

    x = dc.read_values(idx=x_ids, session=test_session_tmpfile)
    y = dc.read_values(idx=y_ids, session=test_session_tmpfile)

    xy = x.union(y)

    def expand(idx):
        yield f"val-{idx}"

    generated = xy.settings(parallel=2).gen(
        gen=expand,
        params=("idx",),
        output={"val": str},
    )

    values = generated.to_values("val")

    assert len(values) == 2 * n
    assert set(values) == {f"val-{i}" for i in range(2 * n)}


@pytest.mark.parametrize("full", [False, True])
def test_gen_works_after_merge(test_session_tmpfile, monkeypatch, full):
    """Merge drops sys columns as well; ensure UDF generation still works."""
    monkeypatch.setattr("datachain.query.dispatch.DEFAULT_BATCH_SIZE", 5, raising=False)
    n = 30

    idxs = list(range(n))

    left = dc.read_values(
        idx=idxs,
        left_value=[f"left-{i}" for i in idxs],
        session=test_session_tmpfile,
    )
    right = dc.read_values(
        idx=idxs,
        right_value=[f"right-{i}" for i in idxs],
        session=test_session_tmpfile,
    )

    merged = left.merge(right, on="idx", full=full)

    def expand(idx, left_value, right_value):
        yield f"val-{idx}-{left_value}-{right_value}"

    generated = merged.settings(parallel=2).gen(
        gen=expand,
        params=("idx", "left_value", "right_value"),
        output={"val": str},
    )

    values = generated.to_values("val")

    assert len(values) == n
    expected = {f"val-{i}-left-{i}-right-{i}" for i in idxs}
    assert set(values) == expected


def test_agg_works_after_union(test_session_tmpfile, monkeypatch):
    """Union must preserve sys columns for aggregations with functional partitions."""
    from datachain import func

    monkeypatch.setattr("datachain.query.dispatch.DEFAULT_BATCH_SIZE", 5, raising=False)

    groups = 5
    n = 30

    x_paths = [f"group-{i % groups}/item-{i}" for i in range(n)]
    y_paths = [f"group-{i % groups}/item-{n + i}" for i in range(n)]

    x = dc.read_values(path=x_paths, session=test_session_tmpfile)
    y = dc.read_values(path=y_paths, session=test_session_tmpfile)

    xy = x.union(y)

    def summarize(paths):
        group = paths[0].split("/")[0]
        yield group, len(paths)

    aggregated = xy.settings(parallel=2).agg(
        summarize,
        params=("path",),
        output={"partition": str, "count": int},
        partition_by=func.parent("path"),
    )

    records = aggregated.to_records()
    expected_counts = {f"group-{g}": 2 * n // groups for g in range(groups)}
    assert {row["partition"]: row["count"] for row in records} == expected_counts


@pytest.mark.parametrize("full", [False, True])
def test_agg_works_after_merge(test_session_tmpfile, monkeypatch, full):
    """Ensure merge keeps sys columns for aggregations with functional partitions."""
    from datachain import func

    monkeypatch.setattr("datachain.query.dispatch.DEFAULT_BATCH_SIZE", 5, raising=False)

    groups = 5
    n = 30
    idxs = list(range(n))

    left = dc.read_values(
        idx=idxs,
        left_path=[f"group-{i % groups}/left-{i}" for i in idxs],
        session=test_session_tmpfile,
    )
    right = dc.read_values(
        idx=idxs,
        right_value=idxs,
        session=test_session_tmpfile,
    )

    merged = left.merge(right, on="idx", full=full)

    def summarize(left_path, right_value):
        group = left_path[0].split("/")[0]
        yield group, sum(right_value)

    aggregated = merged.settings(parallel=2).agg(
        summarize,
        params=("left_path", "right_value"),
        output={"partition": str, "total": int},
        partition_by=func.parent("left_path"),
    )

    records = aggregated.to_records()
    expected_totals = {
        f"group-{g}": sum(val for val in idxs if val % groups == g)
        for g in range(groups)
    }
    assert {row["partition"]: row["total"] for row in records} == expected_totals


def test_agg_list_file_and_map_count(tmp_dir, test_session):
    names = [
        "hotdogs.txt",
        "dogs.txt",
        "dog.txt",
        "1dog.txt",
        "dogatxt.txt",
        "dog.txtx",
    ]

    for name in names:
        (tmp_dir / name).write_text(name, encoding="utf-8")

    base_chain = dc.read_storage(tmp_dir.as_uri(), session=test_session).order_by(
        "file.path"
    )

    expected_files: list[File] = []
    for (file_obj,) in base_chain.select("file").to_iter():
        assert isinstance(file_obj, File)
        expected_files.append(file_obj)

    def collect_files(file: list[File]) -> Iterator[list[File]]:
        # Return the full collection for the partition
        yield file

    def count_files(files: list[File]) -> int:
        return len(files)

    (
        base_chain.agg(files=collect_files)
        .map(num_files=count_files)
        .save("temp_udf_types")
    )

    # Validate result
    ds = dc.read_dataset("temp_udf_types", session=test_session)
    rows = ds.select("num_files").to_list()
    assert rows == [(len(expected_files),)]


def test_agg_list_file_persist_and_read(tmp_dir, test_session):
    names = ["a.txt", "b.txt", "c.txt"]

    for name in names:
        (tmp_dir / name).write_text(name, encoding="utf-8")

    base_chain = dc.read_storage(tmp_dir.as_uri(), session=test_session).order_by(
        "file.path"
    )

    expected_files: list[File] = []
    for (file_obj,) in base_chain.select("file").to_iter():
        assert isinstance(file_obj, File)
        expected_files.append(file_obj)

    def collect_files(file: list[File]) -> Iterator[list[File]]:
        yield file

    (base_chain.agg(files=collect_files).save("temp_files_only"))

    # When reading back, we should get a list of File objects
    ds = dc.read_dataset("temp_files_only", session=test_session)
    vals = ds.select("files").to_list()
    assert len(vals) == 1
    files_list = vals[0][0]
    assert isinstance(files_list, list)
    assert all(isinstance(f, File) for f in files_list)

    expected_sorted: list[File] = sorted(expected_files, key=lambda f: f.path)
    actual_sorted: list[File] = sorted(files_list, key=lambda f: f.path)

    assert [f.model_dump() for f in actual_sorted] == [
        f.model_dump() for f in expected_sorted
    ]
