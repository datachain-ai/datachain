import asyncio
import io
import pickle
import tarfile
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from cloudpickle import dumps, loads
from fsspec.callbacks import DEFAULT_CALLBACK, TqdmCallback
from pydantic import BaseModel

import datachain as dc
from datachain import Generator, Mapper
from datachain.cache import Cache
from datachain.client.fsspec import Client
from datachain.lib import arrow
from datachain.lib.data_model import DataModel
from datachain.lib.file import ArrowRow, File
from datachain.lib.settings import AUTO_PREFETCH, AUTO_PREFETCH_BYTES, DEFAULT_PREFETCH
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.tar import build_tar_member
from datachain.lib.udf import (
    JsonSerializationError,
    UDFBase,
    UdfError,
    UdfRunError,
    _PrefetchBudget,
)
from datachain.lib.utils import DataChainError
from datachain.query import dataset

from .test_udf_signature import get_sign


def test_udf_error():
    orig_err = UdfError("test error")
    for err in (orig_err, loads(dumps(orig_err))):
        assert err.message == "test error"
        assert str(err) == "UdfError: test error"


def test_json_serialization_error_pickle():
    err = JsonSerializationError("bad value", "col_a", repr([1, 2]))
    restored = pickle.loads(pickle.dumps(err))  # noqa: S301
    assert str(restored) == str(err)
    assert restored.column_name == "col_a"
    assert restored.value_repr == repr([1, 2])


def test_json_serialization_error_pickle_unpicklable_value():
    val = object()
    err = JsonSerializationError("bad value", "col_b", repr(val))
    restored = pickle.loads(pickle.dumps(err))  # noqa: S301
    assert str(restored) == str(err)
    assert restored.column_name == "col_b"
    assert "object" in restored.value_repr


@pytest.mark.parametrize(
    "error,stacktrace,udf_name,expected_str,expected_type",
    [
        (
            "test error",
            None,
            None,
            "UdfRunError: test error",
            str,
        ),
        (
            "test error",
            "Traceback (most recent call last): ...",
            None,
            "UdfRunError: test error",
            str,
        ),
        (
            "test error",
            None,
            "MyUDF",
            "UdfRunError: test error",
            str,
        ),
        (
            "test error",
            "Traceback (most recent call last): ...",
            "MyUDF",
            "UdfRunError: test error",
            str,
        ),
        (
            ValueError("invalid value"),
            "Traceback (most recent call last): ...",
            "MyUDF",
            "ValueError: invalid value",
            ValueError,
        ),
        (
            UdfRunError("invalid value"),
            "Traceback (most recent call last): ...",
            "MyUDF",
            "UdfRunError: invalid value",
            UdfRunError,
        ),
        (
            UdfRunError(UdfRunError("invalid value")),
            "Traceback (most recent call last): ...",
            "MyUDF",
            "UdfRunError: invalid value",
            UdfRunError,
        ),
    ],
)
def test_udf_run_error(error, stacktrace, udf_name, expected_str, expected_type):
    orig_err = UdfRunError(error, stacktrace=stacktrace, udf_name=udf_name)
    for err in (orig_err, loads(dumps(orig_err))):
        assert isinstance(err.error, expected_type)
        assert err.stacktrace == stacktrace
        assert err.udf_name == udf_name
        assert str(err) == expected_str


def test_udf_verbose_name_class():
    class MyMapper(Mapper):
        def process(self, key: str) -> int:
            return len(key)

    sign = get_sign(MyMapper, output="res")
    udf = UDFBase._create(sign, sign.output_schema)
    assert udf.verbose_name == "MyMapper"


def test_udf_verbose_name_func():
    def process(key: str) -> int:
        return len(key)

    sign = get_sign(process, output="res")
    udf = UDFBase._create(sign, sign.output_schema)
    assert udf.verbose_name == "process"


def test_udf_verbose_name_lambda():
    sign = get_sign(lambda key: len(key), output="res")  # noqa: PLW0108
    udf = UDFBase._create(sign, sign.output_schema)
    assert udf.verbose_name == "<lambda>"


def test_udf_verbose_name_unknown():
    sign = get_sign(lambda key: len(key), output="res")  # noqa: PLW0108
    udf = UDFBase._create(sign, sign.output_schema)
    udf._func = None
    assert udf.verbose_name == "<unknown>"


def test_udf_verbose_name_multi_signal_mapper(test_session):
    chain = dc.read_values(name=["foo.txt"], session=test_session).map(
        stem=lambda name: name.rsplit(".", 1)[0],
        ext=lambda name: name.rsplit(".", 1)[1],
    )
    udf = chain._query.steps[-1].udf.inner
    assert udf.verbose_name == "stem, ext"


def test_udf_does_not_traverse_setup_value():
    value = {}
    value["self"] = value
    udf = UDFBase()
    udf.params = SignalSchema({"config": str}, setup={"config": lambda: value})

    assert udf._input_reader([], object(), False, DEFAULT_CALLBACK, include_id=False)(
        ()
    ) == [value]


def test_udf_sets_stream_on_setup_file():
    file = File(path="reference.txt")
    udf = UDFBase()
    udf.params = SignalSchema({"ref": str}, setup={"ref": lambda: file})
    catalog = object()

    udf._input_reader([], catalog, False, DEFAULT_CALLBACK, include_id=False)(())

    assert file._catalog is catalog


def test_udf_sets_stream_in_setup_model_collection():
    class Bundle(BaseModel):
        files: list[File]

    file = File(path="reference.txt")
    udf = UDFBase()
    udf.params = SignalSchema({"ref": str}, setup={"ref": lambda: Bundle(files=[file])})
    catalog = object()

    udf._input_reader([], catalog, False, DEFAULT_CALLBACK, include_id=False)(())

    assert file._catalog is catalog


def test_udf_does_not_traverse_bare_setup_collection():
    files = [File(path="reference.txt")]
    udf = UDFBase()
    udf.params = SignalSchema({"ref": str}, setup={"ref": lambda: files})

    udf._input_reader([], object(), False, DEFAULT_CALLBACK, include_id=False)(())

    assert files[0]._catalog is None


def test_udf_output_type_error_message(monkeypatch, test_session):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    chain = dc.read_values(a=["ok"], session=test_session)

    with pytest.raises(DataChainError) as excinfo:
        list(
            chain.map(
                measurement_ids=lambda a: "2",
                params="a",
                output={"measurement_ids": list[str]},
            ).to_list()
        )

    msg = str(excinfo.value)

    # Example message:
    # UdfError: UDF returned an invalid value for output column 'measurement_ids'.
    # Expected list[str], got '2' (type: str).
    assert "invalid value" in msg
    assert "measurement_ids" in msg
    assert "Expected list[str]" in msg
    assert "got '2'" in msg
    assert "type: str" in msg


def test_udf_output_type_error_message_scalar(monkeypatch, test_session):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    chain = dc.read_values(a=["ok"], session=test_session)

    with pytest.raises(DataChainError) as excinfo:
        list(chain.map(my_int=lambda a: "2", params="a", output=int).to_list())

    msg = str(excinfo.value)

    # Example message:
    # UdfError: UDF returned an invalid value for output column 'my_int'.
    # Expected int, got '2' (type: str).
    assert "invalid value" in msg
    assert "my_int" in msg
    assert "Expected int" in msg
    assert "got '2'" in msg
    assert "type: str" in msg


def test_udf_output_type_error_message_includes_missing_outputs(
    monkeypatch, test_session
):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    chain = dc.read_values(a=["ok"], session=test_session)

    # Output expects two columns, but UDF returns a single scalar.
    with pytest.raises(DataChainError) as excinfo:
        list(
            chain.map(
                lambda a: "2",
                params="a",
                output={"measurement_ids": list[str], "x": int},
            ).to_list()
        )

    msg = str(excinfo.value)

    # Example message:
    # UdfError: UDF returned an invalid value for output column 'measurement_ids'.
    # Expected list[str], got '2' (type: str). Note: UDF call returned 1 value
    # while 2 are expected per output definition.
    assert "measurement_ids" in msg
    assert "Expected list[str]" in msg
    assert "UDF call returned 1 value" in msg
    assert "while 2 are expected per output definition" in msg


def test_udf_output_type_error_message_agg_returning_tuple(monkeypatch, test_session):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    chain = dc.read_values(a=["ok"], session=test_session)

    # This mirrors an aggregation mistake:
    # returning a single tuple value instead of yielding rows.
    def bad_agg(a):
        return ("2",)

    with pytest.raises(DataChainError) as excinfo:
        list(
            chain.agg(
                func=bad_agg,
                params="a",
                output={"measurement_ids": list[str], "x": int},
            ).to_list()
        )

    msg = str(excinfo.value)

    # Example message:
    # UdfError: UDF returned an invalid value for output column 'measurement_ids'.
    # Expected list[str], got '2' (type: str). Note: UDF call returned 1 value
    # while 2 are expected per output definition, agg() UDFs usually use yield
    # and have return type Iterator.
    assert "measurement_ids" in msg
    assert "Expected list[str]" in msg
    assert "got '2'" in msg
    assert "type: str" in msg
    assert "UDF call returned 1 value" in msg
    assert "usually use yield" in msg


def test_udf_extra_return_values_raise(monkeypatch, test_session):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    chain = dc.read_values(a=["ok"], session=test_session)

    # A UDF returning more values than declared outputs is a misalignment, not a
    # silent truncation.
    with pytest.raises((ValueError, DataChainError), match="declared in output"):
        list(
            chain.map(
                lambda a: (1, 2, 3),
                params="a",
                output={"x": int, "y": int},
            ).to_list()
        )


def test_udf_output_type_error_message_json_serialization_failure(
    monkeypatch, test_session
):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    chain = dc.read_values(a=["ok"], session=test_session)

    # Create an object that can't be serialized to JSON
    class NonSerializable:
        pass

    def bad_func(a):
        return {"key": NonSerializable()}

    with pytest.raises(DataChainError) as exc_info:
        list(
            chain.map(
                bad_func,
                params="a",
                output={"data": dict},
            ).to_list()
        )

    msg = str(exc_info.value)

    # Example message:
    # UdfError: UDF returned an invalid value for output column 'data'.
    # Expected JSON-serializable dict.
    # JSON serialization error: Object of type NonSerializable is not JSON serializable
    assert "invalid value" in msg
    assert "data" in msg
    assert "JSON-serializable dict" in msg
    assert "JSON serialization error" in msg
    assert "not JSON serializable" in msg

    # The exception chain still preserves the underlying error
    # UdfError -> JsonSerializationError -> TypeError
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, JsonSerializationError)
    assert exc_info.value.__cause__.__cause__ is not None
    assert isinstance(exc_info.value.__cause__.__cause__, TypeError)


@pytest.mark.parametrize("bad_path", [".", "..", "dir/../etc"])
def test_map_prefetch_skips_invalid_path_files(
    monkeypatch, test_session, caplog, bad_path
):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    files = [File(source="s3://bucket", path=bad_path) for _ in range(3)]

    def get_path(file: File) -> str:
        return file.path

    with caplog.at_level("WARNING", logger="datachain"):
        result = (
            dc.read_values(file=files, session=test_session)
            .settings(prefetch=2)
            .map(get_path, params="file", output={"p": str})
            .to_list("p")
        )

    assert result == [(bad_path,)] * 3
    assert any("Skipping prefetch" in m for m in caplog.messages)


def test_output_schema_is_built_once_per_run(test_session, mocker):
    spec = mocker.spy(SignalSchema, "to_udf_spec")
    chain = dc.read_values(x=list(range(200)), session=test_session)
    chain.map(y=lambda x: x + 1, output=int).gen(
        z=lambda y: iter([y]), output=int
    ).save("udf_spec")
    assert spec.call_count < 20


def test_an_insert_error_closes_the_udf_outputs(test_session, mocker):
    closed = []

    def numbers(n: int) -> Iterator[int]:
        try:
            yield from range(n)
        finally:
            closed.append(n)

    def failing_insert(self, table, columns, rows, parts=None, ledger=()):
        raise OSError(28, "No space left on device")

    chain = dc.read_values(n=[1000], session=test_session).save("inputs")
    mocker.patch.object(
        type(test_session.catalog.warehouse), "insert_block", failing_insert
    )
    mocker.patch.object(type(test_session.catalog.warehouse), "INSERT_BATCH_SIZE", 10)
    with pytest.raises(OSError, match="No space left") as error:
        chain.gen(x=numbers, output=int).save("numbers")
    # the error's traceback holds the generator: it must be closed already
    assert closed == [1000]
    assert error.value.__traceback__ is not None


UDF_CLASSES = {
    "map": dc.Mapper,
    "batch_map": dc.BatchMapper,
    "gen": dc.Generator,
    "agg": dc.Aggregator,
}


def _apply(chain, verb, **signal_map):
    kwargs = {"batch": 2} if verb == "batch_map" else {}
    return getattr(chain, verb)(output=int, **kwargs, **signal_map)


@pytest.mark.parametrize("verb", list(UDF_CLASSES))
@pytest.mark.parametrize("process_error", [False, True])
def test_teardown_runs_once_and_keeps_process_error(
    test_session, caplog, verb, process_error
):
    calls = []

    class Broken(UDF_CLASSES[verb]):
        def process(self, value):
            if process_error:
                raise ValueError("boom-process")
            return value if verb in ("map", "batch_map") else [1]

        def teardown(self):
            calls.append("teardown")
            raise RuntimeError("boom-teardown")

    chain = dc.read_values(value=[1, 2], session=test_session)
    with pytest.raises(
        ValueError if process_error else RuntimeError,
        match="boom-process" if process_error else "boom-teardown",
    ):
        _apply(chain, verb, result=Broken()).to_values("result")
    assert calls == ["teardown"]
    assert ("Failed to tear down" in caplog.text) == process_error


@pytest.mark.parametrize("verb", list(UDF_CLASSES))
def test_udf_instance_shared_by_chains(test_session, verb):
    process = {
        "map": lambda value: value * 2,
        "batch_map": lambda value: [v * 2 for v in value],
        "gen": lambda value: [value * 2],
        "agg": lambda value: [v * 2 for v in value],
    }[verb]

    class Twice(UDF_CLASSES[verb]):
        def process(self, value):
            return process(value)

    udf = Twice()
    chain = dc.read_values(value=[1, 2, 3], session=test_session)
    first = _apply(chain, verb, first=udf)
    second = _apply(chain, verb, second=udf)
    assert sorted(first.to_values("first")) == [2, 4, 6]
    assert sorted(second.to_values("second")) == [2, 4, 6]
    assert udf.params is None


@pytest.mark.parametrize(
    "output,value,stored",
    [(list[float], [1, 2.5], "[1.0, 2.5]"), (list[list[float]], [[1]], "[[1.0]]")],
)
def test_gen_tuple_rows_convert_list_items_to_the_column_type(
    test_session, output, value, stored
):
    def rows(n: int) -> Iterator[tuple]:
        yield (value,)

    chain = dc.read_values(n=[1], session=test_session).gen(v=rows, output=output)

    assert repr(chain.to_values("v")[0]) == stored


def test_gen_reports_generated_rows_once_per_interval(
    test_session, mocker, monkeypatch
):
    now = 0.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    callbacks = mocker.spy(dataset, "get_generated_callback")
    updates = mocker.spy(TqdmCallback, "relative_update")

    def rows(n: int) -> Iterator[int]:
        nonlocal now
        for i, seconds in enumerate([0, 0, 1, 1, 1, 2, 2]):
            now = seconds
            yield i

    chain = dc.read_values(n=[1], session=test_session).gen(v=rows, output=int)

    assert chain.count() == 7
    generated = callbacks.spy_return
    reported = [c.args[1] for c in updates.call_args_list if c.args[0] is generated]
    assert (len(reported), sum(reported)) == (4, 7)


MIN_HELD = 2 * DEFAULT_PREFETCH + 1  # rows prefetch=2 can hold
BIG = 4 * AUTO_PREFETCH_BYTES


def _files(*sizes, location=None):
    return [
        File(source="s3://bucket", path=f"f{i}", size=s, location=location)
        for i, s in enumerate(sizes)
    ]


def test_default_prefetch_raises_a_failed_download(test_session, monkeypatch):
    async def fake_prefetch(self, download_cb=None):
        if self.path == "f3":
            raise RuntimeError("download failed")
        return True

    monkeypatch.setattr(File, "_prefetch", fake_prefetch)
    chain = dc.read_values(file=_files(*[BIG] * 10), session=test_session).map(
        p=lambda file: file.path, output=str
    )
    with pytest.raises((RuntimeError, UdfRunError), match="download failed"):
        chain.to_values("p")


@pytest.mark.parametrize(
    "method,settings,expected",
    [
        ("map", {}, (AUTO_PREFETCH, AUTO_PREFETCH_BYTES, MIN_HELD)),
        ("gen", {}, (AUTO_PREFETCH, AUTO_PREFETCH_BYTES, MIN_HELD)),
        ("map", {"prefetch": True}, (AUTO_PREFETCH, AUTO_PREFETCH_BYTES, MIN_HELD)),
        ("map", {"prefetch": 2}, (2, None, 1)),
        ("map", {"prefetch": 0}, (0, None, 1)),
        ("map", {"prefetch_bytes": 1000}, (AUTO_PREFETCH, 1000, 1)),
        ("map", {"prefetch": 1, "prefetch_bytes": 1000}, (1, 1000, 1)),
        ("map", {"prefetch": 2, "prefetch_bytes": 1000}, (2, 1000, 1)),
        ("gen", {"prefetch": 4, "prefetch_bytes": 1000}, (4, 1000, 1)),
        ("batch_map", {}, (DEFAULT_PREFETCH, None, 1)),
        ("batch_map", {"prefetch_bytes": 1000}, (DEFAULT_PREFETCH, None, 1)),
    ],
)
def test_prefetch_settings(test_session, method, settings, expected):
    files = [File(path="a"), File(path="b")]
    chain = dc.read_values(file=files, session=test_session).settings(**settings)
    if method == "map":
        chain = chain.map(y=lambda file: file.path, output=str)
    elif method == "gen":
        chain = chain.gen(y=lambda file: iter([file.path]), output=str)
    else:
        chain = chain.batch_map(
            y=lambda file: [f.path for f in file], output=str, batch=2
        )
    udf = chain._query.steps[-1].udf.inner
    assert (udf.prefetch, udf.prefetch_bytes, udf._prefetch_min_rows) == expected


def test_default_prefetch_without_files(test_session):
    chain = dc.read_values(x=[1, 2], session=test_session).map(
        y=lambda x: x, output=int
    )
    udf = chain._query.steps[-1].udf.inner
    assert (udf.prefetch, udf.prefetch_bytes) == (DEFAULT_PREFETCH, None)


@pytest.mark.parametrize(
    "file_type,other,expected",
    [
        (File | None, None, ["a", "none", "b"]),
        (File | str, "skip", ["a", "skip", "b"]),
    ],
)
def test_union_file_parameters_prefetch_only_file_arm(
    test_session, monkeypatch, file_type, other, expected
):
    fetched = []

    async def fake_prefetch(self, download_cb=None):
        fetched.append(self.path)
        return True

    monkeypatch.setattr(File, "_prefetch", fake_prefetch)
    values = [
        File(source="s3://bucket", path="a", size=10),
        other,
        File(source="s3://bucket", path="b", size=20),
    ]
    chain = dc.read_values(
        file=values,
        output={"file": file_type},
        session=test_session,
    ).map(
        y=lambda file: file.path if isinstance(file, File) else (file or "none"),
        params="file",
        output=str,
    )
    udf = chain._query.steps[-1].udf.inner
    assert (udf.prefetch, udf.prefetch_bytes) == (AUTO_PREFETCH, AUTO_PREFETCH_BYTES)
    assert chain.to_values("y") == expected
    assert sorted(fetched) == ["a", "b"]


def test_nested_data_model_file_parameters_do_not_enable_auto_prefetch(test_session):
    class Holder(DataModel):
        item: File

    holder = Holder(item=File(source="s3://bucket", path="a", size=10))
    chain = dc.read_values(holder=[holder], session=test_session)
    mapped = chain.map(y=lambda holder: holder.item.path, params="holder", output=str)
    udf = mapped._query.steps[-1].udf.inner
    assert (udf.prefetch, udf.prefetch_bytes, udf._effective_prefetch) == (
        DEFAULT_PREFETCH,
        None,
        0,
    )
    assert mapped.to_values("y") == ["a"]


def test_explicit_prefetch_fetches_a_setup_file(test_session, monkeypatch):
    fetched = []

    async def fake_prefetch(self, download_cb=None):
        fetched.append(self.path)
        return True

    monkeypatch.setattr(File, "_prefetch", fake_prefetch)
    ref = File(source="s3://bucket", path="ref", size=10)
    res = (
        dc.read_values(x=[1, 2], session=test_session)
        .setup(ref=lambda: ref)
        .settings(prefetch=2)
        .map(y=lambda ref: ref.path, output=str)
        .to_values("y")
    )
    assert res == ["ref", "ref"]
    assert set(fetched) == {"ref"}


def test_a_prefetched_file_reads_its_cached_copy(test_session, monkeypatch):
    buf = io.BytesIO()
    pq.write_table(pa.table({"a": [1, 2]}), buf)
    data = buf.getvalue()

    async def fake_put_in_cache(self, file, callback=None):
        self.cache.store_data(file, data)

    def read_storage(*args, **kwargs):
        raise AssertionError("read the storage again")

    monkeypatch.setattr(Client, "_put_in_cache", fake_put_in_cache)
    monkeypatch.setattr(arrow, "pyarrow_fs", read_storage)
    res = (
        dc.read_values(
            file=[File(source="s3://bucket", path="t.parquet", size=len(data))],
            session=test_session,
        )
        .settings(prefetch=2)
        .map(
            a=lambda file: ArrowRow(file=file, index=1, kwargs={}).read()["a"],
            output=int,
        )
        .to_values("a")
    )
    assert res == [2]


@pytest.mark.parametrize(
    "files,settings,expected_while_first_in_udf",
    [
        # Default 512 MiB target holds 8 files of 64 MiB each
        (_files(*[AUTO_PREFETCH_BYTES // 8] * 12), {}, 8),
        # Files larger than the 512 MiB target still keep the 5-row default minimum
        (_files(*[BIG] * 10), {}, MIN_HELD),
        # Unsized files under default settings keep the 5-row default minimum
        (_files(*[0] * 10), {}, MIN_HELD),
        # Archive members are budgeted by their parent archive size
        (
            [
                File(
                    source="s3://bucket",
                    path=f"f{i}",
                    size=2048,
                    location=[
                        {
                            "vtype": "tar",
                            "parent": {
                                "source": "s3://bucket",
                                "path": f"archive-{i}.tar",
                                "size": BIG,
                            },
                        }
                    ],
                )
                for i in range(10)
            ],
            {},
            MIN_HELD,
        ),
        # Members sharing one archive count the archive size once
        (
            _files(
                *[2048] * 10,
                location=[
                    {
                        "vtype": "tar",
                        "parent": {
                            "source": "s3://bucket",
                            "path": "archive.tar",
                            "size": AUTO_PREFETCH_BYTES // 2,
                        },
                    }
                ],
            ),
            {},
            10,
        ),
        # Unsized files (size=0, omitted size with etag, or size=-1) claim budget
        (_files(0, 0, 0, 0), {"prefetch": 2, "prefetch_bytes": 100}, 1),
        (
            [
                File(source="s3://bucket", path=f"f{i}", etag=f"etag-{i}")
                for i in range(6)
            ],
            {"prefetch": 2, "prefetch_bytes": 100},
            1,
        ),
        (_files(-1, -1, -1, -1), {"prefetch": 2, "prefetch_bytes": 100}, 1),
        # Listed empty files take 0 bytes and hit max_rows (2*1+1=3)
        (
            [
                File(
                    source="s3://bucket",
                    path=f"f{i}",
                    size=0,
                    etag="d41d8cd98f00b204e9800998ecf8427e",
                    last_modified=datetime(2026, 1, 1, tzinfo=UTC),
                )
                for i in range(6)
            ],
            {"prefetch": 1, "prefetch_bytes": 100},
            3,
        ),
        # Explicit smaller prefetch=1 bounds rows to 2*1+1=3 even when bytes fit
        (_files(*[200] * 6), {"prefetch": 1, "prefetch_bytes": 1000}, 3),
        # Explicit prefetch_bytes keeps a single-row minimum, whatever the count
        *(
            (_files(*[2000] * 6), {"prefetch": n, "prefetch_bytes": 1000}, 1)
            for n in (1, 2, 4, AUTO_PREFETCH)
        ),
    ],
    ids=[
        "default-medium",
        "default-big",
        "default-unsized",
        "in-archive",
        "shared-archive",
        "unsized-zero",
        "omitted-size-with-etag",
        "unsized-neg",
        "listed-empty",
        "row-bound",
        *(f"single-row-minimum-{n}" for n in (1, 2, 4, AUTO_PREFETCH)),
    ],
)
def test_prefetch_budget_bounds_in_map(
    test_session, monkeypatch, files, settings, expected_while_first_in_udf
):
    started = 0
    target_reached = threading.Event()
    in_udf = []

    async def fake_prefetch(self, download_cb=None):
        nonlocal started
        started += 1
        if started >= expected_while_first_in_udf:
            target_reached.set()
        return True

    def inspect_first(file: File) -> str:
        if file.path == "f0":
            assert target_reached.wait(2)
            time.sleep(0.02)
            in_udf.append(started)
        return file.path

    monkeypatch.setattr(File, "_prefetch", fake_prefetch)
    res = (
        dc.read_values(file=files, session=test_session)
        .settings(**settings)
        .map(p=inspect_first, output=str)
        .to_values("p")
    )
    assert len(res) == len(files)
    assert in_udf == [expected_while_first_in_udf]


def test_mutated_file_in_udf_does_not_corrupt_prefetch_budget(
    test_session, monkeypatch
):
    removed = []
    orig_remove = Cache.remove

    async def fake_prefetch(self, download_cb=None):
        return True

    def track_remove(self, file):
        removed.append(file.path)
        orig_remove(self, file)

    def mutate(file: File) -> int:
        file.size = 0
        file.path = ""
        return 1

    monkeypatch.setattr(File, "_prefetch", fake_prefetch)
    monkeypatch.setattr(Cache, "remove", track_remove)
    res = (
        dc.read_values(file=_files(*[600] * 6), session=test_session)
        .settings(prefetch=2, prefetch_bytes=1000)
        .map(seen=mutate, output=int)
        .to_values("seen")
    )
    assert res == [1] * 6
    assert sorted(removed) == [f"f{i}" for i in range(6)]


@pytest.mark.parametrize(
    "method,settings",
    [
        ("map", {}),
        ("map", {"prefetch": 2}),
        ("map", {"prefetch": 2, "prefetch_bytes": 10000}),
        ("batch_map", {}),
    ],
    ids=["map-default", "map-prefetch-2", "map-prefetch-bytes", "batch-map-default"],
)
def test_archive_prefetch_downloads_once_and_keeps_parent_across_rows(
    test_session, monkeypatch, small_tar_member_data, method, settings
):
    data, raw = small_tar_member_data
    parent = File(source="s3://bucket", path="shard.tar", size=len(raw))
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        members = [
            build_tar_member(parent, archive.getmember(name))
            for name in ("prefix.bin", "member.bin") * 2
        ]
    expected = [b"prefix", data] * 2
    rows = len(members) if method == "map" else len(members) // 2

    downloads = 0
    entered = admitted = 0
    both_entered = threading.Event()
    all_admitted = threading.Event()
    removed = []
    orig_download = Client._download
    orig_remove = Cache.remove
    orig_take = _PrefetchBudget.take

    def counted_take(self, item):
        nonlocal admitted
        orig_take(self, item)
        admitted += 1
        if admitted == rows:
            all_admitted.set()

    async def synced_download(self, file, *, callback=None):
        nonlocal entered
        entered += 1
        if entered >= 2:
            both_entered.set()
        await orig_download(self, file, callback=callback)

    async def fake_put_in_cache(self, file, callback=None):
        nonlocal downloads
        downloads += 1
        # a second download waits on the lock, and every row holds the archive
        assert await asyncio.to_thread(both_entered.wait, 10)
        assert await asyncio.to_thread(all_admitted.wait, 10)
        self.cache.store_data(file, raw)

    def read_storage(self, file, block_size=None):
        raise AssertionError(f"read {file.path} from the storage")

    def track_remove(self, file):
        removed.append(file.path)
        orig_remove(self, file)

    monkeypatch.setattr(_PrefetchBudget, "take", counted_take)
    monkeypatch.setattr(Client, "_download", synced_download)
    monkeypatch.setattr(Client, "_put_in_cache", fake_put_in_cache)
    monkeypatch.setattr(Client, "_open_remote", read_storage)
    monkeypatch.setattr(Cache, "remove", track_remove)

    chain = dc.read_values(file=members, session=test_session).settings(**settings)
    if method == "map":
        chain = chain.map(data=lambda file: file.read(), output=bytes)
    else:
        chain = chain.batch_map(
            data=lambda file: [f.read() for f in file], output=bytes, batch=2
        )

    assert chain.order_by("sys.id").to_values("data") == expected
    assert downloads == 1
    assert removed == ["shard.tar"]


def test_archive_prefetch_skips_parents_over_max_size(test_session, monkeypatch):
    parent = File(source="s3://bucket", path="shard.tar", size=4096)
    loc = [{"vtype": "tar", "parent": parent.model_dump()}]
    downloads = []

    async def fake_put_in_cache(self, file, callback=None):
        downloads.append(file.path)

    monkeypatch.setattr(Client, "_put_in_cache", fake_put_in_cache)

    class SmallOnlyGen(Generator):
        prefetch_max_size = 100

        def process(self, file: File) -> Iterator[int]:
            yield file.size

    res = (
        dc.read_values(file=_files(10, 20, location=loc), session=test_session)
        .gen(sz=SmallOnlyGen(), output=int)
        .to_values("sz")
    )
    assert sorted(res) == [10, 20]
    assert downloads == []


def test_prefetch_logs_a_failed_removal(test_session, monkeypatch, caplog):
    async def fake_put_in_cache(self, file, callback=None):
        self.cache.store_data(file, b"data")

    def fail_remove(self, _target):
        raise OSError("busy")

    monkeypatch.setattr(Client, "_put_in_cache", fake_put_in_cache)
    monkeypatch.setattr(Cache, "remove", fail_remove)
    res = (
        dc.read_values(file=_files(10), session=test_session)
        .settings(prefetch=1)
        .map(sz=lambda file: file.size, output=int)
        .to_values("sz")
    )
    assert res == [10]
    assert "Failed to remove prefetched item" in caplog.text
