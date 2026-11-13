import asyncio
import pickle
import threading
import time
from collections.abc import Iterator
from contextlib import closing

import pytest
from cloudpickle import dumps, loads
from fsspec.callbacks import DEFAULT_CALLBACK, TqdmCallback
from pydantic import BaseModel

import datachain as dc
from datachain import Mapper
from datachain.dataset import RowDict
from datachain.lib.file import File
from datachain.lib.settings import AUTO_PREFETCH, AUTO_PREFETCH_BYTES, DEFAULT_PREFETCH
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf import (
    FlatRow,
    JsonSerializationError,
    UDFBase,
    UdfError,
    UdfRunError,
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

    assert udf._parse_row(RowDict(), object(), False, DEFAULT_CALLBACK) == [value]


def test_udf_sets_stream_on_setup_file():
    file = File(path="reference.txt")
    udf = UDFBase()
    udf.params = SignalSchema({"ref": str}, setup={"ref": lambda: file})
    catalog = object()

    udf._parse_row(RowDict(), catalog, False, DEFAULT_CALLBACK)

    assert file._catalog is catalog


def test_udf_sets_stream_in_setup_model_collection():
    class Bundle(BaseModel):
        files: list[File]

    file = File(path="reference.txt")
    udf = UDFBase()
    udf.params = SignalSchema({"ref": str}, setup={"ref": lambda: Bundle(files=[file])})
    catalog = object()

    udf._parse_row(RowDict(), catalog, False, DEFAULT_CALLBACK)

    assert file._catalog is catalog


def test_udf_does_not_traverse_bare_setup_collection():
    files = [File(path="reference.txt")]
    udf = UDFBase()
    udf.params = SignalSchema({"ref": str}, setup={"ref": lambda: files})

    udf._parse_row(RowDict(), object(), False, DEFAULT_CALLBACK)

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

    def failing_insert(self, table, columns, rows, batch_size=None):
        next(iter(rows))
        raise OSError(28, "No space left on device")

    chain = dc.read_values(n=[1000], session=test_session).save("inputs")
    mocker.patch.object(
        type(test_session.catalog.warehouse), "insert_tuples", failing_insert
    )
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
def test_gen_flat_rows_convert_list_items_to_the_column_type(
    test_session, output, value, stored
):
    def rows(n: int) -> Iterator[list]:
        yield FlatRow((value,))  # type: ignore[misc]

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


def _read_ahead(monkeypatch, files, fail_path=None):
    """The default read-ahead over `files` with fake downloads, its download
    counters and the input rows it has taken."""
    from datachain.lib.udf import _prefetch_inputs

    state = {"started": 0, "in_flight": 0, "peak": 0}

    async def fake_prefetch(self, download_cb=None):
        if self.path == fail_path:
            raise RuntimeError("download failed")
        state["started"] += 1
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0.05)  # longer than the 15.6 ms Windows timer tick
        state["in_flight"] -= 1
        return True

    taken = []

    def rows():
        for row in enumerate(files):
            taken.append(row)
            yield row

    monkeypatch.setattr(File, "_prefetch", fake_prefetch)
    inputs = _prefetch_inputs(rows(), AUTO_PREFETCH, max_bytes=AUTO_PREFETCH_BYTES)
    return inputs, state, taken


@pytest.mark.parametrize(
    "files,min_in_flight,max_held",
    [
        (_files(*[2048] * 300), MIN_HELD + 1, 2 * AUTO_PREFETCH + 1),
        (_files(*[AUTO_PREFETCH_BYTES // 8] * 60), 2, 8),
        (_files(*[BIG] * 40), 2, MIN_HELD),
        (_files(*[0] * 40), 2, MIN_HELD),
        (_files(*[2048] * 40, location=[{"vtype": "tar"}]), 2, MIN_HELD),
    ],
    ids=["small", "medium", "big", "unsized", "in-archive"],
)
def test_default_prefetch_holds_files_within_the_limit(
    monkeypatch, files, min_in_flight, max_held
):
    inputs, state, _ = _read_ahead(monkeypatch, files)
    held = 0
    for done, _ in enumerate(inputs):
        held = max(held, state["started"] - done)
        time.sleep(0.001)
    assert state["peak"] >= min_in_flight
    assert held <= max_held


@pytest.mark.parametrize(
    "size,max_taken", [(2048, 2 * AUTO_PREFETCH + 2), (BIG, MIN_HELD + 1)]
)
def test_default_prefetch_takes_no_more_input_rows_than_it_holds(
    monkeypatch, size, max_taken
):
    inputs, _, taken = _read_ahead(monkeypatch, _files(*[size] * 200))
    with closing(inputs):
        next(inputs)
        time.sleep(0.3)  # the UDF is busy with the first row
        assert len(taken) <= max_taken


def test_default_prefetch_raises_a_failed_download(monkeypatch):
    inputs, _, _ = _read_ahead(monkeypatch, _files(*[BIG] * 50), fail_path="f3")
    errors = []

    def run():
        try:
            for _ in inputs:
                time.sleep(0.05)
        except RuntimeError as e:
            errors.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(10)
    assert not thread.is_alive()
    assert len(errors) == 1


@pytest.mark.parametrize(
    "method,settings,expected",
    [
        ("map", {}, (AUTO_PREFETCH, AUTO_PREFETCH_BYTES)),
        ("gen", {}, (AUTO_PREFETCH, AUTO_PREFETCH_BYTES)),
        ("map", {"prefetch": True}, (AUTO_PREFETCH, AUTO_PREFETCH_BYTES)),
        ("map", {"prefetch": 2}, (2, None)),
        ("map", {"prefetch": 0}, (0, None)),
        ("map", {"prefetch_bytes": 1000}, (AUTO_PREFETCH, 1000)),
        ("gen", {"prefetch": 4, "prefetch_bytes": 1000}, (4, 1000)),
        ("batch_map", {}, (DEFAULT_PREFETCH, None)),
        ("batch_map", {"prefetch_bytes": 1000}, (DEFAULT_PREFETCH, None)),
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
    assert (udf.prefetch, udf.prefetch_bytes) == expected


def test_default_prefetch_without_files(test_session):
    chain = dc.read_values(x=[1, 2], session=test_session).map(
        y=lambda x: x, output=int
    )
    udf = chain._query.steps[-1].udf.inner
    assert (udf.prefetch, udf.prefetch_bytes) == (DEFAULT_PREFETCH, None)
