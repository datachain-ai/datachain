import os
import pickle
import subprocess
import sys

import pytest
from cloudpickle import dumps, loads
from fsspec.callbacks import DEFAULT_CALLBACK
from pydantic import BaseModel

import datachain as dc
from datachain import Mapper
from datachain.dataset import RowDict
from datachain.hash_utils import hash_value
from datachain.lib.file import File
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf import JsonSerializationError, UDFBase, UdfError, UdfRunError
from datachain.lib.utils import DataChainError
from tests.utils import is_sha256_hex

from .test_udf_signature import get_sign


class _OpaqueConstructorValue:
    pass


class _HashableConstructorValue:
    def __hash__(self):
        return 1


class _CallableConstructorValue:
    def __call__(self, value):
        return value


def _constructor_function(value):
    return value


def _make_constructor_closure(captured):
    def closure(value):
        return value, captured

    return closure


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

    sign = get_sign(MyMapper, params=[], output="res")
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


def test_class_udf_hash_varies_with_instance_state():
    class Limited(Mapper):
        def __init__(self, limit: int):
            super().__init__()
            self.limit = limit

        def process(self, x: int) -> int:
            return x + self.limit

    sign_a = get_sign(Limited(0), output="y")
    sign_b = get_sign(Limited(3), output="y")
    udf_a = Mapper._create(sign_a, sign_a.output_schema)
    udf_b = Mapper._create(sign_b, sign_b.output_schema)
    assert udf_a.hash() != udf_b.hash()


def test_class_udf_hash_is_deterministic_across_instances():
    class Limited(Mapper):
        def __init__(self, limit: int):
            super().__init__()
            self.limit = limit

        def process(self, x: int) -> int:
            return x + self.limit

    sign_a = get_sign(Limited(3), output="y")
    sign_b = get_sign(Limited(3), output="y")
    udf_a = Mapper._create(sign_a, sign_a.output_schema)
    udf_b = Mapper._create(sign_b, sign_b.output_schema)
    assert udf_a.hash() == udf_b.hash()


@pytest.mark.parametrize(
    "first_config,second_config",
    [
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        (
            {"options": {"a": 1, "b": 2}},
            {"options": {"b": 2, "a": 1}},
        ),
    ],
)
def test_class_udf_hash_preserves_constructor_dict_order(first_config, second_config):
    class Configured(Mapper):
        def __init__(self, config):
            self.config = config

        def process(self, x: int) -> str:
            return ",".join(self.config)

    first = Configured(first_config)
    second = Configured(second_config)
    sign_a = get_sign(first, output="y")
    sign_b = get_sign(second, output="y")
    udf_a = Mapper._create(sign_a, sign_a.output_schema)
    udf_b = Mapper._create(sign_b, sign_b.output_schema)

    assert udf_a.hash() != udf_b.hash()


def test_class_udf_unsupported_constructor_value_disables_cache_reuse(caplog):
    class Opaque:
        def __repr__(self):
            return "Opaque()"

    class Configured(Mapper):
        def __init__(self, config: Opaque):
            self.config = config

        def process(self, x: int) -> int:
            return x

    first = Configured(Opaque())
    second = Configured(Opaque())
    sign_a = get_sign(first, output="y")
    sign_b = get_sign(second, output="y")
    udf_a = Mapper._create(sign_a, sign_a.output_schema)
    udf_b = Mapper._create(sign_b, sign_b.output_schema)

    assert udf_a.hash() == udf_a.hash()
    assert udf_a.hash() != udf_b.hash()
    assert "cache reuse across UDF instances is disabled" in caplog.text


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_OpaqueConstructorValue(), id="custom-object"),
        pytest.param(_HashableConstructorValue(), id="object-with-hash"),
        pytest.param(_constructor_function, id="function"),
        pytest.param(lambda value: value, id="lambda"),
        pytest.param(_make_constructor_closure("captured"), id="closure"),
        pytest.param(_CallableConstructorValue(), id="callable-object"),
        pytest.param(
            {"options": [{"client": _OpaqueConstructorValue()}]},
            id="nested-custom-object",
        ),
    ],
)
def test_class_udf_unsupported_constructor_values_do_not_reuse_cache(config):
    class Configured(Mapper):
        def __init__(self, value):
            self.value = value

        def process(self, x: int) -> int:
            return x

    first = Configured(config)
    second = Configured(config)
    sign_a = get_sign(first, output="y")
    sign_b = get_sign(second, output="y")
    udf_a = Mapper._create(sign_a, sign_a.output_schema)
    udf_b = Mapper._create(sign_b, sign_b.output_schema)

    assert udf_a.hash() == udf_a.hash()
    assert udf_a.hash() != udf_b.hash()


def test_class_udf_cyclic_constructor_value_disables_cache_reuse():
    class Configured(Mapper):
        def __init__(self, config):
            self.config = config

        def process(self, x: int) -> int:
            return x

    config = {}
    config["self"] = config
    first = Configured(config)
    second = Configured(config)
    sign_a = get_sign(first, output="y")
    sign_b = get_sign(second, output="y")
    udf_a = Mapper._create(sign_a, sign_a.output_schema)
    udf_b = Mapper._create(sign_b, sign_b.output_schema)

    assert udf_a.hash() == udf_a.hash()
    assert udf_a.hash() != udf_b.hash()


@pytest.mark.parametrize(
    "first_key,second_key,matches",
    [
        ("tokenizer-v1", "tokenizer-v1", True),
        ("tokenizer-v1", "tokenizer-v2", False),
    ],
)
def test_class_udf_identity_hash_overrides_opaque_constructor_fallback(
    first_key, second_key, matches, caplog
):
    class Opaque:
        pass

    class Configured(Mapper):
        def __init__(self, config: Opaque, cache_key: str):
            self.config = config
            self.cache_key = cache_key

        def identity_hash(self) -> str:
            return hash_value(self.cache_key)

        def process(self, x: int) -> int:
            return x

    first = Configured(Opaque(), first_key)
    second = Configured(Opaque(), second_key)
    sign_a = get_sign(first, output="y")
    sign_b = get_sign(second, output="y")
    udf_a = Mapper._create(sign_a, sign_a.output_schema)
    udf_b = Mapper._create(sign_b, sign_b.output_schema)

    assert (udf_a.hash() == udf_b.hash()) is matches
    assert "cache reuse across UDF instances is disabled" not in caplog.text


def test_class_udf_identity_hash_replaces_automatic_constructor_hash():
    class Configured(Mapper):
        def __init__(self, limit: int, extra: str):
            self.limit = limit
            self.extra = extra

        def identity_hash(self) -> str:
            return hash_value(self.extra)

        def process(self, x: int) -> int:
            return x + self.limit

    first = Configured(3, "shared")
    second = Configured(5, "shared")
    sign_a = get_sign(first, output="y")
    sign_b = get_sign(second, output="y")
    udf_a = Mapper._create(sign_a, sign_a.output_schema)
    udf_b = Mapper._create(sign_b, sign_b.output_schema)

    assert udf_a.hash() == udf_b.hash()


def test_class_udf_hash_override_can_call_super():
    class Configured(Mapper):
        def __init__(self, limit: int):
            self.limit = limit

        def process(self, x: int) -> int:
            return x + self.limit

        def hash(self, include_body: bool = True) -> str:
            return super().hash(include_body)

    first = Configured(3)
    second = Configured(5)
    sign_a = get_sign(first, output="y")
    sign_b = get_sign(second, output="y")
    udf_a = Mapper._create(sign_a, sign_a.output_schema)
    udf_b = Mapper._create(sign_b, sign_b.output_schema)

    assert udf_a.hash() != udf_b.hash()


@pytest.mark.parametrize(
    "invalid_hash",
    ["tokenizer-v1", None, "ab" * 16 + " " * 32],
)
def test_class_udf_identity_hash_rejects_invalid_hash(invalid_hash):
    class Configured(Mapper):
        def identity_hash(self) -> str:
            return invalid_hash

        def process(self, x: int) -> int:
            return x

    udf = Configured()
    sign = get_sign(udf, output="y")
    udf = Mapper._create(sign, sign.output_schema)

    with pytest.raises(
        ValueError,
        match=r"identity_hash\(\) must return a SHA-256 hexadecimal string",
    ):
        udf.hash()


@pytest.mark.parametrize(
    "args,kwargs,matches_default",
    [
        ((), {}, True),
        ((3,), {}, True),
        ((), {"limit": 3}, True),
        ((4,), {}, False),
    ],
)
def test_class_udf_captures_normalized_constructor_arguments(
    args, kwargs, matches_default
):
    class Limited(Mapper):
        def __init__(self, limit: int = 3):
            self.limit = limit

        def process(self, x: int) -> int:
            return x + self.limit

    baseline = Limited(limit=3)
    udf = Limited(*args, **kwargs)

    assert (
        udf._constructor_identity_hash == baseline._constructor_identity_hash
    ) is matches_default


class _PydanticShapeA(BaseModel):
    value: int


class _PydanticShapeB(BaseModel):
    value: str  # different shape


@pytest.mark.parametrize(
    "first_arg,second_arg,matches",
    [
        (_PydanticShapeA, _PydanticShapeA, True),
        (_PydanticShapeA, _PydanticShapeB, False),
        ([_PydanticShapeA], [_PydanticShapeA], True),
        ({"s": _PydanticShapeA}, {"s": _PydanticShapeA}, True),
        ((_PydanticShapeA,), (_PydanticShapeA,), True),
    ],
    ids=["same-class", "different-shape", "in-list", "in-dict", "in-tuple"],
)
def test_class_udf_hashes_pydantic_class_arg(first_arg, second_arg, matches, caplog):
    class Configured(Mapper):
        def __init__(self, schema):
            self.schema = schema

        def process(self, x: int) -> int:
            return x

    first = Configured(first_arg)
    second = Configured(second_arg)

    assert (
        first._constructor_identity_hash == second._constructor_identity_hash
    ) is matches
    assert "cache reuse across UDF instances is disabled" not in caplog.text


def test_class_udf_hash_survives_subclass_custom_new():
    class Custom(Mapper):
        def __new__(cls, limit):
            return super().__new__(cls)

        def __init__(self, limit: int):
            self.limit = limit

        def process(self, x: int) -> int:
            return x + self.limit

    sign = get_sign(Custom(3), output="y")
    udf = Mapper._create(sign, sign.output_schema)

    assert is_sha256_hex(udf.hash())


def test_class_udf_constructor_hash_survives_cloudpickle_roundtrip():
    class Limited(Mapper):
        def __init__(self, limit: int):
            self.limit = limit

        def process(self, x: int) -> int:
            return x + self.limit

    udf = Limited(3)
    restored = loads(dumps(udf))

    assert restored._constructor_identity_hash == udf._constructor_identity_hash


def test_class_udf_hash_is_deterministic_across_processes():
    code = """
from datachain import Mapper
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf_signature import UdfSignature

class Limited(Mapper):
    def __init__(self, limit=3, labels=frozenset({"a", "b", "c"})):
        self.limit = limit
        self.labels = labels

    def process(self, x):
        return x + self.limit

udf = Limited()
sign = UdfSignature(
    udf,
    SignalSchema({"x": int}),
    SignalSchema({"y": int}),
)
print(Mapper._create(sign, sign.output_schema).hash())
"""

    hashes = {
        subprocess.check_output(  # noqa: S603
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("1", "2", "random")
    }

    assert len(hashes) == 1


def test_class_udf_hash_without_body_ignores_process_implementation():
    def make_udf(add: bool):
        class Stateful(Mapper):
            def __init__(self, limit: int = 3):
                self.limit = limit

            if add:

                def process(self, x: int) -> int:
                    return x + self.limit

            else:

                def process(self, x: int) -> int:
                    return x * self.limit

        sign = get_sign(Stateful(), output="y")  # type: ignore[arg-type]
        return Mapper._create(sign, sign.output_schema)

    added = make_udf(add=True)
    multiplied = make_udf(add=False)

    assert added.hash(include_body=False) == multiplied.hash(include_body=False)
    assert added.hash() != multiplied.hash()


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
