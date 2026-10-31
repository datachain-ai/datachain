from collections.abc import Iterator

import pytest

import datachain as dc
from datachain.lib.signal_schema import SignalResolvingError
from datachain.lib.udf_signature import UdfSignature, UdfSignatureError
from datachain.query.batch import Batch, NoBatching


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (list[int], int),
        (list[list[float]], list[float]),
        (Iterator[int], int),
        (Iterator[list[float]], list[float]),
    ],
)
def test_map_batch_output_signature(annotation, expected):
    def fn(value: list[int]):
        return value

    fn.__annotations__["return"] = annotation
    sign = UdfSignature.parse("", {"result": fn}, is_generator=False, is_batch_map=True)
    assert sign.output_schema.values == {"result": expected}


@pytest.mark.parametrize("annotation", [int, tuple[int, int], list])
def test_map_batch_rejects_scalar_output(annotation):
    def fn(value: list[int]):
        return value

    fn.__annotations__["return"] = annotation
    with pytest.raises(UdfSignatureError, match=r"list\[T\] or Iterator\[T\]"):
        UdfSignature.parse("", {"result": fn}, is_generator=False, is_batch_map=True)


def test_map_batch_rejects_scalar_input(test_session):
    def fn(value: int) -> list[int]:
        return [value]

    with pytest.raises(SignalResolvingError, match="is not a list"):
        dc.read_values(value=[1], session=test_session).batch_map(result=fn, batch=2)


def test_map_batch_hash_and_strategy(test_session):
    def fn(value):
        return value

    chain = dc.read_values(value=[1], session=test_session)
    scalar = chain.map(result=fn, output=int)._query.steps[-1].udf
    adapters = [
        chain.batch_map(result=fn, output=int, batch=b)._query.steps[-1].udf
        for b in (1, 2, 3)
    ]
    assert scalar.hash() == scalar.inner.hash()
    assert isinstance(scalar.get_batching(), NoBatching)
    assert len({a.hash() for a in (scalar, *adapters)}) == 4
    assert len({a.hash(include_body=False) for a in (scalar, *adapters)}) == 4
    for batch, adapter in enumerate(adapters, 1):
        assert isinstance(adapter.get_batching(), Batch)
        assert adapter.get_batching().count == batch


@pytest.mark.parametrize("batch", [1, 3])
def test_map_batch_checkpoint_incomplete_group(test_session, batch):
    from datachain.query.dataset import UDFSignal

    def fn(value: list[int]) -> list[int]:
        return value

    chain = dc.read_values(value=[1], session=test_session).batch_map(
        result=fn, batch=batch
    )
    step = chain._query.steps[-1]
    assert isinstance(step, UDFSignal)
    table = step.create_output_table("udf_batch_incomplete")
    warehouse = test_session.catalog.warehouse
    warehouse.insert_rows(
        table,
        [
            {"sys__id": 1, "sys__input_id": 1, "sys__partial": True, "result": 1},
            {"sys__id": 2, "sys__input_id": 1, "sys__partial": False, "result": 2},
            {"sys__id": 3, "sys__input_id": 3, "sys__partial": True, "result": 3},
            {"sys__id": 4, "sys__input_id": 3, "sys__partial": True, "result": 4},
        ],
    )
    warehouse.insert_rows_done(table)
    try:
        assert step.find_incomplete_inputs(table) == [3]
    finally:
        warehouse.db.drop_table(table)


@pytest.mark.parametrize(
    "method,kwargs,hint",
    [
        ("map", {"batch": 32}, "Use batch_map"),
        ("map", {"batch_size": 32}, "database insert buffering"),
        ("batch_map", {"batch_size": 32}, "Use batch_map"),
        ("batch_map", {"parallel": 2}, "Use settings"),
        ("batch_map", {"prefetch": 8}, "Use settings"),
    ],
)
def test_map_options_have_actionable_errors(test_session, method, kwargs, hint):
    from datachain.lib.utils import DataChainParamsError

    def encode(value: list[int]) -> list[int]:
        return value

    source = dc.read_values(value=[1], session=test_session)
    with pytest.raises(DataChainParamsError, match=hint) as exc:
        getattr(source, method)(result=encode, **kwargs)
    assert next(iter(kwargs)) in str(exc.value)


def test_batch_map_multi_output_hint_is_literal(test_session):
    from datachain.lib.utils import DataChainParamsError

    def other(value: list[int]) -> list[int]:
        return value

    source = dc.read_values(value=[1], session=test_session)
    with pytest.raises(DataChainParamsError, match=r"output=\{\.\.\.\}"):
        source.map(
            result=dc.llm.complete("value", include_usage=True),
            other=other,
        )


def test_batch_map_can_name_output_batch_size(test_session):
    def encode(value: list[int]) -> list[int]:
        return value

    result = (
        dc.read_values(value=[1, 2], session=test_session)
        .batch_map(batch_size=encode, batch=1)
        .order_by("value")
        .to_values("batch_size")
    )
    assert result == [1, 2]


@pytest.mark.parametrize("multi", [False, True])
def test_batch_map_validates_parameter_annotations_with_bindings(test_session, multi):
    def bad(value: int) -> list[int]:
        return [value]

    def good(value: list[int]) -> list[int]:
        return value

    source = dc.read_values(value=[1], session=test_session)
    with pytest.raises(SignalResolvingError, match="is not a list"):
        if multi:
            source.batch_map(bad=bad, good=good)
        else:
            source.batch_map(result=bad, params=["value"])


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("operation", [dc.llm.embed, dc.llm.complete])
def test_batch_map_rejects_scalar_llm_operations(test_session, multi, operation):
    from datachain.llm.spec import LLMConfigError

    def good(text: list[str]) -> list[str]:
        return text

    source = dc.read_values(text=["hello"], session=test_session)
    with pytest.raises(LLMConfigError, match=r"use \.map\(\)"):
        if multi:
            source.batch_map(result=operation("text"), good=good)
        else:
            source.batch_map(result=operation("text"))


def test_cyclic_string_annotation_does_not_hang():
    import subprocess
    import sys

    script = (
        "from datachain.lib.udf_signature import _resolve_annotation\n"
        "Alias = 'Alias'\n"
        "def fn(): pass\n"
        "assert _resolve_annotation('Alias', fn) == 'Alias'\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)  # noqa: S603
