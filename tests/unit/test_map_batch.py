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
    "rows,batch,workers,expected",
    [(500, 32, 4, 3), (3, 32, 4, 1), (100000, 32, 4, 312), (500, 1, 4, 125)],
)
def test_map_batch_dispatch_task_size(rows, batch, workers, expected):
    from datachain.query.dispatch import UDFDispatcher

    dispatcher = UDFDispatcher.__new__(UDFDispatcher)
    dispatcher.rows_total = rows
    dispatcher.batching = Batch(batch) if batch > 1 else NoBatching()
    assert dispatcher.input_batch_size(workers) == expected
