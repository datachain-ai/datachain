import importlib
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import datachain as dc
from datachain.lib.udf import UdfError
from datachain.lib.utils import DataChainParamsError


@pytest.mark.parametrize(
    "rows,batch,sizes",
    [(7, 3, [3, 3, 1]), (1001, 1000, [1000, 1]), (3, 1, [1, 1, 1]), (0, 4, [])],
)
def test_batch_map_call_sizes(test_session, rows, batch, sizes):
    calls = []

    class Identity(dc.BatchMapper):
        def setup(self):
            calls.append("setup")

        def process(self, value: list[int]) -> Iterator[int]:
            calls.append(len(value))
            yield from value

        def teardown(self):
            calls.append("teardown")

    result = (
        dc.read_values(value=list(range(1001)), session=test_session)
        .order_by("value")
        .limit(rows)
        .batch_map(result=Identity(), batch=batch)
        .to_values("result")
    )
    assert sorted(result) == list(range(rows))
    assert calls == (["setup", *sizes, "teardown"] if sizes else [])


@pytest.mark.parametrize("batch", [32, 1000])
def test_batch_map_one_worker_bounds_input_chunks(
    test_session_tmpfile, tmp_path, monkeypatch, batch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sitecustomize.py").write_text(
        "from datachain.query import dispatch\ndispatch.cpu_count = lambda: 1\n"
    )
    marker = tmp_path / "chunks"

    class Bounded(dc.BatchMapper):
        def setup(self):
            from unittest.mock import patch

            from datachain.data_storage.warehouse import AbstractWarehouse

            original = AbstractWarehouse.dataset_rows_select_from_ids

            def bounded(warehouse, query, ids, is_batched):
                ids = list(ids)
                rows = sum(len(group) for group in ids)
                assert rows <= 10000
                with marker.open("a") as output:
                    output.write(f"{rows}\n")
                return original(warehouse, query, ids, is_batched)

            self.patch = patch.object(
                AbstractWarehouse, "dataset_rows_select_from_ids", bounded
            )
            self.patch.start()

        def process(self, value: list[int]) -> list[int]:
            return value

        def teardown(self):
            self.patch.stop()

    values = list(range(20001))
    result = (
        dc.read_values(value=values, session=test_session_tmpfile)
        .settings(parallel=True, prefetch=0)
        .batch_map(result=Bounded(), batch=batch)
        .to_values("result")
    )
    assert sorted(result) == values
    assert marker.read_text().splitlines()


@pytest.mark.parametrize("parallel", [False, 2])
def test_batch_map_matches_map(test_session_tmpfile, parallel):
    def twice(value: list[int]) -> list[int]:
        assert isinstance(value, list)
        assert 0 < len(value) <= 4
        return [v * 2 for v in value]

    def scalar(value: int) -> int:
        return value * 2

    chain = dc.read_values(value=list(range(11)), session=test_session_tmpfile)
    expected = chain.map(result=scalar).order_by("value").to_list()
    actual = (
        chain.settings(parallel=parallel)
        .batch_map(result=twice, batch=4)
        .order_by("value")
        .to_list()
    )
    assert actual == expected


@pytest.mark.parametrize("batch,size", [(1, 0), (1, 2), (3, 0), (3, 2), (3, 4)])
def test_batch_map_wrong_length(test_session, batch, size):
    calls = []

    class Wrong(dc.BatchMapper):
        def process(self, value: list[int]) -> list[int]:
            return [0] * size

        def teardown(self):
            calls.append("teardown")

    got = size if size < batch else f"more than {batch}"
    with pytest.raises(UdfError, match=f"got {got} outputs for {batch} inputs"):
        dc.read_values(value=[1, 2, 3], session=test_session).batch_map(
            result=Wrong(), batch=batch
        ).to_list()
    assert calls == ["teardown"]


@pytest.mark.parametrize("multi", [False, True])
def test_batch_map_reads_at_most_one_extra_result(test_session, multi):
    pulled = []
    closed = []

    def too_many(value: list[int]) -> Iterator[int]:
        try:
            for i in range(100):
                pulled.append(i)
                yield i
        finally:
            closed.append(True)

    def other(value: list[int]) -> list[int]:
        return value

    functions = {"result": too_many, "other": other} if multi else {"result": too_many}
    label = "entry 'result'" if multi else "function 'too_many'"
    with pytest.raises(
        UdfError,
        match=f"{label} must return one output per input row: "
        "got more than 2 outputs for 2 inputs",
    ):
        dc.read_values(value=[1, 2], session=test_session).batch_map(
            **functions, batch=2
        ).to_list()
    assert pulled == [0, 1, 2]
    assert closed == [True]


@pytest.mark.parametrize("batch", [0, -1, True, 2.5, "2", None])
def test_batch_map_invalid_size(test_session, batch):
    with pytest.raises(DataChainParamsError, match="batch must be a positive integer"):
        dc.read_values(value=[1], session=test_session).batch_map(
            result=lambda value: value, batch=batch
        )


def test_batch_map_nested_outputs_and_params(test_session):
    def embeddings(
        values: list[int], names: list[str]
    ) -> list[tuple[list[float], str]]:
        return [
            ([float(v), 0.5], n.upper()) for v, n in zip(values, names, strict=True)
        ]

    result = (
        dc.read_values(value=[1, 2, 3], name=["a", "b", "c"], session=test_session)
        .batch_map(
            embeddings,
            params=["value", "name"],
            output=["embedding", "upper"],
            batch=2,
        )
        .order_by("value")
        .to_list("embedding", "upper")
    )
    assert result == [([1.0, 0.5], "A"), ([2.0, 0.5], "B"), ([3.0, 0.5], "C")]


@pytest.mark.parametrize("prefetch", [0, 2])
@pytest.mark.parametrize("parallel", [False, 2])
def test_batch_map_files(test_session_tmpfile, tmp_path, prefetch, parallel):
    for i in range(5):
        (tmp_path / f"{i}.txt").write_text(str(i))

    previous = []

    def read(file: list[dc.File]) -> list[str]:
        for f in previous:
            assert not f._catalog.cache.contains(f)
        for f in file:
            assert f._catalog.cache.contains(f) == bool(prefetch)
        previous[:] = file
        result = []
        for f in file:
            with f.open() as stream:
                result.append(stream.read().decode())
        return result

    files = [dc.File(source=tmp_path.as_uri(), path=f"{i}.txt") for i in range(5)]
    result = (
        dc.read_values(file=files, session=test_session_tmpfile)
        .settings(prefetch=prefetch, parallel=parallel)
        .batch_map(text=read, batch=3)
        .to_values("text")
    )
    assert sorted(result) == [str(i) for i in range(5)]


def test_batch_map_multi_signal_dependencies(test_session):
    def twice(value: list[int]) -> list[int]:
        return [v * 2 for v in value]

    def increment(doubled: list[int]) -> Iterator[int]:
        yield from (v + 1 for v in doubled)

    result = (
        dc.read_values(value=list(range(7)), session=test_session)
        .batch_map(final=increment, doubled=twice, batch=3)
        .order_by("value")
        .to_list("doubled", "final")
    )
    assert result == [(v * 2, v * 2 + 1) for v in range(7)]


@pytest.mark.parametrize("reads", ["value", "first"])
def test_batch_map_multi_signal_entries_get_own_lists(test_session, reads):
    def first(value: list[int]) -> list[int]:
        return value

    def sort_value(value: list[int]) -> list[int]:
        original = value.copy()
        value.sort()
        return [v * 10 for v in original]

    def sort_first(first: list[int]) -> list[int]:
        original = first.copy()
        first.sort()
        return [v * 10 for v in original]

    result = (
        dc.read_values(value=[3, 1, 2], session=test_session)
        .batch_map(
            first=first,
            scaled=sort_first if reads == "first" else sort_value,
            batch=3,
        )
        .order_by("value")
        .to_list("first", "scaled")
    )
    assert result == [(1, 10), (2, 20), (3, 30)]


def test_batch_map_delta(test_session):
    calls = []

    def twice(value: list[int]) -> list[int]:
        calls.extend(value)
        return [v * 2 for v in value]

    def run():
        return (
            dc.read_dataset(
                "batch_source", session=test_session, delta=True, delta_on="value"
            )
            .batch_map(result=twice, batch=3)
            .save("batch_delta")
        )

    dc.read_values(value=[1, 2, 3, 4], session=test_session).save("batch_source")
    run()
    assert sorted(calls) == [1, 2, 3, 4]
    calls.clear()
    dc.read_values(value=[1, 2, 3, 4, 5, 6], session=test_session).save("batch_source")
    result = run().order_by("value").to_values("result")
    assert result == [2, 4, 6, 8, 10, 12]
    assert sorted(calls) == [5, 6]


@pytest.mark.parametrize("parallel", [False, 2])
def test_batch_map_model_output_roundtrip(test_session_tmpfile, parallel):
    class BatchResult(dc.DataModel):
        value: int
        vector: list[float]

    def encode(value: list[int]) -> list[BatchResult]:
        return [BatchResult(value=v, vector=[float(v)]) for v in value]

    result = (
        dc.read_values(value=[1, 2, 3], session=test_session_tmpfile)
        .settings(parallel=parallel)
        .batch_map(result=encode, batch=2)
        .save("batch_models")
    )
    loaded = dc.read_dataset(result.name, session=test_session_tmpfile)
    assert loaded.order_by("value").to_values("result") == [
        BatchResult(value=v, vector=[float(v)]) for v in [1, 2, 3]
    ]


def test_batch_map_nullable_input(test_session):
    def encode(value: list[int | None]) -> list[int]:
        return [v if v is not None else -1 for v in value]

    chain = dc.read_values(
        value=[1, None, 3], output={"value": int | None}, session=test_session
    )
    assert sorted(chain.batch_map(result=encode, batch=2).to_values("result")) == [
        -1,
        1,
        3,
    ]


@pytest.mark.parametrize("verb", ["map", "batch_map", "agg"])
def test_batch_map_reads_null_models_like_map_and_agg(test_session, verb):
    class Item(dc.DataModel):
        value: int | None = None

    seen = []

    def one(item: Item | None) -> int:
        seen.append(item)
        return 0

    def many(item: list[Item]) -> list[int]:
        seen.extend(item)
        return [0] * len(item)

    def group(item: list[Item]) -> Iterator[int]:
        seen.extend(item)
        yield 0

    chain = dc.read_values(
        item=[Item(value=2), None], output={"item": Item | None}, session=test_session
    )
    if verb == "map":
        chain.map(result=one).to_list()
    elif verb == "batch_map":
        chain.batch_map(result=many, batch=2).to_list()
    else:
        chain.agg(result=group).to_list()
    assert sorted(seen, key=repr) == [Item(value=2), Item(value=None)]


def test_batch_map_list_input(test_session):
    def total(values: list[list[int]]) -> list[int]:
        return [sum(v) for v in values]

    result = (
        dc.read_values(values=[[1, 2], [], [3]], session=test_session)
        .batch_map(result=total, batch=2)
        .to_values("result")
    )
    assert sorted(result) == [0, 3, 3]


def test_batch_map_no_annotation_explicit_output(test_session):
    def twice(value):
        return [v * 2 for v in value]

    result = (
        dc.read_values(value=[1, 2, 3], session=test_session)
        .batch_map(result=twice, output=int, batch=2)
        .order_by("value")
        .to_values("result")
    )
    assert result == [2, 4, 6]


def test_batch_map_distributes_inference_to_all_workers(test_session_tmpfile, tmp_path):
    marker_dir = tmp_path / "workers"
    marker_dir.mkdir()

    def worker_id(value: list[int]) -> list[int]:
        pid = os.getpid()
        Path(marker_dir, str(pid)).touch()
        deadline = time.monotonic() + 20
        while len(list(marker_dir.iterdir())) < 4:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "inference batches were not distributed to all workers"
                )
            time.sleep(0.01)
        return [pid] * len(value)

    result = (
        dc.read_values(value=list(range(64)), session=test_session_tmpfile)
        .settings(parallel=4, prefetch=0)
        .batch_map(worker=worker_id, batch=4)
        .to_values("worker")
    )
    assert len(result) == 64
    assert len(set(result)) == 4


def test_batch_map_and_map_reject_each_others_mapper_class(test_session):
    source = dc.read_values(value=[1], session=test_session)
    with pytest.raises(DataChainParamsError, match="requires a BatchMapper instance"):
        source.batch_map(result=dc.Mapper(), output=int, batch=2)
    with pytest.raises(DataChainParamsError, match="requires a Mapper instance"):
        source.map(result=dc.BatchMapper(), output=int)


@pytest.mark.parametrize("parallel", [False, 2])
@pytest.mark.parametrize("multi", [False, True])
def test_batch_map_setup_resources(test_session_tmpfile, parallel, multi):
    class Model:
        def encode(self, values):
            return [v * 2 for v in values]

    def encode(value: list[int], model, offsets: list[int]) -> list[int]:
        return [v + offsets[0] for v in model.encode(value)]

    def increment(embedding: list[int]) -> list[int]:
        return [v + 1 for v in embedding]

    source = (
        dc.read_values(value=list(range(7)), session=test_session_tmpfile)
        .settings(parallel=parallel)
        .setup(model=Model, offsets=lambda: [10])
    )
    functions = {"embedding": encode}
    if multi:
        functions["incremented"] = increment
    result = source.batch_map(**functions, batch=3).order_by("value")
    assert result.to_values("embedding") == [v * 2 + 10 for v in range(7)]
    if multi:
        assert result.to_values("incremented") == [v * 2 + 11 for v in range(7)]


def test_batch_map_does_not_prefetch_setup_collections(
    test_session, tmp_path, monkeypatch
):
    async def unexpected_prefetch(self, download_cb=None):
        pytest.fail("a setup collection must not be walked for prefetch")

    monkeypatch.setattr(dc.File, "_prefetch", unexpected_prefetch)

    def identity(value: list[int], references: list[dc.File]) -> list[int]:
        assert len(references) == 1
        assert references[0].path == "reference.txt"
        return value

    result = (
        dc.read_values(value=[1, 2, 3], session=test_session)
        .setup(
            references=lambda: [dc.File(source=tmp_path.as_uri(), path="reference.txt")]
        )
        .batch_map(result=identity, batch=2)
        .order_by("value")
        .to_values("result")
    )
    assert result == [1, 2, 3]


@pytest.mark.parametrize("output", [list[float], {"embedding": list[float]}])
def test_batch_map_explicit_output_overrides_array_annotation(test_session, output):
    import numpy as np

    def encode(value: list[int]) -> np.ndarray:
        return np.array([[v, 0.5] for v in value])

    result = (
        dc.read_values(value=[1, 2, 3], session=test_session)
        .batch_map(embedding=encode, batch=2, output=output)
        .order_by("value")
        .to_values("embedding")
    )
    assert result == [[1.0, 0.5], [2.0, 0.5], [3.0, 0.5]]


@pytest.mark.parametrize(
    "multi,invalid",
    [
        (False, "abc"),
        (False, b"abc"),
        (False, {"a": 1}),
        (False, {1, 2}),
        (False, None),
        (False, 42),
        (True, "abc"),
    ],
)
def test_batch_map_rejects_non_batch_returns(test_session, multi, invalid):
    def bad(value):
        return invalid

    def good(value: list[str]) -> list[str]:
        return value

    source = dc.read_values(value=["x", "y", "z"], session=test_session)
    with pytest.raises(UdfError, match=r"batch_map\(\).*row results"):
        if multi:
            source.batch_map(broken=bad, good=good, batch=3).to_list()
        else:
            source.batch_map(result=bad, batch=3).to_list()


@pytest.mark.parametrize("style", ["plain", "partial", "callable"])
def test_batch_map_resolves_nested_annotations(
    test_session, tmp_path, monkeypatch, style
):
    from functools import partial

    module_name = f"typed_batch_{style}"
    (tmp_path / f"{module_name}.py").write_text(
        "from __future__ import annotations\n"
        "import datachain as dc\n"
        "class Result(dc.DataModel):\n"
        "    value: int\n"
        "def encode(offset: int, value: list['int']) -> list['Result']:\n"
        "    return [Result(value=v + offset) for v in value]\n"
        "def plain(value: list['int']) -> list['Result']:\n"
        "    return encode(0, value)\n"
        "class Encoder:\n"
        "    def __call__(self, value: list['int']) -> list['Result']:\n"
        "        return encode(0, value)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(module_name)
    functions = {
        "plain": module.plain,
        "partial": partial(module.encode, 0),
        "callable": module.Encoder(),
    }
    result = (
        dc.read_values(value=[1, 2, 3], session=test_session)
        .batch_map(result=functions[style], params=["value"], batch=2)
        .save(f"typed_{style}")
        .order_by("value")
        .to_values("result")
    )
    assert result == [module.Result(value=v) for v in [1, 2, 3]]


def test_batch_map_accepts_any_element_annotation(test_session):
    from typing import Any

    def encode(value: list[Any]) -> list[int]:
        return [v * 2 for v in value]

    assert (
        dc.read_values(value=[1, 2, 3], session=test_session)
        .batch_map(result=encode, batch=2)
        .order_by("value")
        .to_values("result")
    ) == [2, 4, 6]
