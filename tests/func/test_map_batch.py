import importlib
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

import datachain as dc
from datachain.lib.udf import UdfError
from datachain.lib.utils import DataChainParamsError


@pytest.mark.parametrize("batch", [1, 2, 4, 32])
@pytest.mark.parametrize("parallel", [False, 2])
def test_map_batch_matches_rows(test_session_tmpfile, batch, parallel):
    def twice(value: list[int]) -> list[int]:
        assert isinstance(value, list)
        assert 0 < len(value) <= batch
        return [v * 2 for v in value]

    def scalar(value: int) -> int:
        return value * 2

    chain = dc.read_values(value=list(range(11)), session=test_session_tmpfile)
    expected = chain.map(result=scalar).order_by("value").to_list()
    actual = (
        chain.settings(parallel=parallel)
        .batch_map(result=twice, batch=batch)
        .order_by("value")
        .to_list()
    )
    assert actual == expected


def test_map_batch_partial_and_lifecycle(test_session):
    calls = []

    class Offset(dc.BatchMapper):
        def setup(self):
            calls.append("setup")
            self.offset = 10

        def process(self, value: list[int]) -> Iterator[int]:
            calls.append(len(value))
            yield from (v + self.offset for v in value)

        def teardown(self):
            calls.append("teardown")

    result = (
        dc.read_values(value=list(range(7)), session=test_session)
        .batch_map(result=Offset(), batch=3)
        .order_by("value")
        .to_values("result")
    )
    assert result == list(range(10, 17))
    assert calls == ["setup", 3, 3, 1, "teardown"]


@pytest.mark.parametrize("batch,size", [(1, 0), (1, 2), (3, 0), (3, 2), (3, 4)])
def test_map_batch_wrong_length(test_session, batch, size):
    calls = []

    class Wrong(dc.BatchMapper):
        def process(self, value: list[int]) -> list[int]:
            return [0] * size

        def teardown(self):
            calls.append("teardown")

    with pytest.raises(UdfError, match=f"got {size} outputs for {batch} inputs"):
        dc.read_values(value=[1, 2, 3], session=test_session).batch_map(
            result=Wrong(), batch=batch
        ).to_list()
    assert calls == ["teardown"]


@pytest.mark.parametrize("batch", [0, -1, True, 2.5, "2", None])
def test_map_batch_invalid_size(test_session, batch):
    with pytest.raises(DataChainParamsError, match="batch must be a positive integer"):
        dc.read_values(value=[1], session=test_session).batch_map(
            result=lambda value: value, batch=batch
        )


def test_map_remains_scalar(test_session):
    def scalar(value: int) -> list[int]:
        return [value, value + 1]

    chain = dc.read_values(value=[1, 2], session=test_session)
    assert chain.map(result=scalar).order_by("value").to_values("result") == [
        [1, 2],
        [2, 3],
    ]


@pytest.mark.parametrize("batch", [1, 2])
def test_map_batch_nested_outputs_and_params(test_session, batch):
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
            batch=batch,
        )
        .order_by("value")
        .to_list("embedding", "upper")
    )
    assert result == [([1.0, 0.5], "A"), ([2.0, 0.5], "B"), ([3.0, 0.5], "C")]


@pytest.mark.parametrize("prefetch", [0, 2])
@pytest.mark.parametrize("file_type", [dc.File, dc.ImageFile, dc.VideoFile])
@pytest.mark.parametrize("parallel", [False, 2])
def test_map_batch_files(test_session_tmpfile, tmp_path, prefetch, file_type, parallel):
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

    files = [file_type(source=tmp_path.as_uri(), path=f"{i}.txt") for i in range(5)]
    result = (
        dc.read_values(file=files, session=test_session_tmpfile)
        .settings(prefetch=prefetch, parallel=parallel)
        .batch_map(text=read, batch=3)
        .to_values("text")
    )
    assert sorted(result) == [str(i) for i in range(5)]


@pytest.mark.parametrize("batch", [1, 3])
def test_map_batch_multi_signal_dependencies(test_session, batch):
    def twice(value: list[int]) -> list[int]:
        return [v * 2 for v in value]

    def increment(doubled: list[int]) -> Iterator[int]:
        yield from (v + 1 for v in doubled)

    result = (
        dc.read_values(value=list(range(7)), session=test_session)
        .batch_map(final=increment, doubled=twice, batch=batch)
        .order_by("value")
        .to_list("doubled", "final")
    )
    assert result == [(v * 2, v * 2 + 1) for v in range(7)]


def test_map_batch_multi_signal_wrong_length(test_session):
    def one(value: list[int]) -> list[int]:
        return [0]

    with pytest.raises(UdfError, match="entry 'a' returned 1 outputs for 3 inputs"):
        dc.read_values(value=[1, 2, 3], session=test_session).batch_map(
            a=one, b=one, batch=3
        ).to_list()


def test_map_batch_empty_input(test_session):
    def no_call(value: list[int]) -> list[int]:
        raise AssertionError("empty input should not invoke mapper")

    assert (
        dc.read_values(value=[1], session=test_session)
        .limit(0)
        .batch_map(result=no_call, batch=4)
        .to_list()
        == []
    )


@pytest.mark.parametrize("batch", [1, 3])
def test_map_batch_delta(test_session, batch):
    calls = []

    def twice(value: list[int]) -> list[int]:
        calls.extend(value)
        return [v * 2 for v in value]

    def run():
        return (
            dc.read_dataset(
                "batch_source", session=test_session, delta=True, delta_on="value"
            )
            .batch_map(result=twice, batch=batch)
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
def test_map_batch_model_output_roundtrip(test_session_tmpfile, parallel):
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


def test_map_batch_nullable_input(test_session):
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


def test_map_batch_list_input(test_session):
    def total(values: list[list[int]]) -> list[int]:
        return [sum(v) for v in values]

    result = (
        dc.read_values(values=[[1, 2], [], [3]], session=test_session)
        .batch_map(result=total, batch=2)
        .to_values("result")
    )
    assert sorted(result) == [0, 3, 3]


def test_map_batch_no_annotation_explicit_output(test_session):
    def twice(value):
        return [v * 2 for v in value]

    result = (
        dc.read_values(value=[1, 2, 3], session=test_session)
        .batch_map(result=twice, output=int, batch=2)
        .order_by("value")
        .to_values("result")
    )
    assert result == [2, 4, 6]


def test_map_batch_distributes_inference_to_all_workers(test_session_tmpfile, tmp_path):
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


@pytest.mark.parametrize("batches", [(1, 2), (2, 1)])
def test_batch_map_shared_mapper_instance(test_session, batches):
    class Twice(dc.BatchMapper):
        def process(self, value: list[int]) -> list[int]:
            return [v * 2 for v in value]

    mapper = Twice()
    source = dc.read_values(value=[1, 2, 3], session=test_session)
    chains = {
        name: source.batch_map(**{name: mapper}, batch=batch)
        for name, batch in zip(("first", "second"), batches, strict=True)
    }
    for name, chain in chains.items():
        assert sorted(chain.to_values(name)) == [2, 4, 6]
    assert mapper.params is None
    assert mapper.output is None


def test_map_batch_postponed_annotations(test_session, tmp_path, monkeypatch):
    (tmp_path / "batch_udfs.py").write_text(
        "from __future__ import annotations\n\n\n"
        "def twice(value: list[int]) -> list[int]:\n"
        "    return [v * 2 for v in value]\n\n\n"
        "def thrice(value: 'list[int]') -> 'list[int]':\n"
        "    return [v * 3 for v in value]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    udfs = importlib.import_module("batch_udfs")

    result = (
        dc.read_values(value=[1, 2, 3], session=test_session)
        .batch_map(twice=udfs.twice, batch=2)
        .batch_map(thrice=udfs.thrice, batch=2)
        .order_by("value")
        .to_list("twice", "thrice")
    )
    assert result == [(2, 3), (4, 6), (6, 9)]


def test_map_batch_local_model_with_explicit_output(test_session):
    class Doubled(BaseModel):
        value: int

    def twice(value: "list[int]") -> "list[Doubled]":
        return [Doubled(value=v * 2) for v in value]

    result = (
        dc.read_values(value=[1, 2, 3], session=test_session)
        .batch_map(doubled=twice, output=Doubled, batch=2)
        .order_by("value")
        .to_values("doubled.value")
    )
    assert result == [2, 4, 6]


def test_map_scalar_ignores_lists_in_setup_values(test_session):
    cycle: list = []
    cycle.append(cycle)

    def size(value: int, state: list) -> int:
        return value + len(state)

    result = (
        dc.read_values(value=[1, 2], session=test_session)
        .setup(state=lambda: cycle)
        .map(result=size)
        .order_by("value")
        .to_values("result")
    )
    assert result == [2, 3]


def test_batch_map_default_and_positional_batch(test_session):
    calls = []

    def sizes(value: list[int]) -> list[int]:
        calls.append(len(value))
        return value

    source = dc.read_values(value=list(range(1001)), session=test_session)
    assert sorted(source.batch_map(result=sizes).to_values("result")) == list(
        range(1001)
    )
    assert calls == [1000, 1]
    calls.clear()
    assert sorted(
        source.limit(3).batch_map(sizes, None, {"result": int}, 1).to_values("result")
    ) == [0, 1, 2]
    assert calls == [1, 1, 1]


@pytest.mark.parametrize("batched", [False, True])
def test_map_rejects_wrong_mapper_class(test_session, batched):
    mapper = dc.Mapper() if batched else dc.BatchMapper()
    source = dc.read_values(value=[1], session=test_session)
    method = source.batch_map if batched else source.map
    expected = "BatchMapper" if batched else "Mapper"
    with pytest.raises(DataChainParamsError, match=f"requires a {expected} instance"):
        method(result=mapper, output=int)


def test_scalar_map_can_create_batch_column(test_session):
    result = (
        dc.read_values(value=[1, 2], session=test_session)
        .map(batch=lambda value: value * 2, output=int)
        .order_by("value")
        .to_values("batch")
    )
    assert result == [2, 4]


@pytest.mark.parametrize("batch", [1, 3])
def test_batch_map_reuses_mapper_across_multi_signal_chains(test_session, batch):
    calls = []

    class Twice(dc.BatchMapper):
        def setup(self):
            calls.append("setup")
            self.factor = 2

        def process(self, value: list[int]) -> list[int]:
            return [v * self.factor for v in value]

        def teardown(self):
            calls.append("teardown")

    def increment(first: list[int]) -> list[int]:
        return [v + 1 for v in first]

    mapper = Twice()
    source = dc.read_values(value=[1, 2, 3], session=test_session)
    combined = source.batch_map(first=mapper, final=increment, batch=batch)
    separate = source.batch_map(other=mapper, batch=1)
    assert combined.order_by("value").to_list("first", "final") == [
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    assert separate.order_by("value").to_values("other") == [2, 4, 6]
    assert calls == ["setup", "teardown", "setup", "teardown"]
    assert mapper.params is None
    assert mapper.output is None


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("parallel", [False, 2])
@pytest.mark.parametrize("multi", [False, True])
def test_batch_map_setup_resources(test_session_tmpfile, batch, parallel, multi):
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
    result = source.batch_map(**functions, batch=batch).order_by("value")
    assert result.to_values("embedding") == [v * 2 + 10 for v in range(7)]
    if multi:
        assert result.to_values("incremented") == [v * 2 + 11 for v in range(7)]


def test_agg_keeps_nullable_model_input_behavior(test_session):
    class Item(dc.DataModel):
        value: int | None = None

    def total(item: list[Item]) -> Iterator[int]:
        yield sum(obj.value or 0 for obj in item)

    result = (
        dc.read_values(
            item=[Item(value=2), None],
            output={"item": Item | None},
            session=test_session,
        )
        .agg(total=total)
        .to_values("total")
    )
    assert result == [2]


def test_batch_map_multi_signal_inputs_are_isolated(test_session):
    def uppercase(text: list[str]) -> list[str]:
        original = text.copy()
        text.sort(key=len)
        return [value.upper() for value in original]

    def lengths(text: list[str]) -> list[int]:
        return [len(value) for value in text]

    result = (
        dc.read_values(text=["ccc", "a", "bb"], session=test_session)
        .batch_map(upper=uppercase, length=lengths, batch=3)
        .order_by("text")
        .to_list("text", "upper", "length")
    )
    assert result == [("a", "A", 1), ("bb", "BB", 2), ("ccc", "CCC", 3)]


def test_batch_map_multi_signal_dependencies_are_isolated(test_session):
    def identity(value: list[int]) -> list[int]:
        return value

    def scale(first: list[int]) -> list[int]:
        original = first.copy()
        first.sort()
        return [value * 10 for value in original]

    result = (
        dc.read_values(value=[3, 1, 2], session=test_session)
        .batch_map(first=identity, scaled=scale, batch=3)
        .order_by("value")
        .to_list("first", "scaled")
    )
    assert result == [(1, 10), (2, 20), (3, 30)]


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
        .to_values("result")
    )
    assert result == [1, 2, 3]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("process_error", [False, True])
def test_mapper_teardown_preserves_process_error(test_session, batched, process_error):
    base = dc.BatchMapper if batched else dc.Mapper

    class Broken(base):
        def process(self, value):
            if process_error:
                raise ValueError("boom-process")
            return value

        def teardown(self):
            raise RuntimeError("boom-teardown")

    source = dc.read_values(value=[1, 2], session=test_session)
    method = source.batch_map if batched else source.map
    with pytest.raises(
        ValueError if process_error else RuntimeError,
        match="boom-process" if process_error else "boom-teardown",
    ):
        method(result=Broken(), output=int).to_values("result")


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


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize(
    "invalid", ["abc", b"abc", {"a": 1, "b": 2, "c": 3}, {1, 2, 3}, None, 42]
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
        .to_values("result")
    ) == [2, 4, 6]
