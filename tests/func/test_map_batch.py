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


@pytest.mark.parametrize("batch", [2, 4, 32])
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
        .map(result=twice, batch=batch)
        .order_by("value")
        .to_list()
    )
    assert actual == expected


def test_map_batch_partial_and_lifecycle(test_session):
    calls = []

    class Offset(dc.Mapper):
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
        .map(result=Offset(), batch=3)
        .order_by("value")
        .to_values("result")
    )
    assert result == list(range(10, 17))
    assert calls == ["setup", 3, 3, 1, "teardown"]


@pytest.mark.parametrize("size", [0, 2, 4])
def test_map_batch_wrong_length(test_session, size):
    calls = []

    class Wrong(dc.Mapper):
        def process(self, value: list[int]) -> list[int]:
            return [0] * size

        def teardown(self):
            calls.append("teardown")

    with pytest.raises(UdfError, match=f"got {size} outputs for 3 inputs"):
        dc.read_values(value=[1, 2, 3], session=test_session).map(
            result=Wrong(), batch=3
        ).to_list()
    assert calls == ["teardown"]


@pytest.mark.parametrize("batch", [0, -1, True, 2.5, "2", None])
def test_map_batch_invalid_size(test_session, batch):
    with pytest.raises(DataChainParamsError, match="batch must be a positive integer"):
        dc.read_values(value=[1], session=test_session).map(
            result=lambda value: value, batch=batch
        )


def test_map_batch_one_is_scalar(test_session):
    def scalar(value: int) -> list[int]:
        return [value, value + 1]

    chain = dc.read_values(value=[1, 2], session=test_session)
    assert chain.map(result=scalar, batch=1).order_by("value").to_values("result") == [
        [1, 2],
        [2, 3],
    ]


def test_map_batch_nested_outputs_and_params(test_session):
    def embeddings(
        values: list[int], names: list[str]
    ) -> list[tuple[list[float], str]]:
        return [
            ([float(v), 0.5], n.upper()) for v, n in zip(values, names, strict=True)
        ]

    result = (
        dc.read_values(value=[1, 2, 3], name=["a", "b", "c"], session=test_session)
        .map(
            embeddings, params=["value", "name"], output=["embedding", "upper"], batch=2
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
        .map(text=read, batch=3)
        .to_values("text")
    )
    assert sorted(result) == [str(i) for i in range(5)]


def test_map_batch_multi_signal_dependencies(test_session):
    def twice(value: list[int]) -> list[int]:
        return [v * 2 for v in value]

    def increment(doubled: list[int]) -> Iterator[int]:
        yield from (v + 1 for v in doubled)

    result = (
        dc.read_values(value=list(range(7)), session=test_session)
        .map(final=increment, doubled=twice, batch=3)
        .order_by("value")
        .to_list("doubled", "final")
    )
    assert result == [(v * 2, v * 2 + 1) for v in range(7)]


def test_map_batch_multi_signal_wrong_length(test_session):
    def one(value: list[int]) -> list[int]:
        return [0]

    with pytest.raises(UdfError, match="entry 'a' returned 1 outputs for 3 inputs"):
        dc.read_values(value=[1, 2, 3], session=test_session).map(
            a=one, b=one, batch=3
        ).to_list()


def test_map_batch_empty_input(test_session):
    def no_call(value: list[int]) -> list[int]:
        raise AssertionError("empty input should not invoke mapper")

    assert (
        dc.read_values(value=[1], session=test_session)
        .limit(0)
        .map(result=no_call, batch=4)
        .to_list()
        == []
    )


def test_map_batch_delta(test_session):
    calls = []

    def twice(value: list[int]) -> list[int]:
        calls.extend(value)
        return [v * 2 for v in value]

    def run():
        return (
            dc.read_dataset(
                "batch_source", session=test_session, delta=True, delta_on="value"
            )
            .map(result=twice, batch=3)
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
        .map(result=encode, batch=2)
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
    assert sorted(chain.map(result=encode, batch=2).to_values("result")) == [-1, 1, 3]


def test_map_batch_list_input(test_session):
    def total(values: list[list[int]]) -> list[int]:
        return [sum(v) for v in values]

    result = (
        dc.read_values(values=[[1, 2], [], [3]], session=test_session)
        .map(result=total, batch=2)
        .to_values("result")
    )
    assert sorted(result) == [0, 3, 3]


def test_map_batch_no_annotation_explicit_output(test_session):
    def twice(value):
        return [v * 2 for v in value]

    result = (
        dc.read_values(value=[1, 2, 3], session=test_session)
        .map(result=twice, output=int, batch=2)
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
        .map(worker=worker_id, batch=4)
        .to_values("worker")
    )
    assert len(result) == 64
    assert len(set(result)) == 4


@pytest.mark.parametrize("batched_first", [True, False])
def test_map_batch_shared_mapper_instance(test_session, batched_first):
    class Twice(dc.Mapper):
        def process(self, value):
            if isinstance(value, list):
                return [v * 2 for v in value]
            return value * 2

    mapper = Twice()
    source = dc.read_values(value=[1, 2, 3], session=test_session)
    batches = {"batched": 2, "scalar": 1}
    order = ["batched", "scalar"] if batched_first else ["scalar", "batched"]

    chains = {
        name: source.map(**{name: mapper}, output=int, batch=batches[name])
        for name in order
    }
    for name, chain in chains.items():
        assert sorted(chain.to_values(name)) == [2, 4, 6]


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
        .map(twice=udfs.twice, batch=2)
        .map(thrice=udfs.thrice, batch=2)
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
        .map(doubled=twice, output=Doubled, batch=2)
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
