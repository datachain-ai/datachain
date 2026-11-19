import shutil
import sys
from datetime import UTC

import lance
import pyarrow as pa
import pytest
from lance.file import LanceFileWriter

import datachain as dc
from datachain.client.http import HTTPClient
from datachain.lib import arrow
from datachain.lib.arrow import ArrowGenerator, _lance_file_reader, _LanceDataset
from datachain.lib.dc import C, DatasetPrepareError
from datachain.lib.file import ArrowRow, File
from datachain.utils import safe_closing


def test_read_lance_select_and_filter_pushdown(
    tmp_dir, test_session, mocker, monkeypatch
):
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "lance.file", None)
        with pytest.raises(ImportError, match="datachain\\[lance\\]"):
            _lance_file_reader()

    path = tmp_dir / "rows.lance"
    table = pa.table(
        {
            "a": [3, 2, 1, 1],
            "b": [30, 20, 10, 10],
            "payload": ["p0", "p1", "p2", "p3"],
        }
    )
    with LanceFileWriter(str(path), table.schema) as writer:
        writer.write_batch(table)

    monkeypatch.setattr(arrow, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    prefetch = mocker.spy(File, "_prefetch")
    read_spy = mocker.spy(_LanceDataset, "_read")
    chain = dc.read_lance(path.as_uri(), session=test_session)

    assert chain.select("b").to_values("b") == [30, 20, 10, 10]
    assert prefetch.call_count == 1
    assert [call.args[1] for call in read_spy.call_args_list] == [["b"]]
    assert dc.read_lance(
        path.as_uri(),
        columns=["payload", "a"],
        output={"p": str, "x": int},
        session=test_session,
    ).to_list("p", "x") == [("p0", 3), ("p1", 2), ("p2", 1), ("p3", 1)]

    read_spy.reset_mock()
    rows = chain.filter(C("a") == 1).to_list("source", "payload")
    assert [(s.index, p) for s, p in rows] == [(2, "p2"), (3, "p3")]
    source = rows[0][0]
    assert source.read() == {"a": 1, "b": 10, "payload": "p2"}
    with safe_closing(_LanceDataset(source.file)) as ds:
        assert ds.take([2, 0, 2]).to_pydict()["payload"] == ["p2", "p0", "p2"]
    with pytest.raises(IndexError, match="out of range"):
        ArrowRow(file=source.file, index=99, kwargs=source.kwargs).read()
    assert [call.args[1] for call in read_spy.call_args_list[:2]] == [
        ["a"],
        ["payload"],
    ]
    assert read_spy.call_args_list[1].args[2].to_pylist() == [2, 3]

    read_spy.reset_mock()
    assert chain.filter(C("a") == 999).to_values("payload") == []
    assert [call.args[1] for call in read_spy.call_args_list] == [["a"]]

    monkeypatch.setattr(ArrowGenerator, "DEFAULT_BATCH_SIZE", 2)
    assert chain.filter(C("a") >= 1).select("b").to_values("b") == [30, 20, 10, 10]


def test_read_lance_directory_schema_drift_and_rejects_table_root(
    tmp_dir, test_session
):
    ds_dir = tmp_dir / "dataset_table"
    table = pa.table({"a": [10, 20, 30, 40], "text": ["w0", "w1", "w2", "w3"]})
    ds = lance.write_dataset(table, ds_dir, max_rows_per_file=2)
    ds.delete("a = 20")

    for uri in (ds_dir.as_uri(), f"{ds_dir.as_uri()}/data/*.lance"):
        with pytest.raises(DatasetPrepareError, match="Lance table"):
            dc.read_lance(uri, session=test_session, settings={"cache": True})

    frag = next((ds_dir / "data").glob("*.lance"))
    cached_file = File(
        source=ds_dir.as_uri(),
        path=f"data/{frag.name}",
        size=frag.stat().st_size,
    )
    cached_file._set_stream(test_session.catalog, caching_enabled=True)
    cached_file.ensure_cached()
    remote_table_file = File(
        source="http://mock",
        path=f"table.lance/data/{frag.name}",
        size=frag.stat().st_size,
    )
    for bad_file in (cached_file, remote_table_file):
        with pytest.raises(DatasetPrepareError, match="Lance table"):
            dc.read_values(file=[bad_file], session=test_session).parse_tabular(
                format="lance"
            )

    for folder in ("_indices", "_versions", "data"):
        user_dir = tmp_dir / f"user_{folder}" / folder
        user_dir.mkdir(parents=True)
        if folder == "data":
            (user_dir.parent / "_versions").write_text("not a dir")
        user_file = user_dir / "a.lance"
        t = pa.table({"a": [1, 2]})
        with LanceFileWriter(str(user_file), t.schema) as w:
            w.write_batch(t)
        assert dc.read_lance(user_file.as_uri(), session=test_session).to_values(
            "a"
        ) == [1, 2]

    blob_ds_dir = tmp_dir / "blob_table.lance"
    blob_field = pa.field(
        "payload", pa.large_binary(), metadata={b"lance-encoding:blob": b"true"}
    )
    nested_arr = pa.StructArray.from_arrays(
        [pa.array([b"hello", b"world"], type=pa.large_binary())], fields=[blob_field]
    )
    lance.write_dataset(pa.table({"id": [1, 2], "nested": nested_arr}), blob_ds_dir)
    blob_frag = next((blob_ds_dir / "data").glob("*.lance"))
    standalone_blob = tmp_dir / "standalone_blob.lance"
    shutil.copy(blob_frag, standalone_blob)
    with pytest.raises(
        DatasetPrepareError, match=r"Lance blob column 'nested\.payload'"
    ):
        dc.read_lance(standalone_blob.as_uri(), session=test_session)

    drift_dir = tmp_dir / "lance_drift"
    drift_dir.mkdir()
    s1 = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=False),
            pa.field("extra", pa.string(), nullable=False),
        ]
    )
    s2 = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    t1 = pa.table({"a": [1, 2], "extra": ["u", "v"]}, schema=s1)
    t2 = pa.table({"a": [3, 4]}, schema=s2)
    t3 = pa.table({"a": pa.array([], type=pa.int64())}, schema=s2)
    with LanceFileWriter(str(drift_dir / "f1.lance"), s1) as w:
        w.write_batch(t1)
    with LanceFileWriter(str(drift_dir / "f2.lance"), s2) as w:
        w.write_batch(t2)
    with LanceFileWriter(str(drift_dir / "f3.lance"), s2) as w:
        w.write_batch(t3)

    drift = dc.read_lance(drift_dir.as_uri(), session=test_session)
    assert drift.order_by("a").select("extra").to_values("extra") == [
        "u",
        "v",
        None,
        None,
    ]
    assert drift.filter(C("extra").is_(None) & (C("a") == 4)).to_list(
        "source.index", "a", "extra"
    ) == [(1, 4, None)]
    assert drift.filter(C("a") == 4).select("extra").to_values("extra") == [None]


def test_read_lance_special_column_names_and_null_boolean_predicates(
    tmp_dir, test_session
):
    path = tmp_dir / "special.lance"
    table = pa.table(
        {
            "": [0, 1, 2, 3],
            "`a`": [10, 20, 30, 40],
            "a": [100, 200, 300, 400],
            "b[0]": ["x", "y", "z", "w"],
            "flag": pa.array([True, None, False, True], type=pa.bool_()),
        }
    )
    with LanceFileWriter(str(path), table.schema) as w:
        w.write_batch(table)

    chain = dc.read_lance(path.as_uri(), session=test_session)
    assert chain.select("c0", "c1_a", "a", "b_0").to_list("c0", "c1_a", "a", "b_0") == [
        (0, 10, 100, "x"),
        (1, 20, 200, "y"),
        (2, 30, 300, "z"),
        (3, 40, 400, "w"),
    ]
    assert chain.select("c1_a").to_values("c1_a") == [10, 20, 30, 40]
    assert chain.select("a").to_values("a") == [100, 200, 300, 400]

    pred = (C("flag") == False) | (C("c0") == 3)  # noqa: E712
    assert chain.filter(pred).to_values("c0") == [2, 3]


def test_read_lance_remote_binary_and_embeddings(tmp_dir, test_session, monkeypatch):
    path = tmp_dir / "wide.lance"
    b0, b1 = b"A" * 100_000, b"B" * 100_000
    table = pa.table(
        {
            "id": [1, 2],
            "emb": pa.array(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                type=pa.list_(pa.float32(), 4),
            ),
            "tags": pa.array([["t1", "t2"], ["t3"]], type=pa.large_list(pa.string())),
            "payload": pa.array([b0, b1], type=pa.large_binary()),
        }
    )
    with LanceFileWriter(str(path), table.schema) as w:
        w.write_batch(table)
    raw_bytes = path.read_bytes()
    downloads = 0

    async def fake_get_file(self, lpath, rpath, callback, version_id=None):
        nonlocal downloads
        downloads += 1
        shutil.copy(path, rpath)

    monkeypatch.setattr(HTTPClient, "get_file", fake_get_file)
    file = File(source="http://mock", path="wide.lance", size=len(raw_bytes))
    chain = dc.read_values(
        file=[file], session=test_session, settings={"prefetch": 0}
    ).parse_tabular(format="lance")

    rows = (
        chain.filter(C("id") == 2)
        .select("source", "emb", "tags", "payload")
        .to_list("source", "emb", "tags", "payload")
    )
    assert [(emb, tags, payload) for _, emb, tags, payload in rows] == [
        ([5.0, 6.0, 7.0, 8.0], ["t3"], b1)
    ]
    for _ in range(3):
        assert rows[0][0].read() == {
            "id": 2,
            "emb": [5.0, 6.0, 7.0, 8.0],
            "tags": ["t3"],
            "payload": b1,
        }
    assert downloads == 1

    f2 = File(source="http://mock", path="wide2.lance", size=len(raw_bytes))
    assert dc.read_values(
        file=[file, f2], session=test_session, settings={"prefetch": 0}
    ).parse_tabular(format="lance", columns=["id"], output={"id": int}).to_values(
        "id"
    ) == [1, 2, 1, 2]
    assert downloads == 2


def test_to_lance_roundtrip(tmp_dir, test_session):
    from datetime import datetime
    from typing import Literal

    from pydantic import ConfigDict

    class Item(dc.DataModel):
        model_config = ConfigDict(extra="allow")
        a: int
        b: str | None = None
        dt: datetime | None = None

    class Inner(dc.DataModel):
        tag: str | int = 0
        foo__bar: int = 12
        mode: Literal["fast", "slow"] | None = None

    class Outer(dc.DataModel):
        inner: Inner = Inner()
        note: str | None = None
        items: list[Item] | None = None
        matrix: list[list[int]] | None = None
        meta: dict | None = None
        scores: dict[int, int] | None = None
        ts: datetime | None = None

    class Alt(dc.DataModel):
        code: int = 0

    t1 = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    rows_in = [
        Outer(
            inner=Inner(tag="x", foo__bar=91, mode=None),
            note="n1",
            items=[Item(a=1, b=None, extra_k="e1")],
            matrix=[[]],
            meta=None,
            scores=None,
        ),
        Alt(code=42),
        Outer(
            inner=Inner(tag=7, foo__bar=55, mode="fast"),
            note=None,
            items=[Item(a=2, b="z", dt=t1)],
            matrix=[[10, 20]],
            meta={"a": 1},
            scores={1: 10},
            ts=t1,
        ),
        Outer(
            inner=Inner(tag=8, foo__bar=66, mode="slow"),
            note="n3",
            items=[],
            matrix=[],
            meta={"b": "two", "nested": [1, 2]},
            scores={2: 20},
            ts=t1,
        ),
    ]
    chain = dc.read_values(
        id=[1, 2, 3, 4],
        o=rows_in,
        u=[100, "hello", 200, "world"],
        output={"id": int, "o": Outer | Alt, "u": int | str},
        session=test_session,
    ).order_by("id")

    path = tmp_dir / "exported.lance"
    chain.to_lance(path, chunk_size=1)
    assert path.is_file()
    back = dc.read_lance(path.as_uri(), session=test_session)
    assert back.order_by("id").to_list("id", "o", "u") == [
        (1, rows_in[0], 100),
        (2, rows_in[1], "hello"),
        (3, rows_in[2], 200),
        (4, rows_in[3], "world"),
    ]
    assert dc.read_lance(
        path.as_uri(),
        columns=["o.Outer.inner.tag", "o.Outer.inner.foo__bar", "u"],
        session=test_session,
    ).to_list("o.Outer.inner.tag", "o.Outer.inner.foo__bar", "u") == [
        ("x", 91, 100),
        (None, None, "hello"),
        (7, 55, 200),
        (8, 66, "world"),
    ]

    empty_path = tmp_dir / "empty.lance"
    chain.filter(C("id") < 0).to_lance(empty_path)
    assert dc.read_lance(empty_path.as_uri(), session=test_session).count() == 0
