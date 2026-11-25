import enum
import errno
import io
import json
import math
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

import lance
import pyarrow as pa
import pytest
import sqlalchemy as sa
from lance.file import LanceFileWriter
from pydantic import ConfigDict

import datachain as dc
from datachain.client.http import HTTPClient
from datachain.lib import arrow
from datachain.lib.dc import C, DatasetPrepareError
from datachain.lib.file import ArrowRow, File
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.storage_config import session_for
from datachain.utils import safe_closing
from tests.utils import skip_if_not_sqlite


def _write_lance(path, data, schema=None):
    table = data if isinstance(data, pa.Table) else pa.table(data, schema=schema)
    with LanceFileWriter(str(path), table.schema) as writer:
        writer.write_batch(table)


def test_read_lance_select_and_filter_pushdown(
    tmp_dir, test_session, mocker, monkeypatch
):
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "lance.file", None)
        with pytest.raises(ImportError, match="datachain\\[lance\\]"):
            arrow._lance_file_reader()

    path = tmp_dir / "rows.lance"
    _write_lance(
        path,
        {"a": [3, 2, 1, 1], "b": [30, 20, 10, 10], "payload": ["p0", "p1", "p2", "p3"]},
    )
    monkeypatch.setattr(arrow, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    prefetch = mocker.spy(File, "_prefetch")
    rl = partial(dc.read_lance, session=test_session)
    chain = rl(path.as_uri())

    assert chain.select("b").to_values("b") == [30, 20, 10, 10]
    assert prefetch.call_count == 1
    picked = rl(path.as_uri(), columns=["payload", "a"], output={"p": str, "x": int})
    assert picked.to_list("p", "x") == [("p0", 3), ("p1", 2), ("p2", 1), ("p3", 1)]

    rows = chain.filter(C("a") == 1).to_list("source", "payload")
    assert [(s.index, p) for s, p in rows] == [(2, "p2"), (3, "p3")]
    source = rows[0][0]
    assert source.read() == {"a": 1, "b": 10, "payload": "p2"}
    rv = dc.read_values(
        file=[source.file], lance_fragment=["chapter-1"], session=test_session
    )
    assert rv.parse_tabular(format="lance").to_values("b") == [30, 20, 10, 10]
    with safe_closing(arrow._LanceDataset(source.file)) as ds:
        assert ds.take([2, 0, 2]).to_pydict()["payload"] == ["p2", "p0", "p2"]
    with pytest.raises(IndexError, match="out of range"):
        ArrowRow(file=source.file, index=99, kwargs=source.kwargs).read()

    assert chain.filter(C("a") == 999).to_values("payload") == []

    monkeypatch.setattr(arrow.ArrowGenerator, "DEFAULT_BATCH_SIZE", 2)
    res = chain.filter(C("a") >= 1).select("b").to_values("b")
    assert sorted(res) == [10, 10, 20, 30]

    (zero_path := tmp_dir / "zero.lance").write_bytes(b"")
    with pytest.raises(DatasetPrepareError, match="is 0 bytes"):
        rl(zero_path.as_uri())


def test_read_lance_table_versions_and_pushdown(
    tmp_dir, test_session, mocker, monkeypatch
):
    d = ds_dir = tmp_dir / "dataset_table.lance"
    rl = partial(dc.read_lance, session=test_session)
    b, cache = pa.array([1, 2, 3, 4], pa.uint8()), test_session.catalog.cache
    t = pa.table({"a": [10, 20, 30, 40], "b": b, "text": ["w0", "w1", "w2", "w3"]})
    ds = lance.write_dataset(t, ds_dir, max_rows_per_file=2, enable_stable_row_ids=True)
    ds.tags.create("v1_tag", 1)
    ds.delete("a = 20")
    det_t = pa.table({"a": [777], "b": pa.array([7], pa.uint8()), "text": ["det"]})
    det_op = lance.LanceOperation.Overwrite(
        t.schema, lance.fragment.write_fragments(det_t, ds_dir, schema=t.schema)
    )
    det_ds = lance.LanceDataset.commit(ds_dir, det_op, read_version=2, detached=True)

    monkeypatch.setattr(arrow, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    scanner_spy = mocker.spy(lance.LanceFragment, "scanner")
    read_storage_spy = mocker.spy(dc.lib.dc.storage, "read_storage")

    for ver in (1, "v1_tag"):
        assert set(rl(ds_dir, version=ver).to_values("a")) == {10, 20, 30, 40}
    assert rl(ds_dir, version=det_ds.version).to_values("a") == [777]

    pinned = rl(f"{ds_dir}/", settings={"cache": True})
    lance.write_dataset(t.slice(0, 1), ds_dir, mode="append")
    assert pinned.order_by("a").to_values("text") == ["w0", "w2", "w3"]
    assert not read_storage_spy.called and cache.get_total_size() == 0

    scanner_spy.reset_mock()
    v2 = rl(ds_dir.as_uri(), version=2)
    rows = v2.filter(C("a") >= 30).select("source", "text").to_list("source", "text")
    assert sorted(s.read()["a"] for s, _ in rows) == [30, 40]
    assert sorted((s.index, txt) for s, txt in rows) == [(2, "w2"), (3, "w3")]
    calls = [c.kwargs for c in scanner_spy.call_args_list]
    assert [c["columns"] for c in calls] == [["a", "text"]] * 2
    assert all(c["filter"] is not None for c in calls)
    for expr in (C("b") == 3, C("b").in_([3])):
        assert v2.filter(expr).to_values("a") == [30]
    r0 = rows[0][0]
    assert r0.file.read()[-4:] == b"LANC"
    with pytest.raises(ValueError, match="without a catalog"):
        ArrowRow.model_validate(r0.model_dump()).read()
    for bad_idx in (-1, 1, 99, 99 << 32):
        with pytest.raises(IndexError, match=r"out of range|Invalid"):
            ArrowRow(file=r0.file, index=bad_idx, kwargs=r0.kwargs).read()

    with pytest.raises(TypeError, match="not bool"):
        rl(d, version=True)
    with pytest.raises(
        DatasetPrepareError,
        match="Lance tag '1' not found; pass an int for a version number",
    ):
        rl(d, version="1")
    for kw in ({"bogus": 1}, {"delta": True}, {"update": True}, {"type": "image"}):
        with pytest.raises(TypeError):
            rl(d, **kw)
    g = f"{d.as_uri()}/data/*.lance"
    for u, v in ((d, 99), ([d, d], 1), (g, 1)):
        with pytest.raises(DatasetPrepareError):
            rl(u, version=v)
    for col in ("", "a`b"):
        lance.write_dataset(pa.table({col: [1]}), bad_col := tmp_dir / "bad_col.lance")
        with pytest.raises(DatasetPrepareError, match="not supported"):
            rl(bad_col)
        shutil.rmtree(bad_col)

    shutil.rmtree(ds_dir)
    lance.write_dataset(pa.table({"a": [99], "text": ["recreated"]}), ds_dir)
    stale_src = (stale := rl(ds_dir)).to_values("source")[0]
    shutil.rmtree(ds_dir)
    lance.write_dataset(pa.table({"a": [100], "text": ["v1_new"]}), ds_dir)
    with pytest.raises(FileNotFoundError, match="changed on the source"):
        stale_src.read()
    with pytest.raises(Exception, match="changed on the source"):
        stale.to_values("a")
    assert rl(ds_dir).to_values("a") == [100]


@skip_if_not_sqlite
def test_read_lance_table_parallel_over_fragments(tmp_dir, test_session_tmpfile):
    ds_dir = tmp_dir / "parallel_table.lance"
    tbl = pa.table({"n": list(range(80)), "v": [f"s{i}" for i in range(80)]})
    lance.write_dataset(tbl, ds_dir, max_rows_per_file=20)
    res = dc.read_lance(
        ds_dir, session=test_session_tmpfile, settings={"parallel": 2}
    ).save("parallel_lance")
    assert res.order_by("n").to_values("n") == list(range(80))
    sources = res.order_by("n").limit(3).to_values("source")
    assert [s.read()["v"] for s in sources] == ["s0", "s1", "s2"]


@pytest.mark.parametrize("folder", ["_indices", "_versions", "_refs", "data"])
def test_read_lance_allows_user_folder_names(tmp_dir, test_session, folder):
    user_dir = tmp_dir / f"user_{folder}" / folder
    user_dir.mkdir(parents=True)
    if folder == "data":
        (user_dir.parent / "_versions").write_text("not a dir")
    _write_lance(user_file := user_dir / "a.lance", {"a": [1, 2]})
    assert dc.read_lance(user_file, session=test_session).to_values("a") == [1, 2]


def test_read_lance_rejects_table_internals_blobs_and_clones(tmp_dir, test_session):
    rl = partial(dc.read_lance, session=test_session)
    parent_dir = tmp_dir / "tables"
    tagged_dir = parent_dir / "tagged.lance"
    ds = lance.write_dataset(pa.table({"a": [10, 20]}), tagged_dir)
    ds.tags.create("v1", 1)
    frag = next((tagged_dir / "data").glob("*.lance"))
    for bad_path in (parent_dir, frag, f"{tagged_dir.as_uri()}/data/*.lance"):
        with pytest.raises(
            DatasetPrepareError, match="pass the table path to read_lance"
        ):
            rl(bad_path)

    ds.shallow_clone(str(clone_dir := tmp_dir / "clone.lance"), "v1")
    with pytest.raises(DatasetPrepareError, match="shallow-cloned"):
        rl(clone_dir)

    blob_ds_dir = tmp_dir / "blob_table.lance"
    meta, lb = {b"lance-encoding:blob": b"true"}, pa.large_binary()
    blob_f = pa.field("payload", lb, metadata=meta)
    arr = pa.StructArray.from_arrays([pa.array([b"a"], lb)], fields=[blob_f])
    lance.write_dataset(pa.table({"id": [1], "nested": arr}), blob_ds_dir)
    blob_frag = next((blob_ds_dir / "data").glob("*.lance"))
    shutil.copy(blob_frag, standalone_blob := tmp_dir / "standalone_blob.lance")
    for bad_blob in (blob_ds_dir, standalone_blob.as_uri()):
        with pytest.raises(DatasetPrepareError, match=r"blob column 'nested\.payload'"):
            rl(bad_blob)

    ms, pfx = test_session.catalog.metastore, test_session.get_temp_prefix()
    n_tmp = len(list(ms.list_datasets_by_prefix(pfx)))
    with pytest.raises(DatasetPrepareError, match=r"tagged\.lance.*missing"):
        rl(tagged_dir, columns=["missing"])
    assert len(list(ms.list_datasets_by_prefix(pfx))) == n_tmp


def test_read_lance_directory_schema_drift(tmp_dir, test_session):
    drift_dir = tmp_dir / "lance_drift"
    drift_dir.mkdir()
    s1 = pa.schema([("a", pa.int64(), False), ("extra", pa.string(), False)])
    s2 = pa.schema([("a", pa.int64(), False)])
    _write_lance(drift_dir / "f1.lance", {"a": [1, 2], "extra": ["u", "v"]}, schema=s1)
    _write_lance(drift_dir / "f2.lance", {"a": [3, 4]}, schema=s2)
    _write_lance(drift_dir / "f3.lance", {"a": pa.array([], pa.int64())}, schema=s2)

    drift = dc.read_lance(f"{drift_dir.as_uri()}/", session=test_session)
    assert drift.order_by("a").to_values("extra") == ["u", "v", None, None]
    hit = drift.filter(C("extra").is_(None) & (C("a") == 4))
    assert hit.to_list("source.index", "a", "extra") == [(1, 4, None)]
    assert drift.filter(C("a") == 4).select("extra").to_values("extra") == [None]


def test_read_lance_special_column_names_and_null_boolean_predicates(
    tmp_dir, test_session
):
    path = tmp_dir / "special.lance"
    _write_lance(
        path,
        {
            "": [0, 1, 2, 3],
            "`a`": [10, 20, 30, 40],
            "a": [100, 200, 300, 400],
            "b[0]": ["x", "y", "z", "w"],
            "flag": pa.array([True, None, False, True], type=pa.bool_()),
        },
    )
    chain, cols = (
        dc.read_lance(path.as_uri(), session=test_session),
        ("c0", "c1_a", "a", "b_0"),
    )
    exp = [(0, 10, 100, "x"), (1, 20, 200, "y"), (2, 30, 300, "z"), (3, 40, 400, "w")]
    assert chain.select(*cols).to_list(*cols) == exp
    assert chain.select("c1_a").to_values("c1_a") == [10, 20, 30, 40]
    assert chain.select("a").to_values("a") == [100, 200, 300, 400]
    assert chain.filter((C("flag") == False) | (C("c0") == 3)).to_values("c0") == [2, 3]  # noqa: E712


def test_read_lance_http_without_ranges(
    tmp_path, test_session, http_server_without_ranges
):
    _write_lance(tmp_path / "plain.lance", {"id": [1, 2]})
    url = f"{http_server_without_ranges}/plain.lance"
    assert dc.read_lance(url, session=test_session).to_values("id") == [1, 2]


@pytest.mark.skipif(os.name == "nt", reason="no sparse metadata copy on Windows")
def test_read_lance_remote_metadata_and_cache(tmp_dir, test_session, monkeypatch):
    path = tmp_dir / "wide.lance"
    b0, b1 = os.urandom(100_000), os.urandom(100_000)
    emb = pa.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], pa.list_(pa.float32(), 4)
    )
    tags, pay = (
        pa.array([["t1", "t2"], ["t3"]], pa.large_list(pa.string())),
        pa.array([b0, b1], pa.large_binary()),
    )
    _write_lance(path, {"id": [1, 2], "emb": emb, "tags": tags, "payload": pay})
    raw_bytes = path.read_bytes()
    downloads = read_bytes = 0
    remote_bytes = raw_bytes

    class Stream(io.BytesIO):
        def read(self, n=-1):
            nonlocal read_bytes
            chunk = super().read(n)
            read_bytes += len(chunk)
            return chunk

    async def fake_get_file(self, lpath, rpath, callback, version_id=None):
        nonlocal downloads
        downloads += 1
        shutil.copy(path, rpath)

    monkeypatch.setattr(HTTPClient, "get_file", fake_get_file)
    monkeypatch.setattr(
        HTTPClient,
        "_open_remote",
        lambda self, file, block_size=None: Stream(remote_bytes),
    )
    cache = test_session.catalog.cache
    rv = partial(dc.read_values, session=test_session)
    file = File(source="http://mock", path="wide.lance", size=len(raw_bytes))
    f2 = File(source="http://mock", path="wide2.lance", size=len(raw_bytes))

    chain = rv(file=[file], settings={"prefetch": 0}).parse_tabular(format="lance")
    assert downloads == 0 and read_bytes <= 64 * 1024 < len(raw_bytes)
    assert cache.get_total_size() == 0 and not list(Path(cache.tmp_dir).glob("lance-*"))

    cols = ("source", "emb", "tags", "payload")
    ((src, e_val, t_val, p_val),) = (
        chain.filter(C("id") == 2).select(*cols).to_list(*cols)
    )
    assert (e_val, t_val, p_val) == ([5.0, 6.0, 7.0, 8.0], ["t3"], b1)
    assert (downloads, cache.get_total_size()) == (1, 0)
    exp_row = {"id": 2, "emb": [5.0, 6.0, 7.0, 8.0], "tags": ["t3"], "payload": b1}
    assert src.read() == exp_row and (downloads, cache.get_total_size()) == (2, 0)

    with monkeypatch.context() as mp:
        mp.setattr(os, "name", "nt")
        c_nt = rv(file=[file], settings={"prefetch": 0}).parse_tabular(format="lance")
        assert c_nt.to_values("id") == [1, 2] and downloads == 4

    c_cached = rv(
        file=[file, f2], settings={"cache": True, "prefetch": 0}
    ).parse_tabular(format="lance", columns=["id"], output={"id": int})
    assert sorted(c_cached.to_values("id")) == [1, 1, 2, 2]
    assert (downloads, cache.get_total_size()) == (6, 2 * len(raw_bytes))
    for _ in range(2):
        assert src.read()["id"] == 2
    assert downloads == 6


def test_to_lance_roundtrip_and_struct_layout(tmp_dir, test_session, monkeypatch):
    rl = partial(dc.read_lance, session=test_session)
    rv = partial(dc.read_values, session=test_session)

    class Item(dc.DataModel):
        model_config = ConfigDict(extra="allow")
        a: int
        b: str | None = None
        dt: datetime | None = None

    class Inner(dc.DataModel):
        tag: str | int = 0
        foo__bar: Annotated[int, "meta"] = 12
        mode: Literal["fast", "slow"] = "fast"

    class Outer(dc.DataModel):
        inner: Inner = Inner()
        note: str | None = None
        items: list[Item] = []  # noqa: RUF012
        matrix: list[list[int]] = []  # noqa: RUF012
        dts: list[datetime] = []  # noqa: RUF012
        meta: dict = {}  # noqa: RUF012
        scores: dict[int, int] = {}  # noqa: RUF012
        ts: datetime | None = None

    class Alt(dc.DataModel):
        code: int = 0

    t1, it7 = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC), Item(a=7, b="ok")
    it1, it2 = Item(a=1, extra_k="e1"), Item(a=2, b="z", dt=t1)
    i0, i2 = Inner(tag="x", foo__bar=91), Inner(tag=7, foo__bar=55)
    i3, m3 = Inner(tag=8, foo__bar=66, mode="slow"), {"b": "two", "nested": [1, 2]}
    kw2 = {"meta": {"a": 1}, "scores": {1: 10}, "ts": t1}
    rows_in = [
        Outer(inner=i0, items=[it1], matrix=[[]], dts=[t1]),
        Alt(code=42),
        Outer(inner=i2, note="n2", items=[it2], matrix=[[10, 20]], **kw2),
        Outer(inner=i3, dts=[t1], meta=m3, scores={2: 20}, ts=t1),
    ]
    opt_in = [it7, None, it7, None]
    chain = rv(
        id=[1, 2, 3, 4],
        f=[float("nan"), 1.5, 2.5, float("inf")],
        o=rows_in,
        m=opt_in,
        u=[100, "hello", 200, "world"],
        output={
            "id": int,
            "f": float,
            "o": Outer | Alt,
            "m": Item | None,
            "u": int | str,
        },
    ).order_by("id")
    cols = ["o.Outer.inner.tag", "o.Outer.inner.foo__bar", "u"]
    exp_rows = list(
        zip([1, 2, 3, 4], rows_in, opt_in, [100, "hello", 200, "world"], strict=True)
    )
    exp_cols = [("x", 91, 100), (None, None, "hello"), (7, 55, 200), (8, 66, "world")]

    for is_tbl in (True, False):
        p = tmp_dir / f"exported_{is_tbl}.lance"
        chain.to_lance(
            p,
            chunk_size=1,
            table=is_tbl,
            **({"max_rows_per_file": 2} if is_tbl else {}),
        )
        assert p.is_dir() if is_tbl else p.is_file()
        with pytest.raises(FileExistsError):
            chain.to_lance(p, table=is_tbl)
        with pytest.raises(FileExistsError):
            chain.to_lance(p, table=not is_tbl, mode="overwrite")
        ord_b = rl(p.as_uri()).order_by("id")
        f_vals = ord_b.to_values("f")
        sources = ord_b.to_values("source")
        s0, s2 = sources[0].read(), sources[2].read()
        assert ord_b.to_list("id", "o", "m", "u") == exp_rows
        assert (
            math.isnan(s0["f"])
            and f_vals[1:] == [1.5, 2.5, float("inf")]
            and (s2["o.Outer.meta"], s2["o.Outer.scores"]) == ({"a": 1}, {"1": 10})
        )
        assert rl(p.as_uri(), columns=cols).to_list(*cols) == exp_cols

    tbl_p, file_p = tmp_dir / "exported_True.lance", tmp_dir / "exported_False.lance"
    ds_tbl = lance.dataset(tbl_p)
    assert [f.count_rows() for f in ds_tbl.get_fragments()] == [2, 2]
    proj = ds_tbl.to_table(columns=["o.Outer.inner.foo__bar", "o.Outer.dts"]).schema
    assert (
        proj.field("o.Outer.inner.foo__bar").type,
        proj.field("o.Outer.dts").type,
    ) == (pa.int64(), pa.list_(pa.timestamp("us", tz="UTC")))

    _write_lance(
        stripped := tmp_dir / "stripped.lance",
        ds_tbl.to_table().replace_schema_metadata(None),
    )
    assert rl([file_p.as_uri(), stripped.as_uri()]).to_values("m").count(None) == 4

    with monkeypatch.context() as mp:
        mp.setattr(
            os,
            "link",
            lambda *a: (_ for _ in ()).throw(OSError(errno.EPERM, "no hardlinks")),
        )
        rv(id=[9]).to_lance(no_link := tmp_dir / "no_link.lance", table=False)
    assert rl(no_link).to_values("id") == [9]
    rv(id=[9]).to_lance(file_p, table=False, mode="overwrite")
    assert rl(file_p).to_values("id") == [9]

    chain.filter(C("id") < 0).to_lance(empty_path := tmp_dir / "empty.lance")
    assert rl(empty_path.as_uri()).count() == 0

    class Kind(enum.Enum):
        A = 1

    rv(id=[1, 2]).mutate(k=sa.cast(sa.literal(1), sa.Enum(Kind))).to_lance(
        enum_path := tmp_dir / "enum.lance"
    )
    assert [
        type(dc.json.loads(v))
        for v in lance.dataset(enum_path).to_table().column("k").to_pylist()
    ] == [str, str]


def test_to_lance_table_external_mutations_and_error_cleanup(tmp_dir, test_session):
    rl = partial(dc.read_lance, session=test_session)
    rv = partial(dc.read_values, session=test_session)

    p = tmp_dir / "mutated.lance"
    rv(x=[1, 2], s=["a", "b"]).to_lance(p, chunk_size=1)
    assert len(lance.dataset(p).get_fragments()) == 1
    rv(x=[10], s=["z"]).to_lance(p, mode="overwrite")
    assert (lance.dataset(p).version, rl(p, version=1).to_values("x")) == (2, [1, 2])

    lance.write_dataset(
        pa.table(
            {"x": pa.array([None], pa.int64()), "s": pa.array([None], pa.string())}
        ),
        p,
        mode="append",
    )
    assert rl(p).order_by("x").to_list("x", "s") == [(10, "z"), (None, None)]

    ds = lance.dataset(p)
    ds.add_columns({"z": "x * 2"})
    assert rl(p).order_by("x").to_list("x", "s", "z") == [
        (10, "z", 20),
        (None, None, None),
    ]
    lance.dataset(p).alter_columns({"path": "z", "name": "zz"})
    assert sorted(rl(p).to_values("zz"), key=lambda v: v or 0) == [None, 20]
    lance.dataset(p).drop_columns(["s", "zz"])
    assert rl(p).order_by("x").to_values("x") == [10, None]

    with pytest.raises(ValueError, match="Unsupported mode"):
        rv(x=[0]).to_lance(p, mode="append")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="table=False"):
        rv(x=[1]).to_lance("hf://org/repo/d.lance")

    fail_path = tmp_dir / "failed_create.lance"
    f_fail = File.at(fail_path, session=test_session)
    with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError)):
        arrow.write_lance(
            f_fail,
            SignalSchema({"w": int}),
            iter([[[1]], [[1.75]]]),
            max_rows_per_file=1,
        )
    assert not fail_path.exists()

    race_path = tmp_dir / "race_create.lance"
    f_race = File.at(race_path, session=test_session)
    race_chunks = (
        x
        for x in [[[1]]]
        if lance.write_dataset(pa.table({"w": [99]}), race_path) or True
    )
    with pytest.raises(FileExistsError, match="Lance table already exists"):
        arrow.write_lance(f_race, SignalSchema({"w": int}), race_chunks)
    assert rl(race_path).to_values("w") == [99]


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_lance_s3_and_storage_options(
    cloud_test_catalog, test_session, tmp_dir, monkeypatch
):
    ctc = cloud_test_catalog
    rv = partial(dc.read_values, session=ctc.session)
    uri, s_uri = f"{ctc.src_uri}/events.lance", f"{ctc.src_uri}/shards/s.lance"
    lance_opts = arrow._lance_storage_options(ctc.src_uri, ctc.client_config)
    rv(id=[1, 2, 3], v=[10, 20, 30]).to_lance(
        uri,
        max_rows_per_file=2,
        storage_options=lance_opts,
    )
    bad = {"aws_endpoint_url": "http://127.0.0.1:9"}
    for scfg, kw in (
        (ctc.client_config, {}),
        (bad, {"client_config": ctc.client_config}),
        (ctc.client_config, {"storage_options": lance_opts}),
    ):
        s = session_for(test_session, bad, False)
        s.catalog.source_client_configs[ctc.src_uri] = scfg
        ch = dc.read_lance(uri, session=s, anon=False, **kw)
        ((src, val),) = ch.filter(C("id") == 3).to_list("source", "v")
        assert (val, src.read()["v"]) == (30, 30)
        assert "storage_options" not in json.dumps(src.kwargs)
    ch_cfg = dc.read_lance(uri, session=test_session, client_config=ctc.client_config)
    assert ch_cfg.session.catalog.client_config_for("s3://other") == ctc.client_config

    rl = partial(dc.read_lance, session=ctc.session, anon=False)
    rv(id=[7], v=[70]).to_lance(s_uri, table=False, fs_kwargs=ctc.client_config)
    with pytest.raises(FileExistsError):
        rv(id=[8], v=[80]).to_lance(s_uri, table=False)
    rv(id=[7], v=[70]).to_lance(s_uri, table=False, mode="overwrite")
    assert rl(f"{ctc.src_uri}/shards/").to_values("v") == [70]
    with pytest.raises(DatasetPrepareError, match="union"):
        rl([uri, f"{ctc.src_uri}/shards/"])

    monkeypatch.setenv("DATACHAIN_GCP_CREDENTIALS", '{"a":1}')
    s3_cfg = {
        "aws_endpoint_url": "http://bad",
        "client_kwargs": {"endpoint_url": "http://ok"},
        "config_kwargs": {"region_name": "r1"},
        "anon": True,
    }
    s3_exp = {
        "aws_endpoint": "http://ok",
        "aws_region": "r1",
        "aws_skip_signature": "true",
        "allow_http": "true",
    }
    az_cfg = {
        "account_name": "a",
        "endpoint_url": "http://h/a",
        "sas_token": "?sv=1",
    }
    az_exp = {
        "azure_storage_account_name": "a",
        "azure_storage_endpoint": "http://h/a",
        "azure_storage_sas_key": "sv=1",
        "allow_http": "true",
        "timeout": "30s",
    }
    for u, cfg, so, expected in (
        ("s3://b", None, None, None),
        ("s3://b", s3_cfg, None, s3_exp),
        ("gs://b", None, None, {"google_service_account_key": '{"a":1}'}),
        ("az://b", az_cfg, {"timeout": "30s"}, az_exp),
    ):
        assert arrow._lance_storage_options(u, cfg, so) == expected
    for scheme in ("http", "https", "hf"):
        with pytest.raises(DatasetPrepareError, match="not supported"):
            dc.read_lance(f"{scheme}://h/t.lance", version=1, session=test_session)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset")
def test_lance_commit_time_is_timezone_independent(tmp_dir, monkeypatch):
    path = tmp_dir / "tz.lance"
    lance.write_dataset(pa.table({"a": [1]}), path)
    stamps = []
    for tz in ("America/Los_Angeles", "Asia/Tokyo"):
        monkeypatch.setenv("TZ", tz)
        time.tzset()
        stamps.append(arrow._lance_commit_time(lance.dataset(path)))
    monkeypatch.undo()
    time.tzset()
    assert stamps[0] == stamps[1]
