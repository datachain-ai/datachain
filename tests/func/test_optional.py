"""Functional tests for ``Optional[X]`` / nullable-object support.

Covers ``Optional[DataModel]`` and ``Optional[basic]`` end to end: the #1055
regression, save round-trips, the ``_type_tag`` discriminator, mutate/filter/order_by/
group_by/select on optional leaves, parquet/JSON export, union of optional and
plain models, and aggregates over absent groups. These run on both SQLite and
ClickHouse (via ``scripts/run-clickhouse-tests.sh``).

Extracted from ``test_udf.py`` to keep nullable-object coverage in one place.
"""

import json
from typing import Optional

import pandas as pd

import datachain as dc
from datachain.lib.data_model import DataModel, unwrap_optional
from datachain.lib.file import File
from tests.utils import skip_if_not_sqlite


class _Inner(DataModel):
    score: int = 0
    label: str = ""


class _WithFloat(DataModel):
    a: int = 0
    x: float = 0.0


class _WithList(DataModel):
    a: int = 0
    tags: list[str] = []  # noqa: RUF012


class _ServerToolUsage(DataModel):
    """Stub of anthropic.types.ServerToolUsage (no SDK dependency)."""

    web_search_requests: int = 0


class _Usage(DataModel):
    """Stub of anthropic.types.Usage from issue #1055: a DataModel with
    Optional[int] fields and an Optional[DataModel] field."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    server_tool_use: Optional[_ServerToolUsage] = None
    service_tier: Optional[str] = None


def test_issue_1055_optional_datamodel_output(test_session):
    """#1055 original repro: a UDF returning Optional[DataModel] must not crash
    UDF prep, and None must survive."""

    def maybe(id: int) -> Optional[_Inner]:
        return _Inner(score=id, label=f"n{id}") if id else None

    chain = dc.read_values(id=[0, 1, 2], session=test_session).map(item=maybe)
    by_id = dict(chain.order_by("id").to_list("id", "item"))
    assert by_id[0] is None
    assert by_id[1] == _Inner(score=1, label="n1")
    assert by_id[2] == _Inner(score=2, label="n2")


def test_issue_1055_anthropic_usage_roundtrip(test_session):
    """#1055: an Anthropic-Usage-shaped model (Optional[int] fields + a nested
    Optional[DataModel] field) round-trips through save() on both backends —
    present values preserved, absent optionals come back as None (not 0/'')."""

    usages = [
        _Usage(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=None,
            server_tool_use=_ServerToolUsage(web_search_requests=2),
            service_tier="standard",
        ),
        _Usage(
            input_tokens=20,
            output_tokens=7,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            server_tool_use=None,
            service_tier=None,
        ),
    ]
    (
        dc.read_values(
            id=[1, 2],
            usage=usages,
            output={"id": int, "usage": _Usage},
            session=test_session,
        ).save("issue_1055_usage")
    )

    by_id = dict(
        dc.read_dataset("issue_1055_usage", session=test_session).to_list("id", "usage")
    )
    u1, u2 = by_id[1], by_id[2]

    assert (u1.input_tokens, u1.output_tokens) == (10, 5)
    assert u1.cache_creation_input_tokens == 3
    assert u1.cache_read_input_tokens is None  # Optional[int]=None survives save()
    assert u1.server_tool_use == _ServerToolUsage(web_search_requests=2)
    assert u1.service_tier == "standard"

    assert (u2.input_tokens, u2.output_tokens) == (20, 7)
    assert u2.cache_creation_input_tokens is None
    assert u2.cache_read_input_tokens is None
    assert u2.server_tool_use is None  # absent nested Optional[DataModel]
    assert u2.service_tier is None


def test_read_values_infers_optional_scalar_from_none(test_session):
    """read_values infers Optional[scalar] from a None in the sequence, so the
    None round-trips as NULL instead of the type default."""
    chain = dc.read_values(id=[1, 2, 3], a=[10, None, 30], session=test_session).save(
        "infer_opt_scalar"
    )
    _, is_optional = unwrap_optional(chain.signals_schema.values["a"])
    assert is_optional
    saved = dc.read_dataset("infer_opt_scalar", session=test_session)
    assert {r["id"]: r["a"] for r in saved.to_records()} == {1: 10, 2: None, 3: 30}


def test_optional_basic_scalar_roundtrips_none(test_session):
    """An explicit Optional[scalar] column stores/reads None as NULL on both
    backends (without it, ClickHouse coerces None to the type default 0/"")."""

    def num(id: int) -> Optional[int]:
        return None if id == 2 else id * 10

    def txt(id: int) -> Optional[str]:
        return None if id == 2 else f"n{id}"

    def flag(id: int) -> Optional[bool]:
        return None if id == 2 else id % 2 == 0

    def score(id: int) -> Optional[float]:
        return None if id == 2 else id * 1.5

    chain = (
        dc.read_values(id=[1, 2, 3], session=test_session)
        .map(num=num)
        .map(txt=txt)
        .map(flag=flag)
        .map(score=score)
    )
    rows = {
        r["id"]: (r["num"], r["txt"], r["flag"], r["score"]) for r in chain.to_records()
    }
    assert rows[1] == (10, "n1", False, 1.5)
    assert rows[2] == (None, None, None, None)
    assert rows[3] == (30, "n3", False, 4.5)


def test_optional_float_none_consistent_across_backends(test_session):
    """isnone/count/sum over an Optional[float] None agree on every backend, with
    the same SQL null semantics as Optional[int]."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2, 3],
        score=[0.97, None, 0.55],
        output={"id": int, "score": Optional[float]},
        session=test_session,
    )
    assert chain.filter(func.isnone("score")).count() == 1
    assert chain.filter(func.not_(func.isnone("score"))).count() == 2
    assert sorted(chain.to_values("score"), key=lambda v: (v is None, v)) == [
        0.55,
        0.97,
        None,
    ]
    agg = (
        chain.mutate(g=1)
        .group_by(s_sum=func.sum("score"), s_cnt=func.count("score"), partition_by="g")
        .to_records()
    )
    assert agg == [{"g": 1, "s_sum": 1.52, "s_cnt": 2}]


@skip_if_not_sqlite
def test_optional_float_stored_nan_collapses_to_none_on_sqlite():
    """SQLite stores NaN as NULL, so a stored NaN reads back as None and counts as
    null. The other backends keep NaN distinct (IEEE) — an inherent SQLite limit,
    not a DataChain bug."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2],
        score=[float("nan"), 0.5],
        output={"id": int, "score": Optional[float]},
    )
    assert chain.filter(func.isnone("score")).count() == 1
    assert sorted(chain.to_values("score"), key=lambda v: (v is None, v)) == [0.5, None]


def test_optional_basic_scalar_roundtrips_none_through_save(test_session):
    """An Optional[scalar] None must survive save()/read_dataset on both backends.
    The dc_nullable marker is carried through the serialized dataset schema so the
    persisted ClickHouse column stays Nullable and reads back None, not 0/"" """

    def num(id: int) -> Optional[int]:
        return None if id == 2 else id * 10

    def txt(id: int) -> Optional[str]:
        return None if id == 2 else f"n{id}"

    (
        dc.read_values(id=[1, 2, 3], session=test_session)
        .map(num=num)
        .map(txt=txt)
        .save("opt_scalar_save_roundtrip")
    )

    chain = dc.read_dataset("opt_scalar_save_roundtrip", session=test_session)
    rows = {r["id"]: (r["num"], r["txt"]) for r in chain.to_records()}
    assert rows[1] == (10, "n1")
    assert rows[2] == (None, None)
    assert rows[3] == (30, "n3")


def test_optional_datamodel_leaf_null_through_save(test_session):
    """A scalar leaf under an Optional[DataModel] is Nullable, so an absent-parent
    row stores/reads real NULL (not 0/"") through save() on both backends."""
    items = [_Inner(score=1, label="a"), None, _Inner(score=3, label="c")]
    (
        dc.read_values(
            id=[1, 2, 3],
            item=items,
            output={"id": int, "item": Optional[_Inner]},
            session=test_session,
        ).save("opt_dm_leaf_save")
    )
    chain = dc.read_dataset("opt_dm_leaf_save", session=test_session)

    def _s(v):  # CH returns str leaves as bytes in this raw flat path (pre-existing)
        return v.decode() if isinstance(v, bytes) else v

    by_id = {
        r["id"]: (r["item__score"], _s(r["item__label"])) for r in chain.to_records()
    }
    assert by_id[1] == (1, "a")
    assert by_id[2] == (None, None)  # absent parent → NULL leaves on both backends
    assert by_id[3] == (3, "c")


def test_optional_datamodel_mutate_leaf(test_session):
    """mutate of a leaf under Optional[DataModel] reads NULL for the absent-parent
    row on both backends."""
    from datachain import C

    items = [_Inner(score=1, label="a"), None, _Inner(score=3, label="c")]
    chain = dc.read_values(
        id=[1, 2, 3],
        item=items,
        output={"id": int, "item": Optional[_Inner]},
        session=test_session,
    ).mutate(s=C("item.score"))
    by_id = {r["id"]: r["s"] for r in chain.to_records()}
    assert by_id[1] == 1
    assert by_id[2] is None
    assert by_id[3] == 3


def test_mutate_func_over_absent_leaf_is_none(test_session):
    """A row-level func over a leaf under Optional[DataModel] returns None for the
    absent-parent row on both backends (and survives save). The func result column
    is marked nullable; without it ClickHouse coerces the NULL result to 0/''."""
    from datachain import func

    items = [_Inner(score=1, label="aa"), None, _Inner(score=3, label="ccc")]
    chain = dc.read_values(
        id=[1, 2, 3],
        item=items,
        output={"id": int, "item": Optional[_Inner]},
        session=test_session,
    ).mutate(ln=func.string.length("item.label"))

    by_id = {r["id"]: r["ln"] for r in chain.to_records()}
    assert by_id == {1: 2, 2: None, 3: 3}

    chain.save("mutate_func_absent")
    saved = {
        r["id"]: r["ln"]
        for r in dc.read_dataset(
            "mutate_func_absent", session=test_session
        ).to_records()
    }
    assert saved == {1: 2, 2: None, 3: 3}  # None survives save() on both backends


def test_mutate_expr_over_optional_scalar_is_nullable(test_session):
    from datachain import C

    chain = dc.read_values(
        id=[1, 2, 3],
        a=[10, None, 30],
        b=[1, 2, 3],
        output={"id": int, "a": Optional[int], "b": int},
        session=test_session,
    )

    mutated = chain.mutate(sa=C("a") + 1, sb=C("b") + 1)
    _, sa_opt = unwrap_optional(mutated.signals_schema.values["sa"])
    _, sb_opt = unwrap_optional(mutated.signals_schema.values["sb"])
    assert sa_opt and not sb_opt

    mutated.save("mutate_expr_optional")
    saved = dc.read_dataset("mutate_expr_optional", session=test_session)
    assert {r["id"]: r["sa"] for r in saved.to_records()} == {1: 11, 2: None, 3: 31}


def test_cast_optional_scalar(test_session):
    """func.cast over an Optional column keeps None (casts to the Nullable target,
    so ClickHouse doesn't fail casting NULL to a non-Nullable type)."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2, 3],
        x=[1, None, 3],
        output={"id": int, "x": Optional[int]},
        session=test_session,
    )
    mutated = chain.mutate(y=func.cast("x", str))
    _, is_optional = unwrap_optional(mutated.signals_schema.values["y"])
    assert is_optional
    assert dict(mutated.order_by("id").to_list("id", "y")) == {1: "1", 2: None, 3: "3"}


def test_list_optional_element_roundtrip(test_session):
    """A list with None elements (list[Optional[int]]) round-trips on both
    backends — the Array element is nullable (ClickHouse: Array(Nullable(T)))."""
    chain = dc.read_values(
        id=[1, 2],
        x=[[1, None, 3], [4]],
        output={"id": int, "x": list[Optional[int]]},
        session=test_session,
    )
    chain.save("list_opt_elem")
    saved = dc.read_dataset("list_opt_elem", session=test_session)
    assert dict(saved.order_by("id").to_list("id", "x")) == {1: [1, None, 3], 2: [4]}


def test_composed_func_over_absent_leaf_is_none(test_session):
    from datachain import func

    chain = (
        dc.read_values(
            id=[1, 2],
            item=[_Inner(score=1, label="aa"), None],
            output={"id": int, "item": Optional[_Inner]},
            session=test_session,
        )
        .mutate(ln2=func.string.length("item.label") + 1)
        .order_by("id")
    )
    assert chain.to_values("ln2") == [3, None]


class _Deep(DataModel):
    name: str = ""
    inner: Optional[_Inner] = None


def test_optional_datamodel_parquet_roundtrip(test_session, tmp_path):
    """to_parquet/read_parquet of an Optional[DataModel] preserves present objects
    and reads an absent parent back as None."""
    presents = {
        0: _Inner(score=0, label=""),
        1: _Inner(score=10, label="a"),
        3: _Inner(score=30, label="c"),
    }
    items = [presents.get(i) for i in [0, 1, 2, 3]]
    path = str(tmp_path / "opt.parquet")
    dc.read_values(
        id=[0, 1, 2, 3],
        item=items,
        output={"id": int, "item": Optional[_Inner]},
        session=test_session,
    ).order_by("id").to_parquet(path)

    back = dc.read_parquet(path, session=test_session)
    got = {
        i: (None if it is None else (it.score, it.label))
        for i, it in back.to_list("id", "item")
    }
    assert got == {0: (0, ""), 1: (10, "a"), 2: None, 3: (30, "c")}


def test_optional_datamodel_parquet_roundtrip_nested(test_session, tmp_path):
    """Nested Optional[DataModel] survives a parquet round trip: present,
    present-with-absent-inner, and fully-absent rows all reconstruct."""
    vals = [
        _Deep(name="x", inner=_Inner(score=5, label="z")),
        _Deep(name="y", inner=None),
        None,
    ]
    path = str(tmp_path / "deep.parquet")
    dc.read_values(
        id=[1, 2, 3],
        d=vals,
        output={"id": int, "d": Optional[_Deep]},
        session=test_session,
    ).order_by("id").to_parquet(path)

    back = dc.read_parquet(path, session=test_session)
    got = {}
    for i, d in back.to_list("id", "d"):
        got[i] = (
            None
            if d is None
            else (d.name, None if d.inner is None else (d.inner.score, d.inner.label))
        )
    assert got == {1: ("x", (5, "z")), 2: ("y", None), 3: None}


def test_optional_datamodel_parquet_roundtrip_only_signal(test_session, tmp_path):
    """An Optional[DataModel] as the chain's only signal round-trips through
    parquet (the absent parent reads back as None via the heuristic)."""
    path = str(tmp_path / "only.parquet")
    dc.read_values(
        m=[_Inner(score=1, label="a"), None],
        output={"m": Optional[_Inner]},
        session=test_session,
    ).to_parquet(path)
    got = [
        None if m is None else (m.score, m.label)
        for (m,) in dc.read_parquet(path, session=test_session).to_list("m")
    ]
    assert (1, "a") in got and None in got and len(got) == 2


def test_optional_datamodel_float_leaf_roundtrip(test_session, tmp_path):
    """Optional[DataModel] with a float leaf: the absent parent round-trips as
    None through parquet, and to_json emits null (not literal NaN, invalid JSON)."""
    chain = dc.read_values(
        id=[1, 2],
        m=[_WithFloat(a=1, x=1.5), None],
        output={"id": int, "m": Optional[_WithFloat]},
        session=test_session,
    ).order_by("id")

    pq = str(tmp_path / "f.parquet")
    chain.to_parquet(pq)
    got = {
        i: (None if m is None else (m.a, m.x))
        for i, m in dc.read_parquet(pq, session=test_session).to_list("id", "m")
    }
    assert got == {1: (1, 1.5), 2: None}

    pj = str(tmp_path / "f.json")
    chain.to_json(pj)
    with open(pj) as fh:
        raw = fh.read()
    assert "NaN" not in raw  # RFC-valid JSON
    by_id = {r["id"]: r["m"] for r in json.loads(raw)}
    assert by_id[1] == {"a": 1, "x": 1.5}
    assert by_id[2] is None


def test_optional_datamodel_list_leaf_to_json_null(test_session, tmp_path):
    """Optional[DataModel] with a list leaf collapses to null in to_json for the
    absent parent (the `_type_tag` decides, not the leaf value)."""
    pj = str(tmp_path / "l.json")
    dc.read_values(
        id=[1, 2],
        m=[_WithList(a=1, tags=["x"]), None],
        output={"id": int, "m": Optional[_WithList]},
        session=test_session,
    ).order_by("id").to_json(pj)

    with open(pj) as fh:
        by_id = {r["id"]: r["m"] for r in json.loads(fh.read())}
    assert by_id[1] == {"a": 1, "tags": ["x"]}
    assert by_id[2] is None


def test_optional_file_sets_catalog(test_session, tmp_path):
    """A File under Optional[...] gets its catalog/stream set on hydration, so
    .read_text() works."""
    (tmp_path / "a.txt").write_text("hello")

    class _Wrap(DataModel):
        f: Optional[File] = None

    chain = dc.read_values(
        w=[_Wrap(f=File(path="a.txt", source=tmp_path.as_uri()))],
        output={"w": _Wrap},
        session=test_session,
    )
    w = chain.to_values("w")[0]
    assert w.f.read_text() == "hello"


def test_union_optional_and_plain_datamodel(test_session):
    """union of an Optional[DataModel] chain with a plain DataModel chain yields
    Optional[DataModel]: the plain side is promoted (a present sentinel is added)
    so the column sets align."""

    def optional_chain():
        return dc.read_values(
            id=[1, 2],
            item=[_Inner(score=1, label="x"), None],
            output={"id": int, "item": Optional[_Inner]},
            session=test_session,
        )

    def plain_chain():
        return dc.read_values(
            id=[3],
            item=[_Inner(score=9, label="z")],
            output={"id": int, "item": _Inner},
            session=test_session,
        )

    def rows(chain):
        return sorted(
            (i, None if it is None else (it.score, it.label))
            for i, it in chain.to_list("id", "item")
        )

    merged = optional_chain().union(plain_chain())
    inner, is_optional = unwrap_optional(merged.signals_schema.values["item"])
    assert is_optional and inner is _Inner
    assert rows(merged) == [(1, (1, "x")), (2, None), (3, (9, "z"))]

    # order-independent: plain.union(optional) is equivalent
    assert rows(plain_chain().union(optional_chain())) == [
        (1, (1, "x")),
        (2, None),
        (3, (9, "z")),
    ]


def test_to_json_absent_optional_datamodel_is_null(test_session, tmp_path):
    """to_json/to_jsonl serialize an absent Optional[DataModel] as null (not an
    object of nulls), consistent with hydration. A present object whose leaves
    are the type defaults (0/"") is kept — only an all-NULL parent collapses."""
    presents = {
        0: _Inner(score=0, label=""),
        1: _Inner(score=10, label="a"),
        3: _Inner(score=30, label="c"),
    }
    chain = dc.read_values(
        id=[0, 1, 2, 3],
        item=[presents.get(i) for i in [0, 1, 2, 3]],
        output={"id": int, "item": Optional[_Inner]},
        session=test_session,
    ).order_by("id")

    p = str(tmp_path / "out.json")
    chain.to_json(p)
    with open(p) as f:
        rows = json.load(f)
    by_id = {r["id"]: r["item"] for r in rows}
    assert by_id[0] == {"score": 0, "label": ""}  # present default kept, not None
    assert by_id[1] == {"score": 10, "label": "a"}
    assert by_id[2] is None  # absent parent -> null
    assert by_id[3] == {"score": 30, "label": "c"}

    # nested: present-with-absent-inner and fully-absent
    vals = [
        _Deep(name="x", inner=_Inner(score=5, label="z")),
        _Deep(name="y", inner=None),
        None,
    ]
    nchain = dc.read_values(
        id=[1, 2, 3],
        d=vals,
        output={"id": int, "d": Optional[_Deep]},
        session=test_session,
    ).order_by("id")
    pn = str(tmp_path / "nested.json")
    nchain.to_json(pn)
    with open(pn) as f:
        nrows = {r["id"]: r["d"] for r in json.load(f)}
    assert nrows[1] == {"name": "x", "inner": {"score": 5, "label": "z"}}
    assert nrows[2] == {"name": "y", "inner": None}
    assert nrows[3] is None


def test_group_by_partition_optional_datamodel_leaf(test_session):
    """group_by partitioning on a leaf under Optional[DataModel] works."""
    from datachain import func

    items = [
        _Inner(score=0, label="x"),
        _Inner(score=10, label="y"),
        _Inner(score=10, label="z"),
    ]
    chain = dc.read_values(
        id=[1, 2, 3],
        item=items,
        output={"id": int, "item": Optional[_Inner]},
        session=test_session,
    )
    rows = chain.group_by(cnt=func.count(), partition_by="item.score").to_records()
    assert sorted((r.get("item__score"), r["cnt"]) for r in rows) == [(0, 1), (10, 2)]


def test_select_partial_optional_datamodel_leaf(test_session):
    """select() of a single leaf under Optional[DataModel] builds a partial model
    and preserves None for the absent parent (same to_partial path as group_by)."""
    items = [_Inner(score=5, label="a"), None]
    chain = dc.read_values(
        id=[1, 2],
        item=items,
        output={"id": int, "item": Optional[_Inner]},
        session=test_session,
    )
    got = {
        r["id"]: r.get("item__score")
        for r in chain.select("id", "item.score").to_records()
    }
    assert got == {1: 5, 2: None}


def test_udf_returns_optional_datamodel_mixed_none(test_session):
    def maybe(score: int) -> _Inner | None:
        return _Inner(score=score, label="ok") if score > 0 else None

    chain = (
        dc.read_values(score=[1, 0, 2], session=test_session)
        .map(item=maybe)
        .order_by("score")
    )

    score_to_item = dict(chain.to_list())
    assert score_to_item[0] is None
    assert score_to_item[1] == _Inner(score=1, label="ok")
    assert score_to_item[2] == _Inner(score=2, label="ok")


def test_udf_returns_optional_datamodel_all_none(test_session):
    def always_none(score: int) -> _Inner | None:
        return None

    chain = dc.read_values(score=[1, 2, 3], session=test_session).map(item=always_none)
    rows = list(chain.to_list())
    assert all(item is None for _, item in rows)


def test_read_values_with_optional_datamodel(test_session):
    items = [_Inner(score=1, label="a"), None, _Inner(score=3, label="c")]
    chain = dc.read_values(
        items=items,
        output={"items": Optional[_Inner]},
        session=test_session,
    )
    scores = [(item.score if item else None) for (item,) in chain.to_list()]
    assert sorted(s for s in scores if s is not None) == [1, 3]
    assert scores.count(None) == 1


def test_read_values_optional_datamodel_multi_column(test_session):
    """read_values with an Optional[DataModel] *alongside another column* — the
    multi-output flatten path, which must still emit the sentinel."""
    from datachain import func

    items = [_Inner(score=1, label="a"), None, _Inner(score=3, label="c")]
    chain = dc.read_values(
        id=[1, 2, 3],
        item=items,
        output={"id": int, "item": Optional[_Inner]},
        session=test_session,
    ).save("rv_multi_opt")

    by_id = {r[0]: r[1] for r in chain.select("id", "item").to_list()}
    assert by_id[1] == _Inner(score=1, label="a")
    assert by_id[2] is None
    assert by_id[3] == _Inner(score=3, label="c")

    present = chain.group_by(n=func.count("item"), partition_by="id")
    assert {r["id"]: r["n"] for r in present.to_records()} == {1: 1, 2: 0, 3: 1}
    assert chain.filter(func.isnone("item")).count() == 1


def test_read_values_optional_datamodel_inferred(test_session):
    """A DataModel column with some None values inferred as Optional[DataModel]
    round-trips through save() with None preserved on both backends. (The pure
    type-inference assertion lives in tests/unit/lib/test_optional.py.)"""
    items = [_Inner(score=1, label="a"), None, _Inner(score=3, label="c")]
    saved = dc.read_values(id=[1, 2, 3], item=items, session=test_session).save(
        "rv_inferred_opt"
    )
    by_id = {r[0]: r[1] for r in saved.select("id", "item").to_list()}
    assert by_id[1] == _Inner(score=1, label="a")
    assert by_id[2] is None
    assert by_id[3] == _Inner(score=3, label="c")


def test_multi_output_map_optional_datamodel(test_session):
    """A multi-output map returning Optional[DataModel] in one slot must emit the
    sentinel for that slot (single-output already worked; multi-output did not)."""

    def split(id: int):
        return id * 10, (None if id == 2 else _Inner(score=id, label=f"x{id}"))

    chain = dc.read_values(id=[1, 2, 3], session=test_session).map(
        split, output={"big": int, "item": Optional[_Inner]}
    )

    by_id = {r[0]: (r[1], r[2]) for r in chain.select("id", "big", "item").to_list()}
    assert by_id[1] == (10, _Inner(score=1, label="x1"))
    assert by_id[2] == (20, None)
    assert by_id[3] == (30, _Inner(score=3, label="x3"))


def test_nested_optional_datamodel_in_outer_model(test_session):
    class Outer(DataModel):
        name: str
        inner: _Inner | None = None

    def make_outer(n: str) -> Outer:
        return Outer(
            name=n,
            inner=None if n == "b" else _Inner(score=len(n), label=n),
        )

    chain = (
        dc.read_values(n=["a", "b", "cc"], session=test_session)
        .map(out=make_outer)
        .order_by("n")
    )

    rows = list(chain.to_list())
    name_to_out = {r[0]: r[1] for r in rows}
    assert name_to_out["a"].inner == _Inner(score=1, label="a")
    assert name_to_out["b"].inner is None
    assert name_to_out["cc"].inner == _Inner(score=2, label="cc")


def test_isnone_filters_optional_datamodel(test_session):
    from datachain import func

    def maybe(score: int) -> _Inner | None:
        return _Inner(score=score, label="ok") if score > 0 else None

    chain = dc.read_values(score=[1, 0, 2, 0, 3], session=test_session).map(item=maybe)
    assert chain.filter(func.not_(func.isnone("item"))).count() == 3
    assert chain.filter(func.isnone("item")).count() == 2


def test_filter_optional_datamodel_leaf_excludes_absent(test_session):
    """A predicate on a leaf under Optional[DataModel] must not match absent-parent
    rows. The leaf is genuinely Nullable, so an absent parent's leaf is NULL on both
    backends and NULL never satisfies an equality/comparison predicate."""
    from datachain import func

    # id=2 -> absent. id=3 has a *real present* score of 0.
    presents = {
        1: _Inner(score=10, label="a"),
        3: _Inner(score=0, label="c"),
        4: _Inner(score=7, label="d"),
    }

    def pick(id: int) -> _Inner | None:
        return presents.get(id)

    chain = dc.read_values(id=[1, 2, 3, 4], session=test_session).map(item=pick)

    def ids(c):
        return sorted(r["id"] for r in c.select("id").to_records())

    assert ids(chain.filter(dc.C("item.score") == 0)) == [3]
    assert ids(chain.filter(dc.C("item.label") == "")) == []
    assert ids(chain.filter(dc.C("item.score") > 5)) == [1, 4]
    assert ids(chain.filter(func.isnone("item"))) == [2]


def test_order_by_optional_datamodel_leaf_nulls_last(test_session):
    """order_by on a leaf under Optional[DataModel] sorts absent rows last on both
    backends. The leaf is genuinely Nullable (absent -> NULL); order_by emits an
    explicit NULLS LAST so SQLite (which sorts NULL first by default) agrees with
    ClickHouse."""

    # id=2 -> absent. Present scores: id1=10, id3=0, id4=7.
    presents = {
        1: _Inner(score=10, label="a"),
        3: _Inner(score=0, label="c"),
        4: _Inner(score=7, label="d"),
    }

    def pick(id: int) -> _Inner | None:
        return presents.get(id)

    chain = dc.read_values(id=[1, 2, 3, 4], session=test_session).map(item=pick)

    def order(*, descending: bool, use_c: bool):
        col = dc.C("item.score") if use_c else "item.score"
        return [
            r["id"]
            for r in chain.order_by(col, descending=descending)
            .select("id")
            .to_records()
        ]

    # ascending: real 0,7,10 then the absent row (NULL) last.
    assert order(descending=False, use_c=False) == [3, 4, 1, 2]
    assert order(descending=False, use_c=True) == [3, 4, 1, 2]
    # descending: real 10,7,0 then the absent row (NULL) last.
    assert order(descending=True, use_c=False) == [1, 4, 3, 2]


def test_order_by_optional_basic_nulls_last(test_session):
    """order_by on a top-level Optional[basic] column sorts NULL rows last on both
    backends. The column stores genuine NULL, and order_by emits an explicit
    NULLS LAST so SQLite (which sorts NULL first by default) agrees with
    ClickHouse."""
    chain = dc.read_values(
        id=[1, 2, 3, 4],
        score=[10, None, 0, 7],
        output={"id": int, "score": Optional[int]},
        session=test_session,
    )

    def order(*, descending):
        return [
            r["id"]
            for r in chain.order_by("score", descending=descending)
            .select("id")
            .to_records()
        ]

    # ascending: present 0,7,10 then the NULL row (id=2) last.
    assert order(descending=False) == [3, 4, 1, 2]
    # descending: present 10,7,0 then the NULL row last.
    assert order(descending=True) == [1, 4, 3, 2]


def test_merge_unmatched_right_is_none(test_session):
    """A left merge widens the right side to Optional and reads unmatched rows as
    None on both backends (ClickHouse must not coerce the NULL to 0); an inner
    join leaves the right side non-optional."""
    left = dc.read_values(k=[1, 2, 3], output={"k": int}, session=test_session)
    right = dc.read_values(
        k=[1], rv=[99], output={"k": int, "rv": int}, session=test_session
    )

    merged = left.merge(right, on="k", rname="r_")
    _, is_optional = unwrap_optional(merged.signals_schema.values["rv"])
    assert is_optional

    merged.save("merge_unmatched")
    saved = dc.read_dataset("merge_unmatched", session=test_session)
    assert {r["k"]: r["rv"] for r in saved.to_records()} == {1: 99, 2: None, 3: None}

    _, inner_opt = unwrap_optional(
        left.merge(right, on="k", rname="r_", inner=True).signals_schema.values["rv"]
    )
    assert not inner_opt


def test_subtract_optional_key_null_safe(test_session):
    """subtract treats NULL keys as equal (NULL == NULL) on both backends, so a
    None-key row present in both is removed."""
    a = dc.read_values(
        k=[1, None, 3], output={"k": Optional[int]}, session=test_session
    )
    b = dc.read_values(k=[None], output={"k": Optional[int]}, session=test_session)
    assert sorted(str(v) for v in a.subtract(b, on=["k"]).to_values("k")) == ["1", "3"]


def test_subtract_optional_multikey_null_safe(test_session):
    """Multi-key subtract is NULL-safe per key: a row matches only when every key
    is equal (NULLs included), so (1, None) survives."""
    a = dc.read_values(
        x=[1, 1, None],
        y=["a", None, "c"],
        output={"x": Optional[int], "y": Optional[str]},
        session=test_session,
    )
    b = dc.read_values(
        x=[1, None],
        y=["a", "c"],
        output={"x": Optional[int], "y": Optional[str]},
        session=test_session,
    )
    assert sorted(str(t) for t in a.subtract(b, on=["x", "y"]).to_list("x", "y")) == [
        "(1, None)"
    ]


def test_window_order_by_optional_nulls_last(test_session):
    """row_number().over(order_by=Optional) ranks NULL rows last on both backends
    (SQLite orders NULLs first by default, ClickHouse last)."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2, 3, 4],
        g=[1, 1, 1, 1],
        s=[3, None, 1, None],
        output={"id": int, "g": int, "s": Optional[int]},
        session=test_session,
    )
    win = func.window(partition_by="g", order_by="s")
    rn = dict(chain.mutate(rn=func.row_number().over(win)).to_list("id", "rn"))
    # present rows (s=1,3) ranked first; the two NULL rows (ids 2, 4) last.
    assert rn[3] == 1 and rn[1] == 2
    assert {rn[2], rn[4]} == {3, 4}


def test_flat_export_optional_datamodel_leaf_none(test_session):
    """Flat exports (to_records/to_pandas/...) return None for an absent-parent
    row's nested leaf cells on both backends. The leaves are genuinely Nullable,
    so an absent parent stores real NULL (not the ''/0 type default ClickHouse
    used to coerce non-nullable columns to)."""

    # id=2 -> absent. id=3 has a *real present* score of 0 / label "".
    presents = {
        1: _Inner(score=10, label="a"),
        3: _Inner(score=0, label=""),
    }

    def pick(id: int) -> _Inner | None:
        return presents.get(id)

    chain = dc.read_values(id=[1, 2, 3], session=test_session).map(item=pick)

    def norm(v):
        # ClickHouse returns String leaves as bytes in the raw flat path; the
        # bytes-vs-str divergence is orthogonal to the absent-leaf fix here.
        return v.decode() if isinstance(v, bytes) else v

    rows = {
        r["id"]: (r["item__score"], norm(r["item__label"])) for r in chain.to_records()
    }
    assert rows[1] == (10, "a")
    assert rows[2] == (None, None)  # absent parent -> NULL leaves, not 0/""
    assert rows[3] == (0, "")  # present parent keeps its real 0/"" values

    df = chain.order_by("id").to_pandas()
    score = df["item"]["score"].tolist()
    label = [norm(v) for v in df["item"]["label"].tolist()]
    assert score[0] == 10 and score[2] == 0
    assert label[0] == "a" and label[2] == ""
    # absent parent -> null leaf; pandas spells it None or NaN depending on
    # version/dtype, so use pandas' own null check.
    assert pd.isna(score[1])
    assert pd.isna(label[1])


def test_type_tag_excluded_from_exports(test_session, tmp_path):
    """The `_type_tag` discriminator never appears in exports; absent parents
    round-trip to None via the all-leaves-None heuristic."""
    presents = {1: _Inner(score=10, label="a")}

    def pick(id: int) -> _Inner | None:
        return presents.get(id)

    chain = (
        dc.read_values(id=[1, 2], session=test_session).map(item=pick).order_by("id")
    )

    assert all("_type_tag" not in k for k in chain.to_records()[0])
    assert not any("_type_tag" in c for c in chain.to_pandas().columns)
    cols = chain.to_pandas(include_hidden=True).columns
    assert not any("_type_tag" in c for c in cols)

    pj = str(tmp_path / "out.json")
    chain.to_json(pj)
    with open(pj) as f:
        by_id = {r["id"]: r["item"] for r in json.load(f)}
    assert by_id[1] == {"score": 10, "label": "a"}
    assert by_id[2] is None

    pp = str(tmp_path / "out.parquet")
    chain.to_parquet(pp)
    got = dict(
        dc.read_parquet(pp, session=test_session).order_by("id").to_list("id", "item")
    )
    assert got[1] == _Inner(score=10, label="a")
    assert got[2] is None


def test_count_optional_datamodel_uses_sentinel(test_session):
    """``func.count("opt_model")`` returns the number of rows whose parent is
    present — same value on SQLite and ClickHouse, regardless of how CH
    coerces absent-row leaves."""
    from datachain import func

    def maybe(score: int) -> _Inner | None:
        return _Inner(score=score, label="ok") if score > 0 else None

    chain = (
        dc.read_values(score=[1, 0, 2, 0, 3], session=test_session)
        .map(item=maybe)
        .mutate(grp=1)
    )

    rows = chain.group_by(
        present=func.count("item"), total=func.count(), partition_by="grp"
    ).to_records()
    assert rows == [{"grp": 1, "present": 3, "total": 5}]


def test_label_preserves_sentinel_aware_func(test_session):
    """func.isnone and func.count over an Optional[DataModel] resolve via the
    model's hidden _type_tag (the model itself has no real column on disk)."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2],
        item=[_Inner(score=5, label="abc"), None],
        output={"id": int, "item": Optional[_Inner]},
        session=test_session,
    )
    got = sorted(chain.mutate(x=func.isnone("item").label("x")).to_values("x"))
    assert got == [False, True]

    rows = chain.group_by(
        c=func.count("item").label("c"), partition_by="id"
    ).to_records()
    assert sorted((r["id"], r["c"]) for r in rows) == [(1, 1), (2, 0)]


def test_count_optional_datamodel_empty_is_zero(test_session):
    """A global ``func.count("opt_model")`` over an empty chain returns 0, not
    NULL. The sentinel-based count uses SUM, which is NULL over zero rows; a
    COALESCE keeps count()'s 0-for-empty contract."""
    from datachain import func

    def maybe(score: int) -> _Inner | None:
        return _Inner(score=score, label="ok") if score > 0 else None

    chain = (
        dc.read_values(score=[1, 2, 3], session=test_session)
        .map(item=maybe)
        .filter(dc.C("score") > 100)  # matches nothing -> empty input
    )
    rows = chain.group_by(present=func.count("item")).to_records()
    assert rows == [{"present": 0}]


def test_aggregates_over_optional_datamodel_leaf(test_session):
    """SUM/AVG/MIN/MAX on a leaf under ``Optional[DataModel]`` skip absent
    rows on both SQLite and ClickHouse: the leaf is genuinely Nullable, so an
    absent parent is NULL and SQL aggregates skip NULL natively."""
    from datachain import func

    def maybe(score: int) -> _Inner | None:
        return _Inner(score=score, label="ok") if score > 0 else None

    chain = (
        dc.read_values(score=[10, 0, 30], session=test_session)
        .map(item=maybe)
        .mutate(grp=1)
    )
    rows = chain.group_by(
        n=func.count("item"),
        s=func.sum("item.score"),
        a=func.avg("item.score"),
        mn=func.min("item.score"),
        mx=func.max("item.score"),
        partition_by="grp",
    ).to_records()
    assert rows == [{"grp": 1, "n": 2, "s": 40, "a": 20.0, "mn": 10, "mx": 30}]


def test_extra_aggregates_over_optional_datamodel_leaf(test_session):
    """any_value/concat/xor_agg also skip absent-parent rows."""
    from datachain import func

    def maybe(score: int) -> _Inner | None:
        return _Inner(score=score, label=f"l{score}") if score > 0 else None

    chain = (
        dc.read_values(score=[10, 0, 30], session=test_session)
        .map(item=maybe)
        .mutate(grp=1)
    )
    rows = chain.group_by(
        any_label=func.any_value("item.label"),
        concat_labels=func.concat("item.label", separator=","),
        xor_score=func.xor_agg("item.score"),
        partition_by="grp",
    ).to_records()
    assert len(rows) == 1
    [r] = rows
    # any_value returns one of the present labels (non-deterministic, but
    # never the empty-string default that CH would emit for the absent row).
    assert r["any_label"] in {"l10", "l30"}
    # concat skips absent rows on both backends.
    assert set(r["concat_labels"].split(",")) == {"l10", "l30"}
    # xor of {10, 30} excluding the absent 0 (which XORs to itself).
    assert r["xor_score"] == 10 ^ 30


def test_aggregates_over_all_absent_partition_return_none(test_session):
    """sum/min/max over a group whose every row's Optional[DataModel] parent is
    absent return None on both backends (and survive save()). The aggregate
    result column is marked nullable; without it ClickHouse coerces the NULL
    aggregate to the type default (0)."""
    from datachain import func

    # group g=1 has present items; group g=2 is entirely absent.
    items = [_Inner(score=10, label="a"), _Inner(score=20, label="b"), None, None]
    chain = dc.read_values(
        id=[1, 2, 3, 4],
        g=[1, 1, 2, 2],
        item=items,
        output={"id": int, "g": int, "item": Optional[_Inner]},
        session=test_session,
    )
    chain.group_by(
        sm=func.sum("item.score"),
        mn=func.min("item.score"),
        mx=func.max("item.score"),
        cnt=func.count("item"),
        partition_by="g",
    ).save("agg_all_absent")

    rows = {
        r["g"]: (r["sm"], r["mn"], r["mx"], r["cnt"])
        for r in dc.read_dataset("agg_all_absent", session=test_session).to_records()
    }
    assert rows[1] == (30, 10, 20, 2)
    assert rows[2] == (None, None, None, 0)  # all-absent group -> NULL, not 0


def test_aggregates_over_top_level_optional_scalar_keep_none(test_session):
    """sum/min/max over a top-level Optional[scalar] keep None for an all-NULL
    group, with the result column typed Optional."""
    from datachain import func

    # group g=2 has only NULL x values.
    chain = dc.read_values(
        id=[1, 2, 3, 4],
        g=[1, 1, 2, 2],
        x=[10, 20, None, None],
        output={"id": int, "g": int, "x": Optional[int]},
        session=test_session,
    )
    grouped = chain.group_by(
        sm=func.sum("x"),
        mn=func.min("x"),
        mx=func.max("x"),
        partition_by="g",
    )
    # the result columns stay nullable in the schema, not downgraded to int.
    for col in ("sm", "mn", "mx"):
        _, is_optional = unwrap_optional(grouped.signals_schema.values[col])
        assert is_optional, f"{col} should be Optional"

    grouped.save("agg_top_optional")
    rows = {
        r["g"]: (r["sm"], r["mn"], r["mx"])
        for r in dc.read_dataset("agg_top_optional", session=test_session).to_records()
    }
    assert rows[1] == (30, 10, 20)
    assert rows[2] == (None, None, None)  # all-NULL group -> NULL, not 0


def test_count_over_optional_scalar_stays_non_nullable(test_session):
    """count() over a nullable source must stay a plain int, not Optional[int]:
    COUNT never returns NULL (it returns 0), so it is exempt from the
    nullable-result marking that sum/min/max get."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2, 3],
        g=[1, 1, 2],
        x=[10, None, None],
        output={"id": int, "g": int, "x": Optional[int]},
        session=test_session,
    )
    grouped = chain.group_by(c=func.count("x"), partition_by="g")
    _, is_optional = unwrap_optional(grouped.signals_schema.values["c"])
    assert not is_optional, "count column must be plain int, not Optional[int]"
    rows = {r["g"]: r["c"] for r in grouped.to_records()}
    assert rows == {1: 1, 2: 0}  # g=2 has no non-NULL x -> 0, never None


def test_union_plain_left_optional_right_nullable_leaves(test_session):
    """union(plain, Optional[M]) keeps absent-row leaves NULL on CH (no phantom
    0/"" matches)."""
    from datachain import C

    plain = dc.read_values(
        item=[_Inner(score=1, label="a")],
        output={"item": _Inner},
        session=test_session,
    )
    opt = dc.read_values(
        item=[_Inner(score=2, label="b"), None],
        output={"item": Optional[_Inner]},
        session=test_session,
    )
    plain.union(opt).save("u_plain_left")
    back = dc.read_dataset("u_plain_left", session=test_session)
    assert back.count() == 3
    assert back.filter(C("item.label") == "").count() == 0
    assert back.filter(C("item.score") == 0).count() == 0


def test_union_does_not_mutate_input_chains(test_session):
    """union() must not mutate its operands: a chain used as LEFT in one union can
    still be used as RIGHT in another."""

    def mk_plain():
        return dc.read_values(
            item=[_Inner(score=1, label="a")],
            output={"item": _Inner},
            session=test_session,
        )

    def mk_opt():
        return dc.read_values(
            item=[_Inner(score=2, label="b"), None],
            output={"item": Optional[_Inner]},
            session=test_session,
        )

    opt = mk_opt()
    opt.union(mk_plain()).save("u_reuse_1")
    mk_plain().union(opt).save("u_reuse_2")  # opt reused -> must not raise
    assert dc.read_dataset("u_reuse_2", session=test_session).count() == 3


def _none_last(values):
    return sorted(values, key=lambda v: (v is None, v))


def test_union_scalar_promotes_to_optional_both_orders(test_session):
    """union of plain ``int`` with ``Optional[int]`` is ``Optional[int]`` in
    either order, and None survives the round-trip on both backends."""
    plain = dc.read_values(x=[1, 2], output={"x": int}, session=test_session)
    opt = dc.read_values(x=[3, None], output={"x": Optional[int]}, session=test_session)

    for i, merged in enumerate((plain.union(opt), opt.union(plain))):
        _, is_optional = unwrap_optional(merged.signals_schema.values["x"])
        assert is_optional
        merged.save(f"u_scalar_{i}")
        back = dc.read_dataset(f"u_scalar_{i}", session=test_session)
        assert _none_last(back.to_values("x")) == [1, 2, 3, None]


def test_union_collection_promotes_schema(test_session):
    """A collection signal promotes to Optional too, so the schema never claims a
    non-null type (faithful CH round-trip for collections is out of scope)."""
    lplain = dc.read_values(x=[[1, 2]], output={"x": list[int]}, session=test_session)
    lopt = dc.read_values(
        x=[None], output={"x": Optional[list[int]]}, session=test_session
    )
    _, l_opt = unwrap_optional(lplain.union(lopt).signals_schema.values["x"])
    assert l_opt


def test_union_optional_datamodel_schema_both_orders(test_session):
    """union of Optional[M] with M is Optional[M] in either operand order and
    round-trips, with a nullable _type_tag on both sides."""

    def mk_plain():
        return dc.read_values(
            item=[_Inner(score=1, label="a")],
            output={"item": _Inner},
            session=test_session,
        )

    def mk_opt():
        return dc.read_values(
            item=[_Inner(score=2, label="b"), None],
            output={"item": Optional[_Inner]},
            session=test_session,
        )

    for i, merged in enumerate(
        (mk_plain().union(mk_opt()), mk_opt().union(mk_plain()))
    ):
        _, is_optional = unwrap_optional(merged.signals_schema.values["item"])
        assert is_optional
        merged.save(f"u_dm_order_{i}")
        back = dc.read_dataset(f"u_dm_order_{i}", session=test_session)
        assert back.count() == 3
        assert sum(1 for v in back.to_values("item") if v is None) == 1


@skip_if_not_sqlite
def test_sqlite_udf_funcs_none_safe():
    """SQLite Python-UDF funcs return None on a None input instead of crashing
    ("user-defined function raised exception")."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2],
        t=["a-b", None],
        n=[10, None],
        a=[[1, 2], None],
        output={
            "id": int,
            "t": Optional[str],
            "n": Optional[int],
            "a": Optional[list[int]],
        },
    )
    out = chain.mutate(
        sp=func.string.split("t", "-"),
        h=func.int_hash_64("n"),
        ln=func.array.length("a"),
    ).order_by("id")
    rows = {r["id"]: (r["sp"], r["h"], r["ln"]) for r in out.to_records()}
    assert rows[1] == (["a", "b"], rows[1][1], 2)
    assert rows[2] == (None, None, None)


def test_greatest_least_over_optional_propagate_none(test_session):
    """greatest/least yield None on a NULL argument on both backends (CH natively
    ignores NULL, so a custom compile wraps it)."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2, 3],
        n=[10, None, 3],
        m=[5, 5, None],
        output={"id": int, "n": Optional[int], "m": Optional[int]},
        session=test_session,
    )
    out = chain.mutate(
        g=func.greatest("n", "m"),
        ls=func.least("n", "m"),
        # literal args never go NULL; only the column drives propagation
        gl=func.greatest("n", 7),
    ).order_by("id")
    rows = {r["id"]: (r["g"], r["ls"], r["gl"]) for r in out.to_records()}
    assert rows[1] == (10, 5, 10)
    assert rows[2] == (None, None, None)  # n is NULL
    assert rows[3] == (None, None, 7)  # m is NULL (gl ignores m → 7)


def test_case_no_else_is_nullable(test_session):
    """func.case with no else_ yields None for unmatched rows on both backends
    (not 0 on CH)."""
    from datachain import C, func

    chain = dc.read_values(
        id=[1, 2, 3],
        a=[10, None, 3],
        output={"id": int, "a": Optional[int]},
        session=test_session,
    )
    # unmatched (a<=5 or a is NULL) -> None, not 0
    no_else = chain.mutate(x=func.case((C("a") > 5, 100))).order_by("id")
    assert [r["x"] for r in no_else.to_records()] == [100, None, None]
    # an explicit else still fills every row
    with_else = chain.mutate(x=func.case((C("a") > 5, 100), else_=-1)).order_by("id")
    assert [r["x"] for r in with_else.to_records()] == [100, -1, -1]


def test_boolean_logic_over_optional_preserves_none(test_session):
    """and_/or_/not_ keep NULL on both backends (three-valued logic; not False on
    CH). 1/0 vs True/False is an unrelated repr difference, so compare via bool()."""
    from datachain import C, func

    chain = dc.read_values(
        id=[1, 2, 3],
        a=[10, None, 3],
        b=[1, 2, None],
        output={"id": int, "a": Optional[int], "b": Optional[int]},
        session=test_session,
    )
    out = chain.mutate(
        an=func.and_(C("a") > 5, C("b") > 0),
        orr=func.or_(C("a") > 5, C("b") > 0),
        nt=func.not_(C("a") > 5),
    ).order_by("id")

    def norm(v):
        return None if v is None else bool(v)

    rows = {
        r["id"]: (norm(r["an"]), norm(r["orr"]), norm(r["nt"]))
        for r in out.to_records()
    }
    # a NULL operand -> NULL unless the other operand decides the result
    assert rows[1] == (True, True, False)
    assert rows[2] == (None, True, None)
    assert rows[3] == (False, None, True)


def test_collect_over_optional_preserves_none_elements(test_session):
    """func.collect keeps NULL elements on both backends; collected order is
    unspecified, so compare as multisets."""
    from collections import Counter

    from datachain import func

    chain = dc.read_values(
        id=[1, 2, 3],
        g=[1, 1, 1],
        a=[10, None, 30],
        output={"id": int, "g": int, "a": Optional[int]},
        session=test_session,
    )
    rows = chain.group_by(c=func.collect("a"), partition_by="g").to_records()
    assert len(rows) == 1
    # the None element is preserved (length 3, not 2)
    assert Counter(rows[0]["c"]) == Counter([10, None, 30])


def test_array_funcs_over_optional_list_no_crash(test_session):
    """array.* funcs accept an Optional[list] column (the result type unwraps the
    Optional) — previously mutate() raised 'Array column must be of type list'."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2],
        x=[[10, 20, 30], None],
        output={"id": int, "x": Optional[list[int]]},
        session=test_session,
    )
    out = chain.mutate(
        el=func.array.get_element("x", 0),
        sl=func.array.slice("x", 0, 2),
    ).order_by("id")
    rows = {r["id"]: (r["el"], r["sl"]) for r in out.to_records()}
    assert rows[1] == (10, [10, 20])


def test_array_contains_over_null_array_no_crash(test_session):
    """array.contains over a NULL whole-array returns a value instead of crashing
    the SQLite UDF (json.loads(None))."""
    from datachain import func

    chain = dc.read_values(
        a=[["x", "y"], None],
        id=[1, 2],
        output={"a": Optional[list[str]], "id": int},
        session=test_session,
    )
    out = chain.mutate(c=func.array.contains("a", "x")).order_by("id")
    by_id = {r["id"]: r["c"] for r in out.to_records()}
    assert by_id[1]  # present array contains "x"


def test_string_hash_of_none_is_none(test_session):
    """string_hash(None) returns None on both backends (NULL in -> NULL out),
    instead of SQLite hashing the empty string."""
    from datachain import func

    chain = dc.read_values(
        id=[1, 2],
        s=["a", None],
        output={"id": int, "s": Optional[str]},
        session=test_session,
    )
    by_id = {
        r["id"]: r["h"]
        for r in chain.mutate(h=func.string.string_hash("s")).to_records()
    }
    assert by_id[1] is not None
    assert by_id[2] is None


def test_concat_all_null_group_is_none(test_session):
    """concat over an all-NULL group is None on both backends (not '')."""
    from datachain import func

    rows = dict(
        dc.read_values(
            g=[1, 1, 2, 2],
            v=[None, None, "a", "b"],
            output={"g": int, "v": Optional[str]},
            session=test_session,
        )
        .group_by(c=func.concat("v", ","), partition_by="g")
        .order_by("g")
        .to_list("g", "c")
    )
    assert rows[1] is None  # all-NULL group
    assert set(rows[2].split(",")) == {"a", "b"}  # order unspecified


def test_outer_merge_unmatched_model_is_none(test_session):
    """An outer/full merge of a plain DataModel returns None for unmatched rows
    on both backends, not a default-filled instance (the model's leaves widen to
    nullable, so the all-NULL row hydrates as None)."""
    left = dc.read_values(k=[1], session=test_session, output={"k": int})
    right = dc.read_values(
        k=[2],
        m=[_Inner(score=9, label="x")],
        output={"k": int, "m": _Inner},
        session=test_session,
    )
    got = left.merge(right, on="k", full=True).order_by("k").to_values("m")
    assert got == [None, _Inner(score=9, label="x")]
