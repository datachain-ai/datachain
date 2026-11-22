import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import sqlalchemy

import datachain as dc
from datachain import func
from datachain.lib import arrow
from datachain.lib.arrow import ArrowGenerator
from datachain.lib.dc import C
from datachain.lib.file import File
from datachain.sql.types import Int64
from tests.utils import skip_if_not_sqlite


@pytest.fixture
def arrow_reads(monkeypatch):
    reads = []
    for name in ("read_batches", "filter_batches"):
        fn = getattr(arrow, name)

        def record(ds, *args, _fn=fn, **kw):
            reads.append(kw.get("columns") or ds.schema.names)
            return _fn(ds, *args, **kw)

        monkeypatch.setattr(arrow, name, record)
    return reads


@pytest.fixture
def rows_parquet(tmp_dir):
    path = tmp_dir / "rows.parquet"
    pq.write_table(
        pa.table(
            {
                "a": [3, 2, 1, 1],
                "b": [30, 20, 10, 10],
                "c": [100, 200, 300, 400],
            }
        ),
        path,
    )
    return path.as_uri()


@pytest.fixture
def arrow_rows(monkeypatch):
    rows = []
    for name in ("read_batches", "filter_batches"):
        fn = getattr(arrow, name)

        def record(ds, *args, _fn=fn, **kw):
            total = 0
            for idx, batch in _fn(ds, *args, **kw):
                total += batch.num_rows
                yield idx, batch
            rows.append(total)

        monkeypatch.setattr(arrow, name, record)
    return rows


@pytest.fixture
def nulls_parquet(tmp_dir):
    path = tmp_dir / "nulls.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [0, 1, 2, 3],
                "x": pa.array([10, None, 20, 30], type=pa.int64()),
                "s": pa.array(["a", None, "b", "c"], type=pa.large_string()),
                "flag": pa.array([True, None, False, True], type=pa.bool_()),
                "cat": pa.array(["k1", None, "k2", "k1"]).dictionary_encode(),
            }
        ),
        path,
    )
    return path.as_uri()


def test_read_parquet_select_reads_only_selected_columns(
    tmp_dir, test_session, mocker, monkeypatch, arrow_reads
):
    path = tmp_dir / "episodes.parquet"
    pq.write_table(
        pa.table(
            {
                "unused": ["a", "b"],
                "length": pa.array([10, 20], type=pa.int64()),
                "stats/x/max": pa.array([1.5, 2.5], type=pa.float32()),
            }
        ),
        path,
    )
    monkeypatch.setattr(arrow, "PARTIAL_READ_PREFETCH_MAX_SIZE", 1)
    prefetch = mocker.spy(File, "_prefetch")
    spy = mocker.spy(arrow, "file_schemas")
    chain = dc.read_parquet(path.as_uri(), session=test_session)
    selected = chain.select("length", "stats_x_max").to_list()
    assert spy.call_count == 1
    assert prefetch.call_count == 0
    assert chain.count() == 2
    assert arrow_reads == [["length", "stats/x/max"], ["stats/x/max"]]

    full = dc.read_parquet(
        path.as_uri(),
        output={"unused": str, "length": int, "stats_x_max": float},
        session=test_session,
    )
    assert (
        selected
        == full.select("length", "stats_x_max").to_list()
        == [(10, 1.5), (20, 2.5)]
    )
    assert prefetch.call_count == 1
    assert arrow_reads[-1] == ["unused", "length", "stats/x/max"]


def test_read_parquet_select_nested_reads_top_level_column(
    tmp_dir, test_session, arrow_reads
):
    path = tmp_dir / "nested.parquet"
    pq.write_table(pa.table({"a": [{"b": 1, "c": 2}], "d": [3]}), path)
    chain = dc.read_parquet(path.as_uri(), session=test_session)
    ((partial_a,),) = chain.select("a.b").to_list()
    assert partial_a.model_dump() == {"b": 1}
    assert chain.to_list("a.b", "a.c") == [(1, 2)]
    assert arrow_reads == [["a"]]
    ((_, full_a, full_d),) = chain.to_list()
    assert (full_a.b, full_a.c, full_d) == (1, 2, 3)


def test_read_parquet_select_keeps_source_only_when_selected(
    rows_parquet, test_session, arrow_reads
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    rows = chain.select("a", "source.file.path").to_list("a", "source.file.path")
    assert rows == [
        (3, "rows.parquet"),
        (2, "rows.parquet"),
        (1, "rows.parquet"),
        (1, "rows.parquet"),
    ]
    assert chain.to_values("source.index") == [0, 1, 2, 3]
    assert arrow_reads == [["a"]]
    full = dc.read_parquet(
        rows_parquet,
        output={"a": int, "b": int, "c": int},
        session=test_session,
    )
    assert rows == full.to_list("a", "source.file.path")


def test_read_parquet_with_source_column_pushes_down_data_or_read_source(
    tmp_dir, test_session, arrow_reads
):
    path = tmp_dir / "df.parquet"
    pq.write_table(pa.table({"source": ["s3"], "a": [1], "b": [2]}), path)
    chain = dc.read_parquet(path.as_uri(), session=test_session)
    assert chain.to_list("a", "source") == [(1, "s3")]
    assert chain.select("a", "c0_source.file.path").to_list(
        "a", "c0_source.file.path"
    ) == [(1, "df.parquet")]
    assert arrow_reads == [["source", "a"], ["a"]]


def test_read_parquet_reused_for_different_column_selections(
    rows_parquet, test_session, arrow_reads
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    assert chain.to_values("a") == [3, 2, 1, 1]
    assert chain.to_values("c") == [100, 200, 300, 400]
    assert chain.to_list("a", "b", "c") == [
        (3, 30, 100),
        (2, 20, 200),
        (1, 10, 300),
        (1, 10, 400),
    ]
    assert arrow_reads == [["a"], ["c"], ["a", "b", "c"]]


def test_read_parquet_reuses_cached_full_read_before_projecting(
    rows_parquet, test_session, arrow_reads
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    assert [(r[1], r[2], r[3]) for r in chain.to_list()] == [
        (3, 30, 100),
        (2, 20, 200),
        (1, 10, 300),
        (1, 10, 400),
    ]
    assert chain.to_values("a") == [3, 2, 1, 1]
    assert chain.filter(C("b") > 10).count() == 2
    assert arrow_reads == [["a", "b", "c"]]
    assert chain.settings(ephemeral=True).to_values("a") == [3, 2, 1, 1]
    assert arrow_reads == [["a", "b", "c"], ["a"]]


def test_read_parquet_full_read_does_not_invalidate_narrow_map_checkpoint(
    rows_parquet, test_session, arrow_reads
):
    calls = 0

    def slow(a: int) -> int:
        nonlocal calls
        calls += 1
        return a + 1

    chain = dc.read_parquet(rows_parquet, session=test_session)
    mapped = chain.select("a", "b").map(x=slow)
    assert mapped.to_values("x") == [4, 3, 2, 2]
    assert calls == 4
    filtered = chain.filter(C("a") >= 2).map(x=slow)
    assert filtered.to_values("x") == [4, 3]
    assert calls == 6
    assert len(chain.to_list()) == 4
    assert mapped.to_values("x") == [4, 3, 2, 2]
    assert filtered.to_values("x") == [4, 3]
    assert calls == 6
    assert arrow_reads == [["a", "b"], ["a", "b", "c"], ["a", "b", "c"]]


def test_read_csv_and_json_project_selected_columns(tmp_dir, test_session, arrow_reads):
    csv_path = tmp_dir / "rows.csv"
    csv_path.write_text("a,b,c\n3,30,100\n2,20,200\n")
    assert dc.read_csv(csv_path.as_uri(), session=test_session).filter(
        C("b") > 20
    ).to_values("a") == [3]

    jsonl_path = tmp_dir / "rows.jsonl"
    jsonl_path.write_text('{"a": 3, "b": 30, "c": 100}\n{"a": 2, "b": 20, "c": 200}\n')
    assert dc.read_storage(jsonl_path.as_uri(), session=test_session).parse_tabular(
        format="json"
    ).to_values("c") == [100, 200]
    assert arrow_reads == [["a", "b"], ["c"]]


@pytest.mark.parametrize(
    "run,expected,columns",
    [
        pytest.param(
            lambda c: c.to_list("c", "a"),
            [(100, 3), (200, 2), (300, 1), (400, 1)],
            ["a", "c"],
            id="to_list",
        ),
        pytest.param(
            lambda c: c.to_values("c"),
            [100, 200, 300, 400],
            ["c"],
            id="to_values",
        ),
        pytest.param(
            lambda c: c.select("a", "sys.id").to_values("a"),
            [3, 2, 1, 1],
            ["a"],
            id="select_with_sys_id",
        ),
        pytest.param(
            lambda c: c.select("a", d=C("b") + 1).to_list("a", "d"),
            [(3, 31), (2, 21), (1, 11), (1, 11)],
            ["a", "b"],
            id="select_expr",
        ),
        pytest.param(
            lambda c: c.filter(C("b") > 10).select("a").to_values("a"),
            [3, 2],
            ["a", "b"],
            id="filter_then_select",
        ),
        pytest.param(
            lambda c: c.select("a").filter(C("b") > 10).to_values("a"),
            [3, 2],
            ["a", "b"],
            id="select_then_filter",
        ),
        pytest.param(
            lambda c: (
                c.select("a", "b").map(x=lambda a: a + 1, output=int).to_values("x")
            ),
            [4, 3, 2, 2],
            ["a", "b"],
            id="select_then_map",
        ),
        pytest.param(
            lambda c: c.order_by("b").select("a").to_values("a"),
            [1, 1, 2, 3],
            ["a", "b"],
            id="order_by_then_select",
        ),
        pytest.param(
            lambda c: c.mutate(d=C("b") + 1).select("a", "d").to_list("a", "d"),
            [(3, 31), (2, 21), (1, 11), (1, 11)],
            ["a", "b"],
            id="mutate_then_select",
        ),
        pytest.param(
            lambda c: c.limit(2).offset(1).select("a").to_values("a"),
            [2, 1],
            ["a"],
            id="limit_offset_then_select",
        ),
        pytest.param(
            lambda c: sorted(
                c.group_by(total=func.sum("b"), partition_by="a")
                .save("grouped")
                .to_list("a", "total")
            ),
            [(1, 20), (2, 20), (3, 30)],
            ["a", "b"],
            id="group_by",
        ),
        pytest.param(
            lambda c: sorted(
                c.group_by(n=func.count(), partition_by="a").to_list("a", "n")
            ),
            [(1, 2), (2, 1), (3, 1)],
            ["a"],
            id="group_by_count",
        ),
        pytest.param(
            lambda c: sorted(c.distinct("b").select("a").to_values("a")),
            [1, 2, 3],
            ["a", "b"],
            id="distinct_then_select",
        ),
        pytest.param(
            lambda c: sorted(
                c.mutate(
                    r=func.row_number().over(
                        func.window(partition_by="a", order_by="b")
                    )
                )
                .select("a", "r")
                .to_list("a", "r")
            ),
            [(1, 1), (1, 2), (2, 1), (3, 1)],
            ["a", "b"],
            id="window_order_by",
        ),
        pytest.param(
            lambda c: (
                c.gen(lambda b: [(b + 1,)], params=["b"], output={"d": int})
                .save("generated")
                .to_values("d")
            ),
            [31, 21, 11, 11],
            ["b"],
            id="gen_params",
        ),
        pytest.param(
            lambda c: sorted(
                c.agg(
                    lambda b: [(sum(b),)],
                    params=["b"],
                    partition_by="a",
                    output={"total": int},
                ).to_values("total")
            ),
            [20, 20, 30],
            ["a", "b"],
            id="agg_partition_by",
        ),
        pytest.param(
            lambda c: c.count(),
            4,
            ["a"],
            id="count",
        ),
        pytest.param(
            lambda c: c.filter(C("b") > 10).count(),
            2,
            ["b"],
            id="filter_then_count",
        ),
        pytest.param(
            lambda c: c.select("a").order_by(C("b")).limit(2).to_values("a"),
            [1, 1],
            ["a", "b"],
            id="select_then_order_by_dropped_column",
        ),
        pytest.param(
            lambda c: sorted(c._query.group_by([func.avg("b")], ["a"])),
            [(1, 10.0), (2, 20.0), (3, 30.0)],
            ["a", "b"],
            id="query_group_by_function",
        ),
    ],
)
def test_read_parquet_projects_columns_across_steps(
    rows_parquet, test_session, arrow_reads, run, expected, columns
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    full = dc.read_parquet(
        rows_parquet,
        output={"a": int, "b": int, "c": int},
        session=test_session,
    )
    assert run(chain) == expected
    assert arrow_reads == [columns]
    assert run(full) == expected


@pytest.mark.parametrize(
    "run,expected",
    [
        pytest.param(
            lambda c: [(r[1], r[2], r[3]) for r in c.to_list()],
            [(3, 30, 100), (2, 20, 200), (1, 10, 300), (1, 10, 400)],
            id="to_list_all",
        ),
        pytest.param(
            lambda c: [r[-3:] for r in c._query.select()],
            [(3, 30, 100), (2, 20, 200), (1, 10, 300), (1, 10, 400)],
            id="empty_sql_select",
        ),
        pytest.param(
            lambda c: c.save("saved").to_list("a", "b", "c"),
            [(3, 30, 100), (2, 20, 200), (1, 10, 300), (1, 10, 400)],
            id="save",
        ),
        pytest.param(
            lambda c: list(c._query.distinct().select("a")),
            [(3,), (2,), (1,), (1,)],
            id="unscoped_distinct",
        ),
        pytest.param(
            lambda c: c.select_except("c").to_values("a"),
            [3, 2, 1, 1],
            id="select_except",
        ),
        pytest.param(
            lambda c: c.merge(c, on="c").to_values("a"),
            [3, 2, 1, 1],
            id="merge",
        ),
        pytest.param(
            lambda c: c.union(c).to_values("a"),
            [3, 2, 1, 1, 3, 2, 1, 1],
            id="union",
        ),
        pytest.param(
            lambda c: c.filter(
                sqlalchemy.literal_column('"b" > 10', type_=sqlalchemy.Boolean())
            ).to_values("a"),
            [3, 2],
            id="filter_literal_column",
        ),
        pytest.param(
            lambda c: c.mutate(
                x=sqlalchemy.literal_column("b + 1", type_=Int64())
            ).to_values("x"),
            [31, 21, 11, 11],
            id="mutate_literal_column",
        ),
        pytest.param(
            lambda c: list(c._query.filter(sqlalchemy.text('"b" > 10')).select("a")),
            [(3,), (2,)],
            id="filter_text_clause",
        ),
        pytest.param(
            lambda c: (
                c.select("a")
                .filter(C("sys.rand") == c.to_values("sys.rand")[0])
                .to_values("a")
            ),
            [3],
            id="select_then_filter_sys_rand",
        ),
        pytest.param(
            lambda c: c.filter(
                C("sys.rand") == c.to_list("a", "sys.rand")[0][1]
            ).to_values("a"),
            [3],
            id="select_data_and_sys_rand",
        ),
        pytest.param(
            lambda c: (
                [r[1:] for r in c.to_list("a", "sys.id", "sys.rand")]
                == [r[2:] for r in c.to_list("a", "b", "sys.id", "sys.rand")]
            ),
            True,
            id="select_data_and_both_sys_columns",
        ),
    ],
)
def test_read_parquet_fallback_reads_all_columns(
    rows_parquet, test_session, arrow_reads, run, expected
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    full = dc.read_parquet(
        rows_parquet,
        output={"a": int, "b": int, "c": int},
        session=test_session,
    )
    assert run(chain) == run(full) == expected
    assert arrow_reads[0] == ["a", "b", "c"]


def test_map_after_read_parquet_runs_once_for_two_selections(
    rows_parquet, test_session, arrow_reads
):
    def run(chain):
        calls = iter(range(10))
        mapped = chain.map(lambda a: next(calls), output={"tag": int})
        return mapped.to_list("a", "tag"), mapped.to_list("a", "b", "tag")

    first, second = run(dc.read_parquet(rows_parquet, session=test_session))
    full_first, full_second = run(
        dc.read_parquet(
            rows_parquet,
            output={"a": int, "b": int, "c": int},
            session=test_session,
        )
    )
    assert first == full_first == [(3, 0), (2, 1), (1, 2), (1, 3)]
    assert second == full_second == [(3, 30, 0), (2, 20, 1), (1, 10, 2), (1, 10, 3)]
    assert arrow_reads == [["a", "b", "c"], ["a", "b", "c"]]


def test_read_parquet_shuffle_reads_all_columns(
    rows_parquet, test_session, arrow_reads
):
    shuffled = dc.read_parquet(rows_parquet, session=test_session).shuffle().limit(3)
    assert [
        (a, b)
        for (a,), (b,) in zip(shuffled.to_list("a"), shuffled.to_list("b"), strict=True)
    ] == shuffled.to_list("a", "b")
    assert arrow_reads == [["a", "b", "c"]]


def test_parse_tabular_delta_reads_all_columns(rows_parquet, test_session, arrow_reads):
    dc.read_storage(rows_parquet, session=test_session).save("files")

    def read(**kwargs):
        return (
            dc.read_dataset(
                "files",
                delta=True,
                delta_on="file.path",
                delta_result_on="source.file.path",
                session=test_session,
            )
            .parse_tabular(**kwargs)
            .to_values("a")
        )

    assert read() == read(output={"a": int, "b": int, "c": int}) == [3, 2, 1, 1]
    assert arrow_reads[0] == ["a", "b", "c"]


def test_read_parquet_zero_column_count(tmp_dir, test_session):
    path = tmp_dir / "empty_cols.parquet"
    pq.write_table(pa.table({}), path)
    assert dc.read_parquet(path.as_uri(), session=test_session).count() == 0


def test_filter_between_on_values(test_session):
    chain = dc.read_values(x=[1, 2, 3, 4], session=test_session)
    assert chain.filter(C("x").between(2, 3)).order_by("x").to_values("x") == [2, 3]
    assert chain.filter(~C("x").between(2, 3)).order_by("x").to_values("x") == [1, 4]


def test_read_parquet_filter_prunes_row_groups_and_streams_payload(
    tmp_dir, test_session, mocker, monkeypatch
):
    path = tmp_dir / "multi_rg.parquet"
    table = pa.table(
        {
            "a": [1, 2, 10, 20, 100, 200, 100, 100, 200, 100, 100, 100],
            "b": [10, 20, 10, 70, 50, 60, 50, 50, 60, 50, 50, 50],
            "payload": [f"p{i}" for i in range(12)],
        }
    )
    with pq.ParquetWriter(path, table.schema) as writer:
        writer.write_table(table.slice(0, 2))  # rg 0: pruned by stats (a < 10)
        writer.write_table(table.slice(2, 2))  # rg 1: passes stats, 0 hits for b == 60
        writer.write_table(table.slice(4, 8))  # rg 2: 8 rows, hits at pos 1 and 4

    monkeypatch.setattr(ArrowGenerator, "DEFAULT_BATCH_SIZE", 2)
    monkeypatch.setattr(arrow, "SMALL_ROW_GROUP_BYTES", 0)
    scan_rest = mocker.spy(arrow, "_scan_group_rest")
    res = (
        dc.read_parquet(path.as_uri(), session=test_session)
        .filter((C("a") >= 10) & (C("b") == 60))
        .select("payload", "source.index")
        .to_list("payload", "source.index")
    )
    assert sorted(res) == [("p5", 5), ("p8", 8)]
    assert scan_rest.call_count == 1


def test_read_parquet_filter_coalesces_remaining_row_groups_after_two_match(
    tmp_dir, test_session, mocker, monkeypatch
):
    path = tmp_dir / "seven_rg.parquet"
    table = pa.table(
        {
            # rg 0 and rg 1 match (1); rg 2..5 (4-group window) and rg 6 have 0s
            # alongside 2s so statistics keep all 7 groups for C("a") == 1.
            "a": [1, 0, 1, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
            "payload": [f"p{i}" for i in range(14)],
        }
    )
    pq.write_table(table, path, row_group_size=2)
    monkeypatch.setattr(arrow, "SMALL_ROW_GROUP_BYTES", 0)
    scan_spy = mocker.spy(arrow, "_scan_groups")
    rest_spy = mocker.spy(arrow, "_scan_group_rest")

    chain = dc.read_parquet(path.as_uri(), session=test_session)
    res = (
        chain.filter(C("a") == 1)
        .select("payload", "source.index")
        .to_list("payload", "source.index")
    )
    assert sorted(res) == [("p0", 0), ("p2", 2)]
    # rg 0 and rg 1 use _scan_group_rest; rg 2..5 use one 4-group _scan_groups
    # window (0 hits -> resets matched=0); rg 6 probes filter column only (0 hits).
    assert rest_spy.call_count == 2
    assert scan_spy.call_count == 1

    all_res = (
        chain.filter(C("a") >= 0)
        .select("payload", "source.index")
        .to_list("payload", "source.index")
    )
    assert len(all_res) == 14
    # rg 0 and rg 1 use _scan_group_rest; rg 2..5 and rg 6 use two _scan_groups windows.
    assert rest_spy.call_count == 4
    assert scan_spy.call_count == 3


def test_read_parquet_filter_without_select_preserves_source_index_and_all_columns(
    tmp_dir, test_session, arrow_rows
):
    path = tmp_dir / "rg.parquet"
    table = pa.table(
        {
            "id": [10, 20, 30, 40],
            "text": ["w0", "w1", "w2", "w3"],
        }
    )
    pq.write_table(table, path, row_group_size=2)

    chain = (
        dc.read_parquet(path.as_uri(), session=test_session)
        .filter(C("id") == 30)
        .save("filtered_only")
    )
    assert arrow_rows == [1]
    ((source, i, text),) = chain.to_list("source", "id", "text")
    assert (source.index, i, text) == (2, 30, "w2")
    assert source.read() == {"id": 30, "text": "w2"}


def test_read_parquet_filter_with_nested_column_preserves_source_index(
    tmp_dir, test_session
):
    path = tmp_dir / "nested_filter.parquet"
    pq.write_table(
        pa.table({"a": [10, 20, 30], "nested": [{"x": 1}, {"x": 2}, {"x": 3}]}),
        path,
    )
    res = (
        dc.read_parquet(path.as_uri(), session=test_session)
        .filter(C("a") == 20)
        .to_list("source.index", "a", "nested.x")
    )
    assert res == [(1, 20, 2)]


@pytest.mark.parametrize(
    "predicate,expected_a,expected_indices",
    [
        (C("a") == 2, [2], [1]),
        (C("a") != 1, [3, 2], [0, 1]),
        (C("a") < 2, [1, 1], [2, 3]),
        (C("a") <= 2, [2, 1, 1], [1, 2, 3]),
        (C("a") > 1, [3, 2], [0, 1]),
        (C("a") >= 2, [3, 2], [0, 1]),
        ((C("a") == 1) & (C("c") == 400), [1], [3]),
        ((C("a") == 3) | (C("c") == 400), [3, 1], [0, 3]),
        (~(C("a") == 1), [3, 2], [0, 1]),
        (~((C("a") == 1) | (C("a") == 2)), [3], [0]),
        (C("a").in_([2, 3]), [3, 2], [0, 1]),
        (C("a").not_in([1, 2]), [3], [0]),
        (~C("a").in_([1, 2]), [3], [0]),
        (C("a").between(2, 3), [3, 2], [0, 1]),
        (~C("a").between(2, 3), [1, 1], [2, 3]),
    ],
)
def test_read_parquet_filter_pushdown_operators(
    rows_parquet,
    test_session,
    predicate,
    expected_a,
    expected_indices,
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    full = dc.read_parquet(
        rows_parquet,
        output={"a": int, "b": int, "c": int},
        session=test_session,
    )
    expected = list(zip(expected_a, expected_indices, strict=True))
    assert (
        chain.filter(predicate).to_list("a", "source.index")
        == full.filter(predicate).to_list("a", "source.index")
        == expected
    )


@pytest.mark.parametrize(
    "predicate,expected",
    [
        pytest.param(C("x").is_(None), [1], id="is_none"),
        pytest.param(C("x").is_not(None), [0, 2, 3], id="is_not_none"),
        pytest.param(C("x").in_([10, 30]), [0, 3], id="in"),
        pytest.param(C("x").not_in([10, 30]), [2], id="not_in"),
        pytest.param(~C("x").in_([10, 20]), [3], id="not_in_expr"),
        pytest.param(
            (C("x").not_in([10, 30])) | C("x").is_(None),
            [1, 2],
            id="not_in_or_is_none",
        ),
        pytest.param(C("s") == "b", [2], id="large_string_eq"),
        pytest.param(C("cat") == "k2", [2], id="dictionary_eq"),
        pytest.param(C("flag"), [0, 3], id="bool_col"),
    ],
)
def test_read_parquet_filter_pushdown_nulls_and_booleans(
    nulls_parquet, test_session, predicate, expected
):
    chain = dc.read_parquet(nulls_parquet, session=test_session)
    full = dc.read_parquet(
        nulls_parquet,
        output={
            "id": int,
            "x": int | None,
            "s": str | None,
            "flag": bool | None,
            "cat": str | None,
        },
        session=test_session,
    )
    assert (
        chain.filter(predicate).to_values("id")
        == full.filter(predicate).to_values("id")
        == expected
    )


@skip_if_not_sqlite
@pytest.mark.parametrize(
    "predicate,expected",
    [
        pytest.param(C("flag").is_(False), [2], id="is_false"),
        pytest.param(C("flag").is_not(True), [1, 2], id="is_not_true"),
        pytest.param(
            ~(C("flag").is_(True) & (C("id") < 2)),
            [1, 2, 3],
            id="not_is_true_and_lt",
        ),
    ],
)
def test_read_parquet_filter_pushdown_boolean_is_true_and_false(
    nulls_parquet, test_session, predicate, expected
):
    # ClickHouse evaluates IS NOT TRUE on nullable UInt8 without matching NULL.
    chain = dc.read_parquet(nulls_parquet, session=test_session)
    full = dc.read_parquet(
        nulls_parquet,
        output={
            "id": int,
            "x": int | None,
            "s": str | None,
            "flag": bool | None,
            "cat": str | None,
        },
        session=test_session,
    )
    assert (
        chain.filter(predicate).to_values("id")
        == full.filter(predicate).to_values("id")
        == expected
    )


def test_read_parquet_filter_partial_pushdown_with_sql_predicate(
    rows_parquet, test_session
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    res = (
        chain.filter((C("a") == 1) & (C("c") + 1 > 350))
        .select("c", "source.index")
        .to_list("c", "source.index")
    )
    assert res == [(400, 3)]


def test_read_parquet_different_filters_do_not_collide_in_cache(
    rows_parquet, test_session, arrow_reads
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    assert chain.filter(C("a") == 3).to_values("b") == [30]
    assert chain.filter(C("a") == 2).to_values("b") == [20]
    assert chain.to_values("b") == [30, 20, 10, 10]
    assert arrow_reads == [["a", "b"], ["a", "b"], ["b"]]


def test_read_parquet_filters_with_long_in_lists_get_their_own_reads(
    rows_parquet, test_session
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    in_a = [-1000 - i for i in range(100)]
    in_b = in_a.copy()
    in_a[45] = 3
    in_b[45] = 2
    assert chain.filter(C("a").in_(in_a)).to_values("b") == [30]
    assert chain.filter(C("a").in_(in_b)).to_values("b") == [20]


def test_read_parquet_filter_preserves_sys_id_when_selected(rows_parquet, test_session):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    assert chain.filter(C("a") == 1).to_list("a", "source.index", "sys.id") == [
        (1, 2, 3),
        (1, 3, 4),
    ]


def test_read_parquet_filter_preserves_sys_id_when_both_sys_columns_selected(
    rows_parquet, test_session
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    assert [
        (a, idx, sid)
        for a, idx, sid, _ in chain.filter(C("a") == 1).to_list(
            "a", "source.index", "sys.id", "sys.rand"
        )
    ] == [
        (1, 2, 3),
        (1, 3, 4),
    ]


def test_read_parquet_filter_preserves_sys_id_when_filtered(rows_parquet, test_session):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    assert chain.filter(C("a") == 1, C("sys.id") == 3).to_list("a", "source.index") == [
        (1, 2)
    ]


def test_read_parquet_filter_disabled_on_literal_column(rows_parquet, test_session):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    assert chain.filter(
        C("a") < 3,
        sqlalchemy.literal_column("sys__id", type_=sqlalchemy.Integer()) == 3,
    ).to_list("a", "source.index") == [(1, 2)]


def test_read_parquet_filter_joined_on_sys_id(tmp_dir, test_session):
    path = tmp_dir / "join_sys.parquet"
    pq.write_table(
        pa.table({"a": [5, 6, 7, 8, 9, 10], "b": ["p", "q", "r", "s", "t", "u"]}),
        path,
    )
    uri = path.as_uri()
    left = dc.read_parquet(uri, session=test_session).filter(C("a") > 7)
    right = dc.read_parquet(uri, session=test_session).select("sys.id", "b")
    assert left.merge(right, on="sys.id", rname="r_").order_by("a").to_list(
        "a", "r_b"
    ) == [(8, "s"), (9, "t"), (10, "u")]


def test_read_parquet_filter_across_select_and_order_by(
    rows_parquet, test_session, arrow_rows
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    assert chain.select("a", "b").filter(C("a") == 3).to_list("a", "b") == [(3, 30)]
    assert chain.order_by("b").filter(C("b") == 20).to_list("a", "b") == [(2, 20)]
    res = (
        chain.select("a", "b")
        .order_by("b")
        .filter(C("a") < 3)
        .filter(C("b") >= 20)
        .to_list("a", "b")
    )
    assert res == [(2, 20)]
    assert arrow_rows == [1, 1, 1]
    grouped = (
        chain.filter(C("a") > 1)
        .group_by(n=dc.func.count(), partition_by="a")
        .order_by("a")
        .to_list("a", "n")
    )
    assert grouped == [(2, 1), (3, 1)]


def test_read_parquet_filter_does_not_push_across_limit_or_mutate(
    rows_parquet, test_session
):
    chain = dc.read_parquet(rows_parquet, session=test_session)
    assert chain.limit(1).filter(C("a") == 2).to_values("a") == []
    assert chain.mutate(a=C("a") + 10).filter(C("a") == 12).to_values("a") == [12]
    assert [
        r[0] for r in chain._query.select(a=C("a") + 1).select("a").filter(C("a") == 2)
    ] == [2, 2]
    assert [
        r[0] for r in chain._query.select(a=C("b")).select("a").filter(C("a") == 20)
    ] == [20]


def test_read_parquet_filter_skips_nullable_field_with_non_optional_output():
    from datachain.lib.arrow_filter import to_arrow_filter

    nullable_schema = pa.schema([pa.field("x", pa.int64(), nullable=True)])
    non_null_schema = pa.schema([pa.field("x", pa.int64(), nullable=False)])
    assert to_arrow_filter([C("x") == 0], nullable_schema, {"x": int}) is None
    assert (
        to_arrow_filter([C("x") == 0], nullable_schema, {"x": int | None}) is not None
    )
    assert to_arrow_filter([C("x") == 0], non_null_schema, {"x": int}) is not None
    assert to_arrow_filter([C("x") < 2**64], non_null_schema, {"x": int}) is None


@skip_if_not_sqlite
def test_read_parquet_filter_unpushable_types_fall_back_to_sql(tmp_dir, test_session):
    path = tmp_dir / "floats.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [0, 1, 2, 3],
                "f": [1.5, 0.5, float("nan"), None],
                "s": ["0", "5", "9", "9"],
                "n": pa.nulls(4),
                "u": pa.array([0, 1, 2, 3], type=pa.uint64()),
            }
        ),
        path,
    )
    chain = dc.read_parquet(path.as_uri(), session=test_session)
    assert chain.filter(C("f") != 1.5).to_values("id") == [1]
    assert chain.filter(~((C("id") == 0) | (C("f") == 0.5))).to_values("id") == []
    assert chain.filter((C("id") == 0) | (C("f") == 0.5)).to_values("id") == [0, 1]
    assert chain.filter(C("f").is_(None)).to_values("id") == [2, 3]
    assert chain.filter(C("id").not_in([0, 1.5])).to_values("id") == [1, 2, 3]
    assert chain.filter(C("id").in_([1, "2"])).to_values("id") == [1, 2]
    assert chain.filter(C("id").between(0, 1.5)).to_values("id") == [0, 1]
    assert chain.filter(C("id").between(0, "2")).to_values("id") == [0, 1, 2]
    assert chain.filter(C("s")).to_values("id") == [1, 2, 3]
    assert chain.filter(C("id") == True).to_values("id") == [1]  # noqa: E712
    assert chain.filter(C("id") > "5").to_values("id") == []
    assert chain.filter(C("s") == 5).to_values("id") == [1]
    assert chain.filter(C("n").in_(["foo"])).to_values("id") == []
    assert chain.filter(C("u") > -1).to_values("id") == [0, 1, 2, 3]


def test_read_parquet_filter_on_blank_column_name(tmp_dir, test_session):
    blank = tmp_dir / "blank.parquet"
    pq.write_table(pa.table({"": [0, 1, 2], "n": [10, 11, 12]}), blank)
    res = (
        dc.read_parquet(blank.as_uri(), session=test_session)
        .filter(C("c0") == 1)
        .to_list("source.index", "c0", "n")
    )
    assert res == [(1, 1, 11)]


def test_read_parquet_filter_on_column_missing_in_one_file(tmp_dir, test_session):
    drift_dir = tmp_dir / "drift"
    drift_dir.mkdir()
    pq.write_table(
        pa.table({"a": [0, 1, 2, 3]}), drift_dir / "f1.parquet", row_group_size=2
    )
    pq.write_table(
        pa.table({"a": [1, 2, 3, 4], "extra": ["w", "x", "y", "z"]}),
        drift_dir / "f2.parquet",
        row_group_size=2,
    )
    drift = dc.read_parquet(drift_dir.as_uri(), session=test_session)
    assert drift.filter(C("extra").is_(None) & (C("a") == 2)).to_list(
        "source.index", "a", "extra"
    ) == [(2, 2, None)]
