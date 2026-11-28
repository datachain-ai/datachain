import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import sqlalchemy

import datachain as dc
from datachain import func
from datachain.lib import arrow
from datachain.lib.dc import C
from datachain.lib.file import File
from datachain.sql.types import Int64


@pytest.fixture
def arrow_reads(monkeypatch):
    reads = []
    read_batches = arrow.read_batches

    def record(ds, *args, **kw):
        reads.append(kw.get("columns") or ds.schema.names)
        return read_batches(ds, *args, **kw)

    monkeypatch.setattr(arrow, "read_batches", record)
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
    assert len(chain.to_list()) == 4
    assert mapped.to_values("x") == [4, 3, 2, 2]
    assert calls == 4
    assert arrow_reads == [["a", "b"], ["a", "b", "c"]]


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
