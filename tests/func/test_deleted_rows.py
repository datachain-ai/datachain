from datetime import datetime, timezone

import pytest

import datachain as dc
from datachain import func
from datachain.lib.file import File, VFileError

DELETED = [{"vtype": "deleted"}]
T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def source(tmp_path):
    (tmp_path / "a.txt").write_text("v2")
    return tmp_path.as_uri()


@pytest.fixture
def files(source):
    return [
        File(source=source, path="a.txt", version="1", etag="e1", is_latest=False),
        File(source=source, path="a.txt", version="2", etag="e2", size=2),
        File(
            source=source,
            path="b.txt",
            version="m1",
            etag="",
            is_latest=False,
            last_modified=T0,
            location=DELETED,
        ),
    ]


@pytest.fixture
def history(files, test_session):
    return dc.read_values(file=files, session=test_session)


def deleted_paths(chain, column="file"):
    return sorted((f.path, f.version) for f in chain.to_values(column) if f.is_deleted)


def test_delete_row_round_trips_through_save(history):
    (row,) = [
        f
        for f in dc.read_dataset(history.save("history").name).to_values("file")
        if f.version == "m1"
    ]

    assert row.is_deleted
    assert row.location == DELETED
    assert (row.etag, row.size, row.is_latest) == ("", 0, False)
    assert row.last_modified == T0


@pytest.mark.parametrize(
    "expr,versions",
    [
        (lambda: func.is_deleted("file"), ["m1"]),
        (lambda: dc.C("file.etag") == "", ["m1"]),
        (lambda: func.is_deleted("file") == False, ["1", "2"]),  # noqa: E712
        (lambda: ~(dc.C("file.etag") == ""), ["1", "2"]),
        (lambda: func.is_deleted("file") & dc.C("file.path").glob("b*"), ["m1"]),
    ],
)
def test_filter_delete_rows(history, expr, versions):
    assert sorted(history.filter(expr()).to_values("file.version")) == versions


def test_is_deleted_as_a_column(history):
    rows = history.mutate(d=func.is_deleted("file")).order_by("file.version")

    assert rows.to_list("file.version", "d") == [
        ("1", False),
        ("2", False),
        ("m1", True),
    ]


def test_is_deleted_on_a_renamed_column(files, test_session):
    chain = dc.read_values(f=files, session=test_session)

    assert chain.filter(func.is_deleted("f")).to_values("f.version") == ["m1"]
    assert deleted_paths(chain, "f") == [("b.txt", "m1")]


def test_delete_rows_through_chain_ops(history, files, test_session):
    plain = dc.read_values(file=files[:1], session=test_session)
    names = dc.read_values(
        path=["a.txt", "b.txt"], tag=["x", "y"], session=test_session
    )

    assert deleted_paths(history.select("file")) == [("b.txt", "m1")]
    assert deleted_paths(history.union(plain)) == [("b.txt", "m1")]
    assert deleted_paths(plain.union(history)) == [("b.txt", "m1")]
    assert deleted_paths(history.merge(names, on="file.path", right_on="path")) == [
        ("b.txt", "m1")
    ]
    assert deleted_paths(history.distinct("file.path", "file.version")) == [
        ("b.txt", "m1")
    ]
    assert deleted_paths(history.order_by("file.last_modified")) == [("b.txt", "m1")]


def test_map_over_delete_rows_with_prefetch(source, files, test_session):
    live = test_session.catalog.get_client(source).get_file_info("a.txt")
    rows = (
        dc.read_values(file=[live, files[2]], session=test_session)
        .settings(prefetch=2)
        .map(d=lambda file: file.is_deleted, output=bool)
        .order_by("file.path")
    )

    assert rows.to_values("d") == [False, True]


def test_gen_over_delete_rows_passes_them_through(history):
    rows = history.filter(func.is_deleted("file")).gen(
        f=lambda file: [file], output={"f": File}
    )

    assert deleted_paths(rows, "f") == [("b.txt", "m1")]


@pytest.mark.parametrize(
    "use",
    [
        lambda f, _: f.read(),
        lambda f, _: f.read_bytes(),
        lambda f, _: f.read_text(),
        lambda f, _: f.open().__enter__(),
        lambda f, _: f.open("wb").__enter__(),
        lambda f, _: f.ensure_cached(),
        lambda f, _: f.resolve(),
        lambda f, out: f.export(out),
        lambda f, out: f.export(out, link_type="symlink"),
        lambda f, out: f.save(str(out / "b.txt")),
    ],
)
def test_delete_row_has_no_content(history, tmp_path, use):
    (row,) = history.filter(func.is_deleted("file")).to_values("file")

    with pytest.raises(VFileError, match="object was deleted"):
        use(row, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_delete_rows_round_trip_through_parquet(history, tmp_path):
    history.to_parquet(tmp_path / "history.parquet")
    back = dc.read_parquet(str(tmp_path / "history.parquet"), session=history.session)

    assert deleted_paths(back) == [("b.txt", "m1")]
    assert back.filter(func.is_deleted("file")).to_values("file.version") == ["m1"]
