import time
from datetime import UTC, datetime

import pytest

import datachain as dc
from datachain import func
from datachain.lib.file import File, FileError
from tests.utils import make_tar, skip_if_not_sqlite

T0 = datetime(2026, 10, 1, tzinfo=UTC)


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
            is_deleted=True,
        ),
    ]


@pytest.fixture
def history(files, test_session):
    return dc.read_values(file=files, session=test_session)


def deleted_paths(chain, column="file"):
    return sorted((f.path, f.version) for f in chain.to_values(column) if f.is_deleted)


def test_row_for_deleted_object_round_trips_through_save(history):
    (row,) = [
        f
        for f in dc.read_dataset(history.save("history").name).to_values("file")
        if f.version == "m1"
    ]

    assert row.is_deleted
    assert (row.etag, row.size, row.is_latest) == ("", 0, False)
    assert row.last_modified == T0


@pytest.mark.parametrize(
    "expr,versions",
    [
        (lambda: dc.C("file.is_deleted"), ["m1"]),
        (lambda: dc.C("file.is_deleted") == False, ["1", "2"]),  # noqa: E712
        (lambda: dc.C("file.is_deleted") & dc.C("file.path").glob("b*"), ["m1"]),
    ],
)
def test_filter_rows_for_deleted_objects(history, expr, versions):
    assert sorted(history.filter(expr()).to_values("file.version")) == versions


def test_rows_for_deleted_objects_through_chain_ops(history, files, test_session):
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


def test_map_over_rows_for_deleted_objects_with_prefetch(source, files, test_session):
    live = test_session.catalog.get_client(source).get_file_info("a.txt")
    rows = (
        dc.read_values(file=[live, files[2]], session=test_session)
        .settings(prefetch=2)
        .map(d=lambda file: file.is_deleted, output=bool)
        .order_by("file.path")
    )

    assert rows.to_values("d") == [False, True]


def test_gen_over_rows_for_deleted_objects_passes_them_through(history):
    rows = history.filter(dc.C("file.is_deleted")).gen(
        f=lambda file: [file], output={"f": File}
    )

    assert deleted_paths(rows, "f") == [("b.txt", "m1")]


def _open(file, mode="rb"):
    with file.open(mode):
        pass


@pytest.mark.parametrize(
    "use",
    [
        lambda f, _: f.read_bytes(),
        lambda f, _: f.read_text(),
        lambda f, _: _open(f),
        lambda f, _: _open(f, "wb"),
        lambda f, _: f.ensure_cached(),
        lambda f, _: f.resolve(),
        lambda f, out: f.export(out),
        lambda f, out: f.export(out, link_type="symlink"),
        lambda f, out: f.save(str(out / "b.txt")),
    ],
    ids=[
        "read_bytes",
        "read_text",
        "open",
        "open_write",
        "ensure_cached",
        "resolve",
        "export",
        "export_symlink",
        "save",
    ],
)
def test_row_for_deleted_object_has_no_content(history, tmp_path, use):
    (row,) = history.filter(dc.C("file.is_deleted")).to_values("file")

    with pytest.raises(FileError, match="object was deleted"):
        use(row, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_rows_for_deleted_objects_round_trip_through_parquet(history, tmp_path):
    history.to_parquet(tmp_path / "history.parquet")
    back = dc.read_parquet(str(tmp_path / "history.parquet"), session=history.session)

    assert deleted_paths(back) == [("b.txt", "m1")]
    assert back.filter(dc.C("file.is_deleted")).to_values("file.version") == ["m1"]


# phase: (writes before the cutoff, writes after it); "-" deletes
SCENARIOS = {
    "unchanged": ("A", ""),
    "overwritten_after": ("A", "C"),
    "deleted_after": ("A", "-"),
    "deleted_before": ("A-", ""),
    "deleted_before_recreated_after": ("A-", "C"),
    "overwritten_before": ("AB", ""),
    "overwritten_twice_after": ("A", "CD"),
    "overwritten_then_deleted_after": ("A", "C-"),
    "created_after": ("", "C"),
}


def _write_history(root, phases):
    # moto keeps whole seconds: one write per key per second keeps the order
    steps = max(len(p) for p in phases.values())
    for step in range(steps):
        for key, writes in phases.items():
            if step < len(writes):
                if writes[step] == "-":
                    (root / key).unlink()
                else:
                    (root / key).write_text(writes[step])
        time.sleep(1.1)


@pytest.mark.parametrize(
    "cloud_type,version_aware", [("s3", True), ("gs", True)], indirect=True
)
def test_changes_since_a_cutoff(cloud_test_catalog, cloud_prefix):
    ctc = cloud_test_catalog
    prefix = cloud_prefix
    root = ctc.src / prefix
    _write_history(root, {k: before for k, (before, _) in SCENARIOS.items()})
    cutoff = datetime.now(UTC)
    time.sleep(1.1)
    _write_history(root, {k: after for k, (_, after) in SCENARIOS.items()})

    history = dc.read_storage(
        f"{ctc.src_uri}/{prefix}", session=ctc.session, versions="all"
    )
    newest = func.window(
        partition_by="file.path", order_by="file.last_modified", desc=True
    )
    then = (
        history.filter(dc.C("file.last_modified") <= cutoff)
        .mutate(rank=func.row_number().over(newest))
        .filter(dc.C("rank") == 1)
        .filter(dc.C("file.is_deleted") == False)  # noqa: E712
    )
    now = history.filter(dc.C("file.is_latest"))
    diff = then.merge(now, on="file.path", full=True, rname="now_")

    def name(path):
        return path.rsplit("/", 1)[-1]

    changes = {}
    for old, new in diff.to_list("file", "now_file"):
        key = name((old or new).path)
        if not old:
            changes[key] = "created"
        elif not new:
            changes[key] = ("deleted", old.read_text())
        elif old.version != new.version:
            changes[key] = ("overwritten", old.read_text())
        else:
            changes[key] = "unchanged"

    assert changes == {
        "unchanged": "unchanged",
        "overwritten_before": "unchanged",
        "overwritten_after": ("overwritten", "A"),
        "overwritten_twice_after": ("overwritten", "A"),
        "deleted_after": ("deleted", "A"),
        "overwritten_then_deleted_after": ("deleted", "A"),
        "deleted_before_recreated_after": "created",
        "created_after": "created",
    }
    deleted_now = history.group_by(
        latest=func.max(dc.C("file.is_latest")), partition_by="file.path"
    ).filter(dc.C("latest") == False)  # noqa: E712
    assert sorted(name(p) for p in deleted_now.to_values("file.path")) == [
        "deleted_after",
        "deleted_before",
        "overwritten_then_deleted_after",
    ]
    assert sorted(
        name(f.path) for f in history.filter(dc.C("file.is_deleted")).to_values("file")
    ) == [
        "deleted_after",
        "deleted_before",
        "deleted_before_recreated_after",
        "overwritten_then_deleted_after",
    ]


def test_parse_tabular_refuses_a_row_for_a_deleted_object(history):
    deleted = history.filter(dc.C("file.is_deleted"))

    with pytest.raises(FileError, match="object was deleted"):
        deleted.parse_tabular(format="csv").to_list()


@pytest.mark.parametrize(
    "cloud_type,version_aware", [("s3", True), ("gs", True)], indirect=True
)
def test_parse_tabular_reads_each_version(cloud_test_catalog, cloud_prefix):
    ctc = cloud_test_catalog
    for row in ("1", "2"):
        (ctc.src / cloud_prefix / "t.csv").write_text(f"x\n{row}\n")

    rows = (
        dc.read_storage(
            f"{ctc.src_uri}/{cloud_prefix}", session=ctc.session, versions="all"
        )
        .settings(prefetch=0)  # nothing in the cache
        .parse_tabular(format="csv", source=True)
        .to_list("x", "source")
    )

    assert sorted(x for x, _ in rows) == [1, 2]
    assert sorted(source.read()["x"] for _, source in rows) == [1, 2]


@pytest.mark.parametrize("cloud_type,version_aware", [("s3", True)], indirect=True)
def test_all_versions_read_a_noncurrent_null_version(cloud_test_catalog, cloud_prefix):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)

    def versioning(status):
        client.fs.call_s3(
            "put_bucket_versioning",
            Bucket=client.name,
            VersioningConfiguration={"Status": status},
        )

    versioning("Suspended")
    try:
        (ctc.src / cloud_prefix / "a").write_text("before versioning")
        (current,) = dc.read_storage(
            f"{ctc.src_uri}/{cloud_prefix}", session=ctc.session
        ).to_values("file")
        assert current.version == ""
    finally:
        versioning("Enabled")
    (ctc.src / cloud_prefix / "a").write_text("after")

    chain = dc.read_storage(
        f"{ctc.src_uri}/{cloud_prefix}",
        session=ctc.session,
        versions="all",
        update=True,
    )

    assert sorted(
        (f.version == "null", f.read_text()) for f in chain.to_values("file")
    ) == [
        (False, "after"),
        (True, "before versioning"),
    ]


@skip_if_not_sqlite
def test_copy_dataset_refuses_rows_for_deleted_objects(history, tmp_path):
    saved = history.filter(dc.C("file.is_deleted")).save("deleted")

    with pytest.raises(FileError, match="object was deleted"):
        history.session.catalog.cp([f"ds://{saved.name}"], str(tmp_path / "out"))


@pytest.mark.parametrize(
    "cloud_type,version_aware", [("s3", True), ("gs", True)], indirect=True
)
def test_deleted_archive_keeps_retained_version_members_readable(
    cloud_test_catalog, cloud_prefix
):
    from datachain.lib.tar import process_tar

    data = b"retained archive member"
    ctc = cloud_test_catalog
    key = f"{cloud_prefix}/archive.tar"
    (ctc.src / key).write_bytes(make_tar({"member.txt": data}))
    archive = dc.read_storage(f"{ctc.src_uri}/{key}", session=ctc.session)
    (file,) = archive.to_values("file")
    assert file.version
    members = archive.gen(file=process_tar).to_values("file")
    (ctc.src / key).unlink()
    ctc.catalog.cache.clear()
    history = dc.read_storage(
        f"{ctc.src_uri}/{cloud_prefix}/",
        versions="all",
        update=True,
        session=ctc.session,
    )
    (deleted,) = history.filter(dc.C("file.is_deleted")).to_values("file")
    assert deleted.is_deleted
    with pytest.raises(FileError, match="object was deleted"):
        list(process_tar(deleted))
    retained_members = (
        history.filter(
            dc.C("file.is_deleted") == False,  # noqa: E712
            dc.C("file.is_latest") == False,  # noqa: E712
        )
        .gen(file=process_tar)
        .to_values("file")
    )
    assert [m.read() for m in retained_members] == [data]
    assert [m.read() for m in members] == [data]
    assert all(not m.is_deleted for m in retained_members)
