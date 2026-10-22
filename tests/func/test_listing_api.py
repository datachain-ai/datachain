import json
import uuid

import pyarrow as pa
import pytest

import datachain as dc
from datachain.lib.convert.flatten import flatten
from datachain.lib.dc import DataChain
from datachain.lib.file import File
from datachain.lib.listing import get_listing
from datachain.lib.settings import Settings
from datachain.lib.signal_schema import SignalSchema
from datachain.query.dataset import DatasetQuery
from datachain.query.session import Session


def _read_listing(session, uri, versions):
    name, _, _, reused = get_listing(uri, session, versions=versions)
    assert reused
    query = DatasetQuery(name=name, session=session)
    query.resolve_listing()
    schema = SignalSchema({"sys": dc.Sys, "file": File})
    return DataChain(query, Settings(), schema).to_values("file")


@pytest.mark.parametrize("versions", [False, True])
def test_listing_rows_match_saved_listing(cloud_test_catalog, versions):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/cats"

    rows = sorted(dc.list_storage(uri, versions=versions, session=ctc.session))
    chain = dc.list_storage(uri, versions=versions, session=ctc.session).save()

    assert {type(r) for r in rows} == {dc.FileRow}
    assert {r.path.removeprefix("cats/") for r in rows} == {"cat1", "cat2"}
    assert rows == sorted(flatten(f) for f in _read_listing(ctc.session, uri, versions))
    assert sorted(flatten(f) for f in chain.to_values("file")) == [
        r for r in rows if r.is_latest
    ]


@pytest.mark.parametrize("versions", [False, True])
def test_listing_save_is_the_read_storage_cache(cloud_test_catalog, versions):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/dogs/"

    saved = dc.list_storage(uri, versions=versions, session=ctc.session).save()
    _, _, _, reused = get_listing(uri, ctc.session, versions=versions)
    read = dc.read_storage(uri, versions=versions, session=ctc.session)

    assert reused
    assert saved.order_by("file.path").to_values("file") == read.order_by(
        "file.path"
    ).to_values("file")
    names = [lst.name for lst in dc.listings(session=ctc.session).to_values("listing")]
    assert len(names) == 1


def test_listing_to_arrow(cloud_test_catalog):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/dogs/"
    rows = sorted(dc.list_storage(uri, session=ctc.session))
    table = dc.list_storage(uri, session=ctc.session).to_arrow()

    assert table.column_names == list(dc.FileRow._fields)
    assert table.schema.field("size").type == pa.int64()
    assert table.schema.field("last_modified").type == pa.timestamp("us", tz="UTC")
    assert sorted(tuple(r.values()) for r in table.to_pylist()) == rows
    df = dc.list_storage(uri, session=ctc.session).to_pandas()
    assert sorted(df["path"]) == [r.path for r in rows]


def test_listing_to_arrow_location_is_json(test_session, tmp_dir, mocker):
    location = {"vtype": "tar", "offset": 1}
    row = dc.FileRow("file:///d", "a", location=location)
    mocker.patch.object(dc.Listing, "_pages", return_value=iter([[row], []]))

    table = dc.list_storage(tmp_dir, session=test_session).to_arrow()

    assert table.column("location").to_pylist() == [json.dumps(location)]


def test_listing_empty_to_arrow(test_session, tmp_dir):
    (tmp_dir / "empty").mkdir()

    table = dc.list_storage(tmp_dir / "empty", session=test_session).to_arrow()

    assert table.num_rows == 0
    assert table.column_names == list(dc.FileRow._fields)


@pytest.mark.parametrize(
    "use",
    [
        list,
        dc.Listing.to_list,
        dc.Listing.to_objects,
        dc.Listing.to_arrow,
        dc.Listing.to_pandas,
        dc.Listing.save,
    ],
)
def test_listing_is_listed_once(test_session, tmp_dir, use):
    (tmp_dir / "a.txt").write_text("a")
    lst = dc.list_storage(tmp_dir, session=test_session)
    list(lst)

    with pytest.raises(
        RuntimeError, match=r"already listed; call dc\.list_storage\(\)"
    ):
        use(lst)


def test_listing_lists_again_when_called_again(test_session, tmp_dir):
    (tmp_dir / "a.txt").write_text("a")
    assert [r.path for r in dc.list_storage(tmp_dir, session=test_session)] == ["a.txt"]

    (tmp_dir / "b.txt").write_text("bb")
    rows = dc.list_storage(tmp_dir, session=test_session).to_list()

    assert sorted(r.path for r in rows) == ["a.txt", "b.txt"]
    assert not dc.listings(session=test_session).count()


def test_listing_rows_take_file_attributes(test_session, tmp_dir):
    (tmp_dir / "a.txt").write_text("hello")

    (row,) = dc.list_storage(tmp_dir, session=test_session)

    assert row.get_file_ext() == "txt"
    with pytest.raises(AttributeError, match=r"Listing\.to_objects"):
        row.read_text()


def test_listing_to_objects(cloud_test_catalog):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/cats"

    files = dc.list_storage(uri, session=ctc.session).to_objects(type="text")
    files.sort(key=lambda f: f.path)

    assert all(type(f) is dc.TextFile for f in files)
    assert [flatten(f) for f in files] == sorted(
        dc.list_storage(uri, session=ctc.session)
    )
    assert [f.read() for f in files] == ["meow", "mrow"]


def test_listing_to_objects_needs_no_session(tmp_dir, mocker):
    (tmp_dir / "a.txt").write_text("hello")
    mocker.patch.object(Session, "get", side_effect=AssertionError("no session"))

    (file,) = dc.list_storage(tmp_dir).to_objects()

    assert type(file) is File
    assert file.read() == b"hello"


@pytest.mark.parametrize("versions", [False, True])
def test_listing_single_object(cloud_test_catalog, versions):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/description"

    (row,) = dc.list_storage(uri, versions=versions, session=ctc.session)
    (file,) = dc.read_storage(uri, versions=versions, session=ctc.session).to_values(
        "file"
    )

    assert row == flatten(file)
    saved = dc.list_storage(uri, versions=versions, session=ctc.session).save()
    assert saved.to_values("file") == [file]
    assert not dc.listings(session=ctc.session).count()


@pytest.mark.parametrize(
    "pattern",
    [
        "cats/",
        "cats",
        "*",
        "**/*",
        "dogs/*",
        "dogs/**/*",
        "dogs/dog?",
        "**/dog*",
        "{cats,dogs}/*",
        "dogs/dog{1..3}",
        "*/cat1",
        "none/*",
        "/cats/",
        "cats//",
        "dogs/dog1*",
        "dogs/zzz*",
        "d*/dog1",
        "dogs/**/dog?",
        "dogs/others/*",
        "none/zzz*",
    ],
)
@pytest.mark.parametrize("recursive", [True, False])
def test_listing_patterns_match_read_storage(cloud_test_catalog, pattern, recursive):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/{pattern}"

    try:
        expected = sorted(
            flatten(f)
            for f in dc.read_storage(
                uri, recursive=recursive, session=ctc.session
            ).to_values("file")
        )
    except FileNotFoundError:
        with pytest.raises(FileNotFoundError):
            dc.list_storage(uri, recursive=recursive, session=ctc.session).to_list()
        return
    rows = dc.list_storage(uri, recursive=recursive, session=ctc.session)

    assert sorted(r for r in rows if r.is_latest) == expected


@pytest.mark.parametrize(
    "patterns",
    [
        ["cats/", "dogs/"],
        ["cats/", "cats/"],
        ["dogs/*", "dogs/dog1"],
        ["description", "dogs/"],
        ["dogs/dog{1,2}", "cats/"],
        ["dogs/dog{1..3}*"],
        ["dogs/*", "dogs/d*"],
        ["{cats,dogs}/*{1,2}"],
    ],
)
def test_listing_uri_lists_match_read_storage(cloud_test_catalog, patterns):
    ctc = cloud_test_catalog
    uris = [f"{ctc.src_uri}/{p}" for p in patterns]

    expected = sorted(
        flatten(f) for f in dc.read_storage(uris, session=ctc.session).to_values("file")
    )
    rows = dc.list_storage(uris, session=ctc.session)

    assert sorted(r for r in rows if r.is_latest) == expected


def test_listing_save_with_pattern(cloud_test_catalog):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/dogs/dog?"

    saved = dc.list_storage(uri, session=ctc.session).save()

    assert sorted(f.path.rsplit("/", 1)[-1] for f in saved.to_values("file")) == [
        "dog1",
        "dog2",
        "dog3",
    ]


def test_listing_needs_no_session(tmp_dir, mocker):
    (tmp_dir / "a.txt").write_text("a")
    mocker.patch.object(Session, "get", side_effect=AssertionError("no session"))

    assert [r.path for r in dc.list_storage(tmp_dir)] == ["a.txt"]
    assert dc.list_storage(tmp_dir).to_arrow().num_rows == 1


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_listing_save_stores_every_version(cloud_test_catalog_upload):
    ctc = cloud_test_catalog_upload
    uri = f"{ctc.src_uri}/history/"
    File.upload(b"v1", f"{uri}a.txt", ctc.catalog)
    File.upload(b"v2", f"{uri}a.txt", ctc.catalog)

    rows = sorted(dc.list_storage(uri, versions=True, session=ctc.session))
    saved = dc.list_storage(uri, versions=True, session=ctc.session).save()

    assert [r.is_latest for r in rows].count(False) == 1
    assert sorted(flatten(f) for f in _read_listing(ctc.session, uri, True)) == rows
    assert [flatten(f) for f in saved.to_values("file")] == [
        r for r in rows if r.is_latest
    ]


def test_listing_takes_client_config_of_active_session(catalog, tmp_dir):
    config = {"endpoint_url": "https://minio.example.com"}
    configured = catalog.copy()
    configured.client_config = config
    with dc.Session("ingest", catalog=configured):
        assert dc.list_storage(tmp_dir).client_config == config
    own = {"endpoint_url": "https://other.example.com"}
    assert (
        dc.list_storage(tmp_dir, client_config=own).client_config["endpoint_url"]
        == (own["endpoint_url"])
    )


def test_read_storage_reuses_listing_saved_with_defaults(cloud_test_catalog):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/dogs/"

    saved = dc.list_storage(uri, session=ctc.session).save()
    read = dc.read_storage(uri, session=ctc.session)

    assert read.order_by("file.path").to_values("file") == saved.order_by(
        "file.path"
    ).to_values("file")
    assert dc.listings(session=ctc.session).count() == 1


def test_listing_probes_and_keeps_active_session_config(catalog, mocker):
    from datachain.lib.dc import storage

    config = {"endpoint_url": "https://minio.example.com"}
    configured = catalog.copy()
    configured.client_config = config
    probe = mocker.patch.object(storage, "_all_buckets_anonymous", return_value=True)

    with dc.Session("ingest", catalog=configured):
        lst = dc.list_storage("s3://bucket/data/")
        read_session, read_config = storage._storage_session(
            ["s3://bucket/data/"], None, None, None, False, True
        )
        read_session.__exit__(None, None, None)

    assert probe.call_args.args[1] == config
    assert lst.client_config == config | {"anon": True}
    assert read_config == config | {"anon": True}


def test_listing_save_lists_a_shared_listing_once(test_session, tmp_dir, mocker):
    (tmp_dir / "a.txt").write_text("a")
    (tmp_dir / "b.csv").write_text("b")
    from datachain.lib.dc import storage

    write = mocker.spy(storage, "_write_listing")

    saved = dc.list_storage(f"{tmp_dir.as_uri()}/*.{{txt,csv}}", session=test_session)
    chain = saved.save()

    assert write.call_count == 1
    assert sorted(f.path for f in chain.to_values("file")) == ["a.txt", "b.csv"]
    assert dc.listings(session=test_session).count() == 1
    (listing,) = dc.listings(session=test_session).to_values("listing")
    assert listing.version == "1.0.0"


def test_listing_uses_cache_of_active_session(test_session, tmp_dir, mocker):
    from datachain.lib.dc import storage

    (tmp_dir / "a.txt").write_text("a")
    pages = mocker.spy(storage, "list_pages")

    catalog = test_session.catalog.copy()
    with dc.Session("active", catalog=catalog):
        (file,) = dc.list_storage(tmp_dir).to_objects()

    assert pages.call_args.args[1] is catalog.cache
    assert file._catalog.cache is catalog.cache


def _write_tree(tmp_dir):
    for path in ["a.txt", "d/c.txt", "d/e.jpg"]:
        (tmp_dir / path).parent.mkdir(exist_ok=True)
        (tmp_dir / path).write_text(path)


def test_listing_save_with_patterns_keeps_a_newer_parent_listing(test_session, tmp_dir):
    _write_tree(tmp_dir)
    dc.read_storage(f"{tmp_dir.as_uri()}/d/", session=test_session).count()
    dc.read_storage(tmp_dir.as_uri(), session=test_session).count()

    chain = dc.list_storage(f"{tmp_dir.as_uri()}/d/*.{{txt,jpg}}", session=test_session)
    root = dc.read_storage(tmp_dir.as_uri(), session=test_session)

    assert sorted(chain.save().to_values("file.path")) == ["c.txt", "e.jpg"]
    assert sorted(root.to_values("file.path")) == ["a.txt", "d/c.txt", "d/e.jpg"]


def test_expired_parent_listing_is_listed_whole(test_session, tmp_dir, mocker):
    _write_tree(tmp_dir)
    dc.read_storage(tmp_dir.as_uri(), session=test_session).count()
    chain = dc.read_storage(f"{tmp_dir.as_uri()}/d/", session=test_session)

    mocker.patch(
        "datachain.query.dataset.listing_dataset_expired", side_effect=[True, False]
    )
    assert sorted(chain.to_values("file.path")) == ["d/c.txt", "d/e.jpg"]

    root = dc.read_storage(tmp_dir.as_uri(), session=test_session)
    assert sorted(root.to_values("file.path")) == ["a.txt", "d/c.txt", "d/e.jpg"]


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_listing_one_level_reads_nothing_below_it(cloud_test_catalog, mocker):
    from datachain.client.fsspec import RangeListedClient

    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    prefix = f"level-{uuid.uuid4().hex[:8]}"
    top = [f"{prefix}/{i:02}.txt" for i in range(12)]  # pages of 2: past 5 pages
    for key in [*top, f"{prefix}/sub/deep.txt"]:
        client.fs.pipe_file(f"{client.name}/{key}", b"x")
    client.fs.invalidate_cache()
    mocker.patch.object(RangeListedClient, "LIST_PAGE_SIZE", 2)
    pages = mocker.spy(type(client), "_list_page")
    ranges = mocker.spy(RangeListedClient, "_fetch_prefix")
    try:
        rows = dc.list_storage(
            f"{ctc.src_uri}/{prefix}/", recursive=False, session=ctc.session
        )
        paths = sorted(r.path for r in rows)
    finally:
        client.fs.rm(f"{client.name}/{prefix}", recursive=True)
        client.fs.invalidate_cache()

    assert paths == top
    assert not ranges.call_count  # the subfolder is never listed
    assert all(c.kwargs.get("delimiter") == "/" for c in pages.call_args_list)


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
def test_listing_a_missing_bucket_raises(cloud_test_catalog):
    from botocore.exceptions import ClientError

    missing = cloud_test_catalog.src_uri.replace("://", "://missing-")
    lst = dc.list_storage(f"{missing}/a*/**", session=cloud_test_catalog.session)

    # the backend's error (NoSuchBucket on S3), not an empty listing
    with pytest.raises((FileNotFoundError, ClientError)):
        lst.to_list()


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
def test_brace_alternatives_of_one_folder_list_it_once(cloud_test_catalog, mocker):
    from datachain.client.fsspec import RangeListedClient

    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    prefix = f"braces-{uuid.uuid4().hex[:8]}"
    for name in ("a.jpg", "b.png", "c.gif", "d.txt"):
        client.fs.pipe_file(f"{client.name}/{prefix}/{name}", b"x")
    client.fs.invalidate_cache()
    level = mocker.spy(RangeListedClient, "level_pages")
    try:
        rows = dc.list_storage(
            f"{ctc.src_uri}/{prefix}/*.{{jpg,png,gif}}", session=ctc.session
        )
        paths = sorted(r.path for r in rows)
    finally:
        client.fs.rm(f"{client.name}/{prefix}", recursive=True)
        client.fs.invalidate_cache()

    assert paths == [f"{prefix}/a.jpg", f"{prefix}/b.png", f"{prefix}/c.gif"]
    assert level.call_count == 1  # one scan for the three alternatives
