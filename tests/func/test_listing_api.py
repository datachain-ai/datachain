import json

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

    rows = sorted(dc.listing(uri, versions=versions, session=ctc.session))
    chain = dc.listing(uri, versions=versions, session=ctc.session).save()

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

    saved = dc.listing(uri, versions=versions, session=ctc.session).save()
    _, _, _, reused = get_listing(uri, ctc.session, versions=versions)
    read = dc.read_storage(uri, versions=versions, session=ctc.session)

    assert reused
    assert saved.order_by("file.path").to_values("file") == read.order_by(
        "file.path"
    ).to_values("file")
    names = [
        lst.name
        for lst in dc.listing_datasets(session=ctc.session).to_values("listing")
    ]
    assert len(names) == 1


def test_listing_to_arrow(cloud_test_catalog):
    ctc = cloud_test_catalog
    rows = sorted(dc.listing(ctc.src_uri, session=ctc.session))
    table = dc.listing(ctc.src_uri, session=ctc.session).to_arrow().sort_by("path")

    assert table.column_names == list(dc.FileRow._fields)
    assert table.schema.field("size").type == pa.int64()
    assert table.schema.field("last_modified").type == pa.timestamp("us", tz="UTC")
    assert [tuple(r.values()) for r in table.to_pylist()] == rows
    df = dc.listing(ctc.src_uri, session=ctc.session).to_pandas()
    assert sorted(df["path"]) == [r.path for r in rows]


def test_listing_to_arrow_location_is_json(test_session, tmp_dir, mocker):
    location = {"vtype": "tar", "offset": 1}
    row = dc.FileRow("file:///d", "a", location=location)
    mocker.patch.object(dc.Listing, "_pages", return_value=iter([[row], []]))

    table = dc.listing(tmp_dir, session=test_session).to_arrow()

    assert table.column("location").to_pylist() == [json.dumps(location)]


def test_listing_empty_to_arrow(test_session, tmp_dir):
    (tmp_dir / "empty").mkdir()

    table = dc.listing(tmp_dir / "empty", session=test_session).to_arrow()

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
    lst = dc.listing(tmp_dir, session=test_session)
    list(lst)

    with pytest.raises(RuntimeError, match=r"already listed; call dc\.listing\(\)"):
        use(lst)


def test_listing_lists_again_when_called_again(test_session, tmp_dir):
    (tmp_dir / "a.txt").write_text("a")
    assert [r.path for r in dc.listing(tmp_dir, session=test_session)] == ["a.txt"]

    (tmp_dir / "b.txt").write_text("bb")
    rows = dc.listing(tmp_dir, session=test_session).to_list()

    assert sorted(r.path for r in rows) == ["a.txt", "b.txt"]
    assert not dc.listing_datasets(session=test_session).count()


def test_listing_rows_take_file_attributes(test_session, tmp_dir):
    (tmp_dir / "a.txt").write_text("hello")

    (row,) = dc.listing(tmp_dir, session=test_session)

    assert row.get_file_ext() == "txt"
    with pytest.raises(RuntimeError, match="catalog is not set"):
        row.read_text()


def test_listing_to_objects(cloud_test_catalog):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/cats"

    files = dc.listing(uri, session=ctc.session).to_objects(type="text")
    files.sort(key=lambda f: f.path)

    assert all(type(f) is dc.TextFile for f in files)
    assert [flatten(f) for f in files] == sorted(dc.listing(uri, session=ctc.session))
    assert [f.read() for f in files] == ["meow", "mrow"]


def test_listing_to_objects_needs_no_session(tmp_dir, mocker):
    (tmp_dir / "a.txt").write_text("hello")
    mocker.patch.object(Session, "get", side_effect=AssertionError("no session"))

    (file,) = dc.listing(tmp_dir).to_objects()

    assert type(file) is File
    assert file.read() == b"hello"


@pytest.mark.parametrize("versions", [False, True])
def test_listing_single_object(cloud_test_catalog, versions):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/description"

    (row,) = dc.listing(uri, versions=versions, session=ctc.session)
    (file,) = dc.read_storage(uri, versions=versions, session=ctc.session).to_values(
        "file"
    )

    assert row == flatten(file)
    saved = dc.listing(uri, versions=versions, session=ctc.session).save()
    assert saved.to_values("file") == [file]
    assert not dc.listing_datasets(session=ctc.session).count()


@pytest.mark.parametrize("uri", ["s3://b/*.jpg", "s3://b/{a,b}/", "s3://b/x?"])
def test_listing_rejects_globs(test_session, uri):
    with pytest.raises(ValueError, match="glob"):
        dc.listing(uri, session=test_session)


def test_listing_rejects_uri_lists(test_session):
    with pytest.raises(TypeError, match="one URI"):
        dc.listing(["s3://a/", "s3://b/"], session=test_session)  # type: ignore[arg-type]


def test_listings_is_deprecated(test_session, tmp_dir):
    (tmp_dir / "a.txt").write_text("a")
    dc.read_storage(tmp_dir, session=test_session).exec()

    with pytest.warns(DeprecationWarning, match="listing_datasets"):
        old = dc.listings(session=test_session).to_values("listing")

    assert old == dc.listing_datasets(session=test_session).to_values("listing")
    assert len(old) == 1


def test_listing_needs_no_session(tmp_dir, mocker):
    (tmp_dir / "a.txt").write_text("a")
    mocker.patch.object(Session, "get", side_effect=AssertionError("no session"))

    assert [r.path for r in dc.listing(tmp_dir)] == ["a.txt"]
    assert dc.listing(tmp_dir).to_arrow().num_rows == 1


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_listing_save_stores_every_version(cloud_test_catalog_upload):
    ctc = cloud_test_catalog_upload
    uri = f"{ctc.src_uri}/history/"
    File.upload(b"v1", f"{uri}a.txt", ctc.catalog)
    File.upload(b"v2", f"{uri}a.txt", ctc.catalog)

    rows = sorted(dc.listing(uri, versions=True, session=ctc.session))
    saved = dc.listing(uri, versions=True, session=ctc.session).save()

    assert [r.is_latest for r in rows].count(False) == 1
    assert sorted(flatten(f) for f in _read_listing(ctc.session, uri, True)) == rows
    assert [flatten(f) for f in saved.to_values("file")] == [
        r for r in rows if r.is_latest
    ]


def test_listing_takes_client_config_of_active_session(tmp_dir):
    config = {"endpoint_url": "https://minio.example.com"}
    with dc.Session("ingest", client_config=config, in_memory=True):
        assert dc.listing(tmp_dir).client_config == config
    own = {"endpoint_url": "https://other.example.com"}
    assert (
        dc.listing(tmp_dir, client_config=own).client_config["endpoint_url"]
        == (own["endpoint_url"])
    )


def test_read_storage_reuses_listing_saved_with_defaults(cloud_test_catalog):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/dogs/"

    saved = dc.listing(uri, session=ctc.session).save()
    read = dc.read_storage(uri, session=ctc.session)

    assert read.order_by("file.path").to_values("file") == saved.order_by(
        "file.path"
    ).to_values("file")
    assert dc.listing_datasets(session=ctc.session).count() == 1
