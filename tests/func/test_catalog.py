import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

import datachain as dc
import datachain.lib.listing as listing_lib
from datachain.cli import garbage_collect
from datachain.client import Client
from datachain.client.local import FileClient
from datachain.error import DatasetNotFoundError
from datachain.lib.file import File
from datachain.lib.listing import parse_listing_uri
from tests.data import ENTRIES
from tests.utils import DEFAULT_TREE, skip_if_not_sqlite, tree_from_path


def listing_stats(uri, catalog):
    list_dataset_name, _, _ = parse_listing_uri(uri)
    dataset = catalog.get_dataset(list_dataset_name, versions=None)
    dataset_version = dataset.get_version(dataset.latest_version)
    return dataset_version.num_objects, dataset_version.size


@contextmanager
def enlist_source(catalog, *args, **kwargs):
    """Wrapper around catalog.enlist_source that ensures listing is closed."""
    lst, client, path = catalog.enlist_source(*args, **kwargs)
    try:
        yield lst, client, path
    finally:
        if lst:
            lst.close()


@pytest.fixture
def pre_created_ds_name():
    return "pre_created_dataset"


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "gs", "azure"],
    indirect=True,
)
def test_find(cloud_test_catalog, cloud_type):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog
    dirs = ["cats/", "dogs/", "dogs/others/"]
    expected_paths = dirs + [entry.path for entry in ENTRIES]
    assert set(catalog.find([src_uri])) == {
        f"{src_uri}/{path}" for path in expected_paths
    }

    with pytest.raises(FileNotFoundError):
        set(catalog.find([f"{src_uri}/does_not_exist"]))


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "gs", "azure"],
    indirect=True,
)
def test_find_names_paths_size_type(cloud_test_catalog):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    assert set(catalog.find([src_uri], names=["*cat*"])) == {
        f"{src_uri}/cats/",
        f"{src_uri}/cats/cat1",
        f"{src_uri}/cats/cat2",
    }

    assert set(catalog.find([src_uri], names=["*cat*"], typ="dir")) == {
        f"{src_uri}/cats/",
    }

    assert len(list(catalog.find([src_uri], names=["*CAT*"]))) == 0

    assert set(catalog.find([src_uri], inames=["*CAT*"])) == {
        f"{src_uri}/cats/",
        f"{src_uri}/cats/cat1",
        f"{src_uri}/cats/cat2",
    }

    assert set(catalog.find([src_uri], paths=["*cats/cat*"])) == {
        f"{src_uri}/cats/cat1",
        f"{src_uri}/cats/cat2",
    }

    assert len(list(catalog.find([src_uri], paths=["*caTS/CaT**"]))) == 0

    assert set(catalog.find([src_uri], ipaths=["*caTS/CaT*"])) == {
        f"{src_uri}/cats/cat1",
        f"{src_uri}/cats/cat2",
    }

    assert set(catalog.find([src_uri], size="5", typ="f")) == {
        f"{src_uri}/description",
    }

    assert set(catalog.find([src_uri], size="-3", typ="f")) == {
        f"{src_uri}/dogs/dog2",
    }


def test_find_names_columns(cloud_test_catalog, cloud_type):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    assert set(
        catalog.find(
            [src_uri],
            names=["*cat*"],
            columns=["du", "name", "path", "size", "type"],
        )
    ) == {
        "\t".join(columns)
        for columns in [
            ["8", "cats", f"{src_uri}/cats/", "0", "d"],
            ["4", "cat1", f"{src_uri}/cats/cat1", "4", "f"],
            ["4", "cat2", f"{src_uri}/cats/cat2", "4", "f"],
        ]
    }


@pytest.mark.parametrize(
    "recursive,star,dir_exists",
    (
        (True, True, False),
        (True, False, False),
        (True, False, True),
        (False, True, False),
        (False, False, False),
    ),
)
def test_cp_root(cloud_test_catalog, recursive, star, dir_exists, cloud_type):
    src_uri = cloud_test_catalog.src_uri
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = working_dir / "data"

    if star:
        src_path = f"{src_uri}/*"
    else:
        src_path = src_uri
        if cloud_type == "file":
            src_path += "/"

    if star:
        with pytest.raises(FileNotFoundError):
            catalog.cp([src_path], str(dest), recursive=recursive)

    if dir_exists or star:
        dest.mkdir()

    catalog.cp([src_path], str(dest), recursive=recursive)

    if not star and not recursive:
        # The root directory is skipped, so nothing is copied
        expected = {}
    elif recursive:
        expected = DEFAULT_TREE
    else:
        expected = {"description": "Cats and Dogs"}
    assert tree_from_path(dest) == expected


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "gs", "azure"],
    indirect=True,
)
@skip_if_not_sqlite
def test_cp_local_dataset(cloud_test_catalog, dogs_dataset):
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = working_dir / "data"
    dest.mkdir()

    dataset_uri = dogs_dataset.uri(version="1.0.0")

    catalog.cp([dataset_uri], str(dest))

    parsed = urlparse(str(cloud_test_catalog.src))
    netloc = Path(parsed.netloc.strip("/"))
    path = Path(parsed.path.strip("/"))

    assert tree_from_path(dest / netloc / path) == {
        "dogs": {
            "dog1": "woof",
            "dog2": "arf",
            "dog3": "bark",
            "others": {"dog4": "ruff"},
        }
    }


@pytest.mark.parametrize(
    "recursive,star,slash,dir_exists",
    (
        (True, True, False, False),
        (True, False, False, False),
        (True, False, False, True),
        (True, False, True, False),
        (False, True, False, False),
        (False, False, False, False),
        (False, False, True, False),
    ),
)
def test_cp_subdir(cloud_test_catalog, recursive, star, slash, dir_exists):
    if not star and not slash and dir_exists:
        pytest.skip("Fix in https://github.com/datachain-ai/datachain/issues/535")

    src_uri = f"{cloud_test_catalog.src_uri}/dogs"
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = working_dir / "data"

    if star:
        src_path = f"{src_uri}/*"
    elif slash:
        src_path = f"{src_uri}/"
    else:
        src_path = src_uri

    if star:
        with pytest.raises(FileNotFoundError):
            catalog.cp([src_path], str(dest), recursive=recursive)

    if dir_exists or star:
        dest.mkdir()

    catalog.cp([src_path], str(dest), recursive=recursive)

    if not star and not recursive:
        # Directories are skipped, so nothing is copied
        expected = {}
    elif not dir_exists:
        expected = DEFAULT_TREE["dogs"]
        if not recursive:
            expected = expected.copy()
            expected.pop("others")
    else:
        expected = {"dogs": DEFAULT_TREE["dogs"]}
    assert tree_from_path(dest) == expected


@pytest.mark.parametrize(
    "recursive,star,slash",
    (
        (True, True, False),
        (True, False, False),
        (True, False, True),
        (False, True, False),
        (False, False, False),
        (False, False, True),
    ),
)
def test_cp_multi_subdir(cloud_test_catalog, recursive, star, slash, cloud_type):
    if recursive and not star and not slash:
        pytest.skip("Fix in https://github.com/datachain-ai/datachain/issues/535")

    if cloud_type == "file" and recursive and not star and slash:
        pytest.skip("Fix in https://github.com/datachain-ai/datachain/issues/535")

    sources = [
        f"{cloud_test_catalog.src_uri}/cats",
        f"{cloud_test_catalog.src_uri}/dogs",
    ]
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = working_dir / "data"

    if star:
        src_paths = [f"{src}/*" for src in sources]
    elif slash:
        src_paths = [f"{src}/" for src in sources]
    else:
        src_paths = sources

    with pytest.raises(FileNotFoundError):
        catalog.cp(src_paths, str(dest), recursive=recursive)

    dest.mkdir()

    catalog.cp(src_paths, str(dest), recursive=recursive)

    if not star and not recursive:
        # Directories are skipped, so nothing is copied
        expected = {}
    elif star or slash:
        expected = DEFAULT_TREE["dogs"] | DEFAULT_TREE["cats"]
        if not recursive:
            expected = {k: v for k, v in expected.items() if not isinstance(v, dict)}
    else:
        expected = DEFAULT_TREE
    assert tree_from_path(dest) == expected


def test_cp_double_subdir(cloud_test_catalog):
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs/others"
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog
    dest = working_dir / "data"

    catalog.cp([src_path], str(dest), recursive=True)

    assert tree_from_path(dest) == {"dog4": "ruff"}


@pytest.mark.parametrize(
    "src,expected", [("", {"d": {"e.txt": "real"}}), ("d/", {"e.txt": "real"})]
)
@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_cp_skips_keys_with_double_slash(cloud_test_catalog, caplog, src, expected):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    prefix = "cp-double-slash"
    client.fs.pipe_file(f"{client.name}/{prefix}/d/e.txt", b"real")
    client.fs.pipe_file(f"{client.name}/{prefix}/d//e.txt", b"odd")
    client.fs.invalidate_cache()
    dest = ctc.working_dir / "data"
    try:
        with caplog.at_level("WARNING", logger="datachain"):
            ctc.catalog.cp([f"{ctc.src_uri}/{prefix}/{src}"], str(dest), recursive=True)
        files = {
            f.path: f
            for f in dc.read_storage(
                f"{ctc.src_uri}/{prefix}/{src}", session=ctc.session
            ).to_values("file")
        }
    finally:
        client.fs.rm(f"{client.name}/{prefix}", recursive=True)
        client.fs.invalidate_cache()

    assert tree_from_path(dest) == expected
    assert files[f"{prefix}/d//e.txt"].get_local_path() is None  # not downloaded
    assert f"File '{prefix}/d//e.txt' not copied: path has '//'" in caplog.text


@pytest.mark.parametrize("cloud_type,tree", [("s3", {})], indirect=True)
def test_cp_bucket_root_skips_empty_segments_in_existing_directory(cloud_test_catalog):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    client.fs.pipe_file(f"{client.name}/d/e.txt", b"real")
    client.fs.pipe_file(f"{client.name}/d//e.txt", b"odd")
    client.fs.invalidate_cache()
    dest = ctc.working_dir / "out"
    dest.mkdir()
    ctc.catalog.cp([ctc.src_uri], str(dest), recursive=True)
    assert tree_from_path(dest) == {"d": {"e.txt": "real"}}


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_cp_double_slash_key_to_explicit_filename(cloud_test_catalog, force):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    key = "cp-explicit/d//e.txt"
    client.fs.pipe_file(f"{client.name}/{key}", b"content")
    client.fs.invalidate_cache()
    dest = ctc.working_dir / "out.txt"
    if force:
        dest.write_text("existing content")
    try:
        ctc.catalog.cp([f"{ctc.src_uri}/{key}"], str(dest), force=force)
        assert dest.read_bytes() == b"content"
    finally:
        client.fs.rm(f"{client.name}/{key}")
        client.fs.invalidate_cache()


@pytest.mark.parametrize("no_glob", (True, False))
def test_cp_single_file(cloud_test_catalog, no_glob):
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog
    dest = working_dir / "data"
    src_path = f"{cloud_test_catalog.src_uri}/dogs/dog1"
    dest.mkdir()

    catalog.cp([src_path], str(dest / "local_dog"), no_glob=no_glob)

    assert tree_from_path(dest) == {"local_dog": "woof"}


@pytest.mark.parametrize("tree", [{"bar-file": "original"}], indirect=True)
def test_cp_file_storage_mutation(cloud_test_catalog):
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog
    src_path = f"{cloud_test_catalog.src_uri}/bar-file"

    dest = working_dir / "data1"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"))
    assert tree_from_path(dest) == {"local": "original"}

    (cloud_test_catalog.src / "bar-file").write_text("modified")
    dest = working_dir / "data2"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"))
    assert tree_from_path(dest) == {"local": "modified"}

    # For a file we access it directly, we don't take the entry from listing
    # so we don't check the previous etag with the new modified one
    catalog.cache.clear()
    dest = working_dir / "data3"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"))
    assert tree_from_path(dest) == {"local": "modified"}

    catalog.index([src_path], update=True)
    dest = working_dir / "data4"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"))
    assert tree_from_path(dest) == {"local": "modified"}


@pytest.mark.parametrize("tree", [{"foo-file": "original"}], indirect=True)
def test_cp_dir_storage_mutation(cloud_test_catalog, version_aware):
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog
    src_path = f"{cloud_test_catalog.src_uri}/"

    dest = working_dir / "data1"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"), recursive=True)
    assert tree_from_path(dest) == {"local": {"foo-file": "original"}}

    (cloud_test_catalog.src / "foo-file").write_text("modified")
    dest = working_dir / "data2"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"), recursive=True)
    assert tree_from_path(dest) == {"local": {"foo-file": "original"}}

    # For a dir we access files through listing
    # so it finds a etag for the origin file, but it's now not in cache + it
    # is modified on the local storage, so we can't find the file referenced
    # by the listing anymore if FS is not version aware, or we can find
    # the original version and download if FS support versioning
    catalog.cache.clear()
    dest = working_dir / "data3"
    dest.mkdir()
    if version_aware:
        catalog.cp([src_path], str(dest / "local"), recursive=True)
        assert tree_from_path(dest) == {"local": {"foo-file": "original"}}
    else:
        with pytest.raises(FileNotFoundError):
            catalog.cp([src_path], str(dest / "local"), recursive=True)
            assert tree_from_path(dest) == {"local": {}}

    catalog.index([src_path], update=True)
    dest = working_dir / "data4"
    dest.mkdir()
    catalog.cp([src_path], str(dest / "local"), recursive=True)
    assert tree_from_path(dest) == {"local": {"foo-file": "modified"}}


@pytest.mark.parametrize("cloud_type, version_aware", [("file", False)], indirect=True)
def test_cp_symlinks(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    catalog.client_config["use_symlinks"] = True
    src_uri = cloud_test_catalog.src_uri
    work_dir = cloud_test_catalog.working_dir
    dest = work_dir / "data"
    dest.mkdir()
    catalog.cp([f"{src_uri}/dogs/"], str(dest), recursive=True)

    assert (dest / "dog1").is_symlink()
    assert os.path.realpath(dest / "dog1") == str(
        cloud_test_catalog.src / "dogs" / "dog1"
    )
    assert (dest / "dog1").read_text() == "woof"
    assert (dest / "others" / "dog4").is_symlink()
    assert os.path.realpath(dest / "others" / "dog4") == str(
        cloud_test_catalog.src / "dogs" / "others" / "dog4"
    )
    assert (dest / "others" / "dog4").read_text() == "ruff"


def test_du(cloud_test_catalog, cloud_type):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    expected_results = [
        (f"{src_uri}/cats/", 8),
        (f"{src_uri}/dogs/others/", 4),
        (f"{src_uri}/dogs/", 15),
        (f"{src_uri}/", 36),
    ]

    results = catalog.du([src_uri])
    assert set(results) == set(expected_results[3:])

    results = catalog.du([src_uri], depth=1)
    assert set(results) == set(expected_results[:1] + expected_results[2:])

    results = catalog.du([src_uri], depth=5)
    assert set(results) == set(expected_results)


def test_ls_glob(cloud_test_catalog):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    assert sorted(
        (source.node.name, [r[0] for r in results])
        for source, results in catalog.ls([f"{src_uri}/dogs/dog*"], fields=["name"])
    ) == [("dog1", ["dog1"]), ("dog2", ["dog2"]), ("dog3", ["dog3"])]


def test_ls_file(cloud_test_catalog):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    assert sorted(
        (source.node.name, [r[0] for r in results])
        for source, results in catalog.ls([f"{src_uri}/dogs/dog1"], fields=["name"])
    ) == [("dog1", ["dog1"])]


def test_ls_dir_same_name_as_file(cloud_test_catalog, cloud_type):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    path = f"{src_uri}/dogs/dog1"

    # check that file exists
    assert sorted(
        (source.node.name, [r[0] for r in results])
        for source, results in catalog.ls([path], fields=["name"])
    ) == [("dog1", ["dog1"])]

    if cloud_type == "file":
        # should be fixed upstream in fsspec
        # boils down to https://github.com/fsspec/filesystem_spec/pull/1567#issuecomment-2563160414
        # fsspec removes the trailing slash and returns a file, that's why we are
        # are not getting an error here
        assert sorted(
            (source.node.name, [r[0] for r in results])
            for source, results in catalog.ls([f"{path}/"], fields=["name"])
        ) == [("", ["."])]
    else:
        with pytest.raises(FileNotFoundError):
            next(catalog.ls([f"{path}/"], fields=["name"]))


def test_ls_prefix_not_found(cloud_test_catalog):
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog
    with pytest.raises(FileNotFoundError):
        list(catalog.ls([f"{src_uri}/bogus/"], fields=["name"]))


def test_index_error(cloud_test_catalog):
    protocol = cloud_test_catalog.src_uri.split("://", 1)[0]
    # XXX: different clients raise inconsistent exceptions
    with pytest.raises(Exception):  # noqa: B017
        cloud_test_catalog.catalog.index([f"{protocol}://does_not_exist"])


def test_dataset_stats(test_session):
    ids = [1, 2, 3]
    values = tuple(zip(["a", "b", "c"], [1, 2, 3], strict=False))

    ds1 = dc.read_values(
        ids=ids,
        file=[dc.File(path=name, size=size) for name, size in values],
        session=test_session,
    ).persist()
    dataset_version1 = test_session.catalog.get_dataset(
        ds1.name, versions=["1.0.0"]
    ).get_version("1.0.0")
    assert dataset_version1.num_objects == 3
    assert dataset_version1.size == 6

    ds2 = dc.read_values(
        ids=ids,
        file1=[dc.File(path=name, size=size) for name, size in values],
        file2=[dc.File(path=name, size=size * 2) for name, size in values],
        session=test_session,
    ).persist()
    dataset_version2 = test_session.catalog.get_dataset(
        ds2.name, versions=["1.0.0"]
    ).get_version("1.0.0")
    assert dataset_version2.num_objects == 3
    assert dataset_version2.size == 18


def test_ls_datasets_ordered(test_session, ignore_checkpoints):
    ids = [1, 2, 3]
    values = tuple(zip(["a", "b", "c"], ids, strict=False))

    assert not list(test_session.catalog.ls_datasets())

    chain = dc.read_values(
        ids=ids,
        file=[dc.File(path=name, size=size) for name, size in values],
        session=test_session,
    )
    chain.save("cats")
    chain.save("dogs")
    chain.save("cats")
    chain.save("cats")
    chain.save("cats")
    datasets = list(test_session.catalog.ls_datasets())

    # Session datasets should be automatically filtered out by ls_datasets()
    assert [(d.name, v.version) for d in datasets for v in d.versions] == [
        ("cats", "1.0.0"),
        ("cats", "1.0.1"),
        ("cats", "1.0.2"),
        ("cats", "1.0.3"),
        ("dogs", "1.0.0"),
    ]


def test_ls_datasets_no_json(test_session):
    ids = [1, 2, 3]
    values = tuple(zip(["a", "b", "c"], [1, 2, 3], strict=False))

    dc.read_values(
        ids=ids,
        file=[dc.File(path=name, size=size) for name, size in values],
        session=test_session,
    ).persist()
    datasets = test_session.catalog.ls_datasets()
    assert datasets
    for d in datasets:
        assert hasattr(d, "id")
        assert not hasattr(d, "feature_schema")
        assert d.versions
        for v in d.versions:
            assert hasattr(v, "id")
            assert not hasattr(v, "preview")
            assert not hasattr(v, "feature_schema")


def test_listing_stats(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    with pytest.raises(DatasetNotFoundError):
        listing_stats(src_uri, catalog)

    with enlist_source(catalog, src_uri):
        pass
    num_objects, size = listing_stats(src_uri, catalog)
    assert num_objects == 7
    assert size == 36

    with enlist_source(catalog, f"{src_uri}/dogs/", update=True):
        pass
    num_objects, size = listing_stats(src_uri, catalog)
    assert num_objects == 7
    assert size == 36

    num_objects, size = listing_stats(f"{src_uri}/dogs/", catalog)
    assert num_objects == 4
    assert size == 15

    with enlist_source(catalog, f"{src_uri}/dogs/"):
        pass
    num_objects, size = listing_stats(src_uri, catalog)
    assert num_objects == 7
    assert size == 36


def test_enlist_source_handles_slash(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs"

    with enlist_source(catalog, src_path):
        pass
    num_objects, size = listing_stats(src_path, catalog)
    assert num_objects == len(DEFAULT_TREE["dogs"])
    assert size == 15

    src_path = f"{src_uri}/dogs"
    with enlist_source(catalog, src_path, update=True):
        pass
    num_objects, size = listing_stats(src_path, catalog)
    assert num_objects == len(DEFAULT_TREE["dogs"])
    assert size == 15


def test_enlist_source_handles_glob(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs/*.jpg"

    with enlist_source(catalog, src_path):
        pass
    num_objects, size = listing_stats(src_path, catalog)

    assert num_objects == len(DEFAULT_TREE["dogs"])
    assert size == 15


def test_enlist_source_handles_file(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs/dog1"

    with enlist_source(catalog, src_path):
        pass
    with pytest.raises(DatasetNotFoundError):
        listing_stats(src_path, catalog)


@pytest.mark.parametrize("from_cli", [False, True])
def test_garbage_collect_temp_tables(cloud_test_catalog, from_cli, capsys):
    catalog = cloud_test_catalog.catalog
    assert catalog.get_temp_table_names() == []
    temp_tables = [
        "tmp_vc12F",
        "tmp_jh653",
    ]
    for t in temp_tables:
        catalog.warehouse.create_udf_table(name=t)
    assert set(catalog.get_temp_table_names()) == set(temp_tables)
    if from_cli:
        garbage_collect(catalog)
        captured = capsys.readouterr()
        assert "Removed 2 temporary tables." in captured.out
    else:
        catalog.cleanup_tables(temp_tables)
    assert catalog.get_temp_table_names() == []


@pytest.fixture
def gcs_fake_credentials(monkeypatch):
    # For signed URL tests to work we need to setup some fake credentials
    # that looks like real ones
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        os.path.dirname(__file__) + "/fake-service-account-credentials.json",
    )


@pytest.mark.parametrize("is_studio", (False,))
@pytest.mark.parametrize("tree", [{"test-signed-file": "original"}], indirect=True)
@pytest.mark.parametrize(
    "cloud_type, version_aware",
    (["s3", False], ["azure", False], ["gs", False]),
    indirect=True,
)
def test_signed_url(cloud_test_catalog, gcs_fake_credentials):
    signed_url = cloud_test_catalog.catalog.signed_url(
        cloud_test_catalog.src_uri, "test-signed-file"
    )
    content = requests.get(signed_url, timeout=10).text
    assert content == "original"


@pytest.mark.parametrize("is_studio", (False,))
@pytest.mark.parametrize(
    "tree", [{"test-signed-file-versioned": "original"}], indirect=True
)
@pytest.mark.parametrize(
    "cloud_type, version_aware",
    (["s3", True], ["azure", True], ["gs", True]),
    indirect=True,
)
def test_signed_url_versioned(cloud_test_catalog, gcs_fake_credentials):
    file_name = "test-signed-file-versioned"
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog
    client = catalog.get_client(src_uri)

    original_version = client.get_file_info(file_name).version

    (cloud_test_catalog.src / file_name).write_text("modified")

    modified_version = client.get_file_info(file_name).version

    for version, expected in [
        (original_version, "original"),
        (modified_version, "modified"),
    ]:
        signed_url = catalog.signed_url(
            src_uri,
            file_name,
            version_id=version,
        )

        content = requests.get(signed_url, timeout=10).text
        assert content == expected


@pytest.mark.parametrize(
    "cloud_type, version_aware",
    (["s3", False], ["azure", False], ["gs", False]),
    indirect=True,
)
def test_signed_url_with_content_disposition(
    cloud_test_catalog, cloud_type, gcs_fake_credentials
):
    import urllib.parse

    content_disposition = "attachment; filename=test-signed-file"
    quoted_content_disposition = urllib.parse.quote(content_disposition)

    signed_url = cloud_test_catalog.catalog.signed_url(
        cloud_test_catalog.src_uri,
        "test-signed-file",
        content_disposition=content_disposition,
    )
    param = "rscd" if cloud_type == "azure" else "response-content-disposition"
    expected_value = (
        quoted_content_disposition.replace("%20", "+")
        if cloud_type == "gs"
        else quoted_content_disposition
    )
    assert f"{param}={expected_value}" in signed_url


@pytest.mark.parametrize(
    "cloud_type, version_aware",
    (["s3", False], ["azure", False], ["gs", False]),
    indirect=True,
)
def test_signed_url_with_anon_client(cloud_test_catalog, cloud_type):
    catalog = cloud_test_catalog.catalog
    catalog.client_config["anon"] = True
    signed_url = catalog.signed_url(
        cloud_test_catalog.src_uri,
        "test-signed-file",
        content_disposition="attachment; filename=test-signed-file",
    )
    param = "rscd" if cloud_type == "azure" else "response-content-disposition"
    assert param not in signed_url


def test_enlist_source_selects_snapshot_once(cloud_test_catalog, mocker):
    ctc = cloud_test_catalog
    select = mocker.spy(listing_lib, "describe_listing")
    with enlist_source(ctc.catalog, f"{ctc.src_uri}/dogs/") as (listing, _, _):
        assert listing is not None
        assert listing_stats(f"{ctc.src_uri}/dogs/", ctc.catalog) == (4, 15)
    assert select.call_count == 1


@pytest.mark.parametrize("patterns", [("", ""), ("dog?", "dog*")])
def test_enlist_sources_updates_shared_snapshot_once(
    cloud_test_catalog, mocker, patterns
):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/dogs/"
    write = mocker.spy(listing_lib, "write_listing")
    with ctc.catalog.enlist_sources(
        [uri + pattern for pattern in patterns], update=True
    ) as sources:
        assert sources is not None
        half = len(sources) // 2
        assert half == (1 if patterns == ("", "") else 3)
        assert [s.node for s in sources[:half]] == [s.node for s in sources[half:]]
    assert write.call_count == 1


def test_enlist_source_client_config_override(cloud_test_catalog, mocker):
    ctc = cloud_test_catalog
    config = ctc.client_config.copy()
    original = {"endpoint_url": "http://127.0.0.1:1", "anon": True}
    ctc.catalog.client_config = original.copy()
    clients = mocker.spy(Client, "get_client")
    with enlist_source(ctc.catalog, f"{ctc.src_uri}/dogs/", client_config=config) as (
        listing,
        client,
        _,
    ):
        assert listing is not None
        assert listing_stats(f"{ctc.src_uri}/dogs/", ctc.catalog) == (4, 15)
        assert client is listing.client
    assert clients.call_args_list
    assert all(
        c.kwargs.get("endpoint_url") != original["endpoint_url"]
        for c in clients.call_args_list
    )
    assert ctc.catalog.client_config == original


@pytest.mark.parametrize("update", [False, True])
def test_enlist_source_with_glob_in_parent(cloud_test_catalog, update):
    ctc = cloud_test_catalog
    with enlist_source(ctc.catalog, f"{ctc.src_uri}/dogs/*/dog4", update=update) as (
        listing,
        _,
        path,
    ):
        assert listing is not None
        assert path == ("*/dog4" if ctc.server.kind == "file" else "dogs/*/dog4")
        rows = [
            row
            for _, result in ctc.catalog.ls(
                [f"{ctc.src_uri}/dogs/*/dog4"], fields=["path"]
            )
            for row in result
        ]
        expected = "others/dog4" if ctc.server.kind == "file" else "dogs/others/dog4"
        assert rows == [(expected,)]
        assert listing_stats(f"{ctc.src_uri}/dogs/", ctc.catalog) == (4, 15)


@pytest.mark.parametrize("cloud_type", ["file", "s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
@pytest.mark.parametrize("pattern", ["*/dog4", "oth*/dog4", "*/dog4/"])
@pytest.mark.parametrize("operation", ["ls", "cp"])
def test_catalog_literal_file_after_glob(
    cloud_test_catalog, tmp_path, pattern, operation
):
    ctc = cloud_test_catalog
    source = f"{ctc.src_uri}/dogs/{pattern}"
    if operation == "ls":
        rows = [
            row
            for _, result in ctc.catalog.ls([source], fields=["name"])
            for row in result
        ]
        assert rows == ([] if pattern.endswith("/") else [("dog4",)])
    else:
        output = tmp_path / "downloaded"
        if pattern.endswith("/"):
            with pytest.raises(FileNotFoundError):
                ctc.catalog.cp([source], str(output))
        else:
            ctc.catalog.cp([source], str(output))
            assert output.read_text() == DEFAULT_TREE["dogs"]["others"]["dog4"]


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
def test_enlist_source_keeps_repeated_slashes(cloud_test_catalog):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    client.fs.pipe_file(f"{client.name}/dogs//x/dog4", b"9")
    uri = f"{ctc.src_uri}/dogs//*/dog4"
    rows = [
        row for _, result in ctc.catalog.ls([uri], fields=["path"]) for row in result
    ]
    assert rows == [("dogs//x/dog4",)]


def test_grouped_sources_update_shared_snapshot_once(cloud_test_catalog, mocker):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/dogs/"
    write = mocker.spy(listing_lib, "write_listing")
    groups = ctc.catalog.enlist_sources_grouped([uri, uri], update=True)
    try:
        assert len(groups) == 2
        assert write.call_count == 1
    finally:
        for group in groups:
            group.close()


@pytest.mark.parametrize("version_aware", [False], indirect=True)
@pytest.mark.parametrize(
    "tree", [{"dir": {"a{b": "one", "sub{1": {"f": "two"}}}], indirect=True
)
@pytest.mark.parametrize("name,content", [("a{b", "one"), ("sub{1/f", "two")])
@pytest.mark.parametrize("operation", ["ls", "cp"])
def test_catalog_literal_brace_object(
    cloud_test_catalog, tmp_path, name, content, operation
):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/dir/{name}"
    if operation == "ls":
        rows = [list(rows) for _, rows in ctc.catalog.ls([uri], fields=["name"])]
        assert rows == [[(name.rsplit("/", 1)[-1],)]]
    else:
        output = tmp_path / "downloaded"
        ctc.catalog.cp([uri], str(output))
        assert output.read_text() == content


def test_catalog_update_with_glob_directory(cloud_test_catalog):
    ctc = cloud_test_catalog
    groups = [
        (source.node.name, list(rows))
        for source, rows in ctc.catalog.ls(
            [f"{ctc.src_uri}/dogs/*/"], fields=["name"], update=True
        )
    ]
    assert groups == [("others", [("dog4",)])]


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
@pytest.mark.parametrize("config", [None, {}], ids=["inherit", "empty"])
@pytest.mark.parametrize("operation", ["ls", "du", "find", "index", "cp", "clone"])
def test_catalog_client_config_empty_or_inherited(
    cloud_test_catalog, tmp_path, mocker, operation, config
):
    ctc = cloud_test_catalog
    original = {"use_symlinks": True}
    ctc.catalog.client_config = original.copy()
    clients = mocker.spy(Client, "get_client")
    other = ctc.catalog.copy()
    other.client_config = {}
    with dc.Session("other", catalog=other):
        uri = f"{ctc.src_uri}/dogs/"
        if operation == "ls":
            for _, rows in ctc.catalog.ls([uri], fields=["name"], client_config=config):
                assert list(rows)
        elif operation == "index":
            ctc.catalog.index([uri], client_config=config)
        elif operation == "cp":
            output = tmp_path / "downloaded"
            ctc.catalog.cp([f"{uri}dog1"], str(output), client_config=config)
            assert output.read_text() == DEFAULT_TREE["dogs"]["dog1"]
            assert output.is_symlink() == (config is None)
        elif operation == "clone":
            ctc.catalog.clone(
                [uri], "config_clone", no_cp=True, recursive=True, client_config=config
            )
        else:
            assert list(getattr(ctc.catalog, operation)([uri], client_config=config))
    assert clients.call_args_list
    assert all(
        bool(call.kwargs.get("use_symlinks")) == (config is None)
        for call in clients.call_args_list
    )
    assert ctc.catalog.client_config == original
    assert other.client_config == {}


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
@pytest.mark.parametrize("form", ["uri", "absolute", "relative"])
def test_enlist_source_normalizes_local_glob_path(
    cloud_test_catalog, monkeypatch, form
):
    ctc = cloud_test_catalog
    if form == "uri":
        root = ctc.src_uri
    elif form == "absolute":
        root = os.fspath(ctc.src)
    else:
        monkeypatch.chdir(ctc.src.parent)
        root = ctc.src.name
    with enlist_source(ctc.catalog, f"{root}/d*//others/") as (listing, _, path):
        assert listing is not None
        assert path == "d*/others/"


@pytest.mark.skipif(os.name == "nt", reason="POSIX filesystem root")
@pytest.mark.parametrize("source", ["/", "/core*", "file:///", "file:///core*"])
def test_catalog_ls_filesystem_root(catalog, mocker, source):
    scanned = []

    async def scandir(client, start_prefix, method="default"):
        scanned.append((Path(client.name), start_prefix))
        yield [
            File(source=client.uri, path="core-probe", size=1, etag="one"),
            File(source=client.uri, path="other-probe", size=1, etag="two"),
        ]

    mocker.patch.object(FileClient, "scandir", scandir)
    rows = [
        row for _, result in catalog.ls([source], fields=["name"]) for row in result
    ]

    expected = [("core-probe",)]
    if not source.endswith("*"):
        expected.append(("other-probe",))
    assert sorted(rows) == expected
    assert scanned == [(Path("/"), "")]


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
@pytest.mark.parametrize("config", [None, {}], ids=["inherit", "empty"])
@skip_if_not_sqlite
def test_catalog_dataset_client_config_empty_or_inherited(
    cloud_test_catalog, dogs_dataset, mocker, config, tmp_path
):
    ctc = cloud_test_catalog
    original = {"use_symlinks": True}
    ctc.catalog.client_config = original.copy()
    listings = mocker.patch(
        "datachain.catalog.catalog.get_listing", wraps=listing_lib.get_listing
    )
    other = ctc.catalog.copy()
    other.client_config = {}
    with dc.Session("other", catalog=other):
        uri = dogs_dataset.uri(version="1.0.0")
        groups = ctc.catalog.enlist_sources_grouped(
            [uri], update=False, client_config=config
        )
        try:
            assert len(groups) == 1
            assert groups[0].client.use_symlinks == (config is None)
            assert groups[0].client.uri == ctc.src_uri
            assert sorted(source.node.path for source in groups[0].sources) == [
                "dogs/dog1",
                "dogs/dog2",
                "dogs/dog3",
                "dogs/others/dog4",
            ]
        finally:
            for group in groups:
                group.close()
    assert listings.call_args_list
    assert all(
        call.args[1].catalog.client_config == (original if config is None else {})
        for call in listings.call_args_list
    )
    assert ctc.catalog.client_config == original
    assert other.client_config == {}

    output = tmp_path / "copied"
    ctc.catalog.cp([uri], str(output), recursive=True, client_config=config)
    source_path = Path(ctc.src)
    source_prefix = source_path.relative_to(source_path.anchor)
    for relative in ["dogs/dog1", "dogs/dog2", "dogs/dog3", "dogs/others/dog4"]:
        copied = output / source_prefix / relative
        assert copied.is_relative_to(output)
        assert copied.read_text() == (ctc.src / relative).read_text()
        assert copied.is_symlink() == (config is None)


@pytest.mark.skipif(os.name != "nt", reason="drive letters exist only on Windows")
def test_catalog_ls_rejects_file_uri_without_drive(catalog):
    with pytest.raises(ValueError, match="drive letter"):
        list(catalog.ls(["file:///data/dogs/"], fields=["name"]))


@skip_if_not_sqlite
def test_cp_dataset_from_filesystem_root(test_session, tmp_path):
    catalog = test_session.catalog
    source = tmp_path / "data.txt"
    source.write_text("root file contents")
    root = Path(source.anchor)
    client = FileClient.from_source(str(root), catalog.cache)
    path = source.relative_to(root).as_posix()
    file = client.get_file_info(path)
    name, _, _, _ = listing_lib.get_listing(client.uri, test_session)
    dc.read_values(file=[file], session=test_session).settings(
        namespace=catalog.metastore.system_namespace_name,
        project=catalog.metastore.listing_project_name,
    ).save(name, listing=True)
    dataset = dc.read_values(file=[file], session=test_session).save("rootds")
    output = tmp_path / "copied"
    catalog.cp([dataset.dataset.uri(version="1.0.0")], str(output), recursive=True)
    assert (output / path).read_text() == "root file contents"


@pytest.fixture
def literal_brace_folder(cloud_test_catalog, monkeypatch):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}/a{{b/"
    client_cls = Client.get_implementation(uri)
    expected_name, expected_path = client_cls.split_url(uri)
    scandir = client_cls.scandir

    async def scan_literal_folder(client, start_prefix, method="default"):
        # Returning x.txt alone would also pass after scanning the whole parent.
        assert client.name == expected_name
        assert start_prefix == expected_path.rstrip("/")
        async for page in scandir(client, start_prefix, method=method):
            yield page

    monkeypatch.setattr(client_cls, "scandir", scan_literal_folder)
    return ctc, uri


@pytest.mark.parametrize("cloud_type", ["file", "s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
@pytest.mark.parametrize(
    "tree", [{"a{b": {"x.txt": "x"}, "sibling": {"x.txt": "s"}}], indirect=True
)
def test_catalog_glob_lists_only_literal_brace_folder(literal_brace_folder):
    ctc, uri = literal_brace_folder
    rows = [
        row
        for _, result in ctc.catalog.ls([uri + "*.txt"], fields=["name"])
        for row in result
    ]
    assert rows == [("x.txt",)]


@pytest.mark.parametrize("cloud_type", ["file", "s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
@pytest.mark.parametrize(
    "tree", [{"a{b": {"x.txt": "x"}, "sibling": {"x.txt": "s"}}], indirect=True
)
def test_read_glob_lists_only_literal_brace_folder(literal_brace_folder):
    ctc, uri = literal_brace_folder
    (file,) = dc.read_storage(uri + "*.txt", session=ctc.session).to_values("file")
    assert file.name == "x.txt"
    assert file.read_text() == "x"


@pytest.mark.parametrize(
    "config", [None, {}, {"anon": False}], ids=["inherit", "empty", "replace"]
)
def test_signed_url_client_config_empty_or_inherited(catalog, mocker, config):
    catalog.client_config = {"anon": True}
    client = mocker.Mock()
    factory = mocker.patch.object(FileClient, "from_source", return_value=client)
    catalog.signed_url(
        "file:///source",
        "a.txt",
        client_config=config,
        content_disposition="attachment",
    )
    assert factory.call_args.kwargs == ({"anon": True} if config is None else config)
    assert client.url.call_args.kwargs["content_disposition"] == (
        None if config is None else "attachment"
    )
    assert catalog.client_config == {"anon": True}
