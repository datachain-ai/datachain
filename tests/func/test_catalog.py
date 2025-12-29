import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from fsspec.implementations.local import LocalFileSystem

import datachain as dc
from datachain.cli import garbage_collect
from datachain.error import DatasetNotFoundError
from datachain.lib.listing import parse_listing_uri
from tests.data import ENTRIES
from tests.utils import DEFAULT_TREE, skip_if_not_sqlite, tree_from_path


def listing_stats(uri, catalog):
    list_dataset_name, _, _ = parse_listing_uri(uri)
    dataset = catalog.get_dataset(list_dataset_name)
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

    src_uri_path = src_uri
    if cloud_type == "file":
        src_uri_path = LocalFileSystem._strip_protocol(src_uri)

    assert set(
        catalog.find(
            [src_uri],
            names=["*cat*"],
            columns=["du", "name", "path", "size", "type"],
        )
    ) == {
        "\t".join(columns)
        for columns in [
            ["8", "cats", f"{src_uri_path}/cats/", "0", "d"],
            ["4", "cat1", f"{src_uri_path}/cats/cat1", "4", "f"],
            ["4", "cat2", f"{src_uri_path}/cats/cat2", "4", "f"],
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

    src_uri_path = src_uri
    if cloud_type == "file":
        src_uri_path = LocalFileSystem._strip_protocol(src_uri)
    expected_results = [
        (f"{src_uri_path}/cats/", 8),
        (f"{src_uri_path}/dogs/others/", 4),
        (f"{src_uri_path}/dogs/", 15),
        (f"{src_uri_path}/", 36),
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
    dataset_version1 = test_session.catalog.get_dataset(ds1.name).get_version("1.0.0")
    assert dataset_version1.num_objects == 3
    assert dataset_version1.size == 6

    ds2 = dc.read_values(
        ids=ids,
        file1=[dc.File(path=name, size=size) for name, size in values],
        file2=[dc.File(path=name, size=size * 2) for name, size in values],
        session=test_session,
    ).persist()
    dataset_version2 = test_session.catalog.get_dataset(ds2.name).get_version("1.0.0")
    assert dataset_version2.num_objects == 3
    assert dataset_version2.size == 18


def test_ls_datasets_ordered(test_session):
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


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
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


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
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


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_enlist_source_handles_glob(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs/*.jpg"

    with enlist_source(catalog, src_path):
        pass
    num_objects, size = listing_stats(src_path, catalog)

    assert num_objects == len(DEFAULT_TREE["dogs"])
    assert size == 15


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_enlist_source_handles_file(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    src_path = f"{src_uri}/dogs/dog1"

    with enlist_source(catalog, src_path):
        pass
    with pytest.raises(DatasetNotFoundError):
        listing_stats(src_path, catalog)


@pytest.mark.parametrize("from_cli", [False, True])
def test_garbage_collect(cloud_test_catalog, from_cli, capsys):
    catalog = cloud_test_catalog.catalog
    assert catalog.get_temp_table_names() == []
    temp_tables = [
        "tmp_vc12F",
        "udf_jh653",
    ]
    for t in temp_tables:
        catalog.warehouse.create_udf_table(name=t)
    assert set(catalog.get_temp_table_names()) == set(temp_tables)
    if from_cli:
        garbage_collect(catalog)
        captured = capsys.readouterr()
        assert captured.out == "Garbage collecting 2 tables.\n"
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
