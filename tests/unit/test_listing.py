import posixpath

import pytest

import datachain as dc
from datachain.catalog.catalog import DataSource
from datachain.client import Client
from datachain.lib.file import File
from datachain.lib.listing import (
    LISTING_PREFIX,
    get_listing,
    is_listing_dataset,
    listing_uri_from_name,
)
from datachain.listing import Listing
from datachain.node import DirType
from tests.utils import skip_if_not_sqlite

TREE = {
    "dir1": {
        "d2": {None: ["file1.csv", "file2.csv"]},
        None: ["dataset.csv"],
    },
    "dir2": {None: ["diagram.png"]},
    None: ["users.csv"],
}


def _tree_to_entries(tree: dict, path=""):
    for k, v in tree.items():
        if k:
            dir_path = posixpath.join(path, k)
            yield from _tree_to_entries(v, dir_path)
        else:
            for fname in v:
                yield File(path=posixpath.join(path, fname))


@pytest.fixture
def listing(test_session):
    catalog = test_session.catalog
    listing_namespace_name = catalog.metastore.system_namespace_name
    listing_project_name = catalog.metastore.listing_project_name
    dataset_name, _, _, _ = get_listing("file:///whatever", test_session)
    (
        dc.read_values(file=list(_tree_to_entries(TREE)))
        .settings(namespace=listing_namespace_name, project=listing_project_name)
        .save(dataset_name, listing=True)
    )

    with Listing(
        catalog.metastore.clone(),
        catalog.warehouse.clone(),
        Client.get_client("file:///whatever", catalog.cache, **catalog.client_config),
        dataset_name=dataset_name,
        column="file",
    ) as listing_obj:
        yield listing_obj


def test_get_listing_returns_exact_math_on_update(test_session):
    # Context: https://github.com/datachain-ai/datachain/pull/726
    # On update it should be returning the exact match (not a "bigger" one)
    catalog = test_session.catalog
    listing_namespace_name = catalog.metastore.system_namespace_name
    listing_project_name = catalog.metastore.listing_project_name

    dataset_name_dir1, _, _, exists = get_listing("file:///whatever/dir1", test_session)
    (
        dc.read_values(file=list(_tree_to_entries(TREE["dir1"])))
        .settings(namespace=listing_namespace_name, project=listing_project_name)
        .save(dataset_name_dir1, listing=True)
    )
    assert dataset_name_dir1 == f"{LISTING_PREFIX}file:///whatever/dir1/"
    assert not exists

    dataset_name, _, _, exists = get_listing("file:///whatever", test_session)
    (
        dc.read_values(file=list(_tree_to_entries(TREE)))
        .settings(namespace=listing_namespace_name, project=listing_project_name)
        .save(dataset_name, listing=True)
    )
    assert dataset_name == f"{LISTING_PREFIX}file:///whatever/"
    assert not exists

    dataset_name_dir1, _, _, exists = get_listing(
        "file:///whatever/dir1", test_session, update=True
    )
    assert dataset_name_dir1 == f"{LISTING_PREFIX}file:///whatever/dir1/"
    # On update it is always false (there was no reason to complicate it for now)
    assert not exists


def test_resolve_path_in_root(listing):
    node = listing.resolve_path("dir1")
    assert node.dir_type == DirType.DIR
    assert node.is_dir
    assert node.name == "dir1"
    assert node.size == 0


def test_path_resolving_nested(listing):
    node = listing.resolve_path("dir1/d2/file2.csv")
    assert node.dir_type == DirType.FILE
    assert node.name == "file2.csv"
    assert not node.is_dir

    node = listing.resolve_path("dir1/d2")
    assert node.dir_type == DirType.DIR
    assert node.is_dir
    assert node.name == "d2"


def test_resolve_not_existing_path(listing):
    with pytest.raises(FileNotFoundError):
        listing.resolve_path("dir1/fake-file-name")


def test_resolve_root(listing):
    node = listing.resolve_path("")
    assert node.dir_type == DirType.DIR
    assert node.is_dir
    assert node.name == ""
    assert node.size == 0


def test_dir_ends_with_slash(listing):
    node = listing.resolve_path("dir1/")
    assert node.dir_type == DirType.DIR
    assert node.is_dir
    assert node.name == "dir1"


def test_file_ends_with_slash(listing):
    with pytest.raises(FileNotFoundError):
        listing.resolve_path("dir1/dataset.csv/")


def _match_filenames(nodes, expected_names):
    assert len(nodes) == len(expected_names)
    names = (node.name for node in nodes)
    assert set(names) == set(expected_names)


def test_basic_expansion(listing):
    nodes = listing.expand_path("*")
    _match_filenames(nodes, ["dir1", "dir2", "users.csv"])


def test_subname_expansion(listing):
    nodes = listing.expand_path("di*/")
    _match_filenames(nodes, ["dir1", "dir2"])


@skip_if_not_sqlite
def test_multilevel_expansion(listing):
    nodes = listing.expand_path("dir[1,2]/d*")
    _match_filenames(nodes, ["dataset.csv", "diagram.png", "d2"])


def test_expand_root(listing):
    nodes = listing.expand_path("")
    assert len(nodes) == 1
    assert nodes[0].dir_type == DirType.DIR
    assert nodes[0].is_dir


def test_list_dir(listing):
    dir1 = listing.resolve_path("dir1/")
    names = listing.ls_path(dir1, ["path"])
    assert {n[0] for n in names} == {"dir1/d2", "dir1/dataset.csv"}


def test_list_file(listing):
    file = listing.resolve_path("dir1/dataset.csv")
    src = DataSource(listing, listing.client, file)
    results = list(src.ls(["sys__id", "name", "dir_type"]))
    assert {r[1] for r in results} == {"dataset.csv"}
    assert results[0][0] == file.sys__id
    assert results[0][1] == file.name
    assert results[0][2] == DirType.FILE


def test_subtree(listing):
    dir1 = listing.resolve_path("dir1/")
    nodes = listing.subtree_files(dir1)
    subtree_files = ["dataset.csv", "file1.csv", "file2.csv"]
    _match_filenames([nwp.n for nwp in nodes], subtree_files)


def test_subdirs(listing):
    dirs = list(listing.get_dirs_by_parent_path(""))
    _match_filenames(dirs, ["dir1", "dir2"])


@pytest.mark.parametrize(
    "name,is_listing",
    [
        ("lst__s3://my-bucket", True),
        ("lst__file:///my-folder/dir1", True),
        ("s3://my-bucket", False),
        ("my-dataset", False),
    ],
)
def test_is_listing_dataset(name, is_listing):
    assert is_listing_dataset(name) is is_listing


def test_listing_uri_from_name():
    assert listing_uri_from_name("lst__s3://my-bucket") == "s3://my-bucket"
    with pytest.raises(ValueError):
        listing_uri_from_name("s3://my-bucket")
