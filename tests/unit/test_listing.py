import posixpath
import random
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pytest
from sqlalchemy import true

import datachain as dc
from datachain.catalog.catalog import DataSource
from datachain.client import Client
from datachain.fs.utils import path_to_fsspec_uri
from datachain.lib.dc.storage_pattern import (
    convert_globstar_to_glob,
    path_filter,
    should_use_recursion,
    split_uri_pattern,
)
from datachain.lib.file import File
from datachain.lib.listing import (
    LISTING_PREFIX,
    PathFilter,
    _desanitize_ds_name,
    _sanitize_ds_name,
    get_listing,
    glob_match,
    is_listing_dataset,
    listing_uri_from_name,
    parse_listing_uri,
    select_paths,
)
from datachain.listing import SavedListing
from datachain.node import DirType, Node
from datachain.sql.functions import path as pathfunc
from datachain.utils import uses_glob
from tests.utils import skip_if_not_sqlite

TREE = {
    "dir1": {
        "d2": {None: ["file1.csv", "file2.csv"]},
        None: ["dataset.csv"],
    },
    "dir2": {
        "spécial": {
            "dir_%1": {None: ["a.csv", "b.csv"]},
            "dirXX2": {None: ["d.csv"]},
        },
        None: ["diagram.png"],
    },
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
    fake_uri = path_to_fsspec_uri("/whatever")
    catalog = test_session.catalog
    listing_namespace_name = catalog.metastore.system_namespace_name
    listing_project_name = catalog.metastore.listing_project_name
    dataset_name, _, _, _ = get_listing(fake_uri, test_session)
    (
        dc.read_values(file=list(_tree_to_entries(TREE)))
        .settings(namespace=listing_namespace_name, project=listing_project_name)
        .save(dataset_name, listing=True)
    )

    with SavedListing(
        catalog.metastore.clone(),
        catalog.warehouse.clone(),
        Client.get_client(fake_uri, catalog.cache, **catalog.client_config),
        dataset_name=dataset_name,
        column="file",
    ) as listing_obj:
        yield listing_obj


def test_get_listing_returns_exact_math_on_update(test_session):
    # Context: https://github.com/datachain-ai/datachain/pull/726
    # On update it should be returning the exact match (not a "bigger" one)
    fake_uri = path_to_fsspec_uri("/whatever")
    catalog = test_session.catalog
    listing_namespace_name = catalog.metastore.system_namespace_name
    listing_project_name = catalog.metastore.listing_project_name

    dataset_name_dir1, _, _, exists = get_listing(f"{fake_uri}/dir1", test_session)
    (
        dc.read_values(file=list(_tree_to_entries(TREE["dir1"])))
        .settings(namespace=listing_namespace_name, project=listing_project_name)
        .save(dataset_name_dir1, listing=True)
    )
    assert dataset_name_dir1 == f"{LISTING_PREFIX}{fake_uri}/dir1/"
    assert not exists

    dataset_name, _, _, exists = get_listing(fake_uri, test_session)
    (
        dc.read_values(file=list(_tree_to_entries(TREE)))
        .settings(namespace=listing_namespace_name, project=listing_project_name)
        .save(dataset_name, listing=True)
    )
    assert dataset_name == f"{LISTING_PREFIX}{fake_uri}/"
    assert not exists

    dataset_name_dir1, _, _, exists = get_listing(
        f"{fake_uri}/dir1", test_session, update=True
    )
    assert dataset_name_dir1 == f"{LISTING_PREFIX}{fake_uri}/dir1/"
    # On update it is always false (there was no reason to complicate it for now)
    assert not exists


@pytest.mark.parametrize(
    "subdir,expected_list_path",
    [
        ("plain", "plain/"),
        # sanitized in the dataset name as "some_x2edir"; must come back decoded
        ("some.dir", "some.dir/"),
        ("nested/some.dir", "nested/some.dir/"),
        # a dir literally named like an encoding token: pre-escaped to
        # "v_x_x2e0" in the name, so decoding must not turn it into "v.0"
        ("v_x2e0", "v_x2e0/"),
    ],
)
def test_get_listing_reuse_returns_decoded_subpath(
    test_session, subdir, expected_list_path
):
    catalog = test_session.catalog
    fake_uri = path_to_fsspec_uri("/reuse")
    broad_name, _, _, _ = get_listing(fake_uri, test_session)
    (
        dc.read_values(file=[File(path=f"{subdir}/a.csv")])
        .settings(
            namespace=catalog.metastore.system_namespace_name,
            project=catalog.metastore.listing_project_name,
        )
        .save(broad_name, listing=True)
    )

    name, _, list_path, reused = get_listing(f"{fake_uri}/{subdir}", test_session)

    assert reused
    assert name == broad_name
    assert list_path == expected_list_path


def test_get_listing_reuses_broader_listing_for_glob_base(test_session):
    catalog = test_session.catalog
    fake_uri = path_to_fsspec_uri("/globbase")
    broad_name, _, _, _ = get_listing(fake_uri, test_session)
    (
        dc.read_values(file=[File(path="media/music/song.mp3")])
        .settings(
            namespace=catalog.metastore.system_namespace_name,
            project=catalog.metastore.listing_project_name,
        )
        .save(broad_name, listing=True)
    )

    base, pattern = split_uri_pattern(f"{fake_uri}/media/mus*/*.mp3")
    name, _, list_path, reused = get_listing(base, test_session)

    assert pattern == "mus*/*.mp3"
    assert reused
    assert name == broad_name
    assert list_path == "media/"


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


def test_resolve_path_with_wildcard_chars_is_literal(listing):
    node = listing.resolve_path("dir2/spécial/dir_%1")
    assert node.dir_type == DirType.DIR

    # must not resolve via the dirXX2/ sibling, which `dir_%2` matches as a
    # LIKE pattern
    with pytest.raises(FileNotFoundError):
        listing.resolve_path("dir2/spécial/dir_%2")


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


@pytest.mark.parametrize("column", ["file", "document"])
@pytest.mark.parametrize(
    "location",
    [None, {}, [], {"vtype": "tar", "offset": 128}, [{"vtype": "tar", "offset": 128}]],
)
def test_saved_nodes_preserve_file_metadata(test_session, column, location):
    file = File(
        source="s3://bucket",
        path="folder/a.txt",
        size=123,
        version="version-1",
        etag="etag-1",
        last_modified=datetime(2026, 10, 23, 1, 2, 3, tzinfo=UTC),
        location=location,
    )
    chain = dc.read_values(session=test_session, **{column: [file]}).save("nodes")
    catalog = test_session.catalog
    dataset = chain.dataset
    rows = catalog.warehouse.dataset_rows(
        dataset, dataset.latest_version, column=column
    )
    expected_identity = chain.settings(sys=True).to_list("sys.id", "sys.rand")[0]
    nodes = [
        catalog.warehouse.get_node_by_path(rows, file.path),
        *catalog.warehouse.expand_path(rows, "folder/*.txt"),
        *catalog.warehouse.expand_path(rows, "*/a.txt"),
        *(n.n for n in catalog.warehouse.get_subtree_files(rows, Node.root())),
    ]

    assert len(nodes) == 4
    for node in nodes:
        assert node.to_file() == file
        assert node.location == location
        assert (node.sys__id, node.sys__rand) == expected_identity
        assert node.dir_type == DirType.FILE


@pytest.mark.parametrize(
    "pattern,path,dir_type",
    [
        ("dir*/d2", "dir1/d2", DirType.DIR),
        ("dir*/d2/", "dir1/d2", DirType.DIR),
        ("dir*/d2/file1.csv", "dir1/d2/file1.csv", DirType.FILE),
        ("dir*/diagram.png", "dir2/diagram.png", DirType.FILE),
    ],
)
def test_exact_path_after_glob(listing, pattern, path, dir_type):
    nodes = listing.expand_path(pattern)

    assert len(nodes) == 1
    assert nodes[0].path == path
    assert nodes[0].dir_type == dir_type
    assert nodes[0].is_dir == (dir_type == DirType.DIR)


@pytest.mark.parametrize(
    "pattern", ["dir*/missing.txt", "dir*/missing/a.txt", "missing.txt"]
)
def test_missing_exact_path(listing, pattern):
    with pytest.raises(FileNotFoundError, match="Unable to resolve path"):
        listing.warehouse.expand_path(listing.dataset_rows, pattern)


def test_exact_path_after_glob_matches_multiple_directories(test_session):
    chain = dc.read_values(
        session=test_session,
        file=[File(path=path) for path in ("one/a.txt", "two/a.txt", "other/b.txt")],
    ).save("nodes")
    dataset = chain.dataset
    warehouse = test_session.catalog.warehouse
    rows = warehouse.dataset_rows(dataset, dataset.latest_version)

    nodes = warehouse.expand_path(rows, "*/a.txt")

    assert {node.path for node in nodes} == {"one/a.txt", "two/a.txt"}


def test_exact_path_after_glob_with_no_parent_matches(listing):
    assert listing.expand_path("missing*/a.txt") == []


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
    # Encoded name round-trips back to the original URI
    assert listing_uri_from_name("lst__s3://bucket/v1_x2e0/") == "s3://bucket/v1.0/"
    # Escaped _x round-trips
    assert (
        listing_uri_from_name("lst__s3://bucket/export_x_xml/")
        == "s3://bucket/export_xml/"
    )
    with pytest.raises(ValueError):
        listing_uri_from_name("s3://my-bucket")


def test_listing_of_current_objects_has_its_own_name():
    full, lst_uri, path = parse_listing_uri("s3://bucket/v1.0/dir")
    current, cur_uri, cur_path = parse_listing_uri("s3://bucket/v1.0/dir", False)

    assert current == "lst__current__s3://bucket/v1_x2e0/dir/"
    assert (cur_uri, cur_path) == (lst_uri, path)
    assert is_listing_dataset(current)
    assert listing_uri_from_name(current) == listing_uri_from_name(full)
    # neither listing is a prefix of the other, so one is never reused for the other
    assert not current.startswith(full)
    assert not full.startswith(current)


def _ds_name(uri: str) -> str:
    name, _, _ = parse_listing_uri(uri)
    return name


@pytest.mark.parametrize(
    "uri,expected_suffix",
    [
        # Common cloud / local names pass through unchanged
        ("s3://my-bucket/dogs/", "s3://my-bucket/dogs/"),
        # Dots are encoded (_x2e)
        ("s3://bucket/v1.0/", "s3://bucket/v1_x2e0/"),
        # Percent is encoded (_x25)
        ("s3://bucket/dir%25/", "s3://bucket/dir_x2525/"),
        # Hash is encoded (_x23)
        ("s3://bucket/dir%23/", "s3://bucket/dir_x2523/"),
        # Space is encoded (_x20)
        ("s3://bucket/dir%20path/", "s3://bucket/dir_x2520path/"),
        # @ is encoded (_x40)
        ("s3://bucket/user@host/", "s3://bucket/user_x40host/"),
        # Literal _x in path is doubled
        ("s3://bucket/export_xml/", "s3://bucket/export_x_xml/"),
    ],
)
def test_parse_listing_uri_sanitizes_dataset_name(uri, expected_suffix):
    assert _ds_name(uri) == f"{LISTING_PREFIX}{expected_suffix}"


def test_parse_listing_uri_no_collision_percent_vs_literal():
    assert _ds_name("s3://b/dir%25/") != _ds_name("s3://b/dir_x25/")


def test_parse_listing_uri_no_collision_dot_vs_underscore():
    assert _ds_name("s3://b/v1.0/") != _ds_name("s3://b/v1_0/")


@pytest.mark.parametrize(
    "raw",
    [
        "s3://my-bucket/dogs/",
        "s3://bucket/v1.0/",
        "s3://bucket/dir%25/",
        "s3://bucket/user@host/",
        "s3://bucket/export_xml/",
        "s3://my.company.data/path/to/files/",
        "gs://bucket-with_underscores_x_and.dots/",
        "file:///home/user/path with spaces/",
        "s3://b/_x_x_x/edge/",
        # Literal _y and _z in path components (must be escaped, not misread as markers)
        "s3://bucket/_y1a2b_dir/",
        "s3://bucket/_z001f3b_data/",
        "file:///home/user/_yellow_and_zebra/",
        # BMP non-ASCII (U+0100-U+FFFF): Cyrillic, CJK, Latin extended
        "file:///home/user/données/",
        "file:///home/user/\u4e2d\u6587/",
        "s3://bucket/\u00e9l\u00e8ve/",
        # Supplementary (U+10000+): emoji in local paths
        "file:///home/user/\U0001f3b5music/",
        "file:///home/user/\U0001f600dir/",
        # Mixed: ASCII-disallowed + BMP + supplementary in one URI
        "file:///home/user/caf\u00e9 \U0001f600/",  # codespell:ignore caf
    ],
)
def test_sanitize_desanitize_roundtrip(raw):
    """_desanitize_ds_name is the exact inverse of _sanitize_ds_name."""
    assert _desanitize_ds_name(_sanitize_ds_name(raw)) == raw


def test_desanitize_plain_passthrough():
    """Strings with no _xHH sequences pass through unchanged."""
    assert _desanitize_ds_name("s3://my-bucket/dogs/") == "s3://my-bucket/dogs/"


def test_sanitize_non_ascii_bmp_encoding():
    """BMP chars above U+00FF are encoded as _yHHHH (not the old _xHH)."""
    enc = _sanitize_ds_name("lst__s3://bucket.name/\u4e2d\u6587")
    # Must NOT produce the broken 2-digit encoding _x4e2d / _x6587
    assert "_x4e" not in enc and "_x65" not in enc
    # Must produce the short 4-digit _y form
    assert "_y4e2d" in enc
    assert "_y6587" in enc


def test_sanitize_non_ascii_supplementary_encoding():
    """Supplementary chars are encoded as _zHHHHHH."""
    enc = _sanitize_ds_name("file:///\U0001f3b5/")
    assert "_z01f3b5" in enc


def test_desanitize_old_xhh_still_works():
    """Old _xNN-encoded names (stored before the fix) still decode correctly."""
    # Simulate an old stored name: s3://my.bucket/ -> lst__s3://my_x2ebucket/
    old_encoded = "lst__s3://my_x2ebucket/"
    assert _desanitize_ds_name(old_encoded) == "lst__s3://my.bucket/"


_GLOB_PATTERNS = [
    "*",
    "a*",
    "*b",
    "a?c",
    "?",
    "[abc]",
    "[^abc]",
    "[a-c]x",
    "[]a]",
    "[^]a]",
    "[a-]",
    "[-a]",
    "a/*",
    "*/b",
    "a.b",
    "a\\b",
    "[",
    "a[b",
    "[!a]",
    "**",
    "a*b*c",
    "[z-a]",
    "[a-c-e]",
    "[]-a]",
    "[^a-]",
    "*.[ch]",
    "x?y*",
    "a+b",
    "(a)",
    "a$",
    "^a",
]
_GLOB_PATHS = [
    "",
    "a",
    "b",
    "abc",
    "ab",
    "a/b",
    "a/b/c",
    "x/a.b",
    "]",
    "-",
    "!",
    "a-",
    "a\\b",
    "aXc",
    "ac",
    "a[b",
    "c",
    "z",
    "a.c",
    "a.h",
    "xyz",
    "a+b",
    "(a)",
    "a$",
    "^a",
    "\n",
    "a\nb",
]


@pytest.mark.parametrize("pattern", _GLOB_PATTERNS)
def test_glob_match_is_sqlite_glob(pattern):
    with closing(sqlite3.connect(":memory:")) as conn:
        for path in _GLOB_PATHS:
            (expected,) = conn.execute("SELECT ? GLOB ?", (path, pattern)).fetchone()
            assert glob_match(path, pattern) == bool(expected), (path, pattern)


@pytest.mark.parametrize(
    "paths",
    [
        PathFilter(),
        PathFilter.ls("dir/", True),
        PathFilter.ls("dir/", False),
        PathFilter.ls("", False),
        PathFilter.ls("d*/x", True),
        PathFilter.ls("dir/", True).glob("dir/*.txt"),
        PathFilter.ls("", False).glob("a?"),
    ],
)
def test_path_filter_python_matches_sql(test_session, paths):
    names = ["a", "ab", "x.txt", "dir", "dir/x.txt", "dir/a/x.txt", "dirx/x", "d/x"]
    chain = dc.read_values(path=names, session=test_session)
    for condition in paths.sql(dc.C("path")):
        chain = chain.filter(condition)

    assert sorted(chain.to_values("path")) == sorted(filter(paths.matcher(), names))


def test_glob_match_is_sqlite_glob_on_random_patterns():
    rng = random.Random(0)  # noqa: S311
    with closing(sqlite3.connect(":memory:")) as conn:
        for _ in range(3000):
            pattern = "".join(rng.choices("ab-]^[*?/!", k=rng.randint(0, 9)))
            path = "".join(rng.choices("ab-]^[/!", k=rng.randint(0, 9)))
            (expected,) = conn.execute("SELECT ? GLOB ?", (path, pattern)).fetchone()
            assert glob_match(path, pattern) == bool(expected), (path, pattern)


def test_glob_match_is_linear_in_stars():
    path = "a" * 1000 + ".txt"

    assert not glob_match(path, "*a" * 30 + "b.txt")
    assert glob_match(path, "*a" * 30 + "*.txt")


@pytest.mark.parametrize(
    "list_path,pattern,recursive",
    [
        ("dir/", None, True),
        ("dir/", None, False),
        ("", None, True),
        ("d*/x", None, True),
        ("dir/", "*.txt", True),
        ("dir/", "**/*.txt", True),
        ("", "a?", False),
    ],
)
def test_select_paths_keeps_filter_steps(test_session, list_path, pattern, recursive):
    chain = dc.read_values(file=[File(path="dir/x.txt")], session=test_session)
    path = dc.C("file.path")
    rec = should_use_recursion(pattern, recursive) if pattern else recursive
    expected = chain.filter(dc.C("file.is_latest") == true())
    if not rec:
        expected = expected.filter(
            pathfunc.parent(path) == list_path.lstrip("/").rstrip("/*")
        )
    elif list_path and list_path != "/":
        expected = expected.filter(
            path.glob(list_path)
            if uses_glob(list_path)
            else (path == list_path) | path.glob(list_path.rstrip("/") + "/*")
        )
    if pattern:
        glob = convert_globstar_to_glob(pattern)
        if subpath := list_path.strip("/"):
            glob = f"{subpath}/{glob}"
        expected = expected.filter(path.glob(glob))

    actual = select_paths(chain, path_filter(list_path, pattern, recursive))

    assert [s.hash() for s in actual._query.steps] == [
        s.hash() for s in expected._query.steps
    ]
