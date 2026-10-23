import pickle
from datetime import UTC, datetime

import pytest

from datachain.lib.file import File, FileRow
from datachain.node import DirType, Node
from datachain.utils import TIME_ZERO


def test_node_defaults_match_file_metadata():
    node = Node()
    file = File(path="")

    assert {
        name: getattr(node, name) for name in File.model_fields
    } == file.model_dump()
    assert node.to_file() == FileRow("", "").to_file()
    assert node.sys__id == node.sys__rand == 0
    assert node.dir_type == DirType.FILE


@pytest.mark.parametrize("location", [None, {}, [], {"offset": 3}, [{"offset": 3}]])
def test_node_preserves_file_metadata(location):
    file = File(
        source="s3://bucket",
        path="d/a.txt",
        size=42,
        version="v1",
        etag="e1",
        is_latest=False,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        location=location,
    )

    node = Node.from_file(file)

    assert type(node.source) is str
    assert node.location == location
    assert node.to_file() == file
    assert node.to_file("s3://other").source == "s3://other"
    assert node.source == "s3://bucket"
    assert pickle.loads(pickle.dumps(node)) == node  # noqa: S301


@pytest.mark.parametrize("prefix", ["file", "image"])
@pytest.mark.parametrize("location", [{"offset": 3}, [{"offset": 3}]])
def test_node_from_row_preserves_stored_metadata(prefix, location):
    stored = {
        "sys__id": 11,
        "sys__rand": 23,
        f"{prefix}__source": "s3://bucket",
        f"{prefix}__path": "d/a.txt",
        f"{prefix}__location": location,
        f"{prefix}__dir_type": DirType.TAR_ARCHIVE,
    }

    node = Node.from_row(stored, file_prefix=prefix)

    assert (node.sys__id, node.sys__rand) == (11, 23)
    assert node.source == "s3://bucket"
    assert node.path == "d/a.txt"
    assert node.location == location
    assert node.dir_type == DirType.TAR_ARCHIVE
    assert node.to_file().location == location


@pytest.mark.parametrize("include_nulls", [False, True])
def test_node_from_row_defaults_missing_or_null_metadata(include_nulls):
    stored = {
        "sys__id": 11,
        "sys__rand": 23,
        "file__source": "s3://bucket",
        "file__path": "a.txt",
    }
    if include_nulls:
        stored.update({"file__version": None, "file__last_modified": None})

    node = Node.from_row(stored)

    assert node == Node(sys__id=11, sys__rand=23, source="s3://bucket", path="a.txt")
    assert node.version == ""
    assert node.last_modified == TIME_ZERO
    assert node.source == "s3://bucket"
    assert node.to_file().path == "a.txt"


@pytest.mark.parametrize(
    "path,name,parent,downloadable",
    [
        ("a.txt", "a.txt", "", True),
        ("d//a.txt", "a.txt", "d/", True),
        ("d/", "", "d", False),
        ("", "", "", False),
    ],
)
def test_node_tree_helpers_preserve_object_keys(path, name, parent, downloadable):
    node = Node(path=path)

    assert (node.name, node.parent) == (name, parent)
    assert node.is_downloadable is downloadable
    assert node.full_path == path


def test_node_directory_helpers():
    node = Node.from_dir("d//sub")

    assert node.sys__id == -1
    assert node.is_dir and node.is_container
    assert not node.is_downloadable
    assert node.full_path == "d//sub/"
    assert Node.root().full_path == ""


@pytest.mark.parametrize(
    "missing", ["sys__id", "sys__rand", "file__source", "file__path"]
)
def test_node_from_row_requires_identity(missing):
    stored = {
        "sys__id": 11,
        "sys__rand": 23,
        "file__source": "s3://bucket",
        "file__path": "a.txt",
    }
    del stored[missing]

    with pytest.raises(KeyError, match=missing):
        Node.from_row(stored)


def test_node_from_row_rejects_wrong_file_prefix():
    stored = {
        "sys__id": 11,
        "sys__rand": 23,
        "file__source": "s3://bucket",
        "file__path": "a.txt",
    }

    with pytest.raises(KeyError, match="image__source"):
        Node.from_row(stored, file_prefix="image")


def test_node_from_row_normalizes_empty_location_sentinel():
    node = Node.from_row(
        {
            "sys__id": 11,
            "sys__rand": 23,
            "file__source": "s3://bucket",
            "file__path": "a.txt",
            "file__location": "",
        }
    )

    assert node.location is None
    assert node.to_file().location is None


def test_node_positional_identity_is_preserved():
    node = Node(11, 23, "a.txt")

    assert (node.sys__id, node.sys__rand, node.path) == (11, 23, "a.txt")
    assert node.source == ""


def test_node_to_file_accepts_legacy_null_metadata():
    node = Node(path="a.txt", version=None, last_modified=None)  # type: ignore[arg-type]

    assert node.to_file() == File(path="a.txt")
