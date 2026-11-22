import pytest

import datachain as dc
from datachain.lib.file import File
from datachain.node import Node


@pytest.mark.parametrize("location", [None, {}, [], {"offset": 3}, [{"offset": 3}]])
def test_node_from_file_preserves_location(location):
    file = File(
        source="gs://bucket",
        path="archive/member.txt",
        etag="e1",
        md5="d41d8cd98f00b204e9800998ecf8427e",
        location=location,
    )

    node = Node.from_file(file)

    assert node.location == location
    assert node.md5 == file.md5
    assert node.to_file() == file
    assert node.to_file().get_hash() == file.get_hash()


def test_node_from_row_and_expand_path_preserve_md5(test_session):
    file = File(
        source="gs://bucket",
        path="dir/a.txt",
        size=3,
        etag="e1",
        md5="d41d8cd98f00b204e9800998ecf8427e",
    )
    ds = dc.read_values(file=[file], session=test_session).save("node_md5").dataset
    wh = test_session.catalog.warehouse
    rows = wh.dataset_rows(ds, ds.latest_version)

    (expanded,) = wh.expand_path(rows, "dir/a.txt")
    assert expanded.md5 == file.md5
    assert expanded.to_file().get_hash() == file.get_hash()
    assert wh.get_node_by_path(rows, "dir/a.txt").md5 == file.md5

    # Old listings without file__md5 default to "".
    legacy = Node.from_row({"sys__id": 1, "sys__rand": 2, "file__path": "dir/a.txt"})
    assert legacy.md5 == ""
