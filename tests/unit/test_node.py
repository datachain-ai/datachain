import logging
from unittest.mock import Mock

import pytest

from datachain.node import Node, NodeWithPath


@pytest.mark.parametrize(
    "path",
    [
        ["..", "escape.txt"],
        ["bucket", "../escape.txt"],
        ["bucket", "dir/../../escape.txt"],
        ["bucket", "dir/../../../escape.txt"],
        ["bucket", "./escape.txt"],
        ["bucket", ".."],
        ["/escape.txt"],
    ],
)
def test_instantiate_skips_destination_outside_output(tmp_path, caplog, path):
    output = tmp_path / "output"
    output.mkdir()
    client = Mock(uri="s3://bucket")
    node = NodeWithPath(Node(path="escape.txt"), path)

    with caplog.at_level(logging.WARNING, logger="datachain"):
        node.instantiate(client, str(output), None)

    client.instantiate_object.assert_not_called()
    assert "not within output directory" in caplog.text
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize(
    "path,expected",
    [
        (["bucket", "dir/file.txt"], "bucket/dir/file.txt"),
        (["dir", "file.txt"], "dir/file.txt"),
        (["file.txt"], "file.txt"),
        (["bucket", "dir/..file.txt"], "bucket/dir/..file.txt"),
    ],
)
def test_instantiate_writes_destination_inside_output(tmp_path, path, expected):
    output = tmp_path / "output"
    client = Mock(uri="s3://bucket")
    node = NodeWithPath(Node(path="dir/file.txt"), path)

    node.instantiate(client, str(output), None)

    client.instantiate_object.assert_called_once()
    dst = client.instantiate_object.call_args.args[1]
    assert dst == str(output / expected)


def test_instantiate_relative_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = Mock(uri="s3://bucket")

    NodeWithPath(Node(path="a.txt"), ["a.txt"]).instantiate(client, ".", None)
    NodeWithPath(Node(path="b.txt"), ["..", "b.txt"]).instantiate(client, ".", None)

    assert client.instantiate_object.call_count == 1
    assert client.instantiate_object.call_args.args[1] == "./a.txt"
