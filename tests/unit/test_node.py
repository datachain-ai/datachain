import pytest

from datachain.lib.file import File
from datachain.node import Node


@pytest.mark.parametrize("location", [None, {}, [], {"offset": 3}, [{"offset": 3}]])
def test_node_from_file_preserves_location(location):
    file = File(source="s3://bucket", path="archive/member.txt", location=location)

    node = Node.from_file(file)

    assert node.location == location
    assert node.to_file() == file
