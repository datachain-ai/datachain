import pytest

from datachain.node import DirType, Node
from datachain.nodes_thread_pool import NodeChunk


@pytest.fixture
def nodes():
    return iter(
        [
            make_size_node(11, DirType.DIR, "a/f1", 100),
            make_size_node(12, DirType.FILE, "b/f2", 100),
            make_size_node(13, DirType.FILE, "c/f3", 100),
            make_size_node(14, DirType.FILE, "d/", 100),
            make_size_node(15, DirType.FILE, "e/f5", 100),
            make_size_node(16, DirType.DIR, "f/f6", 100),
            make_size_node(17, DirType.FILE, "g/f7", 100),
        ]
    )


def make_size_node(node_id, dir_type, path, size):
    return Node(
        node_id,
        dir_type=dir_type,
        path=path,
        size=size,
    )


class FakeCache:
    def contains(self, _):
        return False


def make_chunks(nodes, *args, **kwargs):
    return NodeChunk(FakeCache(), "s3://foo", nodes, *args, **kwargs)


def test_node_bucket_the_only_item():
    bkt = make_chunks(iter([make_size_node(20, DirType.FILE, "file.csv", 100)]), 201)

    result = next(bkt)
    assert len(result) == 1
    assert next(bkt, None) is None


def test_node_bucket_the_only_item_over_limit():
    bkt = make_chunks(iter([make_size_node(20, DirType.FILE, "file.csv", 100)]), 1)

    result = next(bkt)
    assert len(result) == 1
    assert next(bkt, None) is None


def test_node_bucket_the_last_one():
    bkt = make_chunks(iter([make_size_node(20, DirType.FILE, "file.csv", 100)]), 1)

    next(bkt)
    with pytest.raises(StopIteration):
        next(bkt)


def test_node_bucket_basic(nodes):
    bkt = list(make_chunks(nodes, 201))

    assert len(bkt) == 2
    assert len(bkt[0]) == 3
    assert len(bkt[1]) == 1


def test_node_bucket_full_split(nodes):
    bkt = list(make_chunks(nodes, 0))

    assert len(bkt) == 4
    assert len(bkt[0]) == 1
    assert len(bkt[1]) == 1
    assert len(bkt[2]) == 1
    assert len(bkt[3]) == 1
