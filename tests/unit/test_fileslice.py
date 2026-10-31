import io
import tarfile
from array import array
from ctypes import c_ubyte

import pytest

from datachain.client.fileslice import FileSlice
from datachain.fs.utils import path_to_fsspec_uri
from datachain.lib.file import File
from datachain.lib.tar import process_tar


def make_buffer(kind):
    if kind == "array":
        return array("I", [0] * 4)
    if kind == "typed_view":
        return memoryview(array("I", [0] * 4))
    if kind == "matrix":
        return memoryview(bytearray(16)).cast("B", shape=[4, 4])
    if kind == "empty":
        return bytearray()
    return bytearray(16)


@pytest.mark.parametrize(
    "kind", ["array", "typed_view", "matrix", "bytearray", "empty"]
)
@pytest.mark.parametrize(
    "size, position", [(0, 0), (3, 0), (3, 2), (16, 0), (20, 0), (20, 7), (20, 20)]
)
@pytest.mark.parametrize("local", [False, True])
def test_readinto_buffer_byte_capacity(tmp_path, kind, size, position, local):
    data = bytes(range(size))
    prefix = b"prefix:"
    backing_data = prefix + data + bytes(range(100, 164))
    if local:
        path = tmp_path / "backing.bin"
        path.write_bytes(backing_data)
        backing = path.open("rb")
    else:
        backing = io.BytesIO(backing_data)
    actual = make_buffer(kind)
    expected = make_buffer(kind)
    for buffer in (actual, expected):
        view = memoryview(buffer).cast("B")
        view[:] = b"\xa5" * len(view)

    with FileSlice(backing, len(prefix), size, "member") as stream:
        with io.BytesIO(data) as reference:
            stream.seek(position)
            reference.seek(position)
            assert stream.readinto(actual) == reference.readinto(expected)
            assert stream.tell() == reference.tell()
        assert memoryview(actual).cast("B").tobytes() == (
            memoryview(expected).cast("B").tobytes()
        )
        if position < size:
            assert backing.tell() == len(prefix) + stream.tell()


@pytest.mark.parametrize("shape", [(0, 4), (4, 0)])
@pytest.mark.parametrize("local", [False, True])
def test_readinto_empty_multidimensional_buffer(tmp_path, shape, local):
    buffer = ((c_ubyte * shape[1]) * shape[0])()
    if local:
        path = tmp_path / "backing.bin"
        path.write_bytes(b"prefix:member:after")
        backing = path.open("rb")
    else:
        backing = io.BytesIO(b"prefix:member:after")
    with FileSlice(backing, 7, 6, "member") as stream:
        with io.BytesIO(b"member") as reference:
            assert stream.readinto(buffer) == reference.readinto(buffer) == 0
            assert stream.tell() == reference.tell() == 0
        assert backing.tell() == 7


@pytest.mark.parametrize(
    "kind", ["readonly", "readonly_empty", "strided", "strided_empty"]
)
def test_readinto_invalid_buffer(tmp_path, kind):
    if kind == "readonly":
        buffer = memoryview(b"unchangeable")
    elif kind == "readonly_empty":
        buffer = memoryview(b"")
    elif kind == "strided":
        buffer = memoryview(bytearray(16))[::2]
    else:
        buffer = memoryview(bytearray(16))[::2][:0]
    with io.BytesIO(b"member") as reference:
        with pytest.raises(TypeError):
            reference.readinto(buffer)
    path = tmp_path / "backing.bin"
    path.write_bytes(b"prefix:member:after")
    with FileSlice(path.open("rb"), 7, 6, "member") as stream:
        with pytest.raises(TypeError):
            stream.readinto(buffer)
        assert stream.tell() == 0


@pytest.mark.parametrize("kind", ["array", "matrix"])
@pytest.mark.parametrize("caching_enabled", [False, True])
def test_tar_member_readinto_buffer_byte_capacity(
    tmp_path, catalog, kind, caching_enabled
):
    path = tmp_path / "archive.tar"
    payload = b"a" * 512
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in [("member.bin", payload), ("second.bin", b"b" * 1024)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    parent = File(path=path.name, source=path_to_fsspec_uri(str(tmp_path)))
    parent._set_stream(catalog, caching_enabled)
    member = next(process_tar(parent))
    member._set_stream(catalog, caching_enabled)
    buffer = (
        array("I", [0] * 512)
        if kind == "array"
        else memoryview(bytearray(2048)).cast("B", shape=[16, 128])
    )
    view = memoryview(buffer).cast("B")
    view[:] = b"\xa5" * len(view)

    with member.open() as stream:
        assert stream.readinto(buffer) == len(payload)
        assert stream.tell() == len(payload)
        assert stream.read() == b""
        assert stream.readinto(buffer) == 0
    assert view[: len(payload)].tobytes() == payload
    assert view[len(payload) :].tobytes() == b"\xa5" * (len(view) - len(payload))


def test_positions():
    data = b"0123456789abcdef"
    base = io.BytesIO(data)
    f = FileSlice(base, 5, 5, "foo")
    assert base.tell() == 0
    assert f.readable()
    assert not f.writable()
    assert f.seekable()
    assert f.name == "foo"

    # f.seek() doesn't move the underlying stream
    f.seek(0)
    assert f.tell() == 0
    assert base.tell() == 0

    assert f.read(3) == data[5:8]
    assert f.tell() == 3
    assert base.tell() == 8

    assert f.read(4) == data[8:10]
    assert f.tell() == 5
    assert base.tell() == 10

    b = bytearray(5)
    f.seek(0)
    f.readinto(b)
    assert b == data[5:10]


def test_invalid_slice():
    data = b"0123456789abcdef"
    base = io.BytesIO(data)
    f = FileSlice(base, 10, 10, "foo")
    assert f.read(4) == data[10:14]
    with pytest.raises(RuntimeError):
        f.read()


def test_close():
    data = b"0123456789abcdef"
    base = io.BytesIO(data)
    f = FileSlice(base, 5, 5, "foo")
    assert f.closed is False
    with f:
        assert f.closed is False
    assert f.closed is True
    assert base.closed is True


def test_implicit_close():
    # Assumes refcounting semantics
    data = b"0123456789abcdef"
    base = io.BytesIO(data)
    f = FileSlice(base, 5, 5, "foo")
    assert base.closed is False
    f = None  # noqa: F841
    assert base.closed is True
