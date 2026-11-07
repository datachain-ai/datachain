import io
from array import array

import pytest

from datachain.client.fileslice import FileSlice, FileWrapper


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


def test_file_wrapper_read1_reads_at_most_one_buffer():
    wrapper = FileWrapper(io.BytesIO(b"x" * 3 * io.DEFAULT_BUFFER_SIZE))
    assert len(wrapper.read1()) == io.DEFAULT_BUFFER_SIZE


def test_file_wrapper_read_fills_short_reads():
    class Trickle(io.RawIOBase):
        def __init__(self, data):
            self.data = data

        def readable(self):
            return True

        def read(self, size=-1):
            n = min(3, size)
            chunk, self.data = self.data[:n], self.data[n:]
            return chunk

    wrapper = FileWrapper(Trickle(b"0123456789"))
    assert (wrapper.read(8), wrapper.read(8), wrapper.read(8)) == (
        b"01234567",
        b"89",
        b"",
    )


def test_file_slice_read1_reads_at_most_one_buffer():
    f = FileSlice(io.BytesIO(b"0123456789" * 1000), 5, 9000, "foo")
    assert isinstance(f, io.BufferedIOBase)
    assert f.read(3) == b"567"
    assert len(f.read1()) == min(io.DEFAULT_BUFFER_SIZE, 9000 - 3)


def test_file_slice_readinto_int_array_larger_than_slice_reads_slice():
    f = FileSlice(io.BytesIO(b"prefix" + bytes(range(10)) + b"suffix"), 6, 10, "m")
    buf = array("I", [0] * 4)
    assert f.readinto(buf) == 10
    assert bytes(buf)[:10] == bytes(range(10))
