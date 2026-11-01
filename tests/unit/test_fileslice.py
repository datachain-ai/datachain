import io

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


@pytest.mark.parametrize("prefix", [0, 7])
def test_wrapper_reads_remainder_in_one_call(prefix, mocker):
    from fsspec.callbacks import Callback

    data = b"a" * (256 * 1024)
    source = io.BytesIO(data)
    read = mocker.spy(source, "read")
    callback = Callback()
    with FileWrapper(source, callback) as stream:
        assert stream.read(prefix) == data[:prefix]
        assert stream.read() == data[prefix:]
        read.assert_called_once_with()
        assert callback.value == len(data)
    assert source.closed
