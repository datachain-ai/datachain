import io
from typing import IO

from fsspec.callbacks import DEFAULT_CALLBACK, Callback


class FileWrapper(io.RawIOBase):
    """Instrumented wrapper around an existing file object.

    It wraps the file's read() method to update the callback with the number of
    bytes read.

    It assumes exclusive access to the underlying file object and closes it when it
    gets closed itself.
    """

    def __init__(self, fileobj, callback: Callback = DEFAULT_CALLBACK):
        self.fileobj = fileobj
        self.callback = callback

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return self.fileobj.seekable()

    def tell(self):
        """Return the current file position."""
        return self.fileobj.tell()

    def seek(self, position, whence=io.SEEK_SET):
        """Seek to a position in the file."""
        return self.fileobj.seek(position, whence)

    def readinto(self, b) -> int:
        res = self.fileobj.readinto(b)
        self.callback.relative_update(res)
        return res

    def close(self):
        self.fileobj.close()
        super().close()


class FileSlice(io.RawIOBase):
    """A thin wrapper around an existing file object that provides a part of its data
    as an individual file object.

    It assumes exclusive access to the underlying file object and closes it when it
    gets closed itself.
    """

    def __init__(self, fileobj: IO[bytes], offset: int, size: int, name: str):
        self.fileobj = fileobj
        self.offset = offset
        self.size = size
        self.position = 0
        self.name = name

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return self.fileobj.seekable()

    def tell(self):
        """Return the current file position."""
        return self.position

    def seek(self, position, whence=io.SEEK_SET):
        """Seek to a position in the file."""
        if whence == io.SEEK_SET:
            self.position = min(max(position, 0), self.size)
        elif whence == io.SEEK_CUR:
            if position < 0:
                self.position = max(self.position + position, 0)
            else:
                self.position = min(self.position + position, self.size)
        elif whence == io.SEEK_END:
            self.position = max(min(self.size + position, self.size), 0)
        else:
            raise ValueError("Invalid argument")
        return self.position

    def readinto(self, b):
        max_size = self.size - self.position
        if max_size <= 0:
            return 0
        b = memoryview(b)
        if b.readonly or not b.c_contiguous:
            raise TypeError(
                "readinto() argument must be a writable C-contiguous buffer"
            )
        # Empty multidimensional views cannot be cast to a flat byte view.
        b = b.cast("B") if b.nbytes else memoryview(bytearray())
        self.fileobj.seek(self.offset + self.position)
        if b.nbytes > max_size:
            b = b[:max_size]
        res = self.fileobj.readinto(b)
        if res != b.nbytes:
            raise RuntimeError("unexpected end of data")
        self.position += res
        return res

    def close(self):
        self.fileobj.close()
        super().close()
