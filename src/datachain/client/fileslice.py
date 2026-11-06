import io
from typing import IO

from fsspec.callbacks import DEFAULT_CALLBACK, Callback


class FileWrapper(io.BufferedIOBase):
    """Buffered wrapper that reports bytes read through a callback.

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

    def read(self, size: int | None = -1) -> bytes:
        data = self.fileobj.read(-1 if size is None else size)
        self.callback.relative_update(len(data))
        return data

    def read1(self, size: int | None = -1) -> bytes:
        return self.read(io.DEFAULT_BUFFER_SIZE if size is None or size < 0 else size)

    def close(self):
        self.fileobj.close()
        super().close()


class FileSlice(io.BufferedIOBase):
    """A buffered view of a part of an existing file object, as an individual file.

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

    def read(self, size: int | None = -1) -> bytes:
        left = self.size - self.position
        n = left if size is None or size < 0 else min(size, left)
        buf = bytearray(n)
        return bytes(buf[: self.readinto(buf)])

    def read1(self, size: int | None = -1) -> bytes:
        return self.read(io.DEFAULT_BUFFER_SIZE if size is None or size < 0 else size)

    def readinto(self, b):
        max_size = self.size - self.position
        if max_size <= 0:
            return 0
        self.fileobj.seek(self.offset + self.position)
        if len(b) > max_size:
            b = memoryview(b)[:max_size]
        res = self.fileobj.readinto(b)
        if res != len(b):
            raise RuntimeError("unexpected end of data")
        self.position += res
        return res

    def close(self):
        self.fileobj.close()
        super().close()
