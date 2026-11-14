"""The child side of listing child processes: started by ``listing_worker``, a
child reads the key ranges the parent hands it and sends back what it read. It
imports little, so that it starts fast."""

import asyncio
import os
import pickle
import struct
import sys
import threading
import traceback
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, BinaryIO

from .ranges import KeyRange, Page, RangeReader

if TYPE_CHECKING:
    from datachain.client.fsspec import RangeListedClient

    from .ranges import Lister

HEADER = struct.Struct("<Q")  # the size of the pickle that follows
BATCH_ROWS = 5000
BATCH_PAGES = 16  # pages with few rows, as of delete markers, go too


def dump(message: Any) -> bytes:
    data = pickle.dumps(message, protocol=5)
    return HEADER.pack(len(data)) + data


def load(stream: BinaryIO) -> Any:
    header = stream.read(HEADER.size)
    if len(header) < HEADER.size:
        return None
    (size,) = HEADER.unpack(header)
    return pickle.loads(stream.read(size))  # noqa: S301


class ChildReader(RangeReader):
    """Reads the ranges the parent hands this child process, and tells the parent
    what it read, where it split, and what the checks saw."""

    def __init__(
        self, client: "Lister", prefix: str, write: Callable[[Any], None]
    ) -> None:
        super().__init__(client, prefix)
        self.send = write
        self.room = 0
        self.rids: dict[int, int] = {}  # id(range) -> the parent's range id
        # rows go in batches of several pages: the parent takes each message
        # in turn, and fewer messages cost it less
        self.rows: list[tuple] = []
        self.pages: list[tuple] = []  # what each page of the batch read

    def write(self, message: Any) -> None:
        """Sends ``message`` after the rows read before it."""
        self.flush()
        self.send(message)

    def flush(self) -> None:
        if self.pages:
            # pickled apart: the parent unpickles them where they are read
            rows = pickle.dumps(self.rows, protocol=5)
            self.send(("rows", rows, len(self.rows), self.pages))
            self.rows, self.pages = [], []

    async def read_range(
        self, rid: int, lo: str, hi: str | None, age: int, last: Any, token: Any
    ) -> None:
        key_range = KeyRange(lo, hi, self.order, age, last, token)
        self.rids[id(key_range)] = rid
        try:
            await self._read(key_range)
            self.write(("done", rid))
        except Exception as e:  # noqa: BLE001
            self.rows, self.pages = [], []
            self.send(("failed", rid, "".join(traceback.format_exception(e))))
            return
        finally:
            del self.rids[id(key_range)]

    async def _accept(self, key_range: KeyRange, rows: list[Any], page: Page) -> None:
        r = key_range
        self.rows += map(tuple, rows)
        listed = page.first is not None
        self.pages.append((self.rids[id(r)], r.last, r.token, r.age, listed))
        if len(self.rows) >= BATCH_ROWS or len(self.pages) >= BATCH_PAGES:
            self.flush()

    def _room(self) -> bool:
        return self.room > 0

    def _add(self, key_range: KeyRange, mid: str) -> None:
        self.room -= 1
        self.write(("split", self.rids[id(key_range)], mid, key_range.age))
        key_range.hi = mid

    async def _seam(self, start: str, side: int, key: str) -> None:
        self.write(("seam", start, side, key))

    def _start_ignored(self) -> None:
        self.splits = False
        self.write(("unseeked",))

    def _out_of_order(self) -> None:
        self.splits = False
        self.write(("unordered",))


def main(out_fd: int) -> None:
    """A child's entry: the spec, then requests, on stdin; replies on ``out_fd``."""
    from fsspec.asyn import get_loop, sync

    if sys.platform == "linux":  # a frame of rows in one write: 1 MB, not 64 KB
        import fcntl

        with suppress(OSError):
            fcntl.fcntl(out_fd, 1031, 1 << 20)  # F_SETPIPE_SZ
    out = os.fdopen(out_fd, "wb")
    inp = os.fdopen(os.dup(0), "rb")
    with open(os.devnull, "rb") as devnull:  # a credential process must not read it
        os.dup2(devnull.fileno(), 0)
    try:
        if (spec := load(inp)) is not None:
            cls, args, page_size, prefix = spec
            client = cls(*args)
            client.LIST_PAGE_SIZE = page_size
    except Exception as e:  # noqa: BLE001
        spec = None  # such as a class in the parent's __main__: the parent lists alone
        with suppress(OSError):
            out.write(dump("".join(traceback.format_exception(e))))
            out.flush()
    if spec is not None:
        sync(get_loop(), _serve, client, prefix, inp, out)
    os._exit(0)  # not waiting for threads stuck in the network, such as a DNS lookup


async def _warm_up(client: "RangeListedClient", prefix: str) -> None:
    await client._list_page(prefix, "", None, page_size=1)


async def _serve(
    client: "RangeListedClient", prefix: str, inp: BinaryIO, out: BinaryIO
) -> None:
    loop = asyncio.get_running_loop()
    requests: asyncio.Queue = asyncio.Queue()

    closed = asyncio.Event()

    def put(request: Any) -> None:
        with suppress(RuntimeError):  # the loop is closed
            loop.call_soon_threadsafe(requests.put_nowait, request)

    def pump() -> None:  # blocking reads, also on Windows
        # ends when the parent closes the pipe, is gone or sends garbage
        with suppress(Exception):
            while (request := load(inp)) is not None:
                put(request)
        put(None)
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(closed.set)

    def write_frame(frame: bytes) -> None:
        try:
            out.write(frame)
            out.flush()
        except OSError:  # the parent is gone; Windows says EINVAL, not EPIPE
            os._exit(0)

    def write(message: Any) -> None:
        write_frame(dump(message))

    # read from the start: a parent gone during the warm-up ends this child
    threading.Thread(target=pump, daemon=True).start()
    # warm up DNS, TLS, credentials and the bucket region before taking work
    warm_up = asyncio.ensure_future(_warm_up(client, prefix))
    gone = asyncio.ensure_future(closed.wait())
    await asyncio.wait([warm_up, gone], return_when=asyncio.FIRST_COMPLETED)
    gone.cancel()
    if not warm_up.done():
        warm_up.cancel()
        return
    if error := warm_up.exception():
        write("".join(traceback.format_exception(error)))  # the parent lists alone
        return
    write(None)
    reader = ChildReader(client, prefix, write)
    tasks: set[asyncio.Task] = set()
    while (request := await requests.get()) is not None:
        kind, *args = request
        if kind == "range":
            tasks.add(task := asyncio.create_task(reader.read_range(*args)))
            task.add_done_callback(tasks.discard)
        elif kind == "room":
            reader.room = args[0]
    for task in list(tasks):
        task.cancel()
