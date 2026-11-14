"""The parent side of listing child processes: it starts them and hands them key
ranges (``listing_child`` is the child side)."""

import asyncio
import atexit
import itertools
import logging
import os
import pickle
import signal
import sys
from collections.abc import Iterator, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from datachain.utils import usable_cpu_count

from .listing_child import HEADER, _serve, dump, load
from .ranges import CHILD_WORKERS, KeyRange

if TYPE_CHECKING:
    from datachain.client.fsspec import RangeListedClient

    from .ranges import RangeListing

__all__ = ["HEADER", "Child", "Pool", "_serve", "dump", "load"]

logger = logging.getLogger("datachain")


# a frozen app's executable starts the app, not Python's "-c"
_SPAWNS = bool(sys.executable) and not getattr(sys, "frozen", False)
MAX_PROCESSES = 8


def list_processes(cpus: int) -> int:
    """Listing children for this many CPUs: one per CPU, at most eight, none on one.

    A child waits on the store about half the time, so one per CPU keeps them
    busy. Past eight, the parent, which takes every row, sets the pace; one child
    costs more in pipes than it saves.
    """
    return min(MAX_PROCESSES, cpus) if cpus > 1 else 0


LIST_PROCESSES = list_processes(usable_cpu_count()) if _SPAWNS else 0
# a child that reads ranges and says nothing this long is stuck (stopped,
# deadlocked): the parent kills it and reads its ranges on; a page normally takes
# under a second
READ_LIMIT = 1 << 22
REPLY_TIMEOUT = float(os.environ.get("DATACHAIN_LISTING_REPLY_TIMEOUT", "300"))


async def _receive(stream: asyncio.StreamReader) -> Any:
    (size,) = HEADER.unpack(await stream.readexactly(HEADER.size))
    if size > 1 << 32:  # not a frame: the interpreter printed to stdout at startup
        raise pickle.UnpicklingError(f"a frame of {size} bytes")
    return pickle.loads(await stream.readexactly(size))  # noqa: S301


def _command() -> list[str]:
    path = [p for p in sys.path if isinstance(p, str)]  # the parent's imports
    code = f"""
# before any import: ignore Ctrl-C, and send stray prints to stderr
import os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_IGN)
out = os.dup(1)
os.dup2(2, 1)
sys.path[:] = {path!r}  # also before ctypes: not a ctypes.py in the cwd
if sys.platform == "linux":  # killed when the parent dies, even if stopped or stuck
    __import__("ctypes").CDLL(None).prctl(1, 9)  # PR_SET_PDEATHSIG, SIGKILL
    if os.getppid() != {os.getpid()}:  # the parent died already
        os._exit(0)
from datachain.client.listing_child import main
main(out)
"""
    return [sys.executable, "-c", code]


# children alive in this process; at exit the loop that would close them is gone
_children: set[asyncio.subprocess.Process] = set()


async def _spawn() -> asyncio.subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        *_command(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,  # no terminal: its Ctrl-C and prompts are the parent's
        # a page's frame is some 200 KB: below this the pipe pauses for each one
        limit=READ_LIMIT,
    )
    _children.add(proc)
    return proc


def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kills a child and what it started, such as a credential process: each child
    leads a process group of its own."""
    _children.discard(proc)
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except AttributeError:  # Windows: no process groups
        with suppress(ProcessLookupError):
            proc.kill()
    except OSError:  # no group left, or only a zombie in it (EPERM on macOS)
        # not proc.kill(): its poll() reaps the child before asyncio's watcher,
        # which then reports returncode 255
        with suppress(OSError):
            os.kill(proc.pid, signal.SIGKILL)


@atexit.register
def _kill_children() -> None:
    """Stopped or blocked children don't see the parent exit."""
    for proc in list(_children):
        _kill(proc)


class Rows(Sequence):
    """``FileRow`` rows a child sent, unpickled when first read: by whoever reads
    the listing, not by the event loop, which then keeps up with the children
    instead of contending with the reader for the GIL."""

    __slots__ = ("_count", "_pickled", "_rows")

    def __init__(self, pickled: bytes, count: int) -> None:
        self._pickled = pickled
        self._count = count
        self._rows: list | None = None

    def _load(self) -> list:
        if (rows := self._rows) is None:
            from datachain.lib.file import as_file_row

            # plain tuples come: a row type would unpickle per row in Python
            pickled = pickle.loads(self._pickled)  # noqa: S301
            self._rows = rows = list(map(as_file_row, pickled))
        return rows

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):  # type: ignore[override]
        return self._load()[index]

    def __iter__(self) -> Iterator:
        return iter(self._load())


class Child:
    """A running child process, as the parent sees it: the ranges it reads."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self.proc = proc
        self.ranges: dict[int, tuple[KeyRange, asyncio.Future]] = {}
        self.ids = itertools.count()
        self.room = 0  # how many more ranges it may split off, as it was told
        self.gone = False
        self.heard = 0.0  # when it last said something, or got work while idle

    def send(self, message: Any) -> bool:
        assert self.proc.stdin
        if self.gone or self.proc.stdin.is_closing():
            return False
        self.proc.stdin.write(dump(message))
        return True

    async def read(self, key_range: KeyRange) -> tuple[str, Any]:
        """Has the child read ``key_range``. Returns "done", or "failed" or
        "gone" with the rest of the range: what the child did not pass on."""
        rid = next(self.ids)
        loop = asyncio.get_running_loop()
        reply = loop.create_future()
        if not self.ranges:
            self.heard = loop.time()
        self.ranges[rid] = key_range, reply
        r = key_range
        if not self.send(("range", rid, r.lo, r.hi, r.age, r.last, r.token)):
            del self.ranges[rid]
            return "gone", key_range.rest()
        try:
            return await reply
        finally:
            self.ranges.pop(rid, None)

    def end(self, outcome: str) -> None:
        """The child reads no more: its ranges come back with what is left."""
        self.gone = True
        for key_range, reply in self.ranges.values():
            if not reply.done():
                reply.set_result((outcome, key_range.rest()))


class Pool:
    """Child processes reading key ranges for one ``RangeListing``, which hands
    them out. The parent reads the rest of any range a child fails, so errors
    reach the user as raised, and requeues the rest of a child's ranges when the
    child dies."""

    def __init__(self, client: "RangeListedClient", prefix: str, size: int) -> None:
        self.client, self.prefix = client, prefix
        # the page size travels so that tests can shrink it in children too
        spec = (type(client), (client.name, client.fs_kwargs, client.cache))
        try:
            self.spec = dump((*spec, client.LIST_PAGE_SIZE, prefix))
        except Exception:
            logger.debug(
                "Listing in one process: client config does not pickle", exc_info=True
            )
            size = 0
        self.size = size
        self.listing: RangeListing | None = None
        self.procs: list[asyncio.subprocess.Process] = []
        self.children: list[Child] = []  # running, in the order they came up
        self.tasks: list[asyncio.Task] = []
        self.closed = False

    def start(self) -> None:
        if not self.tasks:
            # a client may know by now how children can list with less: S3's
            # children sign plain requests instead of starting botocore
            if (lean := self.client._child_spec()) is not None:
                self.spec = dump((*lean, self.client.LIST_PAGE_SIZE, self.prefix))
            self.tasks = [asyncio.create_task(self._run()) for _ in range(self.size)]

    async def _run(self) -> None:
        proc = child = None
        try:
            self.procs.append(proc := await _spawn())
            assert proc.stdin
            assert proc.stdout
            proc.stdin.write(self.spec)
            if error := await _receive(proc.stdout):
                logger.debug("Listing process cannot list:\n%s", error)
                return
            child = Child(proc)
            await self._serve(child)
        except (OSError, EOFError, NotImplementedError, pickle.UnpicklingError):
            logger.debug("Listing process failed", exc_info=True)  # exited or never ran
        except Exception:
            logger.warning("Listing process failed", exc_info=True)
        finally:  # also when cancelled
            if child is not None:
                self._drop(child)
            if proc is not None:
                _kill(proc)
            running = [t for t in self.tasks if not t.done()]  # this one included
            failed = self.listing is not None and self.listing.has_failed()
            if not (self.closed or failed or self.children or len(running) > 1):
                logger.warning("Listing in one process: no listing process runs")

    async def _serve(self, child: Child) -> None:
        """Takes a running child's messages until it is gone or stuck."""
        listing = self.listing
        assert listing
        assert child.proc.stdout
        self.children.append(child)
        listing.local.clear()  # this process stops reading ranges
        for _ in range(CHILD_WORKERS):
            listing.start_task(listing.read_in(child))
        listing.changed()
        loop = asyncio.get_running_loop()
        receive: asyncio.Future | None = None
        try:
            while True:
                receive = receive or asyncio.ensure_future(_receive(child.proc.stdout))
                left = REPLY_TIMEOUT - (loop.time() - child.heard)
                # not wait_for: on Python 3.11 it drops a cancel that comes with
                # the message, as when the listing closes
                await asyncio.wait([receive], timeout=max(left, 0.1))
                if not receive.done():
                    if not child.ranges or loop.time() - child.heard < REPLY_TIMEOUT:
                        continue  # idle, or busy for less than the timeout
                    await asyncio.wait([receive], timeout=1)  # a paused parent
                    if not receive.done():
                        logger.warning(
                            "Listing process %s is stuck: listing without it",
                            child.proc.pid,
                        )
                        return
                child.heard = loop.time()
                message, receive = await receive, None
                try:
                    if not await self._take(listing, child, message):
                        return
                except Exception as e:  # noqa: BLE001
                    listing.fail(e)  # a check failed: the listing does
                    return
        finally:
            if receive is not None:
                receive.cancel()

    async def _take(self, listing: "RangeListing", child: Child, message: Any) -> bool:
        """Acts on a child's message; False when the child reads no more."""
        kind, *args = message
        if kind == "rows":  # pages of ranges
            rows, count, pages = args
            for rid, last, token, age, listed in pages:
                listing.progress(child.ranges[rid][0], last, token, age)
                listing.listed_any = listing.listed_any or listed
            if count:
                await listing.emit(Rows(rows, count))
        elif kind == "split":  # sent before the rows of the page it follows
            rid, mid, age = args
            child.room -= 1
            key_range = child.ranges[rid][0]
            key_range.age = age
            listing._add(key_range, mid)
        elif kind == "seam":
            await listing._seam(*args)
        elif kind == "unordered":
            listing._out_of_order()
        elif kind == "unseeked":
            listing._start_ignored()
        elif kind == "done":
            (rid,) = args
            _, reply = child.ranges[rid]
            reply.set_result(("done", None))
        else:  # "failed": the parent reads on itself and raises what it hits
            rid, error = args
            logger.debug("Listing process failed a page:\n%s", error)
            key_range, reply = child.ranges[rid]
            reply.set_result(("failed", key_range.rest()))
            return False
        return True

    def _drop(self, child: Child) -> None:
        if child in self.children:
            self.children.remove(child)
        child.end("gone")
        if self.listing:
            if not self.children:
                self.listing.local.set()  # the parent reads ranges again
            self.listing.changed()

    def share_room(self, room: int) -> None:
        """Shares the room for more ranges among the children that read some:
        only they can split, and the shares add up to the room."""
        busy = [c for c in self.children if c.ranges] or self.children
        share, extra = divmod(max(room, 0), len(busy)) if busy else (0, 0)
        for i, child in enumerate(self.children):
            room_ = share + (i < extra) if child in busy else 0
            if child.room != room_ and child.send(("room", room_)):
                child.room = room_

    def close(self) -> None:
        self.closed = True
        for task in self.tasks:
            task.cancel()
        for proc in self.procs:
            _kill(proc)

    async def wait_closed(self) -> None:
        """Waits a few seconds at most for the children ``close`` killed to exit."""
        waits = [*self.tasks, *(asyncio.ensure_future(p.wait()) for p in self.procs)]
        if waits:
            await asyncio.wait(waits, timeout=5)
