"""Child processes that fetch and parse listing pages; the parent owns the ranges."""

import asyncio
import atexit
import itertools
import logging
import os
import pickle
import signal
import struct
import sys
import threading
import traceback
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, Any, BinaryIO

from datachain.lib.file import FileRow

from .ranges import Page

if TYPE_CHECKING:
    from datachain.client.fsspec import RangeListedClient

logger = logging.getLogger("datachain")


def _cpu_quotas(
    root: str = "/sys/fs/cgroup", membership: str = "/proc/self/cgroup"
) -> list[tuple[int, int]]:
    """``(quota, period)`` of each cgroup this process is in, and their parents."""
    quotas = []
    with suppress(OSError):
        for line in _read_text(membership).splitlines():
            _, controllers, path = line.split(":", 2)
            if not controllers:  # v2
                base, files = root, ["cpu.max"]
            elif "cpu" in controllers.split(","):  # v1
                base = os.path.join(root, controllers)
                files = ["cpu.cfs_quota_us", "cpu.cfs_period_us"]
            else:
                continue
            parts = [p for p in path.split("/") if p]
            for depth in range(len(parts) + 1):
                folder = os.path.join(base, *parts[:depth])
                with suppress(OSError, ValueError):
                    quota, period = " ".join(
                        _read_text(os.path.join(folder, name)) for name in files
                    ).split()
                    if quota not in ("max", "-1"):
                        quotas.append((int(quota), int(period)))
    return quotas


def _read_text(path: str) -> str:
    with open(path) as f:
        return f.read()


def usable_cpus() -> int:
    """The CPUs this process may run on, within its cgroups' CPU quotas."""
    if hasattr(os, "process_cpu_count"):
        cpus = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1
    return max(min([cpus, *(quota // period for quota, period in _cpu_quotas())]), 1)


# a frozen app's executable starts the app, not Python's "-c"
_SPAWNS = bool(sys.executable) and not getattr(sys, "frozen", False)
LIST_PROCESSES = min(4, cpus) if _SPAWNS and (cpus := usable_cpus()) > 1 else 0
HEADER = struct.Struct("<Q")  # the size of the pickle that follows
# a child that doesn't answer a page in time is stuck (stopped, deadlocked): the
# parent kills it and lists the page itself; a page normally takes under a second
REPLY_TIMEOUT = float(os.environ.get("DATACHAIN_LISTING_REPLY_TIMEOUT", "300"))
_as_file_row = partial(tuple.__new__, FileRow)  # skips FileRow's argument handling


def dump(message: Any) -> bytes:
    data = pickle.dumps(message, protocol=5)
    return HEADER.pack(len(data)) + data


def load(stream: BinaryIO) -> Any:
    header = stream.read(HEADER.size)
    if len(header) < HEADER.size:
        return None
    (size,) = HEADER.unpack(header)
    return pickle.loads(stream.read(size))  # noqa: S301


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
from {__name__} import main
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


class Pool:
    """Child processes fetching pages for one ``RangeListing``. The parent fetches
    any page a child fails or loses, so errors reach the user as raised."""

    def __init__(self, client: "RangeListedClient", prefix: str, size: int) -> None:
        # the page size travels so that tests can shrink it in children too
        spec = (type(client), client.name, client.fs_kwargs, client.cache)
        try:
            self.spec = dump((*spec, client.LIST_PAGE_SIZE, prefix))
        except Exception:
            logger.debug(
                "Listing in one process: client config does not pickle", exc_info=True
            )
            size = 0
        self.size = size
        self.procs: list[asyncio.subprocess.Process] = []
        # children that took work, each with its pending replies by request id
        self.children: dict[asyncio.subprocess.Process, dict[int, asyncio.Future]] = {}
        self.tasks: list[asyncio.Task] = []
        self.ids = itertools.count()
        self.closed = False

    def start(self) -> None:
        if not self.tasks:
            self.tasks = [asyncio.create_task(self._run()) for _ in range(self.size)]

    async def _run(self) -> None:
        proc = None
        replies: dict[int, asyncio.Future] = {}
        try:
            self.procs.append(proc := await _spawn())
            assert proc.stdin
            assert proc.stdout
            proc.stdin.write(self.spec)
            if error := await _receive(proc.stdout):
                logger.debug("Listing process cannot list:\n%s", error)
                return
            self.children[proc] = replies
            while True:
                rid, page = await _receive(proc.stdout)
                if (reply := replies.pop(rid, None)) and not reply.done():
                    reply.set_result(page)
                if page is None:  # it failed a page: the parent lists without it
                    return
        except (OSError, EOFError, NotImplementedError, pickle.UnpicklingError):
            logger.debug("Listing process failed", exc_info=True)  # exited or never ran
        except Exception:
            logger.warning("Listing process failed", exc_info=True)
        finally:  # also when cancelled
            if proc is not None:
                self.children.pop(proc, None)
                _kill(proc)
            for reply in replies.values():
                if not reply.done():
                    reply.set_result(None)
            running = [t for t in self.tasks if not t.done()]  # this one included
            if not (self.closed or self.children or len(running) > 1):
                logger.warning("Listing in one process: no listing process runs")

    async def list_page(
        self, start_after: str, token: Any, hi: str | None
    ) -> Page | None:
        """The page after ``token`` (or after ``start_after``) from the least busy
        child; None when no child runs, or the child fails or dies first."""
        if not self.children:
            return None
        proc, replies = min(self.children.items(), key=lambda c: len(c[1]))
        assert proc.stdin
        if proc.stdin.is_closing():
            return None
        rid = next(self.ids)
        replies[rid] = reply = asyncio.get_running_loop().create_future()
        try:
            proc.stdin.write(dump((rid, start_after, token, hi)))
            # not wait_for: on Python 3.11 it drops a cancel that comes with the
            # reply, as when the pool closes, and the cancelled listing hangs
            await asyncio.wait([reply], timeout=REPLY_TIMEOUT)
            if not reply.done():  # a parent paused that long reads its replies first
                await asyncio.wait([reply], timeout=1)
        finally:
            replies.pop(rid, None)
        if not reply.done():
            self._drop(proc, replies)
            return None
        if (page := reply.result()) is None:
            return None
        rows, *rest = page
        return Page(list(map(_as_file_row, rows)), *rest)

    def _drop(self, proc: asyncio.subprocess.Process, replies: dict) -> None:
        """Kills a stuck child; its pending pages come back None now, as the kill
        may not end it (uninterruptible sleep) and it takes no new ones."""
        if self.children.pop(proc, None) is not None:
            logger.warning("Listing process %s is stuck: listing without it", proc.pid)
        _kill(proc)
        for reply in replies.values():
            if not reply.done():
                reply.set_result(None)

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


def main(out_fd: int) -> None:
    """A child's entry: the spec, then requests, on stdin; replies on ``out_fd``."""
    from fsspec.asyn import get_loop, sync

    out = os.fdopen(out_fd, "wb")
    inp = os.fdopen(os.dup(0), "rb")
    with open(os.devnull, "rb") as devnull:  # a credential process must not read it
        os.dup2(devnull.fileno(), 0)
    try:
        if (spec := load(inp)) is not None:
            cls, name, fs_kwargs, cache, page_size, prefix = spec
            client = cls(name, fs_kwargs, cache)
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

    async def reply(rid: int, start_after: str, token: Any, hi: str | None) -> None:
        # None, also when a page fails to pickle: the parent fetches it and raises
        # a lasting error; every request gets an answer
        answer = None
        with suppress(Exception):
            page = await client._list_page(prefix, start_after, token, hi)
            # plain tuples: a row type would unpickle per row in Python
            answer = dump((rid, ([tuple(r) for r in page.rows], *page[1:])))
        write_frame(answer or dump((rid, None)))

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
    tasks: set[asyncio.Task] = set()
    while (request := await requests.get()) is not None:
        tasks.add(task := asyncio.create_task(reply(*request)))
        task.add_done_callback(tasks.discard)
    for task in list(tasks):
        task.cancel()
