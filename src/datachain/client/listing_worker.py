import asyncio
import copy
import logging
import os
import pickle
import signal
import struct
import sys
import threading
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from datachain.client.fsspec import RangeListedClient
    from datachain.client.ranges import Page

logger = logging.getLogger("datachain")


def _cpu_quotas(
    root: str = "/sys/fs/cgroup", membership: str = "/proc/self/cgroup"
) -> list[tuple[int, int]]:
    """``(quota, period)`` of each cgroup this process is in, and their parents."""
    quotas = []
    with suppress(OSError):
        with open(membership) as f:
            lines = f.read().splitlines()
        for line in lines:
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
    for quota, period in _cpu_quotas():
        cpus = min(cpus, quota // period)
    return max(cpus, 1)


LIST_PROCESSES = 0 if usable_cpus() < 2 else min(4, usable_cpus())
FRAME = struct.Struct("<Q")


class ChildDied(Exception):  # noqa: N818
    pass


def _pickles(value: Any) -> bool:
    try:
        pickle.dumps(value)
    except Exception:  # noqa: BLE001
        return False
    return True


def picklable(e: BaseException) -> BaseException:
    """``e``, or a copy of it that pickles: of its nearest class that pickles,
    with each attribute that does not pickle set to ``None``."""
    if _pickles(e):
        return e
    with suppress(Exception):
        err = copy.copy(e)
        if (cls := next(c for c in type(e).__mro__ if _pickles(c))) is not type(e):
            err.__class__ = cls
        for name, value in vars(err).items():
            if not _pickles(value):
                setattr(err, name, None)
        if _pickles(err):
            return err
    with suppress(Exception):
        return RuntimeError(f"{type(e).__name__}: {e}")
    return RuntimeError(type(e).__name__)


def dump(frame: tuple) -> bytes:
    data = pickle.dumps(frame, protocol=5)
    return FRAME.pack(len(data)) + data


def load(stream: BinaryIO) -> Any:
    header = stream.read(FRAME.size)
    if len(header) < FRAME.size:
        return None
    (size,) = FRAME.unpack(header)
    return pickle.loads(stream.read(size))  # noqa: S301


class Child:
    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.replies: dict[int, asyncio.Future] = {}
        self.died = asyncio.Event()

    async def start(self, spec: bytes) -> None:
        # the parent's imports, whatever the cwd holds
        code = f"import sys; sys.path[:] = {sys.path!r}; from {__name__} import main"
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            f"{code}; main()",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        self.send_raw(spec)
        kind, error = await self.read()
        if kind != "ready" or error is not None:
            raise ChildDied from error

    def send_raw(self, data: bytes) -> None:
        assert self.proc and self.proc.stdin
        if self.died.is_set() or self.proc.stdin.is_closing():
            self.kill()
            raise ChildDied
        self.proc.stdin.write(data)

    def send(self, *frame: Any) -> None:
        self.send_raw(dump(frame))

    async def read(self) -> Any:
        assert self.proc and self.proc.stdout
        (size,) = FRAME.unpack(await self.proc.stdout.readexactly(FRAME.size))
        return pickle.loads(await self.proc.stdout.readexactly(size))  # noqa: S301

    async def serve(self) -> None:
        try:
            while True:
                kind, rid, *rest = await self.read()
                if (reply := self.replies.pop(rid, None)) and not reply.done():
                    if kind == "page":
                        reply.set_result(rest)
                    else:
                        reply.set_exception(rest[0])
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.kill()

    async def list_page(self, rid: int, *request: Any) -> Any:
        self.send("page", rid, *request)
        reply = self.replies[rid] = asyncio.get_running_loop().create_future()
        try:
            return await reply
        finally:
            self.replies.pop(rid, None)

    def kill(self) -> None:
        self.died.set()
        for reply in self.replies.values():
            if not reply.done():
                reply.set_exception(ChildDied())
        self.replies.clear()
        if self.proc and self.proc.returncode is None:
            self.proc.kill()


class Pool:
    """Child processes fetching pages for one ``RangeListing``."""

    def __init__(self, client: "RangeListedClient", prefix: str, size: int = 0) -> None:
        try:
            self.spec = dump(
                (
                    type(client),
                    client.name,
                    client.fs_kwargs,
                    client.cache,
                    client.LIST_PAGE_SIZE,
                    prefix,
                    logger.getEffectiveLevel(),
                )
            )
        except Exception:
            logger.debug(
                "Listing in one process: client config does not pickle", exc_info=True
            )
            size = 0
        self.size = size
        self.children: list[Child] = []
        self.tasks: list[asyncio.Task] = []
        self.next_id = 0
        self.closed = self.warned = False

    def start(self) -> None:
        if not self.tasks:
            self.tasks = [asyncio.create_task(self._run()) for _ in range(self.size)]

    async def _run(self) -> None:
        child = Child()
        try:
            await child.start(self.spec)
            self.children.append(child)
            await child.serve()
        except (
            OSError,
            EOFError,
            ChildDied,
            NotImplementedError,
            pickle.UnpicklingError,
        ):
            logger.debug("Listing process failed", exc_info=True)
        except Exception:
            logger.warning("Listing process failed", exc_info=True)
        finally:  # also when cancelled while the child starts
            if child in self.children:
                self.children.remove(child)
            child.kill()
            self._warn_when_alone()

    def _warn_when_alone(self) -> None:
        running = [t for t in self.tasks if not t.done()]  # this one is still running
        if not (self.closed or self.warned or self.children or len(running) > 1):
            self.warned = True
            logger.warning("Listing in one process: no listing process runs")

    def ready(self) -> bool:
        return any(not c.died.is_set() for c in self.children)

    async def list_page(
        self, prefix: str, start_after: str, token: Any, hi: str | None
    ) -> "Page":
        """The page after ``token`` (or after ``start_after``) from a child;
        ``ChildDied`` when the child is gone before it replies."""
        from datachain.lib.file import FileRow

        from .ranges import Page

        live = [c for c in self.children if not c.died.is_set()]
        if not live:
            raise ChildDied
        child = min(live, key=lambda c: len(c.replies))
        rid = self.next_id = self.next_id + 1
        rows, first, last, token = await child.list_page(
            rid, prefix, start_after, token, hi
        )
        return Page(
            list(map(partial(tuple.__new__, FileRow), rows)), first, last, token
        )

    def close(self) -> None:
        self.closed = True
        for task in self.tasks:
            task.cancel()
        for child in self.children:
            child.kill()


async def _serve(
    client: "RangeListedClient", prefix: str, inp: BinaryIO, out: BinaryIO
) -> None:
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()

    def pump() -> None:  # blocking stdin reads, also on Windows
        while (frame := load(inp)) is not None:
            loop.call_soon_threadsafe(inbox.put_nowait, frame)
        loop.call_soon_threadsafe(inbox.put_nowait, None)

    def write(*frame: Any) -> None:
        try:
            out.write(dump(frame))
            out.flush()
        except Exception:  # noqa: BLE001  # the parent is gone or the reply does not pickle
            os._exit(0)

    async def run(rid: int, prefix: str, start_after: str, token: Any, hi) -> None:
        try:
            page = await client._list_page(
                prefix, start_after, token, client._keeps(hi), hi=hi
            )
        except (Exception, asyncio.CancelledError) as e:  # noqa: BLE001
            write("error", rid, picklable(e))
        else:
            # plain tuples: a row type would unpickle per row in Python
            write("page", rid, [tuple(r) for r in page.rows], *page[1:])

    tasks: set[asyncio.Task] = set()
    threading.Thread(target=pump, daemon=True).start()
    try:  # warm up DNS, TLS, credentials and the bucket region before taking work
        await client._probe(prefix, "")
    except Exception as e:
        logger.debug("Listing process warm-up failed", exc_info=True)
        write("ready", picklable(e))
        return
    write("ready", None)
    while (msg := await inbox.get()) is not None:
        _, rid, *request = msg
        tasks.add(task := asyncio.create_task(run(rid, *request)))
        task.add_done_callback(tasks.discard)
    for task in list(tasks):
        task.cancel()


def main() -> None:
    from fsspec.asyn import get_loop, sync

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent closes stdin
    out = os.fdopen(os.dup(1), "wb")
    inp = os.fdopen(os.dup(0), "rb")
    os.dup2(2, 1)  # stray prints must not corrupt the frames
    with open(os.devnull, "rb") as devnull:  # nor a credential process read them
        os.dup2(devnull.fileno(), 0)
    if (spec := load(inp)) is None:
        return
    cls, name, fs_kwargs, cache, page_size, prefix, level = spec
    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(level)
    client = cls(name, fs_kwargs, cache)
    client.LIST_PAGE_SIZE = page_size
    sync(get_loop(), _serve, client, prefix, inp, out)


if __name__ == "__main__":
    main()
