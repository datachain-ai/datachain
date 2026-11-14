import asyncio
import io
import itertools
import os
import pickle
import random
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from typing import ClassVar
from unittest import mock

import pytest
from fsspec.asyn import get_loop, sync

import datachain
from datachain.client import listing_worker
from datachain.client.fsspec import Client, RangeListedClient
from datachain.client.listing_worker import (
    Child,
    Pool,
    _command,
    _receive,
    _serve,
    dump,
    load,
)
from datachain.client.ranges import Keep, KeyRange, Page, RangeListing, code_points
from datachain.lib.file import FileRow

ROOT = os.path.dirname(os.path.dirname(datachain.__file__))


class FakeClient:
    uri = "fake://bucket"
    key_order = staticmethod(str)

    def __init__(self, out, fail=False):
        self.out, self.fail, self.probed_before_ready = out, fail, None
        self.probed = None

    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        if page_size is None:  # a page, not the warm-up: this fake fails it
            raise PermissionError("no list access")
        self.probed_before_ready = not self.out.getvalue()
        self.probed = prefix, start_after, page_size
        if self.fail:
            raise PermissionError("no list access")


def serve(client, out, *requests, then=b""):
    """Sends ``requests``, then ``then``; the parent goes half a second later."""
    read, write = os.pipe()
    os.write(write, b"".join(map(dump, requests)) + then)
    threading.Timer(0.5, os.close, [write]).start()
    with os.fdopen(read, "rb") as inp:
        asyncio.run(_serve(client, "dir/", inp, out))
    out.seek(0)
    return [load(out) for _ in range(1 + len(requests))]


def test_child_warms_up_before_it_reports_ready():
    out = io.BytesIO()
    client = FakeClient(out)

    assert serve(client, out) == [None]
    assert client.probed_before_ready
    assert client.probed == ("dir/", "", 1)


def test_frames_round_trip():
    message = ("rows", 1, [("s3://b", "k")], "k", None, 1, True)
    assert load(io.BytesIO(dump(message))) == message


def test_a_child_reports_a_range_it_fails_and_the_parent_reads_on():
    out = io.BytesIO()

    ready, failed = serve(FakeClient(out), out, ("range", 7, "", None, 0, None, None))

    assert ready is None
    assert failed[:2] == ("failed", 7)
    assert "PermissionError: no list access" in failed[2]


class UnpicklableClient(FakeClient):
    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        if page_size is None:  # a page whose rows don't pickle
            return Page(
                [FileRow("s3://b", "a", 1, "", "", True, lambda: None, None)],
                "a",
                "a",
                None,
            )


@pytest.mark.parametrize("batch_rows", [1, 5000])
def test_a_child_fails_a_range_whose_rows_do_not_pickle(mocker, batch_rows):
    mocker.patch("datachain.client.listing_child.BATCH_ROWS", batch_rows)
    out = io.BytesIO()
    client = UnpicklableClient(out)

    _, failed = serve(client, out, ("range", 7, "", None, 0, None, None))

    assert failed[:2] == ("failed", 7)
    assert "pickle" in failed[2]


def test_a_child_ends_on_a_request_it_cannot_read():
    out = io.BytesIO()
    read, write = os.pipe()
    os.write(write, listing_worker.HEADER.pack(3) + b"bad")
    threading.Timer(3, os.close, [write]).start()  # the parent stays a while
    start = time.monotonic()

    with os.fdopen(read, "rb") as inp:
        asyncio.run(_serve(FakeClient(out), "dir/", inp, out))

    assert time.monotonic() - start < 2
    assert load(io.BytesIO(out.getvalue())) is None


class WarmingClient(FakeClient):
    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        await asyncio.sleep(3600)


def test_a_child_ends_when_the_parent_goes_during_its_warm_up():
    out = io.BytesIO()
    read, write = os.pipe()
    threading.Timer(0.2, os.close, [write]).start()

    with os.fdopen(read, "rb") as inp:
        asyncio.run(_serve(WarmingClient(out), "dir/", inp, out))

    assert not out.getvalue()


def test_a_print_in_the_frames_ends_the_child():
    async def receive():
        stream = asyncio.StreamReader()
        stream.feed_data(b"hello from sitecustomize\n" + dump(None))
        return await _receive(stream)

    with pytest.raises(pickle.UnpicklingError):
        asyncio.run(receive())


class KeysClient(RangeListedClient):
    """Lists (key, version) pairs in key order, in pages of 4."""

    PREFIX = "fake://"
    protocol = "fake"
    LIST_PAGE_SIZE = 4

    def __init__(self, keys: list[tuple[str, str]], delay=None) -> None:
        super().__init__("bucket", {}, None)  # type: ignore[arg-type]
        self.keys, self.delay, self.requests = keys, delay, 0

    def info_to_file(self, v, path):
        raise NotImplementedError

    async def _list_page(
        self, prefix, start_after, token, hi=None, page_size=None, delimiter=""
    ) -> Page:
        self.requests += 1
        if self.delay:
            await asyncio.sleep(self.delay())
        if token is None:
            token = next(
                (i for i, (k, _) in enumerate(self.keys) if k > start_after),
                len(self.keys),
            )
        chunk = self.keys[token : token + (page_size or self.LIST_PAGE_SIZE)]
        end = token + len(chunk)
        keep = Keep(self, start_after, hi)
        return Page(
            [FileRow("fake://bucket", k, 1, v) for k, v in chunk if keep(k)],
            chunk[0][0] if chunk else None,
            max((k for k, _ in chunk), default=None),
            end if end < len(self.keys) else None,
            keep.past_lo,
            keep.past_hi,
        )


def versions_client(versions: int) -> KeysClient:
    """One key with many versions, then one more key."""
    return KeysClient([("a", f"v{i:06}") for i in range(versions)] + [("b", "v0")])


def slow_client(seed: int) -> KeysClient:
    """Keys with a few versions each; every request takes a random while."""
    rng = random.Random(seed)  # noqa: S311
    names = {
        "".join(rng.choices("ab/0123456789", k=rng.randint(1, 6))) for _ in range(300)
    }
    keys = [
        (n, f"v{v}") for n in sorted(names) for v in range(rng.choice([1, 1, 1, 3, 9]))
    ]
    return KeysClient(keys, lambda: rng.random() / 200)


@pytest.mark.parametrize("seed", range(40))
def test_ranges_that_split_while_pages_are_in_flight_are_exact(seed, mocker):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 8)
    mocker.patch("datachain.client.ranges.YOUNG_PAGES", 1)
    client = slow_client(seed)

    pages = _listed(client, processes=False)

    expected = [k for k in client.keys if client._is_valid_key(k[0])]
    assert sorted((r.path, r.version) for p in pages for r in p) == expected


def _listed(client: Client, processes: bool = True) -> list[list]:
    async def pages():
        return [page async for page in client.scandir("")]

    with mock.patch.object(listing_worker, "LIST_PROCESSES", 2 if processes else 0):
        return sync(get_loop(), pages)


class LoopingClient(KeysClient):
    """Returns the page token it was given."""

    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        page = await super()._list_page(prefix, start_after, None, hi, page_size)
        return page._replace(token=0)


def test_a_page_token_that_repeats_fails_the_listing():
    client = LoopingClient([(f"k{i:02}", "v") for i in range(20)])

    with pytest.raises(RuntimeError, match="returned the page token it was given"):
        _listed(client, processes=False)


@pytest.mark.parametrize("processes", [False, True])
def test_versions_of_one_key_take_linear_work_and_bounded_memory(processes, mocker):
    async def starting():
        await asyncio.sleep(3600)

    mocker.patch.object(listing_worker, "_spawn", starting)
    pages = {}
    for versions in (100, 200):
        client = versions_client(versions)
        pages[versions] = _listed(client, processes)
        rows = [(r.path, r.version) for page in pages[versions] for r in page]
        assert sorted(rows) == sorted(client.keys)
        assert client.requests <= versions // 4 + 2

    assert max(len(page) for page in pages[200]) <= KeysClient.LIST_PAGE_SIZE


def test_short_listing_takes_a_request_per_page_and_no_children(mocker):
    start = mocker.spy(Pool, "start")
    client = versions_client(4)  # 5 keys in 4-key pages

    assert sum(map(len, _listed(client))) == 5
    assert client.requests == 2
    assert not start.call_count


@pytest.fixture(autouse=True)
def fake_groups(monkeypatch):
    """A fake child's pid is -2 or lower, and a signal to a pid of 1 or lower
    reaches init or every process: such a signal reaches the fake instead, or
    nobody."""

    def guard(real):
        def guarded(pid, sig):
            if pid in Proc.fakes:
                Proc.fakes[pid].kill()
            elif pid <= 1:
                raise ProcessLookupError(pid)
            else:
                real(pid, sig)

        return guarded

    for name in ("killpg", "kill"):
        if real := getattr(os, name, None):
            monkeypatch.setattr(os, name, guard(real))


class Proc:
    """A child process: its stdout is fed by the test."""

    pids = itertools.count(-2, -1)
    fakes: ClassVar[dict] = {}

    def __init__(self, closing=False):
        self.pid = next(Proc.pids)  # see fake_groups
        Proc.fakes[self.pid] = self
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.stdin = mock.Mock(**{"is_closing.return_value": closing})

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


def test_a_child_with_a_broken_pipe_gives_its_range_back():
    async def check():
        child = Child(Proc(closing=True))  # type: ignore[arg-type]
        key_range = KeyRange("a", "m", code_points, 2, "c", 5)

        outcome, rest = await child.read(key_range)

        assert outcome == "gone"
        assert (rest.lo, rest.hi, rest.age, rest.last, rest.token) == (
            "a", "m", 2, "c", 5,
        )  # fmt: skip
        assert not child.ranges

    asyncio.run(check())


def test_a_gone_childs_ranges_come_back_with_what_is_left():
    async def check():
        child = Child(Proc())  # type: ignore[arg-type]
        read = KeyRange("a", "m", code_points, 0)
        unread = KeyRange("m", None, code_points, 0)
        done = KeyRange("x", None, code_points, 0)
        tasks = [asyncio.create_task(child.read(r)) for r in (read, unread, done)]
        await asyncio.sleep(0)
        read.last, read.token = "c", 5  # as the child's pages said
        done.last, done.token, done.ended = "z", None, True

        child.end("gone")

        (o1, r1), (o2, r2), (o3, r3) = await asyncio.gather(*tasks)
        assert (o1, r1.lo, r1.last, r1.token) == ("gone", "a", "c", 5)
        assert (o2, r2.lo, r2.last, r2.token) == ("gone", "m", None, None)
        assert (o3, r3) == ("gone", None)

    asyncio.run(check())


class ThreadProc:
    """A child process run as threads of this one: ``_serve`` on real pipes with
    the test's client, so the listing runs the real protocol."""

    def __init__(self, client, prefix: str, loop) -> None:
        self.pid = next(Proc.pids)  # see fake_groups
        Proc.fakes[self.pid] = self
        self.returncode: int | None = None
        self.closed = False
        self.loop = loop
        to_child, self.to_child = os.pipe()
        self.from_child, from_child = os.pipe()
        self.stdin = mock.Mock(write=self._write, is_closing=lambda: self.closed)
        self.stdout = asyncio.StreamReader()
        inp, out = os.fdopen(to_child, "rb"), os.fdopen(from_child, "wb")

        def run() -> None:
            with inp, out:
                if load(inp) is not None:  # the spec
                    asyncio.run(_serve(client, prefix, inp, out))

        threading.Thread(target=run, name="listing-child", daemon=True).start()
        threading.Thread(target=self._feed, daemon=True).start()

    def _write(self, data: bytes) -> None:
        if not self.closed:
            os.write(self.to_child, data)

    def _feed(self) -> None:
        """The child's frames to the parent; after a kill they are dropped, but
        still read, so that the child's writes never fail."""
        with os.fdopen(self.from_child, "rb", buffering=0) as frames:
            while chunk := frames.read(1 << 16):
                self.loop.call_soon_threadsafe(self._deliver, chunk)
        self.loop.call_soon_threadsafe(self._deliver, b"")

    def _deliver(self, chunk: bytes) -> None:
        if self.returncode is None and not self.stdout.at_eof():
            if chunk:
                self.stdout.feed_data(chunk)
            else:
                self.stdout.feed_eof()

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9
            self.closed = True
            os.close(self.to_child)  # the child sees the parent go
            self.stdout.feed_eof()

    async def wait(self):
        return self.returncode


@pytest.fixture
def thread_children(mocker):
    """Children as threads of this process, which read with the listing's client."""
    procs: list[ThreadProc] = []

    def use(client, kill_after=None):
        async def spawn():
            procs.append(proc := ThreadProc(client, "", asyncio.get_running_loop()))
            return proc

        mocker.patch.object(listing_worker, "_spawn", spawn)
        mocker.patch("datachain.client.ranges.POOL_ROWS", 0)
        mocker.patch("datachain.client.ranges.CHILD_WORKERS", 4)
        mocker.patch("datachain.client.listing_worker.CHILD_WORKERS", 4)
        if kill_after is not None:
            receive = listing_worker._receive

            async def received(stream):
                message = await receive(stream)
                if message and message[0] == "rows" and kill_after():
                    next(p for p in procs if p.stdout is stream).kill()
                return message

            mocker.patch.object(listing_worker, "_receive", received)
        return procs

    return use


def _rows_from_children(mocker) -> list:
    got: list = []
    take = Pool._take

    async def recorded(self, listing, child, message):
        if message[0] == "rows":
            got.extend(pickle.loads(message[1]))  # noqa: S301
        return await take(self, listing, child, message)

    mocker.patch.object(Pool, "_take", recorded)
    return got


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("deaths", [False, True])
def test_ranges_read_by_children_are_exact(seed, deaths, thread_children, mocker):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)
    mocker.patch("datachain.client.ranges.YOUNG_PAGES", 1)
    client = slow_client(seed)
    rng = random.Random(seed)  # noqa: S311
    procs = thread_children(client, (lambda: rng.random() < 0.05) if deaths else None)
    from_children = _rows_from_children(mocker)
    ready = asyncio.Event()
    serve, list_page = Pool._serve, RangeListing._list_page

    async def serving(self, child):
        ready.set()
        await serve(self, child)

    async def listing(self, *args):
        if self.pool.tasks:
            await asyncio.wait_for(ready.wait(), timeout=10)
        return await list_page(self, *args)

    mocker.patch.object(Pool, "_serve", serving)
    mocker.patch.object(RangeListing, "_list_page", listing)

    pages = _listed(client)

    expected = [k for k in client.keys if client._is_valid_key(k[0])]
    assert sorted((r.path, r.version) for p in pages for r in p) == expected
    assert procs
    if not deaths:
        assert from_children


class ChildFailingClient(KeysClient):
    """Fails pages a listing child reads; with ``everywhere``, every page."""

    def __init__(self, keys, everywhere=False):
        super().__init__(keys)
        self.everywhere = everywhere

    async def _list_page(
        self, prefix, start_after, token, hi=None, page_size=None, delimiter=""
    ) -> Page:
        in_child = threading.current_thread().name == "listing-child"
        if page_size is None and (self.everywhere or in_child) and self.requests > 6:
            raise PermissionError("no list access")
        return await super()._list_page(prefix, start_after, token, hi, page_size)


def test_the_parent_reads_on_where_a_child_failed(thread_children):
    client = ChildFailingClient([(f"k{i:03}", "v") for i in range(200)])
    procs = thread_children(client)

    assert sorted(r.path for p in _listed(client) for r in p) == [
        k for k, _ in client.keys
    ]
    assert procs
    assert all(p.returncode == -9 for p in procs)


def test_a_page_that_fails_everywhere_fails_the_listing_as_raised(thread_children):
    client = ChildFailingClient([(f"k{i:03}", "v") for i in range(200)], True)
    thread_children(client)

    with pytest.raises(PermissionError, match="no list access"):
        _listed(client)


class SilentProc(Proc):
    """A child that is ready, then says nothing: stopped or stuck."""

    def __init__(self):
        super().__init__()
        self.stdout.feed_data(dump(None))


def test_a_stuck_child_is_killed_and_the_parent_reads_its_ranges(mocker, caplog):
    mocker.patch.object(listing_worker, "REPLY_TIMEOUT", 0.1)
    mocker.patch("datachain.client.ranges.POOL_ROWS", 0)
    procs: list = []

    async def spawn():
        procs.append(proc := SilentProc())
        return proc

    mocker.patch.object(listing_worker, "_spawn", spawn)
    client = KeysClient([(f"k{i:03}", "v") for i in range(200)])

    with caplog.at_level("WARNING", logger="datachain"):
        assert sorted(r.path for p in _listed(client) for r in p) == [
            k for k, _ in client.keys
        ]

    assert procs
    assert all(p.returncode == -9 for p in procs)
    assert "is stuck: listing without it" in caplog.text


def test_a_paused_parent_reads_a_message_that_came_meanwhile(mocker):
    mocker.patch.object(listing_worker, "REPLY_TIMEOUT", 0.1)

    async def check():
        client = versions_client(1)
        pool = Pool(client, "", 1)
        listing = RangeListing(client, "", asyncio.Queue(), pool)
        proc = Proc()
        child = Child(proc)  # type: ignore[arg-type]
        serving = asyncio.create_task(pool._serve(child))
        while not child.ranges:  # noqa: ASYNC110
            await asyncio.sleep(0)
        await asyncio.sleep(0.3)  # past the timeout: the parent was paused
        (rid,) = child.ranges
        proc.stdout.feed_data(dump(("done", rid)))
        while child.ranges:  # noqa: ASYNC110
            await asyncio.sleep(0)

        assert not serving.done()
        assert child in pool.children
        for task in [serving, *listing.tasks]:
            task.cancel()
        await asyncio.gather(serving, *listing.tasks, return_exceptions=True)

    asyncio.run(check())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_child_whose_group_refuses_the_signal_is_killed_without_a_poll(mocker):
    # after SIGKILL and before asyncio reaps it, macOS answers killpg with EPERM
    mocker.patch.object(os, "killpg", side_effect=PermissionError(1, "zombie"))
    kill = mocker.patch.object(os, "kill")
    proc = mock.Mock(pid=12345, returncode=None)

    listing_worker._kill(proc)

    kill.assert_called_once_with(12345, signal.SIGKILL)
    proc.kill.assert_not_called()  # its poll() would reap the child


# a module of its own: another process imports it by name, which a test module
# in a "tests" package shared by several repos can't promise
CLIENTS = """
import asyncio

from datachain.client.fsspec import RangeListedClient
from datachain.client.ranges import Page


class IdleClient(RangeListedClient):
    \"\"\"A client a child can rebuild from its spec; every page is empty.\"\"\"

    PREFIX = "fake://"
    protocol = "fake"

    def info_to_file(self, v, path):
        raise NotImplementedError

    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        return Page([], None, None, None)


class WarmingIdleClient(IdleClient):
    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        await asyncio.sleep(3600)


class ThreadIdleClient(IdleClient):  # its warm-up blocks a thread, as DNS can

    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        import time

        await asyncio.get_running_loop().run_in_executor(None, time.sleep, 600)
"""


@pytest.fixture
def clients(tmp_path, monkeypatch):
    (tmp_path / "listing_test_clients.py").write_text(CLIENTS)
    monkeypatch.syspath_prepend(str(tmp_path))
    import listing_test_clients  # type: ignore[import-not-found]

    yield listing_test_clients
    sys.modules.pop("listing_test_clients", None)


PARENT = """
import asyncio, os, signal, sys
import listing_test_clients
from datachain.client.listing_worker import Pool
from datachain.client.ranges import RangeListing

async def main(client, stop):
    lister = getattr(listing_test_clients, client)("bucket", {}, None)
    pool = Pool(lister, "", 2)
    RangeListing(lister, "", asyncio.Queue(), pool)
    pool.start()
    if client == "IdleClient":
        while len(pool.children) < 2:
            await asyncio.sleep(0.05)
    else:  # the children are warming up
        await asyncio.sleep(2)
    for proc in pool.procs:
        if stop:
            os.kill(proc.pid, signal.SIGSTOP)
    print(" ".join(str(p.pid) for p in pool.procs), flush=True)
    await asyncio.sleep(3600)

asyncio.run(main(sys.argv[1], sys.argv[2] == "stop"))
"""


@pytest.mark.parametrize(
    "client,stop",
    [
        ("IdleClient", ""),
        ("WarmingIdleClient", ""),
        pytest.param(
            "IdleClient",
            "stop",
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="only Linux kills an orphan"
            ),
        ),
    ],
)
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_children_exit_when_the_parent_is_killed(client, stop, clients):
    parent = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", PARENT, client, stop],
        stdout=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([os.path.dirname(clients.__file__), ROOT]),
        },
        text=True,
    )
    assert parent.stdout
    pids = [int(p) for p in parent.stdout.readline().split()]
    parent.kill()
    parent.wait()
    parent.stdout.close()

    assert len(pids) == 2
    assert not _alive_after(pids, 20)


def _alive_after(pids: list[int], timeout: float) -> list[int]:
    deadline = time.monotonic() + timeout
    alive = pids
    while alive and time.monotonic() < deadline:
        time.sleep(0.1)
        alive = [p for p in pids if _running(p)]
    return alive


GRANDPARENT = """
import subprocess, sys, time
grandchild = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(600)"], stdout=subprocess.DEVNULL
)
print(grandchild.pid, flush=True)
time.sleep(600)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_killed_child_takes_what_it_started_along():
    async def check():
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            GRANDPARENT,
            stdout=asyncio.subprocess.PIPE,
            start_new_session=True,  # as _spawn does
        )
        assert proc.stdout
        grandchild = int(await proc.stdout.readline())
        listing_worker._kill(proc)
        await proc.wait()
        return grandchild

    assert not _alive_after([asyncio.run(check())], 10)


def test_a_closed_pool_waits_for_its_children_to_exit(clients):
    async def ready(pool):
        while len(pool.children) < 2:  # noqa: ASYNC110
            await asyncio.sleep(0.05)

    async def check():
        lister = clients.IdleClient("bucket", {}, None)
        pool = Pool(lister, "", 2)
        RangeListing(lister, "", asyncio.Queue(), pool)
        pool.start()
        await asyncio.wait_for(ready(pool), 60)
        pool.close()
        await pool.wait_closed()
        return [proc.returncode for proc in pool.procs]

    returncodes = sync(get_loop(), check)  # the loop a listing runs on: it stays open
    assert len(returncodes) == 2
    assert None not in returncodes


def _running(pid: int) -> bool:
    """Whether ``pid`` is a live process, not a zombie."""
    state = subprocess.run(  # noqa: S603
        ["ps", "-o", "stat=", "-p", str(pid)],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_child_reports_why_it_cannot_list():
    out = io.BytesIO()

    (error,) = serve(FakeClient(out, fail=True), out)

    assert "PermissionError: no list access" in error


def child(stdin: bytes, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        _command(),
        input=stdin,
        capture_output=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": ROOT},
        **kwargs,
    )


def test_child_without_a_spec_exits_quietly():
    out = child(b"")

    assert (out.returncode, out.stdout, out.stderr) == (0, b"", b"")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_child_ignores_ctrl_c_while_it_imports():
    proc = subprocess.Popen(  # noqa: S603
        _command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": ROOT},
    )
    time.sleep(0.2)
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(b"", timeout=60)

    assert (proc.returncode, err) == (0, b"")


def test_s3_connections_cover_the_listing_requests_in_flight():
    from datachain.client.ranges import PROCESS_LIST_WORKERS
    from datachain.client.s3 import ClientS3

    fs = ClientS3.create_fs(anon=True, skip_instance_cache=True)
    mine = ClientS3.create_fs(
        anon=True, config_kwargs={"max_pool_connections": 5}, skip_instance_cache=True
    )

    assert fs.config_kwargs["max_pool_connections"] >= PROCESS_LIST_WORKERS
    assert mine.config_kwargs["max_pool_connections"] == 5


def test_child_reports_its_warm_up_error_to_the_parent(tmp_path):
    from datachain.client.local import FileClient

    out = child(dump((FileClient, str(tmp_path), {}, None, 4, "")))

    assert out.returncode == 0
    assert "Traceback" in load(io.BytesIO(out.stdout))


class CancelLosingClient(KeysClient):
    """Loses the first cancel, as a library can."""

    entered: asyncio.Event

    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        self.entered.set()
        with suppress(asyncio.CancelledError):
            await asyncio.sleep(3600)
        await asyncio.sleep(3600)


def test_a_listing_ends_when_a_request_loses_its_cancel():
    async def check():
        client = CancelLosingClient([("a", "v")])
        client.entered = asyncio.Event()
        listing = RangeListing(client, "", asyncio.Queue(), None)
        task = asyncio.create_task(listing.run())
        await client.entered.wait()
        task.cancel()
        await asyncio.wait([task], timeout=10)
        return task.cancelled()

    assert asyncio.run(check())


def test_a_listing_ended_by_a_stray_cancel_ends_its_reader(tmp_path):
    from datachain.client.local import FileClient

    client = FileClient(str(tmp_path), {}, None)

    async def stray(start_prefix, result_queue):
        raise asyncio.CancelledError  # not a cancel of the listing

    client._fetch_stray = stray  # type: ignore[attr-defined]

    async def read():
        try:
            async for _ in client.scandir("", "stray"):
                pass
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():  # type: ignore[union-attr]
                raise
            return "ended"

    assert sync(get_loop(), asyncio.wait_for, read(), 10) == "ended"


class LoopingProbeClient(KeysClient):
    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        return Page([], None, None, "again")


def test_a_probe_that_gets_its_token_back_fails():
    client = LoopingProbeClient([])

    with pytest.raises(RuntimeError, match="returned the page token it was given"):
        sync(get_loop(), client._probe, "", "a")


@pytest.mark.parametrize("cpus,processes", [(1, 0), (2, 2), (4, 4), (8, 8), (32, 8)])
def test_one_child_per_cpu_up_to_eight(cpus, processes):
    assert listing_worker.list_processes(cpus) == processes


def test_a_frozen_app_lists_in_one_process():
    code = (
        "import sys; sys.frozen = True; "
        "from datachain.client import listing_worker; "
        "print(listing_worker.LIST_PROCESSES)"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": ROOT},
    )

    assert out.stdout.strip() == "0"


@pytest.mark.skipif(sys.platform != "linux", reason="ctypes loads on Linux only")
def test_a_ctypes_module_in_the_cwd_does_not_run_in_a_child(tmp_path, clients):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "ctypes.py").write_text("raise SystemExit(3)")
    out = child(dump((clients.IdleClient, ("bucket", {}, None), 4, "")), cwd=cwd)

    assert out.returncode == 0
    assert load(io.BytesIO(out.stdout)) is None  # ready


def test_child_reports_why_it_cannot_build_its_client():
    out = child(dump((int, ("bucket", {}, None), 4, "")))  # int() takes no 3 args

    assert out.returncode == 0
    assert "TypeError" in load(io.BytesIO(out.stdout))
    assert not out.stderr


def test_child_ends_when_the_parent_goes_while_a_thread_is_stuck(clients):
    spec = (clients.ThreadIdleClient, ("bucket", {}, None), 4, "")
    proc = subprocess.Popen(  # noqa: S603
        _command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": ROOT},
    )
    assert proc.stdin
    assert proc.stdout
    proc.stdin.write(dump(spec))
    proc.stdin.flush()
    time.sleep(1)  # the warm-up's thread sleeps
    proc.stdin.close()  # the parent is gone

    assert proc.wait(timeout=20) == 0
    proc.stdout.close()


EXITING = """
import asyncio, os, signal
from fsspec.asyn import get_loop, sync
import listing_test_clients
from datachain.client.listing_worker import Pool
from datachain.client.ranges import RangeListing

async def start():
    lister = listing_test_clients.IdleClient("bucket", {}, None)
    pool = Pool(lister, "", 2)
    RangeListing(lister, "", asyncio.Queue(), pool)
    pool.start()
    while len(pool.children) < 2:
        await asyncio.sleep(0.05)
    for proc in pool.procs:
        os.kill(proc.pid, signal.SIGSTOP)  # it can't see the parent exit
    print(" ".join(str(p.pid) for p in pool.procs), flush=True)
    return pool

pool = sync(get_loop(), start)  # the exit leaves the pool open on fsspec's loop
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_children_die_when_the_parent_exits_with_the_listing_open(clients):
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", EXITING],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([os.path.dirname(clients.__file__), ROOT]),
        },
    )
    pids = [int(p) for p in out.stdout.split()]

    assert len(pids) == 2
    assert not _alive_after(pids, 20)


def test_children_report_keys_out_of_order_after_a_split(thread_children, mocker):
    from tests.unit.test_client import _ReorderingClient

    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)
    client = _ReorderingClient(reversed_from="020")
    thread_children(client)

    with pytest.raises(RuntimeError, match="out of order"):
        _listed(client)


def test_children_report_keys_skipped_past_a_start_key(thread_children, mocker):
    from tests.unit.test_client import _OverSeekingClient

    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)
    mocker.patch("datachain.client.ranges.YOUNG_PAGES", 1)
    client = _OverSeekingClient()
    procs = thread_children(client)

    with pytest.raises(RuntimeError, match="skipped keys"):
        _listed(client)
    assert procs
