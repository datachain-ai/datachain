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
    Pool,
    _command,
    _cpu_quotas,
    _receive,
    _serve,
    dump,
    load,
    usable_cpus,
)
from datachain.client.ranges import Keep, Page, RangeListing
from datachain.lib.file import FileRow

ROOT = os.path.dirname(os.path.dirname(datachain.__file__))


class FakeClient:
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
    assert load(io.BytesIO(dump((1, ([("s3://b", "k")], "k", "k"))))) == (
        1,
        ([("s3://b", "k")], "k", "k"),
    )


def test_a_child_replies_none_to_a_page_it_fails():
    out = io.BytesIO()

    assert serve(FakeClient(out), out, (7, "", None, None)) == [None, (7, None)]


class UnpicklableClient(FakeClient):
    async def _list_page(self, prefix, start_after, token, hi=None, page_size=None):
        if page_size is None:  # a page whose rows don't pickle
            return Page([(lambda: None,)], "a", "a", None)  # type: ignore[list-item]


def test_a_child_replies_none_to_a_page_that_does_not_pickle():
    out = io.BytesIO()
    client = UnpicklableClient(out)

    assert serve(client, out, (7, "", None, None)) == [None, (7, None)]


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


def test_a_child_with_a_broken_pipe_takes_no_more_pages():
    async def check():
        pool = Pool(versions_client(1), "", 1)
        proc = Proc(closing=True)
        pool.children = {proc: {}}

        assert await pool.list_page("", None, None) is None
        assert not pool.children[proc]

    asyncio.run(check())


def test_a_child_that_dies_fails_its_pending_pages(mocker):
    async def check():
        proc = Proc()
        proc.stdout.feed_data(dump(None))  # ready

        async def spawn():
            return proc

        mocker.patch.object(listing_worker, "_spawn", spawn)
        pool = Pool(versions_client(1), "", 1)
        pool.start()
        while not pool.children:  # noqa: ASYNC110
            await asyncio.sleep(0)
        pending = asyncio.create_task(pool.list_page("", None, None))
        await asyncio.sleep(0)
        proc.stdout.feed_eof()

        assert await pending is None
        assert not pool.children
        assert proc.returncode == -9

    asyncio.run(check())


def test_a_stuck_child_is_killed_and_the_parent_lists_its_page(mocker, caplog):
    mocker.patch.object(listing_worker, "REPLY_TIMEOUT", 0.1)

    async def check():
        pool = Pool(versions_client(1), "", 1)
        proc = Proc()  # never answers
        pool.children = {proc: {}}

        assert await pool.list_page("", None, None) is None
        assert proc.returncode == -9

    with caplog.at_level("WARNING", logger="datachain"):
        asyncio.run(check())
    assert "is stuck: listing without it" in caplog.text


def test_a_stuck_child_takes_no_more_pages_and_fails_its_others_at_once(mocker, caplog):
    mocker.patch.object(listing_worker, "REPLY_TIMEOUT", 0.1)

    async def check():
        pool = Pool(versions_client(1), "", 1)
        proc = Proc()  # never answers
        pool.children = {proc: {}}
        first = asyncio.create_task(pool.list_page("", None, None))
        await asyncio.sleep(0.5)  # within the first page's grace
        mocker.patch.object(listing_worker, "REPLY_TIMEOUT", 60)
        second = asyncio.create_task(pool.list_page("", None, None))

        assert await first is None
        # back with the first page's drop, long before a timeout of its own
        await asyncio.wait([second], timeout=30)
        assert second.done()
        assert second.result() is None
        assert not pool.children
        assert await pool.list_page("", None, None) is None

    with caplog.at_level("WARNING", logger="datachain"):
        asyncio.run(check())
    assert caplog.text.count("is stuck") == 1


def test_a_paused_parent_reads_a_reply_that_came_meanwhile(mocker):
    mocker.patch.object(listing_worker, "REPLY_TIMEOUT", 0.1)

    async def check():
        pool = Pool(versions_client(1), "", 1)
        replies: dict = {}
        proc = Proc()
        pool.children = {proc: replies}
        pending = asyncio.create_task(pool.list_page("", None, None))
        await asyncio.sleep(0.3)  # past the timeout: the parent was paused
        next(iter(replies.values())).set_result(([], None, None, None))

        assert await pending == Page([], None, None, None)
        assert proc.returncode is None
        assert proc in pool.children

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


def test_a_page_request_keeps_a_cancel_that_comes_with_its_reply():
    async def check():
        pool = Pool(versions_client(1), "", 1)
        replies: dict = {}
        pool.children = {Proc(): replies}
        pending = asyncio.create_task(pool.list_page("", None, None))
        while not replies:  # noqa: ASYNC110
            await asyncio.sleep(0)
        # the pool closes: the child's pending pages come back None as the
        # listing's workers are cancelled
        next(iter(replies.values())).set_result(None)
        pending.cancel()

        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(check())


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

async def main(client, stop):
    pool = Pool(getattr(listing_test_clients, client)("bucket", {}, None), "", 2)
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
        pool = Pool(clients.IdleClient("bucket", {}, None), "", 2)
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


@pytest.mark.parametrize(
    "files, expected",
    [
        ({"cpu.max": "max 100000"}, []),
        ({"cpu.max": "150000 100000"}, [(150000, 100000)]),
        (
            {"cpu.max": "max 100000", "user.slice/cpu.max": "200000 100000"},
            [(200000, 100000)],
        ),
    ],
)
def test_cpu_quotas_of_cgroup_v2_and_its_parents(files, expected, tmp_path):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    (tmp_path / "cgroup").write_text("0::/user.slice\n")

    assert _cpu_quotas(str(tmp_path), str(tmp_path / "cgroup")) == expected


def test_cpu_quota_of_cgroup_v1(tmp_path):
    folder = tmp_path / "cpu,cpuacct" / "docker"
    folder.mkdir(parents=True)
    (folder / "cpu.cfs_quota_us").write_text("100000\n")
    (folder / "cpu.cfs_period_us").write_text("100000\n")
    (tmp_path / "cgroup").write_text("5:memory:/docker\n4:cpu,cpuacct:/docker\n")

    assert _cpu_quotas(str(tmp_path), str(tmp_path / "cgroup")) == [(100000, 100000)]


@pytest.mark.parametrize(
    "quotas, expected", [([], 16), ([(150000, 100000)], 1), ([(50000, 100000)], 1)]
)
def test_usable_cpus_follow_the_cgroup_quotas(quotas, expected, monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 16, raising=False)
    monkeypatch.setattr(listing_worker, "_cpu_quotas", lambda: quotas)

    assert usable_cpus() == expected


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
    out = child(dump((clients.IdleClient, "bucket", {}, None, 4, "")), cwd=cwd)

    assert out.returncode == 0
    assert load(io.BytesIO(out.stdout)) is None  # ready


def test_child_reports_why_it_cannot_build_its_client():
    out = child(dump((int, "bucket", {}, None, 4, "")))  # int() takes no 3 args

    assert out.returncode == 0
    assert "TypeError" in load(io.BytesIO(out.stdout))
    assert not out.stderr


def test_child_ends_when_the_parent_goes_while_a_thread_is_stuck(clients):
    spec = (clients.ThreadIdleClient, "bucket", {}, None, 4, "")
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

async def start():
    pool = Pool(listing_test_clients.IdleClient("bucket", {}, None), "", 2)
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
