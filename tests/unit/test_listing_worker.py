import asyncio
import io
import os
import pickle
import random
import signal
import subprocess
import sys
import threading
import time
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
from datachain.client.ranges import Keep, Page
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


def serve(client, out):
    inp = io.BytesIO()  # no requests: the child stops at end of input
    asyncio.run(_serve(client, "dir/", inp, out))
    out.seek(0)
    return [load(out)]


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
    read, write = os.pipe()
    os.write(write, dump((7, "", None, None)))
    threading.Timer(0.5, os.close, [write]).start()  # then the parent goes
    out = io.BytesIO()

    with os.fdopen(read, "rb") as inp:
        asyncio.run(_serve(FakeClient(out), "dir/", inp, out))

    out.seek(0)
    assert [load(out), load(out)] == [None, (7, None)]


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
        self, prefix, start_after, token, hi=None, page_size=None
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


class Proc:
    """A child process: its stdout is fed by the test."""

    def __init__(self, closing=False):
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.stdin = mock.Mock(**{"is_closing.return_value": closing})

    def kill(self):
        self.returncode = -9


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
