import asyncio
import io
import logging
import os
import pickle
import random
import subprocess
import sys

import pytest
from fsspec.asyn import get_loop, sync

import datachain
from datachain.client import listing_worker
from datachain.client.fsspec import Client, RangeListedClient
from datachain.client.listing_worker import (
    Child,
    ChildDied,
    Pool,
    _cpu_quotas,
    _serve,
    dump,
    load,
    picklable,
    usable_cpus,
)
from datachain.client.ranges import Page
from datachain.lib.file import FileRow

ROOT = os.path.dirname(os.path.dirname(datachain.__file__))


class FakeClient:
    def __init__(self, out, fail=False):
        self.out, self.fail, self.probed_before_ready = out, fail, None
        self.probed = None

    async def _probe(self, prefix, start_after):
        self.probed_before_ready = not self.out.getvalue()
        self.probed = prefix, start_after
        if self.fail:
            raise PermissionError("no list access")


def serve(client, out):
    inp = io.BytesIO()  # no requests: the child stops at end of input
    asyncio.run(_serve(client, "dir/", inp, out))
    out.seek(0)
    return load(out)


def test_child_warms_up_before_it_reports_ready():
    out = io.BytesIO()
    client = FakeClient(out)

    assert serve(client, out) == ("ready", None)
    assert client.probed_before_ready
    assert client.probed == ("dir/", "")


def test_frames_round_trip():
    assert load(io.BytesIO(dump(("page", 1, [("s3://b", "k")], "k", "k")))) == (
        "page",
        1,
        [("s3://b", "k")],
        "k",
        "k",
    )


def holding(error: Exception) -> Exception:
    error.handle = lambda: None  # type: ignore[attr-defined]
    return error


class TwoArgsError(Exception):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.handle = lambda: None


@pytest.mark.parametrize(
    "error, sent_type",
    [
        (ValueError("x"), ValueError),
        (holding(OSError("x")), OSError),
        (TwoArgsError(1, 2), TwoArgsError),
    ],
)
def test_errors_cross_the_pipe(error, sent_type):
    sent = pickle.loads(pickle.dumps(picklable(error)))  # noqa: S301

    assert type(sent) is sent_type
    assert str(error) in str(sent)


def test_azure_errors_cross_the_pipe():
    from azure.core.exceptions import ServiceRequestError

    try:
        try:
            raise OSError("connection reset")
        except OSError as e:
            raise ServiceRequestError("request failed") from e
    except ServiceRequestError as e:
        error = e

    sent = pickle.loads(pickle.dumps(picklable(error)))  # noqa: S301

    assert type(sent) is ServiceRequestError
    assert "request failed" in str(sent)


class VersionsClient(RangeListedClient):
    """One key with many versions, then one more key, in pages of 4."""

    PREFIX = "fake://"
    protocol = "fake"
    LIST_PAGE_SIZE = 4

    def __init__(self, versions: int) -> None:
        super().__init__("bucket", {}, None)  # type: ignore[arg-type]
        self.keys = [("a", f"v{i:06}") for i in range(versions)] + [("b", "v0")]
        self.requests = 0

    def info_to_file(self, v, path):
        raise NotImplementedError

    async def _list_page(
        self, prefix, start_after, token, keep, page_size=None, hi=None
    ) -> Page:
        self.requests += 1
        if token is None:
            token = next(
                (i for i, (k, _) in enumerate(self.keys) if k > start_after),
                len(self.keys),
            )
        chunk = self.keys[token : token + (page_size or self.LIST_PAGE_SIZE)]
        end = token + len(chunk)
        return Page(
            [FileRow("fake://bucket", k, 1, v) for k, v in chunk if keep(k)],
            chunk[0][0] if chunk else None,
            max((k for k, _ in chunk), default=None),
            end if end < len(self.keys) else None,
        )


class SlowClient(RangeListedClient):
    """Keys with a few versions each; every request takes a random while."""

    PREFIX = "fake://"
    protocol = "fake"
    LIST_PAGE_SIZE = 4

    def __init__(self, seed: int) -> None:
        super().__init__("bucket", {}, None)  # type: ignore[arg-type]
        self.rng = random.Random(seed)  # noqa: S311
        names = {
            "".join(self.rng.choices("ab/0123456789", k=self.rng.randint(1, 6)))
            for _ in range(300)
        }
        self.keys = [
            (name, f"v{v}")
            for name in sorted(names)
            for v in range(self.rng.choice([1, 1, 1, 3, 9]))
        ]

    def info_to_file(self, v, path):
        raise NotImplementedError

    async def _list_page(
        self, prefix, start_after, token, keep, page_size=None, hi=None
    ) -> Page:
        await asyncio.sleep(self.rng.random() / 200)
        if token is None:
            token = next(
                (i for i, (k, _) in enumerate(self.keys) if k > start_after),
                len(self.keys),
            )
        chunk = self.keys[token : token + (page_size or self.LIST_PAGE_SIZE)]
        end = token + len(chunk)
        return Page(
            [FileRow("fake://bucket", k, 1, v) for k, v in chunk if keep(k)],
            chunk[0][0] if chunk else None,
            max((k for k, _ in chunk), default=None),
            end if end < len(self.keys) else None,
        )


@pytest.mark.parametrize("seed", range(40))
def test_ranges_that_split_while_pages_are_in_flight_are_exact(seed, mocker):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 8)
    mocker.patch("datachain.client.ranges.YOUNG_PAGES", 1)
    client = SlowClient(seed)

    pages = _listed(client, processes=False)

    expected = [k for k in client.keys if client._is_valid_key(k[0])]
    assert sorted((r.path, r.version) for p in pages for r in p) == expected


def _listed(client: Client, processes: bool = True) -> list[list]:
    async def pages():
        return [page async for page in client.scandir("", processes=processes)]

    return sync(get_loop(), pages)


@pytest.mark.parametrize("processes", [False, True])
def test_versions_of_one_key_take_linear_work_and_bounded_memory(processes, mocker):
    async def starting(self, spec):
        await asyncio.sleep(3600)

    mocker.patch.object(Child, "start", starting)
    pages = {}
    for versions in (100, 200):
        client = VersionsClient(versions)
        pages[versions] = _listed(client, processes)
        rows = [(r.path, r.version) for page in pages[versions] for r in page]
        assert sorted(rows) == sorted(client.keys)
        assert client.requests <= versions // 4 + 2

    assert max(len(page) for page in pages[200]) <= VersionsClient.LIST_PAGE_SIZE


def test_short_listing_takes_a_request_per_page_and_no_children(mocker):
    start = mocker.spy(Pool, "start")
    client = VersionsClient(4)  # 5 keys in 4-key pages

    assert sum(map(len, _listed(client))) == 5
    assert client.requests == 2
    assert not start.call_count


def test_a_child_with_a_broken_pipe_takes_no_more_pages():
    async def check():
        pool = Pool(VersionsClient(1), "")
        child = Child()
        child.proc = type("Proc", (), {"returncode": 0})()
        child.proc.stdin = type("Pipe", (), {"is_closing": lambda self: True})()
        pool.children = [child]

        with pytest.raises(ChildDied):
            await pool.list_page("", "", None, None)
        assert not pool.ready()
        assert not child.replies
        with pytest.raises(ChildDied):
            await pool.list_page("", "", None, None)

    asyncio.run(check())


def test_a_child_that_dies_fails_its_pending_pages():
    async def check():
        child = Child()
        child.proc = type("Proc", (), {"returncode": 0})()
        child.proc.stdin = type(
            "Pipe", (), {"is_closing": lambda self: False, "write": lambda s, d: None}
        )()
        pending = asyncio.create_task(child.list_page(1, "", "", None, None))
        await asyncio.sleep(0)
        child.kill()

        with pytest.raises(ChildDied):
            await pending

    asyncio.run(check())


def test_child_reports_its_warm_up_error():
    out = io.BytesIO()

    kind, error = serve(FakeClient(out, fail=True), out)

    assert kind == "ready"
    assert isinstance(error, PermissionError)


def test_child_without_a_spec_exits_quietly():
    code = "from datachain.client.listing_worker import main; main()"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        input=b"",
        capture_output=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": ROOT},
    )

    assert (out.returncode, out.stderr) == (0, b"")


def test_child_ignores_ctrl_c():
    code = (
        "import os, signal, sys\n"
        "from datachain.client import listing_worker\n"
        "listing_worker.load = lambda f: signal.raise_signal(signal.SIGINT)\n"
        "listing_worker.main()\n"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        input=b"",
        capture_output=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": ROOT},
    )

    assert (out.returncode, out.stderr) == (0, b"")


class StatusError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.exc_info = (None, None, (lambda: None))


class UnprintableError(Exception):
    def __init__(self):
        super().__init__()
        self.hook = lambda: None

    def __str__(self):
        raise ValueError("no message")


def test_errors_keep_the_attributes_that_pickle():
    sent = pickle.loads(pickle.dumps(picklable(StatusError("denied", 403))))  # noqa: S301

    assert type(sent) is StatusError
    assert (str(sent), sent.status_code, sent.exc_info) == ("denied", 403, None)


def test_errors_of_classes_that_do_not_pickle_keep_their_base_class():
    from botocore.exceptions import ClientError

    error_class = type("NoSuchBucket", (ClientError,), {})
    response = {"Error": {"Code": "NoSuchBucket"}, "ResponseMetadata": {}}
    error = error_class(response, "ListObjectsV2")

    sent = pickle.loads(pickle.dumps(picklable(error)))  # noqa: S301

    assert type(sent) is ClientError
    assert sent.response["Error"]["Code"] == "NoSuchBucket"


def test_an_error_that_neither_pickles_nor_prints_still_crosses():
    sent = pickle.loads(pickle.dumps(picklable(UnprintableError())))  # noqa: S301

    assert isinstance(sent, Exception)


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


@pytest.mark.parametrize("level, logged", [("DEBUG", True), ("WARNING", False)])
def test_child_logs_at_the_parents_level(level, logged, tmp_path):
    from datachain.client.local import FileClient

    spec = dump(
        (FileClient, str(tmp_path), {}, None, 4, "", logging.getLevelName(level))
    )
    code = "from datachain.client.listing_worker import main; main()"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        input=spec,
        capture_output=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": ROOT},
    )

    assert out.returncode == 0
    assert (b"Listing process warm-up failed" in out.stderr) is logged
