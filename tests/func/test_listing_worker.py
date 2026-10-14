import asyncio
import time
from contextlib import aclosing
from unittest import mock

import pytest
from fsspec.asyn import sync

import datachain as dc
from datachain.asyn import get_loop
from datachain.client import Client, listing_worker
from datachain.client.listing_worker import Pool
from datachain.lib.convert.flatten import flatten
from datachain.lib.file import File, FileRow
from datachain.lib.listing import list_bucket

TREE = {
    "a": "1",
    "big": {f"{i:03}": "x" for i in range(60)},
    "flat": {f"{i * 7919 % 4096:03x}": "x" for i in range(30)},
    "z": "1",
}


@pytest.fixture(autouse=True)
def children_from_the_start(monkeypatch):
    monkeypatch.setattr("datachain.client.ranges.POOL_ROWS", 0)


@pytest.fixture
def client(cloud_server, cloud_server_credentials, mocker):
    uri = cloud_server.src_uri
    client = Client.get_implementation(uri).from_source(
        uri, cache=None, **cloud_server.client_config
    )
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch("datachain.client.listing_worker.LIST_PROCESSES", 2)
    return client


@pytest.fixture
def children(client, mocker):
    """Children of the listing; once they are starting, the parent lists no
    further page until one runs, so every listing past two pages moves to them."""
    started: list[Pool] = []
    pool_start, list_page = Pool.start, type(client)._list_page

    def start(self):
        started.append(self)
        pool_start(self)

    async def gated(self, *args, **kwargs):
        deadline = time.monotonic() + 30
        while starting(started) and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        return await list_page(self, *args, **kwargs)

    mocker.patch.object(Pool, "start", start)
    mocker.patch.object(type(client), "_list_page", gated)
    return started


@pytest.fixture
def remote_rows(mocker):
    got: list = []
    list_page = Pool.list_page

    async def recorded(self, *args):
        page = await list_page(self, *args)
        got.extend(page.rows if page else [])
        return page

    mocker.patch.object(Pool, "list_page", recorded)
    return got


def starting(pools):
    pool = pools[-1] if pools else None
    return pool and not pool.children and any(not t.done() for t in pool.tasks)


async def _items(client, prefix):
    items = []
    async with aclosing(client.scandir(prefix)) as pages:
        async for page in pages:
            items.extend(page)
    return items


def items(client, prefix="", processes=True):
    processes_ = listing_worker.LIST_PROCESSES if processes else 0
    with mock.patch.object(listing_worker, "LIST_PROCESSES", processes_):
        return sync(get_loop(), _items, client, prefix)


def listing(client, prefix="", processes=True):
    return rows(items(client, prefix, processes))


def killing(pools, frames, should_kill):
    """``_receive`` that kills the child whose frame ``should_kill`` picks."""
    receive = listing_worker._receive

    async def received(stream):
        frame = await receive(stream)
        if isinstance(frame, tuple) and frame[1] and should_kill(frame[1][0]):
            procs = [p for pool in pools for p in pool.procs if p.stdout is stream]
            frames.append(frame)
            procs[0].kill()
        return frame

    return received


def rows(items):
    return sorted(tuple(i) if isinstance(i, tuple) else flatten(i) for i in items)


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
@pytest.mark.parametrize("prefix", ["", "big", "flat"])
def test_listing_in_children_is_exact(
    client, tree, prefix, children, remote_rows, mocker
):
    died = []
    run = Pool._run

    async def recorded(self):
        await run(self)  # cancelled when the pool closes
        died.append(self)

    mocker.patch.object(Pool, "_run", recorded)

    listed = items(client, prefix)

    assert remote_rows
    assert not died
    assert rows(listed) == listing(client, prefix, processes=False)


# fake-gcs-server repeats a page forever when one key's versions span pages
@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_versions_across_pages_are_exact(client, tree, children, remote_rows):
    for i in range(9):
        client.fs.pipe_file(f"{client.uri}/big/030", str(i).encode())

    listed = items(client)

    assert remote_rows
    assert rows(listed) == listing(client, processes=False)
    assert sum(i.path == "big/030" for i in listed) >= 10


# fake-gcs-server repeats a page forever when one key's versions span pages
@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_a_child_killed_inside_one_keys_versions_loses_none(
    client, tree, children, remote_rows, mocker
):
    for i in range(29):
        client.fs.pipe_file(f"{client.uri}/big/030", str(i).encode())
    killed: list = []

    def inside_the_key(page_rows):
        return any(row[1] == "big/030" for row in page_rows) and len(killed) < 2

    mocker.patch.object(
        listing_worker, "_receive", killing(children, killed, inside_the_key)
    )

    listed = items(client)

    expected = listing(client, processes=False)
    assert killed
    assert rows(listed) == expected
    assert sum(r[1] == "big/030" for r in expected) >= 30


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
@pytest.mark.parametrize("processes", [1, 2])
def test_ranges_of_a_dead_child_move_on(client, tree, processes, children, mocker):
    mocker.patch("datachain.client.listing_worker.LIST_PROCESSES", processes)
    killed: list = []
    mocker.patch.object(
        listing_worker, "_receive", killing(children, killed, lambda _: not killed)
    )

    assert listing(client) == listing(client, processes=False)
    assert killed


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_children_that_never_start_leave_the_listing_exact(client, tree, mocker):
    mocker.patch.object(listing_worker, "_spawn", side_effect=OSError("no process"))

    assert listing(client) == listing(client, processes=False)


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_children_stop_when_listing_is_closed_early(
    client, tree, children, remote_rows, mocker
):
    async def first_child_page():
        async with aclosing(client.scandir("")) as pages:
            async for _ in pages:
                if remote_rows:
                    return

    sync(get_loop(), first_child_page)
    procs = [p for pool in children for p in pool.procs]
    deadline = time.monotonic() + 10
    while any(p.returncode is None for p in procs) and time.monotonic() < deadline:
        time.sleep(0.05)

    assert procs
    assert all(p.returncode is not None for p in procs)


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_config_that_does_not_pickle_lists_in_process(client, tree, remote_rows):
    from aiobotocore.session import AioSession

    client.fs_kwargs = {**client.fs_kwargs, "session": AioSession()}

    assert Pool(client, "", 2).size == 0
    assert items(client)
    assert not remote_rows


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_read_storage_rows_from_children(client, tree, children):
    uri = f"{client.uri}/"
    chain = dc.read_values(seed=[0]).gen(
        file=list_bucket(uri, None, client_config=client.fs_kwargs)
    )

    files = chain.order_by("file.path", "file.version").to_values("file")

    assert children
    assert sorted(flatten(f) for f in files) == listing(client, processes=False)


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_listing_rows_are_normalized_files(client, tree):
    rows = items(client, processes=False)

    assert rows
    assert all(type(r) is FileRow for r in rows)
    assert [flatten(File(**r._asdict())) for r in rows] == [tuple(r) for r in rows]


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
@pytest.mark.parametrize(
    "file_type", [dc.ImageFile, dc.TextFile, dc.VideoFile, dc.AudioFile]
)
def test_file_types_from_children(
    client, tree, children, remote_rows, test_session, file_type
):
    files = (
        dc.read_values(seed=[0], session=test_session)
        .gen(
            file=list_bucket(f"{client.uri}/", None, client_config=client.fs_kwargs),
            output={"file": file_type},
        )
        .to_values("file")
    )

    assert remote_rows
    assert {type(f) for f in files} == {file_type}
    assert sorted(flatten(f) for f in files) == listing(client, processes=False)


@pytest.fixture
def requests_in_flight(client, mocker):
    """The most page requests the parent had open at once."""
    peak = [0]
    now = 0
    list_page = type(client)._list_page

    async def counted(self, *args, **kwargs):
        nonlocal now
        now += 1
        peak[0] = max(peak[0], now)
        try:
            return await list_page(self, *args, **kwargs)
        finally:
            now -= 1

    mocker.patch.object(type(client), "_list_page", counted)
    return peak


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [{f"{i:03}": "x" for i in range(60)}], indirect=True)
def test_parent_lists_with_its_own_budget_when_no_child_runs(
    client, tree, requests_in_flight, mocker, caplog
):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 3)
    mocker.patch.object(listing_worker, "_spawn", side_effect=OSError("no process"))

    with caplog.at_level("WARNING", logger="datachain"):
        assert len(listing(client)) == len(tree)

    assert requests_in_flight[0] <= 3
    warnings = [
        r for r in caplog.records if r.name == "datachain" and r.levelname == "WARNING"
    ]
    assert [r.getMessage() for r in warnings] == [
        "Listing in one process: no listing process runs"
    ]


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_children_import_datachain_whatever_the_cwd_holds(
    client, tree, children, remote_rows, tmp_path, monkeypatch
):
    (tmp_path / "fsspec.py").write_text("raise ImportError('shadowed')\n")
    monkeypatch.chdir(tmp_path)

    assert listing(client) == listing(client, processes=False)
    assert remote_rows


CREDENTIAL_PROCESS = """
import json, os, stat, sys
with open(sys.argv[1], "a") as log:
    log.write(f"{os.getppid()} {stat.S_ISFIFO(os.fstat(0).st_mode)}\\n")
print(json.dumps({"Version": 1, "AccessKeyId": sys.argv[2],
                  "SecretAccessKey": sys.argv[3], "SessionToken": sys.argv[4]}))
"""


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_children_take_credentials_from_a_credential_process(
    client, tree, children, remote_rows, tmp_path, monkeypatch
):
    import os
    import sys

    script, log = tmp_path / "creds.py", tmp_path / "creds.log"
    script.write_text(CREDENTIAL_PROCESS)
    kwargs = dict(client.fs_kwargs)
    keys = [kwargs.pop(k) for k in ("aws_key", "aws_secret", "aws_token")]
    config = tmp_path / "config"
    config.write_text(
        "[profile listing]\n"
        f"credential_process = {sys.executable} {script} {log} {' '.join(keys)}\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    for name in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
        monkeypatch.delenv(name, raising=False)
    client.fs_kwargs = {**kwargs, "profile": "listing"}
    client._fs = None

    assert len(listing(client)) == len(listing(client, processes=False))
    assert remote_rows
    calls = [line.split() for line in log.read_text().splitlines()]
    in_children = [fifo for ppid, fifo in calls if int(ppid) != os.getpid()]
    assert in_children
    assert set(in_children) == {"False"}  # the frame pipe is not their stdin
