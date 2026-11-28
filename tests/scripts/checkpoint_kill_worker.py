import json
import os
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import datachain as dc

root = Path(sys.argv[1])
kind = sys.argv[2]
parallel = int(sys.argv[3])
flush_interval = float(sys.argv[4])
kill = len(sys.argv) > 5
SLOW, KILLED = 6, 7
FLUSH_ENDS = (4, 6)
seen: list[int] = []


def wait_for(name: str) -> None:
    path = root / name
    deadline = time.monotonic() + flush_interval + 30.0
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {name}")
        time.sleep(0.01)


def start(num: int) -> None:
    if kill and seen and seen[-1] in FLUSH_ENDS:
        (root / f"flushed-{seen[-1]}").write_text("1")
    seen.append(num)
    if kill and num > KILLED:
        time.sleep(60)
    with (root / f"calls-{os.getpid()}").open("a") as stream:
        stream.write(f"{num}\n")
    if kill and num == SLOW:
        if parallel > 1:
            (root / "at-6").write_text("1")
            wait_for("at-4")
        # the next row is inserted after the flush interval, with all rows before it
        time.sleep(flush_interval)


def finish(num: int) -> None:
    if kill and parallel > 1 and num == 4:
        (root / "at-4").write_text("1")
        wait_for("at-6")
        time.sleep(flush_interval)


def kill_at(num: int) -> None:
    if kill and num == KILLED:
        if parallel > 1:
            for end in FLUSH_ENDS:
                wait_for(f"flushed-{end}")
        os.killpg(0, signal.SIGKILL)


def mapper(num: int) -> int:
    start(num)
    kill_at(num)
    finish(num)
    return num * 10


def generator(num: int) -> Iterator[int]:
    start(num)
    yield num * 10
    kill_at(num)
    yield num * 10 + 1
    finish(num)


source = dc.read_dataset("nums").order_by("num")
source = source.settings(parallel=parallel or None)
chain = source.map(value=mapper) if kind == "map" else source.gen(value=generator)
result = chain.save("result")
(root / "result.json").write_text(json.dumps(sorted(result.to_values("value"))))
