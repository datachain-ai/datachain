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


def start(num: int) -> None:
    with (root / f"calls-{os.getpid()}").open("a") as stream:
        stream.write(f"{num}\n")
    if kill and num == SLOW:
        # the next row is inserted after the flush interval, with all rows before it
        time.sleep(flush_interval)


def kill_at(num: int) -> None:
    if kill and num == KILLED:
        os.killpg(0, signal.SIGKILL)


def mapper(num: int) -> int:
    start(num)
    kill_at(num)
    return num * 10


def generator(num: int) -> Iterator[int]:
    start(num)
    yield num * 10
    kill_at(num)
    yield num * 10 + 1


source = dc.read_dataset("nums").order_by("num")
source = source.settings(parallel=parallel or None)
chain = source.map(value=mapper) if kind == "map" else source.gen(value=generator)
result = chain.save("result")
(root / "result.json").write_text(json.dumps(sorted(result.to_values("value"))))
