import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import datachain as dc

root = Path(sys.argv[1])
kind = sys.argv[2]
parallel = int(sys.argv[3])
# inputs from this number on never finish: the run can only end by being killed
block_from = int(sys.argv[4]) if len(sys.argv) > 4 else None
delay = 0.32 if parallel else 0.08


def record(event: str, num: int) -> None:
    with (root / f"calls-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps([event, num, time.monotonic()]) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if event == "start" and block_from is not None and num >= block_from:
        time.sleep(3600)


def mapper(file: dc.File) -> int:
    num = int(Path(file.path).stem)
    record("start", num)
    time.sleep(delay)
    record("done", num)
    return num * 10


def generator(file: dc.File) -> Iterator[int]:
    num = int(Path(file.path).stem)
    record("start", num)
    if num % 5 == 0:
        time.sleep(delay)
    else:
        large = kind == "gen-large" and num % (30 if parallel else 120) == (
            14 if parallel else 59
        )
        for offset in range(60 if large else 3):
            if large and block_from is not None and offset == 30:
                time.sleep(3600)  # stop halfway: rows saved, no final marker
            time.sleep(0.08 if large else delay / 3)
            yield num * 100 + offset
    record("done", num)


source = dc.read_storage(str(root / "files")).order_by("file.path")
if parallel:
    source = source.settings(parallel=parallel)
chain = source.map(value=mapper) if kind == "map" else source.gen(value=generator)
result = chain.save("result")
(root / "result.json").write_text(json.dumps(sorted(result.to_values("value"))))
