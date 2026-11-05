import json
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from tests.utils import (
    e2e_subprocess_env,
    run_test_subprocess,
    wait_for_test_subprocess,
)

WORKER = Path(__file__).parents[2] / "scripts" / "checkpoint_kill_worker.py"


def read_calls(root, event):
    return [
        (num, at)
        for path in root.glob("calls-*.jsonl")
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n")
        for kind, num, at in [json.loads(line)]
        if kind == event
    ]


@pytest.mark.skipif(
    sys.platform == "win32", reason="Requires SIGKILL and process groups"
)
@pytest.mark.parametrize("parallel", [0, 4])
@pytest.mark.parametrize("kind", ["map", "gen"])
def test_checkpoint_resume_after_sigkill(tmp_path, catalog_tmpfile, parallel, kind):
    env = e2e_subprocess_env(catalog_tmpfile)
    metastore = catalog_tmpfile.metastore
    first_job = metastore.create_job("checkpoint-kill", "")
    env["DATACHAIN_JOB_ID"] = first_job
    interval = catalog_tmpfile.warehouse.INSERT_FLUSH_INTERVAL
    (tmp_path / "files").mkdir()
    for num in range(120):
        (tmp_path / "files" / f"{num:04d}.txt").write_text(str(num))
    cmd = [
        sys.executable,
        str(WORKER),
        str(tmp_path),
        kind,
        str(parallel),
        str(interval),
    ]
    proc = run_test_subprocess([*cmd, "block"], env, capture_output=False)
    try:
        deadline = time.monotonic() + 90
        while len(read_calls(tmp_path, "start")) < 60 or (
            kind == "gen" and not read_calls(tmp_path, "blocked")
        ):
            assert proc.poll() is None, "Worker exited before the kill point"
            assert time.monotonic() < deadline, "UDF did not reach kill point"
            time.sleep(0.05)
        killed_at = time.monotonic()
    finally:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    assert proc.returncode == -signal.SIGKILL
    saved_by = killed_at - interval - 1
    saved = {num for num, at in read_calls(tmp_path, "done") if at < saved_by}
    assert saved
    blocked = {num for num, _ in read_calls(tmp_path, "blocked")}

    for path in tmp_path.glob("calls-*.jsonl"):
        path.unlink()
    env["DATACHAIN_JOB_ID"] = metastore.create_job(
        "checkpoint-kill", "", rerun_from_job_id=first_job
    )
    proc = run_test_subprocess(cmd, env)
    try:
        rc, _, stderr = wait_for_test_subprocess(proc, 90)
        assert rc == 0, stderr
    finally:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    recomputed = {num for num, _ in read_calls(tmp_path, "start")}
    assert saved.isdisjoint(recomputed)
    assert blocked <= recomputed
    if kind == "map":
        expected = [num * 10 for num in range(120)]
    else:
        expected = [
            num * 100 + offset
            for num in range(120)
            if num % 5
            for offset in range(60 if num == 91 else 3)
        ]
    assert json.loads((tmp_path / "result.json").read_text()) == expected
