import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import closing, suppress
from pathlib import Path

import pytest

WORKER = Path(__file__).parents[2] / "scripts" / "checkpoint_kill_worker.py"


def read_calls(root):
    return [
        json.loads(line)
        for path in root.glob("calls-*.jsonl")
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]


def read_partial_rows(root):
    partial_rows = []
    input_nums = {}
    with closing(sqlite3.connect(root / "db" / ".datachain" / "db")) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'udf_%'"
        ).fetchall()
        for (table,) in tables:
            if table.endswith("_input"):
                columns = {
                    row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')
                }
                if "file__path" in columns:
                    query = f'SELECT sys__id, file__path FROM "{table}"'  # noqa: S608
                    input_nums.update(
                        (row_id, int(Path(path).stem))
                        for row_id, path in conn.execute(query)
                    )
            elif table.endswith("_output_partial"):
                query = f'SELECT * FROM "{table}"'  # noqa: S608
                cursor = conn.execute(query)
                columns = [column[0] for column in cursor.description]
                partial_rows.extend(
                    dict(zip(columns, row, strict=False)) for row in cursor
                )
    return partial_rows, input_nums


@pytest.mark.skipif(
    sys.platform == "win32", reason="Requires SIGKILL and process groups"
)
@pytest.mark.parametrize("parallel", [0, 4])
@pytest.mark.parametrize("kind", ["map", "gen", "gen-large"])
def test_checkpoint_resume_after_sigkill(tmp_path, monkeypatch, parallel, kind):
    monkeypatch.setenv("DATACHAIN_ROOT_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("DATACHAIN_NO_ANALYTICS", "1")
    monkeypatch.delenv("DATACHAIN_TEST", raising=False)
    monkeypatch.delenv("DATACHAIN_JOB_ID", raising=False)
    monkeypatch.delenv("DATACHAIN_IGNORE_CHECKPOINTS", raising=False)
    (tmp_path / "files").mkdir()
    for num in range(120):
        (tmp_path / "files" / f"{num:04d}.txt").write_text(str(num))
    cmd = [sys.executable, str(WORKER), str(tmp_path), kind, str(parallel)]
    with (tmp_path / "first.log").open("w") as log:
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=log, stderr=log, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 90
            while sum(event == "start" for event, _, _ in read_calls(tmp_path)) < 60:
                assert proc.poll() is None, (tmp_path / "first.log").read_text()
                assert time.monotonic() < deadline, "UDF did not reach kill point"
                time.sleep(0.02)
            if kind == "gen-large":
                time.sleep(2.5)
            killed_at = time.monotonic()
        finally:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    assert proc.returncode == -signal.SIGKILL
    first = read_calls(tmp_path)
    old_completed = {
        num for event, num, at in first if event == "done" and at < killed_at - 3
    }
    assert old_completed

    partial_rows, input_nums = read_partial_rows(tmp_path)
    if kind == "map":
        persisted = {row["value"] // 10 for row in partial_rows}
        incomplete = set()
    else:
        persisted = {
            input_nums[row["sys__input_id"]]
            for row in partial_rows
            if not row["sys__partial"]
        }
        incomplete = {
            input_nums[row["sys__input_id"]] for row in partial_rows
        } - persisted
        assert any(row["sys__empty"] for row in partial_rows)
        if kind == "gen-large":
            assert incomplete
    assert old_completed <= persisted

    for path in tmp_path.glob("calls-*.jsonl"):
        path.unlink()
    with (tmp_path / "second.log").open("w") as log:
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=log, stderr=log, start_new_session=True
        )
        try:
            assert proc.wait(timeout=90) == 0, (tmp_path / "second.log").read_text()
        finally:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    recomputed = {num for event, num, _ in read_calls(tmp_path) if event == "start"}
    assert recomputed == set(range(120)) - persisted
    assert incomplete <= recomputed
    if kind == "map":
        expected = [num * 10 for num in range(120)]
    else:
        expected = []
        for num in range(120):
            large = kind == "gen-large" and num % (30 if parallel else 120) == (
                14 if parallel else 59
            )
            if num % 5:
                expected.extend(
                    num * 100 + offset for offset in range(60 if large else 3)
                )
    assert json.loads((tmp_path / "result.json").read_text()) == expected
