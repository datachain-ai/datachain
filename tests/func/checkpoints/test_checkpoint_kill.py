import json
import signal
import sys
from pathlib import Path

import pytest

import datachain as dc
from tests.utils import (
    e2e_subprocess_env,
    run_test_subprocess,
    wait_for_test_subprocess,
)

WORKER = Path(__file__).parents[2] / "scripts" / "checkpoint_kill_worker.py"
EXPECTED = {
    "map": [num * 10 for num in range(10)],
    "gen": [num * 10 + offset for num in range(10) for offset in (0, 1)],
}


@pytest.mark.skipif(
    sys.platform == "win32", reason="Requires SIGKILL and process groups"
)
@pytest.mark.parametrize("parallel", [0, 2])
@pytest.mark.parametrize("kind", ["map", "gen"])
def test_checkpoint_resume_after_sigkill(
    tmp_path, test_session_tmpfile, parallel, kind
):
    catalog = test_session_tmpfile.catalog
    dc.read_values(num=list(range(10)), session=test_session_tmpfile).save("nums")
    env = e2e_subprocess_env(catalog)
    metastore = catalog.metastore
    first_job = metastore.create_job("checkpoint-kill", "")
    env["DATACHAIN_JOB_ID"] = first_job
    interval = str(catalog.warehouse.INSERT_FLUSH_INTERVAL)
    cmd = [sys.executable, str(WORKER), str(tmp_path), kind, str(parallel), interval]

    killed = run_test_subprocess([*cmd, "kill"], env)
    rc, _, stderr = wait_for_test_subprocess(killed, 60)
    assert rc == -signal.SIGKILL, stderr

    for path in tmp_path.glob("calls-*"):
        path.unlink()
    env["DATACHAIN_JOB_ID"] = metastore.create_job(
        "checkpoint-kill", "", rerun_from_job_id=first_job, run_group_id=first_job
    )
    rc, _, stderr = wait_for_test_subprocess(run_test_subprocess(cmd, env), 60)
    assert rc == 0, stderr

    recomputed = {
        int(num)
        for path in tmp_path.glob("calls-*")
        for num in path.read_text().split()
    }
    assert recomputed >= {7, 8, 9}
    assert recomputed.isdisjoint(range(6))
    assert json.loads((tmp_path / "result.json").read_text()) == EXPECTED[kind]
