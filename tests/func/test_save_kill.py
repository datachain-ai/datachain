"""A save killed with SIGKILL leaves no version mid-UDF, and a rerun of the same
job chain reclaims a version killed during the save's copy."""

import signal
import sys
from pathlib import Path

import pytest

import datachain as dc
from datachain.data_storage import JobStatus
from datachain.dataset import DatasetStatus
from datachain.error import DatasetNotFoundError
from tests.utils import (
    e2e_subprocess_env,
    run_test_subprocess,
    wait_for_test_subprocess,
)

WORKER = Path(__file__).parents[1] / "scripts" / "save_kill_worker.py"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Requires SIGKILL")


def _run(catalog, kill_at: str, version: str, parent: str | None = None) -> str:
    metastore = catalog.metastore
    group = metastore.get_job(parent).run_group_id if parent else None
    job_id = metastore.create_job(
        "save-kill", "", rerun_from_job_id=parent, run_group_id=group
    )
    env = e2e_subprocess_env(catalog)
    env["DATACHAIN_JOB_ID"] = job_id
    proc = run_test_subprocess([sys.executable, str(WORKER), kill_at, version], env)
    rc, _, stderr = wait_for_test_subprocess(proc, 90)
    expected = -signal.SIGKILL if kill_at != "none" else 0
    assert rc == expected, stderr
    if rc == -signal.SIGKILL:
        # The supervisor observed process death; a running owner cannot be adopted.
        metastore.set_job_status(job_id, JobStatus.FAILED)
    return job_id


def _versions(catalog):
    ds = catalog.get_dataset("result", versions=None)
    return [(v.version, v.status) for v in ds.versions]


@pytest.mark.parametrize("version", ["auto", "1.0.0"])
def test_kill_mid_udf_leaves_no_version(test_session_tmpfile, version):
    catalog = test_session_tmpfile.catalog
    _run(catalog, "udf", version)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("result")


@pytest.mark.parametrize("version", ["auto", "1.0.0"])
def test_rerun_reclaims_version_killed_during_copy(test_session_tmpfile, version):
    catalog = test_session_tmpfile.catalog
    first = _run(catalog, "copy", version)
    assert _versions(catalog) == [("1.0.0", DatasetStatus.CREATED)]

    second = _run(catalog, "none", version, parent=first)

    # no "already has version", and an auto version does not skip a number
    assert _versions(catalog) == [("1.0.0", DatasetStatus.COMPLETE)]
    ds = catalog.get_dataset("result", versions=None)
    assert ds.get_version("1.0.0").job_id == second
    rows = dc.read_dataset("result", session=test_session_tmpfile)
    assert sorted(rows.to_values("doubled")) == [2, 4, 6, 8, 10]  # not doubled twice


def test_unrelated_job_does_not_reclaim(test_session_tmpfile):
    catalog = test_session_tmpfile.catalog
    _run(catalog, "copy", "1.0.0")
    _run(catalog, "none", "auto")  # a new run, not a rerun
    assert _versions(catalog) == [
        ("1.0.0", DatasetStatus.CREATED),
        ("1.0.1", DatasetStatus.COMPLETE),
    ]
