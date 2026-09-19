"""Tests for jobs skill script helpers."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

# Insert the scripts directory so bare imports work.
SCRIPTS_DIR = str(
    Path(__file__).resolve().parents[2] / "src/datachain/skill/jobs/scripts"
)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from jobs import (  # noqa: E402
    _duration_str,
    _normalize_status,
    _parse_dt,
    _stage_seconds,
    _strip_ordinal,
)

# Also test the duplicated frontmatter parser
GRAPH_SCRIPTS_DIR = str(
    Path(__file__).resolve().parents[2] / "src/datachain/skill/knowledge/scripts"
)
if GRAPH_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, GRAPH_SCRIPTS_DIR)

from utils import read_frontmatter  # noqa: E402


class TestParseDt:
    def test_z_suffix(self):
        result = _parse_dt("2024-06-15T10:30:00Z")
        assert result == datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_offset_suffix(self):
        result = _parse_dt("2024-06-15T10:30:00+00:00")
        assert result == datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_none(self):
        assert _parse_dt(None) is None

    def test_empty(self):
        assert _parse_dt("") is None

    def test_garbage(self):
        assert _parse_dt("not-a-date") is None


class TestNormalizeStatus:
    def test_lowercase(self):
        assert _normalize_status("Complete") == "complete"

    def test_already_lower(self):
        assert _normalize_status("failed") == "failed"

    def test_upper(self):
        assert _normalize_status("RUNNING") == "running"


class TestDurationStr:
    def test_seconds(self):
        assert _duration_str(60) == "60s"

    def test_large(self):
        assert _duration_str(9000) == "9000s"

    def test_zero(self):
        assert _duration_str(0) == "0s"


class TestStripOrdinal:
    def test_third(self):
        assert _strip_ordinal("job-123 (3rd)") == "job-123"

    def test_fourth(self):
        assert _strip_ordinal("run (4th)") == "run"

    def test_no_ordinal(self):
        assert _strip_ordinal("plain-value") == "plain-value"

    def test_empty(self):
        assert _strip_ordinal("") == ""


class TestJobsReadFrontmatter:
    """Test the jobs-local copy of _read_frontmatter."""

    def test_normal(self, tmp_path):
        from jobs import _read_frontmatter

        p = tmp_path / "index.md"
        p.write_text("---\ntotal_jobs: 42\nenriched: true\n---\n## Jobs\n")
        fm = _read_frontmatter(str(p))
        assert fm["total_jobs"] == "42"
        assert fm["enriched"] == "true"

    def test_missing(self):
        from jobs import _read_frontmatter

        assert _read_frontmatter("/nonexistent.md") == {}

    def test_matches_graph_utils(self, tmp_path):
        """Both frontmatter parsers should produce the same result."""
        from jobs import _read_frontmatter as jobs_fm

        content = "---\nname: test\nvalue: hello world\n---\nbody\n"
        p = tmp_path / "test.md"
        p.write_text(content)

        assert jobs_fm(str(p)) == read_frontmatter(str(p))


class TestStageSeconds:
    def test_seconds_per_stage(self):
        stages = _stage_seconds(
            [
                {
                    "name": "waiting",
                    "started_at": "2026-09-16T00:00:00Z",
                    "finished_at": "2026-09-16T00:00:30Z",
                },
                {
                    "name": "running_query",
                    "started_at": "2026-09-16T00:02:30Z",
                    "finished_at": "2026-09-16T00:20:00Z",
                },
            ]
        )

        assert stages == {"waiting": 30, "running_query": 1050}

    def test_open_stage_is_left_out(self):
        """Still running, or a job that stopped mid-stage: not a zero-length stage."""
        stages = _stage_seconds(
            [
                {
                    "name": "waiting",
                    "started_at": "2026-09-16T00:00:00Z",
                    "finished_at": "2026-09-16T00:00:30Z",
                },
                {
                    "name": "running_query",
                    "started_at": "2026-09-16T00:02:30Z",
                    "finished_at": None,
                },
            ]
        )

        assert stages == {"waiting": 30}

    def test_no_steps(self):
        """What a job carries without --enrich."""
        assert _stage_seconds(None) == {}
        assert _stage_seconds([]) == {}


@pytest.fixture
def studio_jobs(mocker):
    """Patch StudioClient, and hand back a setter for the job list it returns."""
    from datachain.remote.studio import Response

    client = mocker.patch("datachain.remote.studio.StudioClient").return_value
    client.get_clusters.return_value = Response([], ok=True, message="", status=200)

    def serve(jobs):
        client.get_jobs.return_value = Response(jobs, ok=True, message="", status=200)
        return client

    return serve


def a_job(index: int, created: datetime, *, stages: bool = True) -> dict:
    """A completed job 20 minutes long: 30s queued, 2m setup, the rest running."""
    finished = created + timedelta(minutes=20)
    job: dict[str, Any] = {
        "id": f"job-{index}",
        "name": f"job-{index}",
        "status": "COMPLETE",
        "created_at": created.isoformat().replace("+00:00", "Z"),
        "finished_at": finished.isoformat().replace("+00:00", "Z"),
        "created_by": "ivan",
    }
    if stages:
        job["steps"] = [
            {
                "name": "waiting",
                "started_at": job["created_at"],
                "finished_at": (created + timedelta(seconds=30))
                .isoformat()
                .replace("+00:00", "Z"),
            },
            {
                "name": "running_query",
                "started_at": (created + timedelta(minutes=2))
                .isoformat()
                .replace("+00:00", "Z"),
                "finished_at": job["finished_at"],
            },
        ]
    return job


class TestFetchStageTimings:
    """Jobs are dated off the fetcher's own clock, so these never age out."""

    def test_every_job_is_timed_in_one_call(self, capsys, studio_jobs):
        """Stages ride on the list call, so no per-job cap can bound them.

        These used to be fetched one job at a time, stopping at 200: past that,
        jobs came back untimed while the output still claimed `enriched: true`,
        so Queue/Run totals quietly dropped them.
        """
        import jobs as jobs_module

        total = 250
        yesterday = jobs_module.datetime.now(tz=timezone.utc) - timedelta(days=1)
        client = studio_jobs([a_job(i, yesterday) for i in range(total)])

        jobs_module.cmd_fetch(days=30, limit=500, enrich=True)
        out = json.loads(capsys.readouterr().out)

        # One call, asking for steps - not one per job.
        client.get_jobs.assert_called_once_with(limit=500, include_steps=True)
        assert len(out["jobs"]) == total
        assert all(j["queue_seconds"] == 30 for j in out["jobs"])
        assert all(j["run_seconds"] == 1080 for j in out["jobs"])

    def test_stages_are_left_out_without_enrich(self, capsys, studio_jobs):
        import jobs as jobs_module

        yesterday = jobs_module.datetime.now(tz=timezone.utc) - timedelta(days=1)
        client = studio_jobs([a_job(1, yesterday, stages=False)])

        jobs_module.cmd_fetch(days=30, limit=500, enrich=False)
        out = json.loads(capsys.readouterr().out)

        client.get_jobs.assert_called_once_with(limit=500, include_steps=False)
        assert out["jobs"][0]["stages"] == {}
        assert out["jobs"][0]["queue_seconds"] is None
        # Duration never depended on stages.
        assert out["jobs"][0]["duration_seconds"] == 1200
