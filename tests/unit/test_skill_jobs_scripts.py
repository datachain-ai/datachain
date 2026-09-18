"""Tests for jobs skill script helpers."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

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


class TestFetchStageTimings:
    def test_every_job_is_timed_past_the_enrich_limit(self, capsys, monkeypatch):
        """Stages ride on the list call, so the per-job enrich cap does not bound them.

        201 completed jobs against a cap of 200 used to leave the last one untimed
        while the output still claimed `enriched: true`, so Queue/Run totals quietly
        dropped it.
        """
        import jobs as jobs_module

        total = jobs_module.ENRICH_LIMIT + 1
        listed = [
            {
                "id": f"job-{i}",
                "name": f"job-{i}",
                "status": "COMPLETE",
                "created_at": "2026-09-16T00:00:00Z",
                "finished_at": "2026-09-16T00:20:00Z",
                "created_by": "ivan",
                "steps": [
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
                ],
            }
            for i in range(total)
        ]

        class Response:
            def __init__(self, data):
                self.ok, self.data, self.message = True, data, ""

        class FakeClient:
            list_calls: ClassVar[list[bool]] = []

            def __init__(self, *args, **kwargs):
                pass

            def get_clusters(self):
                return Response([])

            def get_jobs(self, *args, job_id=None, include_steps=False, **kwargs):
                if job_id is not None:
                    return Response([{"id": job_id, "workers": 2}])
                FakeClient.list_calls.append(include_steps)
                return Response(listed)

        import datachain.remote.studio as studio_module

        monkeypatch.setattr(studio_module, "StudioClient", FakeClient)

        jobs_module.cmd_fetch(days=30, limit=500, enrich=True)
        out = json.loads(capsys.readouterr().out)

        assert FakeClient.list_calls == [True], "the list call must ask for steps"
        assert len(out["jobs"]) == total
        assert all(j["queue_seconds"] == 30 for j in out["jobs"])
        assert all(j["run_seconds"] == 1050 for j in out["jobs"])

    def test_stages_are_left_out_without_enrich(self, capsys, monkeypatch):
        import jobs as jobs_module

        class Response:
            def __init__(self, data):
                self.ok, self.data, self.message = True, data, ""

        class FakeClient:
            include_steps_asked = None

            def __init__(self, *args, **kwargs):
                pass

            def get_clusters(self):
                return Response([])

            def get_jobs(self, *args, job_id=None, include_steps=False, **kwargs):
                FakeClient.include_steps_asked = include_steps
                return Response(
                    [
                        {
                            "id": "job-1",
                            "name": "job-1",
                            "status": "COMPLETE",
                            "created_at": "2026-09-16T00:00:00Z",
                            "created_by": "ivan",
                        }
                    ]
                )

        import datachain.remote.studio as studio_module

        monkeypatch.setattr(studio_module, "StudioClient", FakeClient)

        jobs_module.cmd_fetch(days=30, limit=500, enrich=False)
        out = json.loads(capsys.readouterr().out)

        assert FakeClient.include_steps_asked is False
        assert out["jobs"][0]["stages"] == {}
        assert out["jobs"][0]["queue_seconds"] is None
