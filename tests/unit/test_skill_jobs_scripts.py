"""Tests for jobs skill script helpers."""

import sys
from datetime import datetime, timezone
from pathlib import Path

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
    _strip_ordinal,
)

# Also test the duplicated frontmatter parser
GRAPH_SCRIPTS_DIR = str(
    Path(__file__).resolve().parents[2] / "src/datachain/skill/knowledge/scripts"
)
if GRAPH_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, GRAPH_SCRIPTS_DIR)

from utils import read_frontmatter  # noqa: E402

# ---------------------------------------------------------------------------
# _parse_dt
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _normalize_status
# ---------------------------------------------------------------------------


class TestNormalizeStatus:
    def test_lowercase(self):
        assert _normalize_status("Complete") == "complete"

    def test_already_lower(self):
        assert _normalize_status("failed") == "failed"

    def test_upper(self):
        assert _normalize_status("RUNNING") == "running"


# ---------------------------------------------------------------------------
# _duration_str
# ---------------------------------------------------------------------------


class TestDurationStr:
    def test_seconds(self):
        assert _duration_str(60) == "60s"

    def test_large(self):
        assert _duration_str(9000) == "9000s"

    def test_zero(self):
        assert _duration_str(0) == "0s"


# ---------------------------------------------------------------------------
# _strip_ordinal
# ---------------------------------------------------------------------------


class TestStripOrdinal:
    def test_third(self):
        assert _strip_ordinal("job-123 (3rd)") == "job-123"

    def test_fourth(self):
        assert _strip_ordinal("run (4th)") == "run"

    def test_no_ordinal(self):
        assert _strip_ordinal("plain-value") == "plain-value"

    def test_empty(self):
        assert _strip_ordinal("") == ""


# ---------------------------------------------------------------------------
# _read_frontmatter (duplicated in jobs.py)
# ---------------------------------------------------------------------------


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
