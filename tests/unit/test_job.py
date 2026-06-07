import pytest

from datachain.job import Job


def test_parse():
    """Test that Job.parse returns a valid Job."""
    job = Job.parse(
        id="test-id",
        name="test-job",
        status=1,
        created_at="2024-01-01T00:00:00",
        finished_at=None,
        query="SELECT 1",
        query_type=1,
        workers=1,
        python_version="3.11",
        error_message="",
        error_stack="",
        params="{}",
        metrics="{}",
        parent_job_id=None,
        rerun_from_job_id=None,
        run_group_id="group-1",
    )

    assert job.id == "test-id"
    assert job.name == "test-job"
    assert job.run_group_id == "group-1"
    assert job.rerun_from_job_id is None


@pytest.mark.parametrize(
    "value,expected",
    [
        # SQLite returns JSON columns as raw strings.
        ('{"a": "b"}', {"a": "b"}),
        # PostgreSQL/JSONB deserializes JSON columns to Python objects.
        ({"a": "b"}, {"a": "b"}),
        # Legacy double-encoded rows decode to a string scalar like '{}'.
        ("{}", {}),
        # None / empty-string edge cases (empty JSON in some backends).
        (None, {}),
        ("", {}),
    ],
)
def test_parse_params_metrics_accepts_str_and_dict(value, expected):
    """params/metrics may arrive as a str (SQLite) or a dict (PostgreSQL/JSONB).

    Job.parse must normalize both to a dict without re-decoding a dict.
    """
    job = Job.parse(
        id="test-id",
        name="test-job",
        status=1,
        created_at="2024-01-01T00:00:00",
        finished_at=None,
        query="SELECT 1",
        query_type=1,
        workers=1,
        python_version=None,
        error_message="",
        error_stack="",
        params=value,
        metrics=value,
        parent_job_id=None,
        rerun_from_job_id=None,
        run_group_id=None,
    )

    assert job.params == expected
    assert job.metrics == expected
