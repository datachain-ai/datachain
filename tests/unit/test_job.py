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
