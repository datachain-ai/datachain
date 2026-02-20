from datachain.job import Job


def test_create_ephemeral():
    """Test that create_ephemeral returns a valid in-memory Job."""
    job = Job.create_ephemeral()

    assert job.id
    assert job.run_group_id == job.id
    assert job.rerun_from_job_id is None
    assert job.name == ""


def test_create_ephemeral_unique_ids():
    """Test that each ephemeral job gets a unique id."""
    job1 = Job.create_ephemeral()
    job2 = Job.create_ephemeral()

    assert job1.id != job2.id
    assert job1.run_group_id != job2.run_group_id
