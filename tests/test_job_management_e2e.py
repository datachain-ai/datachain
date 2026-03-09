"""End-to-end tests for JobManager with real script execution."""

import os
import subprocess
import sys
import textwrap

import pytest

from datachain.data_storage import JobStatus
from tests.utils import skip_if_not_sqlite

python_exc = sys.executable or "python3"


@skip_if_not_sqlite
def test_single_job_for_multiple_saves(tmp_path, catalog_tmpfile):
    """
    Test that running a script with multiple .save() calls creates only one Job.

    Simulates: python my_script.py
    """
    script = tmp_path / "test_script.py"
    script_content = textwrap.dedent("""
        import datachain as dc

        dc.read_values(num=[1, 2, 3]).filter(
            dc.C("num") > 1
        ).save("nums_gt_1")

        dc.read_values(val=[10, 20, 30, 40]).filter(
            dc.C("val") < 35
        ).save("vals_lt_35")

        print("COMPLETED")
    """)
    script.write_text(script_content)

    result = subprocess.run(  # noqa: S603
        [python_exc, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
            "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
        },
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "COMPLETED" in result.stdout

    # Verify exactly ONE job was created
    query = catalog_tmpfile.metastore._jobs.select()
    jobs = list(catalog_tmpfile.metastore.db.execute(query))
    assert len(jobs) == 1, f"Expected exactly 1 job, but got {len(jobs)}"

    # Get the job for this script
    job = catalog_tmpfile.metastore.get_last_job_by_name(str(script))
    assert job is not None
    assert job.name == str(script)
    assert job.status == JobStatus.COMPLETE
    assert job.finished_at is not None
    assert job.query == ""

    # Verify both datasets were created with correct job_id
    dataset_versions = list(catalog_tmpfile.list_datasets_versions())
    dataset_names = {ds.name for ds, _, _ in dataset_versions}
    assert "nums_gt_1" in dataset_names
    assert "vals_lt_35" in dataset_names

    # Verify all dataset versions have the correct job_id
    for ds, version, _ in dataset_versions:
        assert version.job_id == job.id, (
            f"Dataset {ds.name} version has job_id={version.job_id}, expected {job.id}"
        )


@skip_if_not_sqlite
@pytest.mark.parametrize(
    "exception_code,expected_status,expected_returncode,check_error_details",
    [
        ("raise RuntimeError('Intentional failure')", JobStatus.FAILED, 1, True),
        ("raise KeyboardInterrupt()", JobStatus.CANCELED, 130, False),
    ],
    ids=["failed", "canceled"],
)
def test_job_marked_on_exception(
    tmp_path,
    catalog_tmpfile,
    exception_code,
    expected_status,
    expected_returncode,
    check_error_details,
):
    """
    Test that when a script raises an exception, the Job is marked appropriately.
    - RuntimeError -> FAILED with error details
    - KeyboardInterrupt -> CANCELED without error details
    Datasets persist even with unhandled exceptions.
    """
    script = tmp_path / "test_exception.py"
    script_content = textwrap.dedent(f"""
        import datachain as dc

        dc.read_values(a=[1, 2, 3]).save("dataset_a")
        dc.read_values(b=[4, 5, 6]).save("dataset_b")
        {exception_code}
    """)
    script.write_text(script_content)

    result = subprocess.run(  # noqa: S603
        [python_exc, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
            "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
        },
    )

    assert result.returncode == expected_returncode
    if check_error_details:
        assert "Intentional failure" in result.stderr
    else:
        assert "KeyboardInterrupt" in result.stderr

    # Verify exactly ONE job was created
    query = catalog_tmpfile.metastore._jobs.select()
    jobs = list(catalog_tmpfile.metastore.db.execute(query))
    assert len(jobs) == 1, f"Expected exactly 1 job, but got {len(jobs)}"

    # Get the job for this script
    job = catalog_tmpfile.metastore.get_last_job_by_name(str(script))
    assert job is not None
    assert job.name == str(script)
    assert job.status == expected_status

    if check_error_details:
        assert "Intentional failure" in job.error_message
        assert "RuntimeError" in job.error_stack
    else:
        # KeyboardInterrupt should not set error message/stack
        assert job.error_message == ""
        assert job.error_stack == ""

    assert job.query == ""

    # Verify datasets persisted even with unhandled exceptions
    dataset_versions = list(catalog_tmpfile.list_datasets_versions())
    dataset_names = {ds.name for ds, _, _ in dataset_versions}
    assert "dataset_a" in dataset_names
    assert "dataset_b" in dataset_names
