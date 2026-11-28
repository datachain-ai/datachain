"""End-to-end tests for JobManager with real script execution."""

import subprocess
import sys
import textwrap

import pytest

from datachain.data_storage import JobStatus
from tests.utils import e2e_subprocess_env, skip_if_not_sqlite

python_exc = sys.executable or "python3"


def _run_script(script, catalog):
    return subprocess.run(  # noqa: S603
        [python_exc, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=e2e_subprocess_env(catalog),
    )


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

    result = _run_script(script, catalog_tmpfile)

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "COMPLETED" in result.stdout

    # Verify exactly ONE job was created
    query = catalog_tmpfile.metastore._jobs.select()
    jobs = list(catalog_tmpfile.metastore.db.execute(query))
    assert len(jobs) == 1, f"Expected exactly 1 job, but got {len(jobs)}"

    job = catalog_tmpfile.metastore.get_last_job_by_name(str(script))
    assert job is not None
    assert job.name == str(script)
    assert job.status == JobStatus.COMPLETE
    assert job.finished_at is not None
    assert job.query == script_content

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


SAVES_IN_DEFAULT_SESSION = """
    import datachain as dc

    dc.read_values(a=[1, 2, 3]).save("dataset_a")
    dc.read_values(b=[4, 5, 6]).save("dataset_b")
    {exception_code}
"""

SAVES_IN_OWN_SESSION = """
    import datachain as dc

    with dc.Session("own"):
        dc.read_values(a=[1, 2, 3]).save("dataset_a")
        dc.read_values(b=[4, 5, 6]).save("dataset_b")
        {exception_code}
"""


@skip_if_not_sqlite
@pytest.mark.parametrize(
    "script_template",
    [SAVES_IN_DEFAULT_SESSION, SAVES_IN_OWN_SESSION],
    ids=["default_session", "own_session"],
)
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
    script_template,
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
    script_content = textwrap.dedent(script_template).format(
        exception_code=exception_code
    )
    script.write_text(script_content)

    result = _run_script(script, catalog_tmpfile)

    assert result.returncode == expected_returncode
    if check_error_details:
        assert "Intentional failure" in result.stderr
    else:
        assert "KeyboardInterrupt" in result.stderr

    # Verify exactly ONE job was created
    query = catalog_tmpfile.metastore._jobs.select()
    jobs = list(catalog_tmpfile.metastore.db.execute(query))
    assert len(jobs) == 1, f"Expected exactly 1 job, but got {len(jobs)}"

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

    assert job.query == script_content

    # Verify datasets persisted even with unhandled exceptions
    dataset_versions = list(catalog_tmpfile.list_datasets_versions())
    dataset_names = {ds.name for ds, _, _ in dataset_versions}
    assert "dataset_a" in dataset_names
    assert "dataset_b" in dataset_names


@skip_if_not_sqlite
def test_script_error_after_wrapping_the_except_hook_marks_job_failed(
    tmp_path, catalog_tmpfile
):
    script = tmp_path / "script.py"
    script.write_text(
        textwrap.dedent("""
        import sys

        import datachain as dc

        chain = dc.read_values(a=[1])
        previous = sys.excepthook

        def wrapper(*args):
            print("wrapper called", file=sys.stderr)
            previous(*args)

        sys.excepthook = wrapper
        chain.save("a")
        raise RuntimeError("Intentional failure")
    """)
    )

    result = _run_script(script, catalog_tmpfile)

    assert result.returncode == 1
    assert result.stderr.count("wrapper called") == 1
    assert "Intentional failure" in result.stderr
    job = catalog_tmpfile.metastore.get_last_job_by_name(str(script))
    assert job is not None
    assert job.status == JobStatus.FAILED, result.stderr


@skip_if_not_sqlite
def test_script_error_after_saving_on_a_worker_thread_marks_job_failed(
    tmp_path, catalog_tmpfile
):
    script = tmp_path / "script.py"
    script.write_text(
        textwrap.dedent("""
        from concurrent.futures import ThreadPoolExecutor

        import datachain as dc

        dc.read_values(a=[1])

        def save():
            session = dc.Session("worker")
            dc.read_values(a=[1], session=session).save("a")
            return session

        with ThreadPoolExecutor() as pool:
            session = pool.submit(save).result()
        raise RuntimeError("Intentional failure")
    """)
    )

    result = _run_script(script, catalog_tmpfile)

    assert result.returncode == 1
    job = catalog_tmpfile.metastore.get_last_job_by_name(str(script))
    assert job is not None
    assert job.status == JobStatus.FAILED, result.stderr


@skip_if_not_sqlite
def test_udf_error_inside_its_own_session_marks_job_failed(tmp_path, catalog_tmpfile):
    script = tmp_path / "script.py"
    script.write_text(
        textwrap.dedent("""
        import datachain as dc

        def fail(a: int) -> int:
            raise RuntimeError("Intentional failure")

        with dc.Session("own"):
            dc.read_values(a=[1, 2, 3]).map(b=fail).save("dataset_a")
    """)
    )

    result = _run_script(script, catalog_tmpfile)

    assert result.returncode == 1
    job = catalog_tmpfile.metastore.get_last_job_by_name(str(script))
    assert job is not None
    assert job.status == JobStatus.FAILED, result.stderr
    assert "Intentional failure" in job.error_message


@skip_if_not_sqlite
def test_script_error_before_any_job_prints_traceback_once_and_stores_no_job(
    tmp_path, catalog_tmpfile
):
    script = tmp_path / "script.py"
    script.write_text(
        textwrap.dedent("""
        import datachain as dc

        dc.read_dataset("missing")
    """)
    )

    result = _run_script(script, catalog_tmpfile)

    assert result.returncode == 1
    assert result.stderr.count("Dataset missing not found") == 1
    assert catalog_tmpfile.metastore.get_last_job_by_name(str(script)) is None
