import io
import os
import sys
from uuid import uuid4

import pytest

from datachain import json
from datachain.catalog.catalog import (
    QUERY_SCRIPT_CANCELED_EXIT_CODE,
    TerminationSignal,
)
from datachain.error import QueryScriptCancelError, QueryScriptRunError


@pytest.fixture
def mock_popen(mocker):
    m = mocker.patch("subprocess.Popen", returncode=0)
    m.return_value.__enter__.return_value = m
    return m


def test(catalog, mock_popen):
    catalog.query("pass")

    expected_env = os.environ | {
        "DATACHAIN_QUERY_PARAMS": "{}",
        "DATACHAIN_JOB_ID": "",
        "DATACHAIN_CHECKPOINTS_RESET": "False",
    }
    mock_popen.assert_called_once_with([sys.executable, "-c", "pass"], env=expected_env)


@pytest.mark.parametrize("reset", [True, False])
def test_args(catalog, mock_popen, reset):
    params = {"param": "value"}
    job_id = str(uuid4())
    env = {"env1": "value1", "env2": "value2"}
    catalog.query(
        "pass",
        env=env,
        python_executable="mypython",
        params=params,
        job_id=job_id,
        reset=reset,
    )

    expected_env = env | {
        "DATACHAIN_QUERY_PARAMS": json.dumps(params),
        "DATACHAIN_JOB_ID": job_id,
        "DATACHAIN_CHECKPOINTS_RESET": str(reset) if reset is not None else str(False),
    }
    mock_popen.assert_called_once_with(["mypython", "-c", "pass"], env=expected_env)


def test_capture_stdout(catalog, mock_popen):
    mock_popen.stdout = io.BytesIO(b"Hello, World!\rLorem Ipsum\nDolor Sit Amet\nconse")
    stdout = []

    catalog.query("pass", stdout_callback=stdout.append)
    assert stdout == ["Hello, World!\r", "Lorem Ipsum\n", "Dolor Sit Amet\n", "conse"]


def test_capture_stderr(catalog, mock_popen):
    mock_popen.stderr = io.BytesIO(b"Hello, World!\rLorem Ipsum\nDolor Sit Amet\nconse")
    stderr = []

    catalog.query("pass", stderr_callback=stderr.append)
    assert stderr == ["Hello, World!\r", "Lorem Ipsum\n", "Dolor Sit Amet\n", "conse"]


def test_capture_output(catalog, mock_popen):
    mock_popen.stdout = io.BytesIO(b"Hello, World!\rLorem Ipsum\nDolor Sit Amet\nconse")
    mock_popen.stderr = io.BytesIO(b"foo\nbar")
    stdout = []
    stderr = []

    catalog.query("pass", stdout_callback=stdout.append, stderr_callback=stderr.append)
    assert stdout == ["Hello, World!\r", "Lorem Ipsum\n", "Dolor Sit Amet\n", "conse"]
    assert stderr == ["foo\n", "bar"]


def test_canceled_by_user(catalog, mock_popen):
    mock_popen.returncode = QUERY_SCRIPT_CANCELED_EXIT_CODE

    with pytest.raises(QueryScriptCancelError) as e:
        catalog.query("pass")
    assert e.value.return_code == QUERY_SCRIPT_CANCELED_EXIT_CODE
    assert "Query script was canceled by user" in str(e.value)


def test_non_zero_exitcode(catalog, mock_popen):
    mock_popen.returncode = 1

    with pytest.raises(QueryScriptRunError) as e:
        catalog.query("pass")
    assert e.value.return_code == 1
    assert "Query script exited with error code 1" in str(e.value)


def test_shutdown_process_on_sigterm(mocker, catalog, mock_popen):
    mock_popen.returncode = -2
    mock_popen.wait.side_effect = [TerminationSignal(15)]
    m = mocker.patch("datachain.catalog.catalog.shutdown_process", return_value=-2)

    with pytest.raises(QueryScriptCancelError) as e:
        catalog.query("pass", interrupt_timeout=0.1, terminate_timeout=0.2)
    assert e.value.return_code == -2
    assert "Query script was canceled by user" in str(e.value)
    m.assert_called_once_with(mock_popen, 0.1, 0.2)
