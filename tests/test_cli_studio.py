import json
import logging
import re
import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest
import requests
import requests_mock
import websockets
from dvc_studio_client.auth import AuthorizationExpiredError
from tabulate import tabulate

from datachain.cli import main
from datachain.config import Config, ConfigLevel
from datachain.job import Job
from datachain.studio import POST_LOGIN_MESSAGE
from datachain.utils import STUDIO_URL
from tests.utils import skip_if_not_sqlite


def mocked_connect(url, additional_headers):
    async def mocked_recv():
        raise websockets.exceptions.ConnectionClosed("Connection closed")

    async def mocked_send(message):
        pass

    async def mocked_close():
        pass

    assert additional_headers == {"Authorization": "token isat_access_token"}
    mocked_websocket = MagicMock()
    mocked_websocket.recv = mocked_recv
    mocked_websocket.send = mocked_send
    mocked_websocket.close = mocked_close
    return mocked_websocket


def test_studio_login_token_check_failed(mocker):
    mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        side_effect=AuthorizationExpiredError,
    )
    assert main(["auth", "login"]) == 1


def test_studio_login_success(mocker):
    mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        return_value=("token_name", "isat_access_token"),
    )

    assert main(["auth", "login"]) == 0

    config = Config().read()
    assert config["studio"]["token"] == "isat_access_token"  # noqa: S105 # nosec B105
    assert config["studio"]["url"] == STUDIO_URL


def test_studio_login_arguments(mocker):
    mock = mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        return_value=("token_name", "isat_access_token"),
    )

    assert (
        main(
            [
                "auth",
                "login",
                "--name",
                "token_name",
                "--hostname",
                "https://example.com",
                "--scopes",
                "experiments",
                "--no-open",
            ]
        )
        == 0
    )

    mock.assert_called_with(
        token_name="token_name",  #  noqa: S106
        hostname="https://example.com",
        scopes="experiments",
        client_name="DataChain",
        open_browser=False,
        post_login_message=POST_LOGIN_MESSAGE,
    )


def test_studio_logout():
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "isat_access_token"}

    assert main(["auth", "logout"]) == 0
    config = Config(ConfigLevel.GLOBAL).read()
    assert "token" not in config["studio"]

    assert main(["auth", "logout"]) == 1


def test_studio_token(capsys):
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "isat_access_token"}

    assert main(["auth", "token"]) == 0
    assert capsys.readouterr().out == "isat_access_token\n"

    with Config(ConfigLevel.GLOBAL).edit() as conf:
        del conf["studio"]["token"]

    assert main(["auth", "token"]) == 1


def test_studio_team_local():
    assert main(["auth", "team", "team_name"]) == 0
    config = Config(ConfigLevel.GLOBAL).read()
    assert config["studio"]["team"] == "team_name"


def test_studio_team_global():
    assert main(["auth", "team", "team_name", "--local"]) == 0
    config = Config(ConfigLevel.LOCAL).read()
    assert config["studio"]["team"] == "team_name"


def test_studio_datasets(capsys, studio_datasets, mocker):
    def list_datasets_local(_, __):
        yield "local.local.local", "1.0.0"
        yield "dev.animals.both", "1.0.0"

    mocker.patch(
        "datachain.cli.commands.datasets.list_datasets_local",
        side_effect=list_datasets_local,
    )
    local_rows = [
        {"Name": "dev.animals.both", "Latest Version": "v1.0.0"},
        {"Name": "local.local.local", "Latest Version": "v1.0.0"},
    ]
    local_output = tabulate(local_rows, headers="keys")

    studio_rows = [
        {"Name": "dev.animals.both", "Latest Version": "v1.0.0"},
        {
            "Name": "dev.animals.cats",
            "Latest Version": "v1.0.0",
        },
        {"Name": "dev.animals.dogs", "Latest Version": "v2.0.0"},
    ]
    studio_output = tabulate(studio_rows, headers="keys")

    both_rows = [
        {"Name": "dev.animals.both", "Studio": "v1.0.0", "Local": "v1.0.0"},
        {"Name": "dev.animals.cats", "Studio": "v1.0.0", "Local": "\u2716"},
        {"Name": "dev.animals.dogs", "Studio": "v2.0.0", "Local": "\u2716"},
        {"Name": "local.local.local", "Studio": "\u2716", "Local": "v1.0.0"},
    ]
    both_output = tabulate(both_rows, headers="keys")

    both_rows_versions = [
        {"Name": "dev.animals.both", "Studio": "v1.0.0", "Local": "v1.0.0"},
        {"Name": "dev.animals.cats", "Studio": "v1.0.0", "Local": "\u2716"},
        {"Name": "dev.animals.dogs", "Studio": "v1.0.0", "Local": "\u2716"},
        {"Name": "dev.animals.dogs", "Studio": "v2.0.0", "Local": "\u2716"},
        {"Name": "local.local.local", "Studio": "\u2716", "Local": "v1.0.0"},
    ]
    both_output_versions = tabulate(both_rows_versions, headers="keys")

    dogs_rows = [
        {"Name": "dogs", "Latest Version": "v1.0.0"},
        {"Name": "dogs", "Latest Version": "v2.0.0"},
    ]
    dogs_output = tabulate(dogs_rows, headers="keys")

    assert main(["dataset", "ls", "--local"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(local_output.splitlines())

    assert main(["dataset", "ls", "--studio"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(studio_output.splitlines())

    assert main(["dataset", "ls", "--local", "--studio"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(both_output.splitlines())

    assert main(["dataset", "ls", "--all"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(both_output.splitlines())

    assert main(["dataset", "ls"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(both_output.splitlines())

    assert main(["dataset", "ls", "--versions"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(both_output_versions.splitlines())

    assert main(["dataset", "ls", "dev.animals.dogs", "--studio"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(dogs_output.splitlines())


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_studio_edit_dataset(capsys, mocker):
    with requests_mock.mock() as m:
        m.post(f"{STUDIO_URL}/api/datachain/datasets", json={})

        # Studio token is required
        assert (
            main(
                [
                    "dataset",
                    "edit",
                    "dev.animals.name",
                    "--new-name",
                    "new-name",
                    "--team",
                    "team_name",
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "Not logged in to Studio" in out

        # Set the studio token
        with Config(ConfigLevel.GLOBAL).edit() as conf:
            conf["studio"] = {"token": "isat_access_token", "team": "team_name"}

        assert (
            main(
                [
                    "dataset",
                    "edit",
                    "dev.animals.name",
                    "--new-name",
                    "new-name",
                    "--team",
                    "team_name",
                ]
            )
            == 0
        )

        assert m.called

        last_request = m.last_request
        assert last_request.json() == {
            "name": "name",
            "namespace": "dev",
            "project": "animals",
            "new_name": "new-name",
            "team_name": "team_name",
            "description": None,
            "attrs": None,
        }

        # With all arguments
        assert (
            main(
                [
                    "dataset",
                    "edit",
                    "dev.animals.name",
                    "--new-name",
                    "new-name",
                    "--description",
                    "description",
                    "--attrs",
                    "attr1",
                    "--team",
                    "team_name",
                ]
            )
            == 0
        )
        last_request = m.last_request
        assert last_request.json() == {
            "name": "name",
            "namespace": "dev",
            "project": "animals",
            "new_name": "new-name",
            "description": "description",
            "attrs": ["attr1"],
            "team_name": "team_name",
        }


@skip_if_not_sqlite
def test_studio_rm_dataset(capsys, mocker):
    with requests_mock.mock() as m:
        m.delete(f"{STUDIO_URL}/api/datachain/datasets", json={})

        # Studio token is required
        assert (
            main(
                ["dataset", "rm", "dev.animals.name", "--team", "team_name", "--studio"]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "Not logged in to Studio" in out

        # Set the studio token
        with Config(ConfigLevel.GLOBAL).edit() as conf:
            conf["studio"] = {"token": "isat_access_token", "team": "team_name"}

        assert (
            main(
                [
                    "dataset",
                    "rm",
                    "dev.animals.name",
                    "--team",
                    "team_name",
                    "--version",
                    "1.0.0",
                    "--force",
                    "--studio",
                ]
            )
            == 0
        )
        assert m.called

        last_request = m.last_request
        assert last_request.json() == {
            "name": "name",
            "namespace": "dev",
            "project": "animals",
            "team_name": "team_name",
            "version": "1.0.0",
            "force": True,
        }


def test_studio_cancel_job(capsys, mocker):
    job_id = "8bddde6c-c3ca-41b0-9d87-ee945bfdce70"
    with requests_mock.mock() as m:
        m.post(f"{STUDIO_URL}/api/datachain/jobs/{job_id}/cancel", json={})

        # Studio token is required
        assert main(["job", "cancel", job_id]) == 1
        out = capsys.readouterr().err
        assert "Not logged in to Studio" in out

        # Set the studio token
        with Config(ConfigLevel.GLOBAL).edit() as conf:
            conf["studio"] = {"token": "isat_access_token", "team": "team_name"}

        assert main(["job", "cancel", job_id]) == 0
        assert m.called


def test_studio_run(capsys, mocker, tmp_dir):
    mocker.patch(
        "datachain.remote.studio.websockets.connect", side_effect=mocked_connect
    )
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "isat_access_token", "team": "team_name"}

    job_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/files?team_name=team_name", json={"id": 1}
        )
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json=[{"status": "COMPLETE"}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={
                "dataset_versions": [
                    {"dataset_name": "dataset_name", "version": "1.0.0"}
                ]
            },
        )

        (tmp_dir / "env_file.txt").write_text("ENV_FROM_FILE=1")
        (tmp_dir / "reqs.txt").write_text("pyjokes")
        (tmp_dir / "file.txt").write_text("file content")
        (tmp_dir / "example_query.py").write_text("print(1)")

        assert (
            main(
                [
                    "job",
                    "run",
                    "example_query.py",
                    "--env-file",
                    "env_file.txt",
                    "--env",
                    "ENV_FROM_ARGS=1",
                    "--env",
                    "ENV2=2",
                    "ENV3=3",
                    "--workers",
                    "2",
                    "--files",
                    "file.txt",
                    "--python-version",
                    "3.12",
                    "--req-file",
                    "reqs.txt",
                    "--req",
                    "stupidity",
                    "--repository",
                    "https://github.com/datachain-ai/datachain",
                    "--cluster",
                    "default",
                    "--credentials-name",
                    "my-credentials",
                ]
            )
            == 0
        )

    out = capsys.readouterr().out
    assert (
        out.strip()
        == f"Job {job_id} created\nOpen the job in Studio at https://example.com\n"
        "========================================\n\n"
        ">>>> Job is now in COMPLETE status.\n\n\n"
        ">>>> Dataset versions created during the job:\n"
        "    - dataset_name@v1.0.0"
    )

    first_request = m.request_history[0]
    second_request = m.request_history[1]

    assert first_request.method == "POST"
    assert (
        first_request.url
        == f"{STUDIO_URL}/api/datachain/jobs/files?team_name=team_name"
    )
    # Check that it's multipart/form-data request
    assert "multipart/form-data" in first_request.headers.get("Content-Type", "")
    # Check query parameters
    assert first_request.qs["team_name"] == ["team_name"]

    assert second_request.method == "POST"
    assert second_request.url == f"{STUDIO_URL}/api/datachain/jobs/"
    assert second_request.json() == {
        "query": "print(1)",
        "query_type": "PYTHON",
        "environment": "ENV_FROM_FILE=1\nENV_FROM_ARGS=1\nENV2=2\nENV3=3",
        "workers": 2,
        "query_name": "example_query.py",
        "files": ["1"],
        "python_version": "3.12",
        "requirements": "pyjokes\nstupidity",
        "team_name": "team_name",
        "repository": "https://github.com/datachain-ai/datachain",
        "priority": 5,
        "compute_cluster_name": "default",
        "start_after": None,
        "rerun_from_job_id": None,
        "reset": False,
        "cron_expression": None,
        "credentials_name": "my-credentials",
    }


def test_studio_run_task(capsys, mocker, tmp_dir, studio_token):
    mocker.patch(
        "datachain.remote.studio.websockets.connect", side_effect=mocked_connect
    )

    job_id = "b2c3d4e5-f6a7-8901-bcde-f12345678901"
    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={
                "dataset_versions": [
                    {"dataset_name": "dataset_name", "version": "1.0.0"}
                ]
            },
        )
        (tmp_dir / "example_query.py").write_text("print(1)")

        assert (
            main(
                [
                    "job",
                    "run",
                    "example_query.py",
                    "--start-time",
                    "tomorrow 3pm",
                    "--cron",
                    "0 0 * * *",
                ]
            )
            == 0
        )
    first_request = m.request_history[0]
    assert first_request.method == "POST"
    assert first_request.url == f"{STUDIO_URL}/api/datachain/jobs/"
    request_json = first_request.json()
    assert request_json["start_after"] is not None
    assert request_json["cron_expression"] is not None

    assert request_json["start_after"] is not None
    assert request_json["cron_expression"] == "0 0 * * *"


@skip_if_not_sqlite
def test_studio_run_reuses_previous_job_for_checkpoints(
    capsys, mocker, tmp_dir, studio_token
):
    mocker.patch(
        "datachain.remote.studio.websockets.connect", side_effect=mocked_connect
    )

    first_job_id = "first-job-uuid-1234"
    second_job_id = "second-job-uuid-5678"

    script_file = tmp_dir / "example_query.py"
    script_file.write_text("print(1)")
    script_path = str(script_file.resolve())

    parent_job = Job(
        id=first_job_id,
        name=script_path,
        status=5,  # COMPLETE
        created_at=datetime.now(),
        query="print(1)",
        query_type=1,
        workers=1,
        params={},
        metrics={},
        is_remote_execution=True,
    )

    get_last_job_calls = []
    create_job_calls = []

    def mock_get_last_job_by_name(name, is_remote_execution=False, conn=None):
        get_last_job_calls.append(
            {"name": name, "is_remote_execution": is_remote_execution}
        )
        if len(get_last_job_calls) == 1:
            return None
        return parent_job

    def mock_create_job(**kwargs):
        create_job_calls.append(kwargs)
        return kwargs.get("job_id", "generated-id")

    mock_metastore = mocker.MagicMock()
    mock_metastore.get_last_job_by_name = mock_get_last_job_by_name
    mock_metastore.create_job = mock_create_job

    mock_catalog = mocker.MagicMock()
    mock_catalog.metastore = mock_metastore

    mocker.patch("datachain.studio.get_catalog", return_value=mock_catalog)

    with requests_mock.mock() as m:
        # First job run - no parent
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={
                "id": first_job_id,
                "url": "https://example.com/job/1",
                "workers": 1,
                "python_version": "3.11",
                "params": {},
                "parent_job_id": None,
                "rerun_from_job_id": None,
                "run_group_id": first_job_id,  # First job has run_group_id = its own id
            },
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json=[{"status": "COMPLETE"}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={first_job_id}&team_name=team_name",
            json={"dataset_versions": []},
        )

        assert main(["job", "run", str(script_file)]) == 0

        first_request = m.request_history[0]
        assert first_request.json()["rerun_from_job_id"] is None

        assert len(get_last_job_calls) == 1
        assert get_last_job_calls[0]["is_remote_execution"] is True
        assert get_last_job_calls[0]["name"] == script_path

        assert len(create_job_calls) == 1
        assert create_job_calls[0]["is_remote_execution"] is True
        assert create_job_calls[0]["job_id"] == first_job_id
        assert create_job_calls[0]["name"] == script_path

        m.reset_mock()

        # Second job run - should find parent
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={
                "id": second_job_id,
                "url": "https://example.com/job/2",
                "workers": 1,
                "python_version": "3.11",
                "params": {},
                "parent_job_id": first_job_id,
                "rerun_from_job_id": first_job_id,
                "run_group_id": first_job_id,  # Same run_group_id as parent
            },
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={second_job_id}&team_name=team_name",
            json={"dataset_versions": []},
        )

        assert main(["job", "run", str(script_file)]) == 0

        second_request = m.request_history[0]
        assert second_request.json()["rerun_from_job_id"] == first_job_id

        assert len(get_last_job_calls) == 2
        assert get_last_job_calls[1]["is_remote_execution"] is True

        assert len(create_job_calls) == 2
        assert create_job_calls[1]["is_remote_execution"] is True
        assert create_job_calls[1]["job_id"] == second_job_id
        assert create_job_calls[1]["rerun_from_job_id"] == first_job_id


@pytest.mark.parametrize(
    "status,expected_exit_code", [("FAILED", 1), ("CANCELED", 2), ("COMPLETE", 0)]
)
def test_studio_run_non_zero_exit_code(
    capsys, mocker, tmp_dir, status, expected_exit_code, studio_token
):
    job_id = str(uuid.uuid4())

    # Mock tail_job_logs to return a status
    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"logs": [{"message": "Starting job...\n"}]}
        yield {"logs": [{"message": "Processing data...\n"}]}
        yield {"job": {"status": status}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": status}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={
                "dataset_versions": [
                    {"dataset_name": "dataset_name", "version": "1.0.0"}
                ]
            },
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        assert (
            main(
                [
                    "job",
                    "run",
                    "example_query.py",
                ]
            )
            == expected_exit_code
        )

    out = capsys.readouterr().out
    assert (
        out.strip()
        == f"Job {job_id} created\nOpen the job in Studio at https://example.com\n"
        "========================================\n"
        "Starting job...\n"
        "Processing data...\n"
        "\n"
        f">>>> Job is now in {status} status.\n\n\n"
        ">>>> Dataset versions created during the job:\n"
        "    - dataset_name@v1.0.0"
    )


def test_studio_run_websocket_disconnect_fetches_status_via_rest(
    capsys, mocker, tmp_dir, studio_token
):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"logs": [{"message": "Starting job...\n"}]}
        yield {"job": {"status": "RUNNING"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "COMPLETE"}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={
                "dataset_versions": [
                    {"dataset_name": "test_dataset", "version": "1.0.0"}
                ]
            },
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        exit_code = main(
            [
                "job",
                "run",
                "example_query.py",
            ]
        )

        assert exit_code == 0

    out = capsys.readouterr().out
    assert ">>>> Job is now in RUNNING status." in out
    assert ">>>> Job is now in COMPLETE status." in out
    assert ">>>> Dataset versions created during the job:" in out


def test_studio_run_websocket_disconnect_job_still_running(
    capsys, mocker, tmp_dir, studio_token
):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"logs": [{"message": "Starting job...\n"}]}
        yield {"job": {"status": "RUNNING"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )

    mocker.patch("datachain.studio.RETRY_MAX_TIMES", 0)
    mocker.patch("datachain.studio.RETRY_SLEEP_SEC", 0.01)

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "RUNNING"}],
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        exit_code = main(
            [
                "job",
                "run",
                "example_query.py",
            ]
        )

        # Should return 1 because job is still running (lost connection)
        assert exit_code == 1

    out = capsys.readouterr().out
    assert ">>>> Job is now in RUNNING status." in out
    assert ">>>> Lost connection." in out
    # Should NOT show dataset versions since job didn't complete
    assert ">>>> Dataset versions created during the job:" not in out


def test_studio_run_invalid_job_status(caplog, capsys, mocker, tmp_dir, studio_token):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"job": {"status": "INVALID_STATUS"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "INVALID_STATUS"}],
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        with caplog.at_level(logging.DEBUG, logger="datachain"):
            exit_code = main(["job", "run", "-v", "example_query.py"])

        assert exit_code == 1

    assert "Job status is not a valid status: INVALID_STATUS" in caplog.text
    assert "Job is not finished: INVALID_STATUS" in caplog.text
    out = capsys.readouterr().out
    assert ">>>> Lost connection." in out


def test_studio_run_tail_job_logs_filters_ping_and_no_follow(
    capsys, mocker, tmp_dir, studio_token
):
    job_id = str(uuid.uuid4())
    messages = [
        json.dumps({"type": "ping"}),
        json.dumps({"logs": [{"message": "Real log\n"}]}),
        json.dumps({"job": {"status": "COMPLETE"}}),
    ]
    call_index = 0

    async def mock_recv():
        nonlocal call_index
        if call_index < len(messages):
            msg = messages[call_index]
            call_index += 1
            return msg
        raise websockets.exceptions.ConnectionClosed("Connection closed")

    captured_url = {}

    class FakeWebsocket:
        recv = staticmethod(mock_recv)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    def mock_connect(url, additional_headers):
        captured_url["url"] = url
        return FakeWebsocket()

    mocker.patch("datachain.remote.studio.websockets.connect", side_effect=mock_connect)

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "COMPLETE"}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={"dataset_versions": []},
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        exit_code = main(["job", "run", "--no-follow", "example_query.py"])

        assert exit_code == 0

    assert "no_follow=true" in captured_url["url"]
    out = capsys.readouterr().out
    assert "ping" not in out
    assert "COMPLETE" in out


def test_studio_run_verbose_finished_status(caplog, mocker, tmp_dir, studio_token):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"job": {"status": "COMPLETE"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "COMPLETE"}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={"dataset_versions": []},
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        with caplog.at_level(logging.DEBUG, logger="datachain"):
            exit_code = main(["job", "run", "-v", "example_query.py"])

        assert exit_code == 0

    assert "Job is in finished status: COMPLETE" in caplog.text


def test_studio_run_verbose_max_retry(caplog, capsys, mocker, tmp_dir, studio_token):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"job": {"status": "RUNNING"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )
    mocker.patch("datachain.studio.RETRY_MAX_TIMES", 0)
    mocker.patch("datachain.studio.RETRY_SLEEP_SEC", 0.01)

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "RUNNING"}],
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        with caplog.at_level(logging.DEBUG, logger="datachain"):
            exit_code = main(["job", "run", "-v", "example_query.py"])

        assert exit_code == 1

    assert "Max retry count reached:" in caplog.text
    assert "Job is not finished: RUNNING." in caplog.text


def test_studio_run_log_blobs(capsys, mocker, tmp_dir, studio_token):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"log_blobs": ["https://example.com/blob1"]}
        yield {"job": {"status": "COMPLETE"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )
    mocker.patch(
        "datachain.studio._fetch_log_blob",
        return_value="fetched log content\n",
    )

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "COMPLETE"}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={"dataset_versions": []},
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        exit_code = main(["job", "run", "example_query.py"])

        assert exit_code == 0

    out = capsys.readouterr().out
    assert "fetched log content" in out


def test_studio_run_log_blobs_fetch_failure(capsys, mocker, tmp_dir, studio_token):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"log_blobs": ["https://example.com/blob1"]}
        yield {"job": {"status": "COMPLETE"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )
    mocker.patch(
        "datachain.studio._fetch_log_blob",
        side_effect=requests.RequestException("connection error"),
    )

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "COMPLETE"}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={"dataset_versions": []},
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        exit_code = main(["job", "run", "example_query.py"])

        assert exit_code == 0

    out = capsys.readouterr().out
    assert "Warning: Failed to fetch logs from studio" in out


def test_studio_get_job_status_exception_returns_none(mocker):
    from datachain.studio import _get_job_status

    client = mocker.MagicMock()
    client.get_jobs.side_effect = requests.RequestException("fail")
    assert _get_job_status(client, "some-job-id") is None


def test_studio_get_job_status_empty_data_returns_none(mocker):
    from datachain.studio import _get_job_status

    client = mocker.MagicMock()
    response = mocker.MagicMock()
    response.ok = True
    response.data = []
    client.get_jobs.return_value = response
    assert _get_job_status(client, "some-job-id") is None


def test_studio_run_rest_status_none(capsys, mocker, tmp_dir, studio_token):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"job": {"status": "COMPLETE"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )
    mocker.patch("datachain.studio._get_job_status", return_value=None)

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={"dataset_versions": []},
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        exit_code = main(["job", "run", "example_query.py"])

        assert exit_code == 0


def test_studio_run_dataset_versions_error(capsys, mocker, tmp_dir, studio_token):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"job": {"status": "COMPLETE"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "COMPLETE"}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={"message": "Internal error"},
            status_code=500,
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        exit_code = main(["job", "run", "example_query.py"])

        assert exit_code == 1

    out = capsys.readouterr().err
    assert "Internal error" in out or "Error" in out


def test_studio_run_task_status_returns_zero(capsys, mocker, tmp_dir, studio_token):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"job": {"status": "TASK"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "TASK"}],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={"dataset_versions": []},
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        exit_code = main(["job", "run", "example_query.py"])

        assert exit_code == 0


def test_unpacker_hook_unknown_ext_type():
    import msgpack

    from datachain.remote.studio import StudioClient

    result = StudioClient._unpacker_hook(99, b"\x01\x02\x03")
    assert isinstance(result, msgpack.ExtType)
    assert result.code == 99
    assert result.data == b"\x01\x02\x03"
