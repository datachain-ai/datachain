from unittest.mock import MagicMock

import pytest
import requests_mock
import websockets
from dvc_studio_client.auth import AuthorizationExpiredError
from tabulate import tabulate

from datachain.cli import main
from datachain.config import Config, ConfigLevel
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

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/files?team_name=team_name", json={"id": 1}
        )
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": 1, "url": "https://example.com"},
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id=1&team_name=team_name",
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
        out.strip() == "Job 1 created\nOpen the job in Studio at https://example.com\n"
        "========================================\n\n\n"
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
        "environment": "ENV_FROM_FILE=1\nENV_FROM_ARGS=1",
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
        "cron_expression": None,
        "credentials_name": "my-credentials",
    }


def test_studio_run_task(capsys, mocker, tmp_dir, studio_token):
    mocker.patch(
        "datachain.remote.studio.websockets.connect", side_effect=mocked_connect
    )

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": 1, "url": "https://example.com"},
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id=1&team_name=team_name",
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


@pytest.mark.parametrize(
    "status,expected_exit_code", [("FAILED", 1), ("CANCELED", 2), ("COMPLETE", 0)]
)
def test_studio_run_non_zero_exit_code(
    capsys, mocker, tmp_dir, status, expected_exit_code
):
    # Mock tail_job_logs to return a status
    async def mock_tail_job_logs(job_id):
        # Simulate some log messages
        yield {"logs": [{"message": "Starting job...\n"}]}
        yield {"logs": [{"message": "Processing data...\n"}]}
        # Return failed status
        yield {"job": {"status": status}}

    mocker.patch(
        "datachain.remote.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )

    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "isat_access_token", "team": "team_name"}

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": 1, "url": "https://example.com"},
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id=1&team_name=team_name",
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
        out.strip() == "Job 1 created\nOpen the job in Studio at https://example.com\n"
        "========================================\n"
        "Starting job...\n"
        "Processing data...\n"
        "\n"
        f">>>> Job is now in {status} status.\n\n\n"
        ">>>> Dataset versions created during the job:\n"
        "    - dataset_name@v1.0.0"
    )
