import asyncio
import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
from datachain.remote.credentials import credentials_path, find_login, load_logins
from datachain.studio import POST_LOGIN_MESSAGE
from datachain.utils import STUDIO_URL
from tests.utils import posix_only, skip_if_not_sqlite, studio_login

OTHER_URL = "https://other.example.com"

pytestmark = pytest.mark.usefixtures("tmp_dir")


@pytest.fixture
def get_access_token(mocker):
    return mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        return_value=("token_name", "isat_access_token"),
    )


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


def test_studio_login_success(get_access_token):
    assert main(["auth", "login"]) == 0

    [login] = load_logins(ConfigLevel.GLOBAL)
    assert login.url == STUDIO_URL
    assert login.token == "isat_access_token"  # noqa: S105
    assert login.teams is None
    assert login.name == "token_name"
    assert "studio" not in Config().read()


def test_studio_login_arguments(get_access_token):
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

    get_access_token.assert_called_with(
        token_name="token_name",  #  noqa: S106
        hostname="https://example.com",
        scopes="experiments",
        team_names=None,
        expires_in_days=365,
        client_name="DataChain",
        open_browser=False,
        post_login_message=POST_LOGIN_MESSAGE,
    )
    assert Config().read()["studio"]["url"] == "https://example.com"


def test_studio_login_with_team_scoping(get_access_token):
    assert (
        main(
            [
                "auth",
                "login",
                "--team",
                "ml-team",
                "--team",
                "data-team",
                "--expires-in",
                "90",
                "--local",
            ]
        )
        == 0
    )

    get_access_token.assert_called_with(
        token_name=None,
        hostname=STUDIO_URL,
        scopes=None,
        team_names=["ml-team", "data-team"],  # Multiple teams
        expires_in_days=90,  # Custom expiration
        client_name="DataChain",
        open_browser=True,
        post_login_message=POST_LOGIN_MESSAGE,
    )

    [login] = load_logins(ConfigLevel.LOCAL)
    assert login.teams == ["ml-team", "data-team"]
    assert load_logins(ConfigLevel.GLOBAL) == []
    assert "token" not in Config(ConfigLevel.LOCAL).read().get("studio", {})


def test_studio_login_default_expiration(get_access_token):
    before = datetime.now(UTC).date()
    assert main(["auth", "login", "--local"]) == 0
    after = datetime.now(UTC).date()

    get_access_token.assert_called_with(
        token_name=None,
        hostname=STUDIO_URL,
        scopes=None,
        team_names=None,  # All teams access
        expires_in_days=365,  # Default expiration
        client_name="DataChain",
        open_browser=True,
        post_login_message=POST_LOGIN_MESSAGE,
    )

    [login] = load_logins(ConfigLevel.LOCAL)
    assert login.token == "isat_access_token"  # noqa: S105
    expected = {(day + timedelta(days=365)).isoformat() for day in (before, after)}
    assert login.expires in expected


def test_studio_login_local_ignores_credentials_in_git(get_access_token, tmp_dir):
    gitignore = tmp_dir / ".datachain" / ".gitignore"
    gitignore.parent.mkdir()
    gitignore.write_text("/cache\n")

    assert main(["auth", "login", "--local"]) == 0
    assert main(["auth", "login", "--local", "--team", "other"]) == 0

    assert gitignore.read_text() == "/cache\n/credentials\n"


@pytest.mark.parametrize("level", [ConfigLevel.GLOBAL, ConfigLevel.LOCAL])
def test_studio_login_single_team_sets_default(get_access_token, capsys, level):
    flags = ["--local"] if level == ConfigLevel.LOCAL else []

    assert main(["auth", "login", "--team", "ml-team", *flags]) == 0

    assert Config(level).read()["studio"]["team"] == "ml-team"
    assert "Set default team to 'ml-team'" in capsys.readouterr().out


def test_studio_login_repeated_team_is_one_team(get_access_token):
    assert main(["auth", "login", "--team", "ml-team", "--team", "ml-team"]) == 0

    [login] = load_logins(ConfigLevel.GLOBAL)
    assert login.teams == ["ml-team"]
    assert Config().read()["studio"]["team"] == "ml-team"


@pytest.mark.parametrize("local", [False, True])
def test_studio_login_team_token_keeps_default_team(mocker, local):
    mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        return_value=("token_name", "customer_token"),
    )
    studio_login(default_team="acme")
    flags = ["--local"] if local else []

    assert main(["auth", "login", "--team", "customer-x", *flags]) == 0

    assert Config().read()["studio"]["team"] == "acme"
    assert find_login(STUDIO_URL, "acme").token == "isat_access_token"  # noqa: S105
    assert find_login(STUDIO_URL, "customer-x").token == "customer_token"  # noqa: S105


def test_studio_login_token_already_exists(capsys):
    studio_login()

    assert main(["auth", "login"]) == 1
    err = capsys.readouterr().err
    assert "Token already exists" in err
    assert "`datachain auth logout`" in err


def test_studio_login_team_token_already_exists(capsys):
    studio_login(teams=["ml-team", "data-team"])

    assert main(["auth", "login", "--team", "data-team"]) == 1
    assert "`datachain auth logout --team data-team`" in capsys.readouterr().err


@posix_only
def test_studio_login_does_not_overwrite_credentials_open_to_others(
    get_access_token, capsys
):
    studio_login(teams=["team-a"])
    path = credentials_path(ConfigLevel.GLOBAL)
    os.chmod(path, 0o644)
    saved = Path(path).read_text()

    assert main(["auth", "login", "--team", "team-b"]) == 1

    get_access_token.assert_not_called()
    assert f"chmod 600 {path}" in capsys.readouterr().err
    assert Path(path).read_text() == saved


def test_studio_login_new_host_resets_team(mocker):
    mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        return_value=("token_name", "new_token"),
    )
    old_url = "https://old-studio.example.com"
    studio_login("old_token", url=old_url, default_team="old-team")
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"]["url"] = old_url

    assert main(["auth", "login", "--hostname", "https://new-studio.example.com"]) == 0

    config = Config(ConfigLevel.GLOBAL).read()
    assert config["studio"] == {"url": "https://new-studio.example.com"}
    logins = {lg.url: lg.token for lg in load_logins(ConfigLevel.GLOBAL)}
    assert logins == {
        old_url: "old_token",
        "https://new-studio.example.com": "new_token",
    }


def test_studio_login_local_new_host_sets_its_own_default_team(get_access_token):
    assert main(["auth", "team", "acme"]) == 0

    url = "https://other.example.com"
    assert main(["auth", "login", "--local", "-H", url, "--team", "x"]) == 0

    assert Config().read()["studio"] == {"url": url, "team": "x"}


def test_studio_old_config_token_is_ignored(get_access_token):
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "old_token", "team": "acme"}

    assert main(["auth", "token"]) == 1

    assert main(["auth", "login"]) == 0
    assert find_login(STUDIO_URL, "acme").token == "isat_access_token"  # noqa: S105


def test_studio_login_http_error_400(mocker, capsys):
    mock_response = mocker.MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"detail": "Invalid token scope"}

    http_error = requests.HTTPError()
    http_error.response = mock_response

    mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        side_effect=http_error,
    )

    assert main(["auth", "login"]) == 1
    captured = capsys.readouterr()
    assert "Failed to authenticate with Studio: Invalid token scope" in captured.err


def test_studio_login_http_error_other(mocker, capsys):
    mock_response = mocker.MagicMock()
    mock_response.status_code = 500

    http_error = requests.HTTPError("Server error")
    http_error.response = mock_response

    mocker.patch(
        "dvc_studio_client.auth.get_access_token",
        side_effect=http_error,
    )

    assert main(["auth", "login"]) == 1
    captured = capsys.readouterr()
    assert "Failed to authenticate with Studio: Server error" in captured.err


def test_studio_logout():
    studio_login()

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/device-logout",
            json={"detail": "Token revoked successfully"},
        )
        assert main(["auth", "logout"]) == 0
        assert m.called
        assert m.last_request.headers["Authorization"] == "token isat_access_token"

    assert load_logins(ConfigLevel.GLOBAL) == []

    assert main(["auth", "logout"]) == 1


def test_studio_logout_keeps_team_tokens():
    studio_login(token="all_token")  # noqa: S106
    studio_login(token="x_token", teams=["x"])  # noqa: S106

    with requests_mock.mock() as m:
        m.post(f"{STUDIO_URL}/api/device-logout", json={})
        assert main(["auth", "logout"]) == 0
        assert m.last_request.headers["Authorization"] == "token all_token"

    [login] = load_logins(ConfigLevel.GLOBAL)
    assert login.token == "x_token"  # noqa: S105


def test_studio_logout_clears_default_team_with_the_last_token():
    studio_login(default_team="acme")
    studio_login(token="x_token", teams=["x"])  # noqa: S106

    with requests_mock.mock() as m:
        m.post(f"{STUDIO_URL}/api/device-logout", json={})
        assert main(["auth", "logout"]) == 0
        assert Config().read()["studio"]["team"] == "acme"

        assert main(["auth", "logout", "--team", "x"]) == 0
        assert "studio" not in Config().read()


def test_studio_logout_without_a_token_for_all_teams(capsys):
    studio_login(teams=["x"])

    assert main(["auth", "logout"]) == 1
    err = capsys.readouterr().err
    assert "only tokens for teams x" in err
    assert "`datachain auth logout --all`" in err


def test_studio_logout_hints_keep_the_other_studio(capsys):
    studio_login(teams=["x"], url=OTHER_URL)

    assert main(["auth", "logout", "-H", OTHER_URL]) == 1
    err = capsys.readouterr().err
    assert f"`datachain auth logout -H {OTHER_URL} --team <team>`" in err
    assert f"`datachain auth logout -H {OTHER_URL} --all`" in err


def test_studio_logout_other_studio_keeps_the_default_team():
    studio_login(url=OTHER_URL)
    studio_login(default_team="acme")

    with requests_mock.mock() as m:
        m.post(f"{OTHER_URL}/api/device-logout", json={})
        assert main(["auth", "logout", "-H", OTHER_URL]) == 0

    assert Config().read()["studio"]["team"] == "acme"


def test_studio_logout_team_without_a_token_for_the_team(capsys):
    studio_login()

    assert main(["auth", "logout", "--team", "x"]) == 1
    assert "No token is saved for team 'x' alone" in capsys.readouterr().err


def test_studio_logout_team_removes_only_its_token():
    studio_login(default_team="acme")
    studio_login(token="customer_token", teams=["customer-x"])  # noqa: S106

    with requests_mock.mock() as m:
        m.post(f"{STUDIO_URL}/api/device-logout", json={})
        assert main(["auth", "logout", "--team", "customer-x"]) == 0
        assert m.call_count == 1
        assert m.last_request.headers["Authorization"] == "token customer_token"

    [login] = load_logins(ConfigLevel.GLOBAL)
    assert login.token == "isat_access_token"  # noqa: S105


def test_studio_logout_team_removes_the_token_for_all_its_teams():
    studio_login(teams=["team-a", "team-b"])

    with requests_mock.mock() as m:
        m.post(f"{STUDIO_URL}/api/device-logout", json={})
        assert main(["auth", "logout", "--team", "team-a"]) == 0
        assert m.call_count == 1

    assert load_logins(ConfigLevel.GLOBAL) == []


def test_studio_logout_all_keeps_tokens_from_the_first_failed_revoke(capsys):
    studio_login(token="all_token")  # noqa: S106
    studio_login(token="a_token", teams=["team-a"])  # noqa: S106
    studio_login(token="b_token", teams=["team-b"])  # noqa: S106

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/device-logout",
            [{"json": {}}, {"json": {}, "status_code": 500}],
        )
        assert main(["auth", "logout", "--all"]) == 1
        sent = [r.headers["Authorization"] for r in m.request_history]

    assert sent == ["token all_token", "token a_token"]
    remaining = [lg.token for lg in load_logins(ConfigLevel.GLOBAL)]
    assert remaining == ["a_token", "b_token"]
    assert "500" in capsys.readouterr().err


def test_studio_logout_token_already_revoked(capsys):
    studio_login()

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/device-logout",
            json={"detail": "Invalid token"},
            status_code=401,
        )
        assert main(["auth", "logout"]) == 0

    assert load_logins(ConfigLevel.GLOBAL) == []

    err = capsys.readouterr().err
    assert "already revoked or is invalid" in err


@pytest.mark.parametrize(
    "url",
    ["https://custom-studio.example.com", "https://custom-studio.example.com/"],
)
def test_studio_logout_custom_url(url):
    studio_login(url=url)
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"url": url}

    with requests_mock.mock() as m:
        m.post(
            "https://custom-studio.example.com/api/device-logout",
            json={"detail": "Token revoked successfully"},
        )
        assert main(["auth", "logout"]) == 0
        assert (
            m.last_request.url == "https://custom-studio.example.com/api/device-logout"
        )

    assert load_logins(ConfigLevel.GLOBAL) == []


def test_studio_logout_url_from_env(monkeypatch):
    env_url = "https://env-studio.example.com"
    config_url = "https://config-studio.example.com"
    monkeypatch.setenv("DATACHAIN_STUDIO_URL", env_url)
    studio_login(token="config_token", url=config_url)  # noqa: S106
    studio_login(token="env_token", url=env_url)  # noqa: S106
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"url": config_url}

    with requests_mock.mock() as m:
        m.post(
            f"{env_url}/api/device-logout",
            json={"detail": "Token revoked successfully"},
        )
        assert main(["auth", "logout"]) == 0
        assert m.last_request.url == f"{env_url}/api/device-logout"
        assert m.last_request.headers["Authorization"] == "token env_token"

    [login] = load_logins(ConfigLevel.GLOBAL)
    assert login.url == config_url


def test_studio_logout_hostname_with_trailing_slash():
    other_url = "https://other-studio.example.com"
    studio_login()
    studio_login(token="other_token", url=other_url)  # noqa: S106

    with requests_mock.mock() as m:
        m.post(f"{other_url}/api/device-logout", json={})
        assert main(["auth", "logout", "-H", f"{other_url}/"]) == 0
        assert m.last_request.headers["Authorization"] == "token other_token"

    [login] = load_logins(ConfigLevel.GLOBAL)
    assert login.url == STUDIO_URL


def test_studio_logout_network_error_aborts(capsys):
    studio_login()

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/device-logout",
            exc=requests.ConnectionError,
        )
        assert main(["auth", "logout"]) == 1

    [login] = load_logins(ConfigLevel.GLOBAL)
    assert login.token == "isat_access_token"  # noqa: S105

    err = capsys.readouterr().err
    assert "Could not reach Studio" in err


def test_studio_logout_server_error_aborts(capsys):
    studio_login()

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/device-logout",
            json={"detail": "Internal server error"},
            status_code=500,
        )
        assert main(["auth", "logout"]) == 1

    [login] = load_logins(ConfigLevel.GLOBAL)
    assert login.token == "isat_access_token"  # noqa: S105

    err = capsys.readouterr().err
    assert "500" in err


def test_studio_logout_local_revokes_local_token():
    studio_login(token="global_token")  # noqa: S106
    studio_login(token="local_token", level=ConfigLevel.LOCAL)  # noqa: S106

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/device-logout",
            json={"detail": "Token revoked successfully"},
        )
        assert main(["auth", "logout", "--local"]) == 0
        assert m.called
        assert m.last_request.headers["Authorization"] == "token local_token"

    assert load_logins(ConfigLevel.LOCAL) == []
    [login] = load_logins(ConfigLevel.GLOBAL)
    assert login.token == "global_token"  # noqa: S105


def test_studio_logout_local_preserves_parent_token(tmp_dir, monkeypatch):
    (tmp_dir / ".git").mkdir()
    (tmp_dir / ".datachain").mkdir()
    (tmp_dir / ".datachain" / "db").touch()
    studio_login(token="parent-token", level=ConfigLevel.LOCAL)  # noqa: S106
    parent_credentials = (tmp_dir / ".datachain" / "credentials").read_text()
    child = tmp_dir / "child"
    (child / ".datachain").mkdir(parents=True)
    # The credentials file alone marks the child's project state.
    (child / ".datachain" / "credentials").touch(mode=0o600)
    monkeypatch.chdir(child)
    studio_login(token="child-token", level=ConfigLevel.LOCAL)  # noqa: S106
    nested = child / "scripts"
    nested.mkdir()
    monkeypatch.chdir(nested)

    with requests_mock.mock() as m:
        m.post(f"{STUDIO_URL}/api/device-logout", json={})
        assert main(["auth", "logout", "--local"]) == 0
        assert m.call_count == 1
        assert m.last_request.headers["Authorization"] == "token child-token"

    assert not (child / ".datachain" / "credentials").exists()
    credentials = (tmp_dir / ".datachain" / "credentials").read_text()
    assert credentials == parent_credentials
    assert not (nested / ".datachain").exists()


def test_studio_token(capsys):
    studio_login()

    assert main(["auth", "token"]) == 0
    assert capsys.readouterr().out == "isat_access_token\n"

    os.remove(credentials_path(ConfigLevel.GLOBAL))

    assert main(["auth", "token"]) == 1
    assert f"Not logged in to Studio at {STUDIO_URL}" in capsys.readouterr().err


def test_studio_token_creates_no_directory(tmp_dir):
    assert main(["auth", "token"]) == 1
    assert not (tmp_dir / ".datachain").exists()


def test_studio_token_for_team(capsys):
    studio_login(default_team="acme")
    studio_login(token="customer_token", teams=["customer-x"])  # noqa: S106

    assert main(["auth", "token", "--team", "customer-x"]) == 0
    assert capsys.readouterr().out == "customer_token\n"

    assert main(["auth", "token"]) == 0
    assert capsys.readouterr().out == "isat_access_token\n"


def test_studio_token_from_env(capsys, monkeypatch):
    monkeypatch.setenv("DATACHAIN_STUDIO_TOKEN", "env_token")

    assert main(["auth", "token"]) == 0
    assert capsys.readouterr().out == "env_token\n"


def test_studio_auth_list(capsys):
    studio_login(default_team="acme")
    studio_login(token="customer_token", teams=["customer-x"])  # noqa: S106

    assert main(["auth", "list"]) == 0

    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines[:2] == [f"Studio URL: {STUDIO_URL}", "Default team: acme"]
    [current] = [line for line in lines if line.startswith("*")]
    assert "all teams" in current
    assert any("customer-x" in line and not line.startswith("*") for line in lines)
    assert "isat_access_token" not in out + err
    assert "customer_token" not in out + err


def test_studio_auth_list_marks_env_token(capsys, monkeypatch):
    studio_login()
    monkeypatch.setenv("DATACHAIN_STUDIO_TOKEN", "env_token")

    assert main(["auth", "list"]) == 0

    out, err = capsys.readouterr()
    [current] = [line for line in out.splitlines() if line.startswith("*")]
    assert "$DATACHAIN_STUDIO_TOKEN" in current
    assert "env_token" not in out + err
    assert "isat_access_token" not in out + err


def test_studio_auth_list_not_logged_in(capsys):
    assert main(["auth", "list"]) == 0
    assert "Not logged in to Studio" in capsys.readouterr().out


def test_studio_team_global():
    assert main(["auth", "team", "team_name"]) == 0
    config = Config(ConfigLevel.GLOBAL).read()
    assert config["studio"]["team"] == "team_name"


def test_studio_team_local():
    assert main(["auth", "team", "team_name", "--local"]) == 0
    config = Config(ConfigLevel.LOCAL).read()
    assert config["studio"]["team"] == "team_name"


def test_studio_team_no_name_and_no_default(capsys):
    assert main(["auth", "team"]) == 1
    captured = capsys.readouterr()
    assert (
        "No default team set. Use `datachain auth team <team_name>` to set one."
        in captured.err
    )


def test_studio_team_no_name_with_existing_default(capsys):
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"team": "existing-team"}

    assert main(["auth", "team"]) == 0
    captured = capsys.readouterr()
    assert "Default team is 'existing-team'" in captured.out


def test_dataset_ls_all_lists_local_only_without_a_token_for_the_team(mocker, capsys):
    studio_login(teams=["team-a"])
    mocker.patch(
        "datachain.cli.commands.datasets.list_datasets_local",
        side_effect=lambda *a, **kw: iter([("local.local.local", "1.0.0", False)]),
    )
    studio = mocker.patch("datachain.cli.commands.datasets.list_datasets_studio")

    assert main(["dataset", "ls", "--all", "--team", "team-b"]) == 0

    studio.assert_not_called()
    assert "local.local.local" in capsys.readouterr().out


def test_expired_token_only_breaks_studio_listings(capsys):
    studio_login(expires="2020-01-01")

    assert main(["ls", "."]) == 0
    assert main(["dataset", "ls"]) == 0
    capsys.readouterr()

    assert main(["ls", ".", "--studio"]) == 1
    assert "expired on 2020-01-01" in capsys.readouterr().err


def test_studio_command_with_team_tokens_but_no_team(capsys):
    studio_login(teams=["a", "b"])

    assert main(["job", "ls"]) == 1
    assert "Studio team is not set" in capsys.readouterr().err


def test_studio_datasets(capsys, studio_datasets, mocker):
    def list_datasets_local(_, __, include_removed=False):
        yield "local.local.local", "1.0.0", False
        yield "dev.animals.both", "1.0.0", False

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
    assert sorted(out.splitlines()) == sorted(local_output.splitlines())

    assert main(["dataset", "ls", "--versions"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(local_output.splitlines())

    assert main(["dataset", "ls", "--versions", "--all"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(both_output_versions.splitlines())

    assert main(["dataset", "ls", "dev.animals.dogs", "--studio"]) == 0
    out = capsys.readouterr().out
    assert sorted(out.splitlines()) == sorted(dogs_output.splitlines())


@pytest.mark.parametrize(
    "extra_cli_args,route,ok_body",
    [
        pytest.param([], "datachain/datasets", [], id="ls-all"),
        pytest.param(
            ["dev.animals.cats"],
            "datachain/datasets/info",
            {
                "versions": [],
                "project": {
                    "created_at": None,
                    "namespace": {"created_at": None},
                },
                "created_at": None,
                "finished_at": None,
            },
            id="ls-by-name",
        ),
    ],
)
def test_dataset_ls_include_removed_flows_to_studio(
    requests_mock, studio_token, extra_cli_args, route, ok_body
):
    """`dataset ls --include-removed --studio` forwards the flag as a
    query parameter to Studio on both the list-all
    (`/datachain/datasets`) and list-by-name (`/datachain/datasets/info`)
    endpoints, and omits it when the flag isn't set."""
    m = requests_mock.get(f"{STUDIO_URL}/api/{route}", json=ok_body)

    assert main(["dataset", "ls", "--studio", *extra_cli_args]) == 0
    assert "include_removed" not in m.last_request.qs

    assert (
        main(["dataset", "ls", "--studio", "--include-removed", *extra_cli_args]) == 0
    )
    assert m.last_request.qs.get("include_removed") == ["true"]


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

        studio_login(default_team="team_name")

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

        studio_login(default_team="team_name")

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


def test_studio_list_jobs(capsys):
    studio_login(default_team="team_name")

    with requests_mock.mock() as m:
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[
                {
                    "id": "8bddde6c-c3ca-41b0-9d87-ee945bfdce70",
                    "name": "on-cluster",
                    "status": "FAILED",
                    "compute_cluster_id": 1,
                    "compute_cluster_name": "prod-cluster",
                    "created_at": "2021-01-01T00:00:00Z",
                    "created_by": "user",
                    "finished_at": "2021-01-01T00:00:20Z",
                    # A job can stop without closing its stages.
                    "steps": [
                        {
                            "name": "waiting",
                            "label": "Waiting in queue",
                            "status": "STARTED",
                            "started_at": "2021-01-01T00:00:00Z",
                            "finished_at": None,
                        },
                    ],
                },
                {
                    "id": "0502eef6-a32e-45fa-8e3b-d20ec0abbcf0",
                    "name": "on-other-cluster",
                    "status": "RUNNING",
                    "compute_cluster_id": 2,
                    "compute_cluster_name": "dev-cluster",
                    "created_at": "2021-01-02T00:00:00Z",
                    "created_by": "user",
                    "started_at": "2021-01-02T00:00:04Z",
                    "finished_at": None,
                    "steps": [
                        {
                            "name": "waiting",
                            "label": "Waiting in queue",
                            "status": "FINISHED",
                            "started_at": "2021-01-02T00:00:00Z",
                            "finished_at": "2021-01-02T00:00:04Z",
                        },
                        {
                            "name": "downloading_files",
                            "label": "Downloading files",
                            "status": "FINISHED",
                            "started_at": "2021-01-02T00:00:04Z",
                            "finished_at": "2021-01-02T01:05:04Z",
                        },
                        {
                            "name": "dw_wake_up",
                            "label": "Waking up data warehouse",
                            "status": "FINISHED",
                            "started_at": None,
                            "finished_at": None,
                        },
                        {
                            "name": "virtualenv",
                            "label": "Installing dependencies",
                            "status": "FINISHED",
                            "started_at": "2021-01-02T01:05:04Z",
                            "finished_at": "2021-01-02T01:07:34Z",
                        },
                        {
                            "name": "running_query",
                            "label": "Running query",
                            "status": "STARTED",
                            "started_at": "2021-01-02T01:07:34Z",
                            "finished_at": None,
                        },
                    ],
                },
            ],
        )

        assert main(["job", "ls"]) == 0
        out = capsys.readouterr().out
        assert "Started at" in out
        assert "2021-01-02T00:00:04Z" in out
        assert "Cluster" not in out
        assert "prod-cluster" not in out
        assert "include_steps" not in m.last_request.qs

        assert main(["job", "ls", "--extended"]) == 0
        out = capsys.readouterr().out

    assert "Cluster" in out
    assert "prod-cluster" in out
    assert m.last_request.qs["include_steps"] == ["true"]
    assert "Waiting in queue: 4s" in out
    assert "Downloading files: 1h 5m" in out
    assert "Installing dependencies: 2m 30s" in out
    assert "Running query: running" in out
    # A stage with no start, and a stopped job's open stage, were never timed.
    assert "Waking up data warehouse: -" in out
    assert "Waiting in queue: -" in out


def test_studio_list_jobs_without_start_time(capsys, studio_token):
    job = {
        "name": "daily",
        "status": "CANCELED",
        "created_at": "2026-09-16T00:00:00Z",
        "created_by": "alice",
    }
    # No worker claimed the first job. An older Studio does not send the field.
    jobs = [
        {**job, "id": "never-claimed", "started_at": None},
        {**job, "id": "old-studio"},
    ]
    with requests_mock.mock() as m:
        m.get(f"{STUDIO_URL}/api/datachain/jobs/", json=jobs)

        assert main(["job", "ls"]) == 0

    rows = [line.split("|")[1:-1] for line in capsys.readouterr().out.splitlines()]
    cells = {row[0].strip(): [cell.strip() for cell in row] for row in rows if row}
    assert cells["ID"][4] == "Started at"
    assert cells["never-claimed"][4] == "-"
    assert cells["old-studio"][4] == "-"


CLUSTER = {
    "id": "k3f9x2mq7a",
    "name": "prod-cluster",
    "status": "ACTIVE",
    "cloud_provider": "AWS",
    "cloud_credentials": "aws-creds",
    "is_active": True,
    "default": True,
    "max_workers": 8,
    "active_workers": 4,
    "busy_workers": 2,
    "cloud_region": "us-west-2",
    "instance_type": "m5.xlarge",
    "compute_class": "Performance",
    "disk_size": "100Gi",
}


def test_studio_clusters_shows_the_machine_and_its_limits(capsys, studio_token):
    """The machine, where it runs, and how many workers it allows."""
    with requests_mock.mock() as m:
        m.get(f"{STUDIO_URL}/api/datachain/clusters/", json=[CLUSTER])

        assert main(["job", "clusters"]) == 0

    out = capsys.readouterr().out
    assert "prod-cluster" in out
    # The id identifies a cluster; names can be reused, so it leads the table.
    assert CLUSTER["id"] in out
    assert re.search(r"\|\s+ID\s+\|", out) is not None
    assert "us-west-2" in out
    assert "m5.xlarge" in out
    assert "Performance" in out
    assert "100Gi" in out
    # busy/active/max, so capacity reads as one column.
    assert "2/4/8" in out


def cell(out: str, column: str) -> str:
    """One named cell of the single rendered row."""
    header, row = [line for line in out.splitlines() if line.startswith("|")][:2]
    index = [h.strip() for h in header.split("|")].index(column)
    return [c.strip() for c in row.split("|")][index]


def test_studio_clusters_unset_fields_read_as_dashes(capsys, studio_token):
    """A cluster that configures none of them. A dash is "not set"."""
    with requests_mock.mock() as m:
        m.get(
            f"{STUDIO_URL}/api/datachain/clusters/",
            json=[
                {
                    **CLUSTER,
                    "name": "plain-cluster",
                    "cloud_region": None,
                    "instance_type": None,
                    "compute_class": None,
                    "disk_size": None,
                }
            ],
        )

        assert main(["job", "clusters"]) == 0

    out = capsys.readouterr().out
    assert "plain-cluster" in out
    assert "us-west-2" not in out
    assert cell(out, "Disk Request") == "-"


def test_studio_clusters_do_not_read_an_id_or_name_as_a_number(capsys, studio_token):
    """Ids come from [a-z0-9], so one can look like scientific notation, and a name is
    whatever someone typed. tabulate would render "12345678e9" as 1.23457e+16 and a
    cluster called "1e5" as 100000 - neither can be pasted back into a command."""
    with requests_mock.mock() as m:
        m.get(
            f"{STUDIO_URL}/api/datachain/clusters/",
            json=[{**CLUSTER, "id": "12345678e9", "name": "1e5"}],
        )

        assert main(["job", "clusters"]) == 0

    out = capsys.readouterr().out
    assert cell(out, "ID") == "12345678e9"
    assert cell(out, "Name") == "1e5"
    # The counts are still reported, so they are still numbers.
    assert "2/4/8" in out


def test_studio_clusters_false_is_not_unknown(capsys, studio_token):
    """Only null reads as unset. A false flag and a zero count are values."""
    with requests_mock.mock() as m:
        m.get(
            f"{STUDIO_URL}/api/datachain/clusters/",
            json=[
                {
                    **CLUSTER,
                    "default": False,
                    "busy_workers": 0,
                    "active_workers": 0,
                    "max_workers": 0,
                }
            ],
        )

        assert main(["job", "clusters"]) == 0

    out = capsys.readouterr().out
    assert cell(out, "Is Default") == "False"
    assert "0/0/0" in out


def test_studio_clusters_json_prints_every_field(capsys, studio_token):
    """`--json` prints the clusters exactly as Studio returned them."""
    with requests_mock.mock() as m:
        m.get(f"{STUDIO_URL}/api/datachain/clusters/", json=[CLUSTER])

        assert main(["job", "clusters", "--json"]) == 0

    assert json.loads(capsys.readouterr().out) == [CLUSTER]


def test_studio_clusters_none_found(capsys, studio_token):
    with requests_mock.mock() as m:
        m.get(f"{STUDIO_URL}/api/datachain/clusters/", json=[])

        assert main(["job", "clusters"]) == 0

    assert "No clusters found" in capsys.readouterr().out


def test_studio_jobs_json_prints_the_response(capsys, studio_token):
    """`--json` changes the format, not what is asked for: `--extended` still rules."""
    job = {
        "id": "0502eef6-a32e-45fa-8e3b-d20ec0abbcf0",
        "name": "daily",
        "status": "COMPLETE",
        "created_at": "2026-09-16T00:00:00Z",
        "finished_at": "2026-09-16T00:20:00Z",
        "created_by": "alice",
        "workers": 4,
        "compute_cluster_name": "prod-cluster",
        "compute_cluster_id": "k3f9x2mq7a",
        "steps": [
            {
                "name": "waiting",
                "label": "Waiting in queue",
                "status": "FINISHED",
                "started_at": "2026-09-16T00:00:00Z",
                "finished_at": "2026-09-16T00:00:04Z",
            }
        ],
    }
    with requests_mock.mock() as m:
        route = m.get(f"{STUDIO_URL}/api/datachain/jobs/", json=[job])

        assert main(["job", "ls", "--json"]) == 0
        assert "include_steps" not in route.last_request.qs
        assert json.loads(capsys.readouterr().out) == [job]

        assert main(["job", "ls", "--json", "--extended", "--limit", "5"]) == 0

    assert route.last_request.qs["include_steps"] == ["true"]
    assert route.last_request.qs["limit"] == ["5"]
    assert json.loads(capsys.readouterr().out) == [job]


@pytest.mark.parametrize(
    "command",
    [
        ["job", "ls"],
        ["job", "cancel", "job-id"],
        ["job", "logs", "job-id"],
        pytest.param(
            ["dataset", "rm", "dev.animals.name", "--studio"],
            marks=skip_if_not_sqlite,
        ),
    ],
)
def test_studio_command_without_login_or_team(capsys, command):
    assert main(command) == 1

    err = capsys.readouterr().err
    assert f"Not logged in to Studio at {STUDIO_URL}" in err
    assert "team is not set" not in err


def test_studio_cancel_job(capsys, mocker):
    job_id = "8bddde6c-c3ca-41b0-9d87-ee945bfdce70"
    with requests_mock.mock() as m:
        m.post(f"{STUDIO_URL}/api/datachain/jobs/{job_id}/cancel", json={})

        # Studio token is required
        assert main(["job", "cancel", job_id]) == 1
        out = capsys.readouterr().err
        assert "Not logged in to Studio" in out

        studio_login(default_team="team_name")

        assert main(["job", "cancel", job_id]) == 0
        assert m.called


def test_studio_run(capsys, mocker, tmp_dir):
    mocker.patch(
        "datachain.remote.studio.websockets.connect", side_effect=mocked_connect
    )
    studio_login(default_team="team_name")

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

    def mock_get_last_job_by_name(name, is_remote_execution=False):
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
        yield {"logs": [{"message": "Starting job...\n", "id": 1}]}
        yield {"logs": [{"message": "Processing data...\n", "id": 2}]}
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
        yield {"logs": [{"message": "Starting job...\n", "id": 1}]}
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
        yield {"logs": [{"message": "Starting job...\n", "id": 1}]}
        yield {"job": {"status": "RUNNING"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )

    mocker.patch("datachain.studio.RECONNECT_MAX_ATTEMPTS", 0)

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
    assert ">>>> Failed to reconnect after" in out
    assert "datachain job logs" in out
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
    assert ">>>> Failed to reconnect after" in out


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
    mocker.patch("datachain.studio.RECONNECT_MAX_ATTEMPTS", 0)

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

    assert "Max reconnect attempts reached:" in caplog.text
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
        return_value=b"fetched log content\n",
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


def test_studio_run_log_blobs_fetches_presigned_bytes(
    capsysbinary, mocker, tmp_dir, studio_token
):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {
            "log_blobs": [
                "https://example.com/blob1",
                "https://example.com/blob2",
            ]
        }
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
        blob1 = m.get(
            "https://example.com/blob1",
            content=b"signal:\xff at 5 \xc2\xb5S",
            headers={"Content-Type": "text/plain"},
        )
        blob2 = m.get(
            "https://example.com/blob2",
            content=b"second blob\n",
            headers={"Content-Type": "text/plain"},
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        exit_code = main(["job", "run", "example_query.py"])

        assert exit_code == 0
        assert "Authorization" not in blob1.last_request.headers
        assert "Authorization" not in blob2.last_request.headers

    out = capsysbinary.readouterr().out
    assert b"signal:\xff at 5 \xc2\xb5S\nsecond blob\n" in out


def test_studio_run_log_blobs_http_error_detail(
    capsys, caplog, mocker, tmp_dir, studio_token
):
    job_id = str(uuid.uuid4())

    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"log_blobs": ["https://example.com/blob1?X-Amz-Signature=secretsig"]}
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
        m.get(
            "https://example.com/blob1?X-Amz-Signature=secretsig",
            status_code=400,
            text="<Error><Code>InvalidArgument</Code></Error>",
        )

        (tmp_dir / "example_query.py").write_text("print(1)")

        with caplog.at_level(logging.DEBUG, logger="datachain"):
            exit_code = main(["job", "run", "-v", "example_query.py"])

        assert exit_code == 0

    out = capsys.readouterr().out
    assert "Warning: Failed to fetch logs from studio (HTTP 400)" in out
    assert "secretsig" not in out
    assert "<Error><Code>InvalidArgument</Code></Error>" in caplog.text


def test_show_log_blobs_propagates_broken_pipe(mocker):
    from datachain.studio import _show_log_blobs

    mocker.patch("datachain.studio._fetch_log_blob", return_value=b"content\n")
    stdout = mocker.patch("datachain.studio.sys.stdout")
    stdout.buffer.write.side_effect = BrokenPipeError

    with pytest.raises(BrokenPipeError):
        asyncio.run(_show_log_blobs(["https://example.com/blob1"], mocker.MagicMock()))


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


def test_studio_run_log_deduplication_on_reconnect(
    capsys, mocker, tmp_dir, studio_token
):
    job_id = str(uuid.uuid4())
    call_count = 0

    async def mock_tail_job_logs(jid, no_follow=False):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {
                "logs": [
                    {"id": 1, "message": "line 1\n"},
                    {"id": 2, "message": "line 2\n"},
                ]
            }
            yield {"job": {"status": "RUNNING"}}
        else:
            yield {
                "logs": [
                    {"id": 1, "message": "line 1\n"},
                    {"id": 2, "message": "line 2\n"},
                    {"id": 3, "message": "line 3\n"},
                ]
            }
            yield {"job": {"status": "COMPLETE"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs", side_effect=mock_tail_job_logs
    )
    mocker.patch("asyncio.sleep")

    with requests_mock.mock() as m:
        m.post(
            f"{STUDIO_URL}/api/datachain/jobs/",
            json={"id": job_id, "url": "https://example.com"},
        )
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            [
                {"json": [{"status": "RUNNING"}]},
                {"json": [{"status": "COMPLETE"}]},
            ],
        )
        m.get(
            f"{STUDIO_URL}/api/datachain/datasets/dataset_job_versions?job_id={job_id}&team_name=team_name",
            json={"dataset_versions": []},
        )
        (tmp_dir / "example_query.py").write_text("print(1)")
        assert main(["job", "run", "example_query.py"]) == 0

    out = capsys.readouterr().out
    assert out.count("line 1\n") == 1
    assert out.count("line 2\n") == 1
    assert out.count("line 3\n") == 1


def test_studio_run_reconnect_resets_counter_on_streaming_data(
    caplog, mocker, tmp_dir, studio_token
):
    job_id = str(uuid.uuid4())
    call_count = 0

    async def mock_tail_job_logs(jid, no_follow=False):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {"logs": [{"id": 1, "message": "output\n"}]}
            yield {"job": {"status": "RUNNING"}}
        else:
            yield {"job": {"status": "RUNNING"}}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs", side_effect=mock_tail_job_logs
    )
    mocker.patch("datachain.studio.RECONNECT_MAX_ATTEMPTS", 1)
    mocker.patch("asyncio.sleep")

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
            assert main(["job", "run", "-v", "example_query.py"]) == 1

    assert "Max reconnect attempts reached:" in caplog.text


def test_studio_job_logs_refused_handshake_aborts(capsys, mocker, studio_token):
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    def mock_connect(url, additional_headers):
        raise InvalidStatus(Response(403, "Forbidden", Headers()))

    mocker.patch("datachain.remote.studio.websockets.connect", side_effect=mock_connect)
    mocker.patch("datachain.studio.RECONNECT_MAX_ATTEMPTS", 0)

    with requests_mock.mock() as m:
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "RUNNING"}],
        )

        exit_code = main(["job", "logs", str(uuid.uuid4())])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Studio refused the log stream connection (HTTP 403)" in captured.err
    assert "team_name" in captured.err
    assert "reconnecting in" not in captured.out


def test_studio_job_logs_transient_handshake_failure_retries(
    capsys, mocker, studio_token
):
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    def mock_connect(url, additional_headers):
        raise InvalidStatus(Response(503, "Service Unavailable", Headers()))

    connect = mocker.patch(
        "datachain.remote.studio.websockets.connect", side_effect=mock_connect
    )
    mocker.patch("datachain.studio.RECONNECT_MAX_ATTEMPTS", 1)
    mocker.patch("datachain.studio.asyncio.sleep")

    with requests_mock.mock() as m:
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            [
                {"json": [{"status": "RUNNING"}]},
                {"json": [{"status": "COMPLETE"}]},
            ],
        )
        m.get(
            re.compile(
                rf"^{re.escape(STUDIO_URL)}/api/datachain/datasets/dataset_job_versions"
            ),
            json={"dataset_versions": []},
        )

        exit_code = main(["job", "logs", str(uuid.uuid4())])

    assert exit_code == 0
    assert connect.call_count == 2
    captured = capsys.readouterr()
    assert ">>>> Job is now in RUNNING status." in captured.out
    assert ">>>> Job is now in COMPLETE status." in captured.out
    assert "refused the log stream connection" not in captured.err


def test_studio_job_logs_terminal_error_on_reconnect_clears_banner(
    capsys, mocker, studio_token
):
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    statuses = [503, 403]

    def mock_connect(url, additional_headers):
        raise InvalidStatus(Response(statuses.pop(0), "", Headers()))

    mocker.patch("datachain.remote.studio.websockets.connect", side_effect=mock_connect)
    mocker.patch("datachain.studio.RECONNECT_BACKOFF_BASE_SEC", 0)

    with requests_mock.mock() as m:
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "RUNNING"}],
        )

        exit_code = main(["job", "logs", str(uuid.uuid4())])

    assert exit_code == 1
    assert not statuses
    captured = capsys.readouterr()
    assert "Studio refused the log stream connection (HTTP 403)" in captured.err
    assert "reconnecting in" in captured.out
    assert re.search(r" {10,}\r", captured.out)


def test_studio_job_logs_interrupt_during_backoff_clears_banner(
    capsys, mocker, studio_token
):
    async def mock_tail_job_logs(jid, no_follow=False):
        return
        yield

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )
    mocker.patch("datachain.studio.asyncio.sleep", side_effect=KeyboardInterrupt)

    with requests_mock.mock() as m:
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "RUNNING"}],
        )

        exit_code = main(["job", "logs", str(uuid.uuid4())])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "reconnecting in" in captured.out
    assert re.search(r" {10,}\r", captured.out)
    assert "Operation cancelled by the user" in captured.err


def test_studio_job_logs_legacy_status_code_attribute_aborts(
    capsys, mocker, studio_token
):
    from websockets.exceptions import WebSocketException

    class LegacyInvalidStatusCode(WebSocketException):
        status_code = 403

    def mock_connect(url, additional_headers):
        raise LegacyInvalidStatusCode

    mocker.patch("datachain.remote.studio.websockets.connect", side_effect=mock_connect)
    mocker.patch("datachain.studio.RECONNECT_MAX_ATTEMPTS", 0)

    with requests_mock.mock() as m:
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "RUNNING"}],
        )

        exit_code = main(["job", "logs", str(uuid.uuid4())])

    assert exit_code == 1
    assert "HTTP 403" in capsys.readouterr().err


def test_studio_job_logs_server_error_frame_aborts(capsys, mocker, studio_token):
    async def mock_tail_job_logs(jid, no_follow=False):
        yield {"message": "Job ID is incorrect or not found"}

    mocker.patch(
        "datachain.studio.StudioClient.tail_job_logs",
        side_effect=mock_tail_job_logs,
    )
    mocker.patch("datachain.studio.RECONNECT_MAX_ATTEMPTS", 0)

    with requests_mock.mock() as m:
        m.get(
            re.compile(rf"^{re.escape(STUDIO_URL)}/api/datachain/jobs/"),
            json=[{"status": "RUNNING"}],
        )

        exit_code = main(["job", "logs", str(uuid.uuid4())])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Job ID is incorrect or not found" in captured.err
    assert "Failed to reconnect" not in captured.out


def test_unpacker_hook_unknown_ext_type():
    import msgpack

    from datachain.remote.studio import StudioClient

    result = StudioClient._unpacker_hook(99, b"\x01\x02\x03")
    assert isinstance(result, msgpack.ExtType)
    assert result.code == 99
    assert result.data == b"\x01\x02\x03"
