import logging
import stat
from datetime import date
from pathlib import Path

import pytest

from datachain.config import Config, ConfigLevel
from datachain.error import DataChainError
from datachain.remote.credentials import (
    Login,
    credentials_path,
    load_logins,
    resolve_token,
)
from datachain.remote.studio import StudioClient, is_token_set
from datachain.utils import STUDIO_URL
from tests.utils import posix_only, studio_login

OTHER_URL = "https://studio.example.com"
LOCAL = ConfigLevel.LOCAL
GLOBAL = ConfigLevel.GLOBAL

pytestmark = pytest.mark.usefixtures("tmp_dir")


def set_studio(level, **values):
    with Config(level).edit() as conf:
        conf["studio"] = values


@pytest.mark.parametrize(
    "logins,team,expected",
    [
        pytest.param(
            [("local-all", None, LOCAL), ("local-x", ["x"], LOCAL)],
            "x",
            "local-x",
            id="local-team-over-local-all-teams",
        ),
        pytest.param(
            [("global-x", ["x"], GLOBAL), ("local-all", None, LOCAL)],
            "x",
            "local-all",
            id="local-all-teams-over-global-team",
        ),
        pytest.param(
            [("global-x", ["x"], GLOBAL), ("local-y", ["y"], LOCAL)],
            "x",
            "global-x",
            id="local-for-another-team-falls-through",
        ),
        pytest.param(
            [("global-all", None, GLOBAL), ("global-x", ["x"], GLOBAL)],
            "x",
            "global-x",
            id="global-team-over-global-all-teams",
        ),
        pytest.param(
            [("global-all", None, GLOBAL), ("global-x", ["x"], GLOBAL)],
            "y",
            "global-all",
            id="all-teams-for-another-team",
        ),
        pytest.param(
            [("global-all", None, GLOBAL), ("global-x", ["x"], GLOBAL)],
            None,
            "global-all",
            id="all-teams-without-a-team",
        ),
        pytest.param(
            [("global-x", ["x"], GLOBAL)],
            "y",
            None,
            id="team-token-only-for-its-teams",
        ),
    ],
)
def test_token_precedence(logins, team, expected):
    for token, teams, level in logins:
        studio_login(token, teams=teams, level=level)

    assert resolve_token(STUDIO_URL, team) == expected


def test_login_for_another_url_falls_through():
    studio_login("local-other", url=OTHER_URL, level=LOCAL)
    studio_login("global")

    assert resolve_token(STUDIO_URL, "x") == "global"
    assert resolve_token(OTHER_URL, "x") == "local-other"


def test_env_token_wins_for_every_team(monkeypatch):
    studio_login("local", teams=["acme"], level=LOCAL)
    studio_login("global")
    monkeypatch.setenv("DATACHAIN_STUDIO_TOKEN", "env")

    assert resolve_token(STUDIO_URL, "acme") == "env"
    assert is_token_set("other-team")


def test_is_token_set_checks_the_team():
    studio_login("team-token", teams=["customer-x"])

    assert is_token_set("customer-x")
    assert not is_token_set("acme")
    assert not is_token_set()

    set_studio(GLOBAL, team="customer-x")
    assert is_token_set()


def test_client_has_no_token_for_a_url_set_by_the_project():
    studio_login("user-token")
    set_studio(LOCAL, url=OTHER_URL, team="acme")

    with pytest.raises(DataChainError, match=f"Not logged in to Studio at {OTHER_URL}"):
        StudioClient().token  # noqa: B018


def test_client_uses_team_to_pick_token():
    studio_login("all", default_team="acme")
    studio_login("team-token", teams=["customer-x"])

    assert StudioClient().token == "all"  # noqa: S105
    assert StudioClient(team="customer-x").token == "team-token"  # noqa: S105


def test_expired_token_fails_instead_of_falling_back():
    studio_login("old", teams=["x"], expires=date(2020, 1, 1))
    studio_login("all", expires=date(2099, 1, 1))

    with pytest.raises(
        DataChainError,
        match=r"expired on 2020-01-01\. Run `datachain auth logout --team x`",
    ):
        resolve_token(STUDIO_URL, "x")
    assert resolve_token(STUDIO_URL, "y") == "all"


def test_login_normalizes_url_and_teams():
    login = Login(url=f"{STUDIO_URL}/", token="token", teams=["a", "a", "b"])  # noqa: S106

    assert (login.url, login.teams) == (STUDIO_URL, ["a", "b"])


def test_url_is_saved_only_for_a_self_hosted_studio():
    studio_login("default")
    studio_login("other", url=OTHER_URL)

    text = Path(credentials_path(GLOBAL)).read_text()
    assert text.startswith('[[token]]\ntoken = "default"')
    assert STUDIO_URL not in text
    assert f'url = "{OTHER_URL}"' in text
    logins = [(lg.url, lg.token) for lg in load_logins(GLOBAL)]
    assert logins == [(STUDIO_URL, "default"), (OTHER_URL, "other")]


@posix_only
def test_saved_file_is_private(global_config_dir):
    studio_login("token")

    path = Path(global_config_dir) / "credentials"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@posix_only
def test_file_open_to_others_is_ignored(caplog, tmp_dir):
    # What a token committed to a repo looks like after a clone.
    studio_login("committed", level=LOCAL)
    (tmp_dir / ".datachain" / "credentials").chmod(0o644)

    with caplog.at_level(logging.WARNING, logger="datachain"):
        assert load_logins(LOCAL) == []
        assert resolve_token(STUDIO_URL, None) is None
    [warning] = caplog.messages
    assert "other users can access it" in warning
