import logging
import stat
from datetime import date
from pathlib import Path

import pytest

from datachain.config import ConfigLevel
from datachain.error import DataChainError
from datachain.remote.credentials import Login, load_logins, resolve_token
from datachain.remote.studio import StudioClient, is_token_set
from datachain.utils import STUDIO_URL
from tests.utils import posix_only, set_studio, studio_login

OTHER_URL = "https://studio.example.com"

pytestmark = pytest.mark.usefixtures("tmp_dir")


@pytest.mark.parametrize(
    "logins,team,expected",
    [
        pytest.param([("all", None), ("x", ["x"])], "x", "x", id="team-over-all-teams"),
        pytest.param(
            [("all", None), ("x", ["x"])], "y", "all", id="all-teams-for-another-team"
        ),
        pytest.param(
            [("all", None), ("x", ["x"])], None, "all", id="all-teams-without-a-team"
        ),
        pytest.param([("x", ["x"])], "y", None, id="team-token-only-for-its-teams"),
    ],
)
def test_token_precedence(logins, team, expected):
    for token, teams in logins:
        studio_login(token, teams=teams)

    assert resolve_token(STUDIO_URL, team) == expected


def test_token_is_picked_by_url():
    studio_login("other", url=OTHER_URL)
    studio_login("default")

    assert resolve_token(STUDIO_URL, "x") == "default"
    assert resolve_token(OTHER_URL, "x") == "other"
    assert resolve_token(f"{OTHER_URL}/", "x") == "other"


def test_project_team_picks_the_user_token():
    studio_login("x-token", teams=["x"])
    studio_login("y-token", teams=["y"], default_team="y")
    set_studio(ConfigLevel.LOCAL, team="x")

    assert StudioClient().token == "x-token"  # noqa: S105


def test_project_credentials_file_is_not_read(tmp_dir):
    path = tmp_dir / ".datachain" / "credentials"
    path.parent.mkdir()
    path.write_text('[[token]]\ntoken = "project"\n')
    path.chmod(0o600)

    assert load_logins() == []
    assert resolve_token(STUDIO_URL, None) is None


def test_env_token_wins_for_every_team(monkeypatch):
    studio_login("team", teams=["acme"])
    studio_login("all")
    monkeypatch.setenv("DATACHAIN_STUDIO_TOKEN", "env")

    assert resolve_token(STUDIO_URL, "acme") == "env"
    assert is_token_set("other-team")


def test_is_token_set_checks_the_team():
    studio_login("team-token", teams=["customer-x"])

    assert is_token_set("customer-x")
    assert not is_token_set("acme")
    assert not is_token_set()

    set_studio(ConfigLevel.GLOBAL, team="customer-x")
    assert is_token_set()


def test_client_has_no_token_for_a_url_set_by_the_project():
    studio_login("user-token")
    set_studio(ConfigLevel.LOCAL, url=OTHER_URL, team="acme")

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


def test_expired_token_names_a_self_hosted_studio():
    studio_login("old", url=OTHER_URL, teams=["x"], expires=date(2020, 1, 1))

    with pytest.raises(DataChainError, match=f"for team 'x' at {OTHER_URL} in ") as exc:
        resolve_token(OTHER_URL, "x")
    # Without -H, the command would remove the token for the default Studio.
    assert f"Run `datachain auth logout -H {OTHER_URL} --team x`" in str(exc.value)


def test_logout_command_quotes_its_arguments():
    login = Login(url="https://studio.example.com/a b", token="t", teams=["Customer A"])  # noqa: S106

    assert login.logout_command(host=True) == (
        "datachain auth logout -H 'https://studio.example.com/a b' --team 'Customer A'"
    )
    plain = Login(url=STUDIO_URL, token="t", teams=["customer-a"])  # noqa: S106
    assert plain.logout_command() == "datachain auth logout --team customer-a"


def test_login_normalizes_url_and_teams():
    login = Login(url=f"{STUDIO_URL}/", token="token", teams=["a", "a", "b"])  # noqa: S106

    assert (login.url, login.teams) == (STUDIO_URL, ["a", "b"])


def test_url_is_saved_only_for_a_self_hosted_studio(global_config_dir):
    studio_login("default")
    studio_login("other", url=OTHER_URL)

    text = (Path(global_config_dir) / "credentials").read_text()
    assert text.startswith('[[token]]\ntoken = "default"')
    assert STUDIO_URL not in text
    assert f'url = "{OTHER_URL}"' in text
    logins = [(lg.url, lg.token) for lg in load_logins()]
    assert logins == [(STUDIO_URL, "default"), (OTHER_URL, "other")]


@pytest.mark.parametrize(
    "content,error",
    [
        ('token = "isat_abc"\n', "each token must be a [[token]] entry"),
        ('token = ["isat_abc"]\n', "each token must be a [[token]] entry"),
        ("[[token]\n", "Unexpected character"),
        ('[[token]]\nname = "x"\n', "`token` must be a non-empty string"),
        ('[[token]]\ntoken = "t"\nurl = 123\n', "`url` must be a string"),
        ('[[token]]\ntoken = "t"\nteams = "acme"\n', "`teams` must be a non-empty"),
        ('[[token]]\ntoken = "t"\nteams = []\n', "`teams` must be a non-empty"),
        ('[[token]]\ntoken = "t"\nexpires = "2020-01-01"\n', "`expires` must be"),
        ('[[token]]\ntoken = "t"\nexpires = 2020-01-01T00:00:00\n', "`expires`"),
    ],
)
def test_invalid_file_is_named(global_config_dir, content, error):
    path = Path(global_config_dir) / "credentials"
    path.write_text(content)
    path.chmod(0o600)

    with pytest.raises(DataChainError, match="Invalid Studio credentials file") as exc:
        load_logins()
    assert str(path) in str(exc.value)
    assert error in str(exc.value)


def test_file_that_is_not_utf8_is_named(global_config_dir):
    path = Path(global_config_dir) / "credentials"
    path.write_bytes(b"\xff\xfe[[token]]")
    path.chmod(0o600)

    with pytest.raises(DataChainError, match="Invalid Studio credentials file") as exc:
        load_logins()
    assert str(path) in str(exc.value)


@posix_only
def test_saved_file_is_private(global_config_dir):
    studio_login("token")

    path = Path(global_config_dir) / "credentials"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@posix_only
def test_file_open_to_others_is_ignored(caplog, global_config_dir):
    studio_login("token")
    (Path(global_config_dir) / "credentials").chmod(0o644)

    with caplog.at_level(logging.WARNING, logger="datachain"):
        assert load_logins() == []
        assert resolve_token(STUDIO_URL, None) is None
    [warning] = caplog.messages
    assert "other users can access it" in warning
