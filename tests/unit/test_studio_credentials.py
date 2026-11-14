import logging
import os
import stat

import pytest

from datachain.config import Config, ConfigLevel
from datachain.error import DataChainError
from datachain.remote.credentials import (
    Login,
    add_login,
    credentials_path,
    find_login,
    has_login,
    load_logins,
    migrate_legacy,
    resolve_token,
)
from datachain.remote.studio import StudioClient
from datachain.utils import STUDIO_URL

OTHER_URL = "https://studio.example.com"


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATACHAIN_STUDIO_TOKEN", raising=False)
    monkeypatch.delenv("DATACHAIN_STUDIO_URL", raising=False)
    monkeypatch.delenv("DATACHAIN_STUDIO_TEAM", raising=False)
    return tmp_path


def login(token, teams=None, url=STUDIO_URL, level=ConfigLevel.GLOBAL):
    add_login(Login(url=url, token=token, teams=teams, level=level))


def set_studio(level, **values):
    with Config(level).edit() as conf:
        conf["studio"] = values


def test_scoped_token_wins_over_all_teams_token():
    login("all")
    login("scoped", teams=["customer-x"])

    assert resolve_token(STUDIO_URL, "customer-x") == "scoped"
    assert resolve_token(STUDIO_URL, "acme") == "all"
    assert resolve_token(STUDIO_URL, None) == "all"


def test_project_token_wins_over_user_token():
    login("user-scoped", teams=["acme"])
    login("project", level=ConfigLevel.LOCAL)

    assert resolve_token(STUDIO_URL, "acme") == "project"


def test_env_token_wins(monkeypatch):
    login("stored")
    monkeypatch.setenv("DATACHAIN_STUDIO_TOKEN", "env")

    assert resolve_token(STUDIO_URL, "acme") == "env"
    assert has_login(OTHER_URL)


def test_scoped_token_not_used_for_other_teams():
    login("scoped", teams=["customer-x"])

    assert resolve_token(STUDIO_URL, "acme") is None
    assert has_login(STUDIO_URL)
    assert not has_login(OTHER_URL)


def test_token_is_never_sent_to_another_url():
    # A repo's config pointing at another host must not receive the user's token.
    login("user-token")
    set_studio(ConfigLevel.LOCAL, url=OTHER_URL, team="acme")

    with pytest.raises(DataChainError, match=f"Not logged in to Studio at {OTHER_URL}"):
        StudioClient().token  # noqa: B018


def test_legacy_token_is_bound_to_its_own_config_url(caplog):
    set_studio(ConfigLevel.GLOBAL, token="legacy-token", team="acme")  # noqa: S106
    set_studio(ConfigLevel.LOCAL, url=OTHER_URL)

    assert find_login(OTHER_URL, "acme") is None
    with caplog.at_level(logging.WARNING, logger="datachain"):
        assert resolve_token(STUDIO_URL, "acme") == "legacy-token"
    assert "datachain auth login" in caplog.text


def test_client_uses_team_to_pick_token():
    login("all")
    login("scoped", teams=["customer-x"])
    set_studio(ConfigLevel.GLOBAL, team="acme")

    assert StudioClient().token == "all"  # noqa: S105
    assert StudioClient(team="customer-x").token == "scoped"  # noqa: S105


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_saved_file_is_private():
    login("token")

    assert stat.S_IMODE(os.stat(credentials_path(ConfigLevel.GLOBAL)).st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_file_open_to_others_is_ignored(caplog):
    # What a token committed to a repo looks like after a clone.
    login("committed", level=ConfigLevel.LOCAL)
    os.chmod(credentials_path(ConfigLevel.LOCAL), 0o644)

    with caplog.at_level(logging.WARNING, logger="datachain"):
        assert load_logins(ConfigLevel.LOCAL) == []
    assert "other users can access it" in caplog.text
    assert resolve_token(STUDIO_URL, None) is None


def test_lookup_does_not_create_project_dir(project):
    assert resolve_token(STUDIO_URL, "acme") is None
    assert not (project / ".datachain").exists()


def test_migrate_legacy_keeps_url_and_team():
    set_studio(
        ConfigLevel.GLOBAL,
        token="legacy-token",  # noqa: S106
        url=OTHER_URL,
        team="acme",
    )

    migrated = migrate_legacy(ConfigLevel.GLOBAL)

    assert migrated is not None
    assert migrated.url == OTHER_URL
    assert Config(ConfigLevel.GLOBAL).read()["studio"] == {
        "url": OTHER_URL,
        "team": "acme",
    }
    [stored] = load_logins(ConfigLevel.GLOBAL)
    assert (stored.url, stored.token, stored.teams) == (OTHER_URL, "legacy-token", None)


def test_migrate_legacy_leaves_token_next_to_existing_login():
    login("new-token")
    set_studio(ConfigLevel.GLOBAL, token="legacy-token")  # noqa: S106

    assert migrate_legacy(ConfigLevel.GLOBAL) is None
    assert Config(ConfigLevel.GLOBAL).read()["studio"]["token"] == "legacy-token"  # noqa: S105
