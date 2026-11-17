import logging
import os
import stat
import threading
import time
from pathlib import Path

import pytest

from datachain.config import Config, ConfigLevel
from datachain.error import DataChainError
from datachain.remote import credentials
from datachain.remote.credentials import (
    TOKEN_ENV,
    Login,
    credentials_path,
    find_login,
    load_logins,
    migrate_legacy,
    remove_login,
    resolve_token,
)
from datachain.remote.studio import StudioClient, is_token_set
from datachain.utils import STUDIO_URL
from tests.utils import studio_login

OTHER_URL = "https://studio.example.com"
LOCAL = ConfigLevel.LOCAL
GLOBAL = ConfigLevel.GLOBAL
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("TOKEN", "URL", "TEAM"):
        monkeypatch.delenv(f"DATACHAIN_STUDIO_{var}", raising=False)
    return tmp_path


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
            id="local-scoped-over-local-all-teams",
        ),
        pytest.param(
            [("global-x", ["x"], GLOBAL), ("local-all", None, LOCAL)],
            "x",
            "local-all",
            id="local-all-teams-over-global-scoped",
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
            id="global-scoped-over-global-all-teams",
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
            id="scoped-only-for-its-teams",
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


def test_env_token_wins(monkeypatch):
    studio_login("local", teams=["acme"], level=LOCAL)
    studio_login("global")
    monkeypatch.setenv(TOKEN_ENV, "env")

    assert resolve_token(STUDIO_URL, "acme") == "env"
    assert is_token_set("other-team")


def test_system_legacy_token_is_the_last_fallback(caplog):
    set_studio(ConfigLevel.SYSTEM, token="system-token")  # noqa: S106

    with caplog.at_level(logging.WARNING, logger="datachain"):
        assert resolve_token(STUDIO_URL, "acme") == "system-token"
    assert "remove `studio.token`" in caplog.text

    studio_login("global")
    assert resolve_token(STUDIO_URL, "acme") == "global"


def test_is_token_set_checks_the_team():
    studio_login("scoped", teams=["customer-x"])

    assert is_token_set("customer-x")
    assert not is_token_set("acme")
    assert not is_token_set()

    set_studio(GLOBAL, team="customer-x")
    assert is_token_set()


def test_token_is_never_sent_to_another_url():
    # A repo's config pointing at another host must not receive the user's token.
    studio_login("user-token")
    set_studio(LOCAL, url=OTHER_URL, team="acme")

    with pytest.raises(DataChainError, match=f"Not logged in to Studio at {OTHER_URL}"):
        StudioClient().token  # noqa: B018


def test_legacy_token_is_bound_to_its_own_config_url(caplog):
    set_studio(GLOBAL, token="legacy-token", team="acme")  # noqa: S106
    set_studio(LOCAL, url=OTHER_URL)

    assert find_login(OTHER_URL, "acme") is None
    with caplog.at_level(logging.WARNING, logger="datachain"):
        assert resolve_token(STUDIO_URL, "acme") == "legacy-token"
    assert "datachain auth login" in caplog.text


def test_client_uses_team_to_pick_token():
    studio_login("all", team="acme")
    studio_login("scoped", teams=["customer-x"])

    assert StudioClient().token == "all"  # noqa: S105
    assert StudioClient(team="customer-x").token == "scoped"  # noqa: S105


def test_login_normalizes_url_and_teams():
    login = Login(url=f"{STUDIO_URL}/", token="token", teams=["a", "a", "b"])  # noqa: S106

    assert (login.url, login.teams) == (STUDIO_URL, ["a", "b"])


@posix_only
def test_saved_file_is_private():
    studio_login("token")

    assert stat.S_IMODE(os.stat(credentials_path(GLOBAL)).st_mode) == 0o600


@posix_only
def test_file_open_to_others_is_ignored(caplog):
    # What a token committed to a repo looks like after a clone.
    studio_login("committed", level=LOCAL)
    os.chmod(credentials_path(LOCAL), 0o644)

    with caplog.at_level(logging.WARNING, logger="datachain"):
        assert load_logins(LOCAL) == []
        assert resolve_token(STUDIO_URL, None) is None
    [warning] = caplog.messages
    assert "other users can access it" in warning


@posix_only
def test_file_open_to_others_is_not_overwritten():
    studio_login("committed", level=LOCAL)
    path = Path(credentials_path(LOCAL))
    path.chmod(0o644)
    saved = path.read_text()

    with pytest.raises(DataChainError, match="chmod 600"):
        studio_login("new", teams=["x"], level=LOCAL)

    assert path.read_text() == saved


def test_local_login_is_ignored_in_git(project):
    gitignore = project / ".datachain" / ".gitignore"
    gitignore.parent.mkdir()
    gitignore.write_text("/cache")

    studio_login("token", level=LOCAL)
    studio_login("other", teams=["x"], level=LOCAL)

    assert gitignore.read_text() == "/cache\n/credentials*\n"


@pytest.mark.parametrize("first_update", ["add", "remove"])
def test_concurrent_updates_keep_each_other(monkeypatch, first_update):
    studio_login("old", teams=["team-old"])
    save = credentials._save_logins
    first_has_read = threading.Event()

    def slow_save(*args):
        # The first update holds its read while the second one starts.
        if threading.current_thread().name == "first":
            first_has_read.set()
            time.sleep(0.3)
        save(*args)

    monkeypatch.setattr(credentials, "_save_logins", slow_save)

    def update():
        if first_update == "add":
            studio_login("a", teams=["team-a"])
        else:
            remove_login(find_login(STUDIO_URL, "team-old"))

    first = threading.Thread(target=update, name="first")
    first.start()
    assert first_has_read.wait(5)
    studio_login("b", teams=["team-b"])
    first.join()

    tokens = sorted(login.token for login in load_logins(GLOBAL))
    assert tokens == (["a", "b", "old"] if first_update == "add" else ["b"])


def test_migrate_legacy_keeps_url_and_team():
    set_studio(
        GLOBAL,
        token="legacy-token",  # noqa: S106
        url=OTHER_URL,
        team="acme",
    )

    migrated = migrate_legacy(GLOBAL)

    assert migrated is not None
    assert migrated.url == OTHER_URL
    assert Config(GLOBAL).read()["studio"] == {"url": OTHER_URL, "team": "acme"}
    [stored] = load_logins(GLOBAL)
    assert (stored.url, stored.token, stored.teams) == (OTHER_URL, "legacy-token", None)


def test_migrate_legacy_leaves_token_next_to_existing_login():
    studio_login("new-token")
    set_studio(GLOBAL, token="legacy-token")  # noqa: S106

    assert migrate_legacy(GLOBAL) is None
    assert Config(GLOBAL).read()["studio"]["token"] == "legacy-token"  # noqa: S105
