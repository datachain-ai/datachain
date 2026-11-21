"""Studio logins: tokens kept outside the config, each bound to one Studio URL.

A login holds a token for one Studio URL, either for a list of teams or, saved
without ``--team``, as the fallback when no token is saved for a team. Which teams
Studio accepts it for is set in Studio when the token is created. Logins live in a
``credentials`` file in the user's DataChain config dir, never in a project: a
project picks its team, and the team picks the token. The file is written
``0600`` and ignored if other users can read it. Each login is a ``[[token]]``
entry; one without ``url`` is for the default Studio.

A command finds its token from the Studio URL and team it acts on:
``DATACHAIN_STUDIO_TOKEN`` first, then a token for the team, then the fallback.
The env token is only sent to a Studio you chose (``DATACHAIN_STUDIO_URL``, your
or the system config, or the default), never to one only a project's config sets.
"""

import logging
import os
import shlex
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import tomlkit
from tomlkit.exceptions import TOMLKitError

from datachain.config import Config, ConfigLevel
from datachain.error import DataChainError
from datachain.utils import STUDIO_URL, global_config_dir

logger = logging.getLogger("datachain")

CREDENTIALS = "credentials"
TOKEN_ENV = "DATACHAIN_STUDIO_TOKEN"  # noqa: S105
URL_ENV = "DATACHAIN_STUDIO_URL"

_warned: set[str] = set()


@dataclass
class Login:
    url: str
    token: str
    # None: saved without --team; the fallback when no token is saved for a team.
    teams: list[str] | None = None
    name: str | None = None
    expires: date | None = None

    def __post_init__(self) -> None:
        self.url = normalize_url(self.url)
        if self.teams is not None:
            self.teams = list(dict.fromkeys(self.teams))

    def covers(self, team: str | None) -> bool:
        return self.teams is None or team in self.teams

    def for_teams(self) -> str:
        """Names the teams in a message; empty for a token saved without --team."""
        if not self.teams:
            return ""
        return f" for team{'s' if len(self.teams) > 1 else ''} {', '.join(self.teams)}"

    def expired(self) -> bool:
        return self.expires is not None and self.expires < datetime.now(UTC).date()

    def logout_command(self, team: str | None = None, host: bool = False) -> str:
        """The `datachain auth logout` command that removes this login.

        For a token for teams, `team` picks the team to name; it defaults to the
        first.
        """
        # Quoted, so the command can be pasted into a shell as is.
        host_flag = f" -H {shlex.quote(self.url)}" if host else ""
        team_flag = (
            f" --team {shlex.quote(team or self.teams[0])}" if self.teams else ""
        )
        return f"datachain auth logout{host_flag}{team_flag}"


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def self_hosted(url: str) -> str:
    """Names a self-hosted Studio in a message; the default Studio goes unnamed."""
    url = normalize_url(url)
    return "" if url == STUDIO_URL else f" at {url}"


def env_token_url() -> str:
    """The Studio `DATACHAIN_STUDIO_TOKEN` may be sent to.

    One you chose: from the env, your or the system config, or the default. A
    project's config, which a cloned repo can contain, doesn't count.
    """
    if url := os.environ.get(URL_ENV):
        return normalize_url(url)
    studio: dict = {}
    for level in (ConfigLevel.SYSTEM, ConfigLevel.GLOBAL):
        studio.update(Config(level).read().get("studio", {}) or {})
    return normalize_url(studio.get("url") or STUDIO_URL)


def credentials_path() -> str:
    return os.path.join(global_config_dir(), CREDENTIALS)


def _open_to_others(path: str) -> bool:
    if os.name == "nt":
        return False
    return bool(os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO))


def _entry_error(item: dict) -> str | None:
    """What is wrong with a `[[token]]` entry, or None."""
    token = item.get("token")
    if not isinstance(token, str) or not token:
        return "`token` must be a non-empty string"
    for key in ("url", "name"):
        if not isinstance(item.get(key, ""), str):
            return f"`{key}` must be a string"
    teams = item.get("teams")
    if teams is not None and not (
        isinstance(teams, list)
        and teams
        and all(isinstance(t, str) and t for t in teams)
    ):
        return "`teams` must be a non-empty list of team names"
    expires = item.get("expires")
    # A TOML datetime is a datetime, which is also a date.
    if expires is not None and (
        not isinstance(expires, date) or isinstance(expires, datetime)
    ):
        return "`expires` must be a date, like 2027-01-31"
    return None


def load_logins(*, for_update: bool = False) -> list[Login]:
    """The saved logins.

    A file other users can access is skipped with a warning, as ssh does with
    private keys; updating it fails, so its tokens are not lost by saving over it.
    """
    path = credentials_path()
    try:
        if _open_to_others(path):
            if for_update:
                raise DataChainError(
                    f"Other users can access {path}, so DataChain does not use the "
                    f"Studio tokens in it. Run `chmod 600 {shlex.quote(path)}` if you "
                    "trust it."
                )
            if path not in _warned:
                _warned.add(path)
                logger.warning(
                    "Ignoring Studio tokens in %s: other users can access it. "
                    "Run `chmod 600 %s` if you trust it.",
                    path,
                    shlex.quote(path),
                )
            return []
        with open(path, encoding="utf-8") as f:
            doc = tomlkit.load(f).unwrap()
    except FileNotFoundError:
        return []
    except (TOMLKitError, UnicodeDecodeError) as exc:
        raise DataChainError(f"Invalid Studio credentials file {path}: {exc}") from exc

    entries = doc.get("token", [])
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise DataChainError(
            f"Invalid Studio credentials file {path}: each token must be a "
            "[[token]] entry."
        )
    for item in entries:
        if error := _entry_error(item):
            raise DataChainError(f"Invalid Studio credentials file {path}: {error}.")
    return [
        Login(
            url=item.get("url") or STUDIO_URL,
            token=item["token"],
            teams=item.get("teams"),
            name=item.get("name"),
            expires=item.get("expires"),
        )
        for item in entries
    ]


def _save_logins(logins: list[Login]) -> str:
    path = credentials_path()
    if not logins:
        Path(path).unlink(missing_ok=True)
        return path

    doc = tomlkit.document()
    entries = tomlkit.aot()
    for login in logins:
        entry = tomlkit.table()
        if login.url != STUDIO_URL:
            entry["url"] = login.url
        if login.teams is not None:
            entry["teams"] = login.teams
        if login.name:
            entry["name"] = login.name
        if login.expires:
            entry["expires"] = login.expires
        entry["token"] = login.token
        entries.append(entry)
    doc["token"] = entries

    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    # mkstemp creates the file 0600, so the token is never readable by others.
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{CREDENTIALS}-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            tomlkit.dump(doc, f)
        os.replace(tmp, path)
    except BaseException:
        os.remove(tmp)
        raise
    return path


def _select_login(logins: list[Login], url: str, team: str | None) -> Login | None:
    """The login to use for this Studio URL and team, from the given ones."""
    url = normalize_url(url)
    matching = [lg for lg in logins if lg.url == url and lg.covers(team)]
    # A login for the team wins over one without teams.
    return min(matching, key=lambda lg: lg.teams is None, default=None)


def resolve_token(url: str, team: str | None) -> str | None:
    url = normalize_url(url)
    if (token := os.environ.get(TOKEN_ENV)) and url == env_token_url():
        return token
    if not (login := _select_login(load_logins(), url, team)):
        return None
    if login.expired():
        # Falling back to another token could switch to another account.
        which = f" for team '{team}'" if login.teams else ""
        # -H, so the command removes this token, not one for the default Studio.
        logout = login.logout_command(team, host=bool(self_hosted(login.url)))
        raise DataChainError(
            f"The Studio token{which}{self_hosted(login.url)} in {credentials_path()} "
            f"expired on {login.expires}. Run `{logout}` and log in again."
        )
    return login.token


def conflicting_login(url: str, teams: list[str] | None) -> Login | None:
    """A saved login that a new login for these teams would overlap.

    Per URL there is at most one login without teams, and each team is in at most
    one team login. Fails like an update would, so `auth login` stops before the
    browser if the login can't be saved.
    """
    url = normalize_url(url)
    for login in load_logins(for_update=True):
        if login.url != url:
            continue
        if teams is None and login.teams is None:
            return login
        if teams and login.teams and set(teams) & set(login.teams):
            return login
    return None


def add_login(login: Login) -> str:
    return _save_logins([*load_logins(for_update=True), login])


def remove_login(login: Login) -> None:
    _save_logins([lg for lg in load_logins(for_update=True) if lg != login])
