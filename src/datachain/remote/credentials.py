"""Studio logins: tokens kept outside the config, each bound to one Studio URL.

A login holds a token for one Studio URL, either for a list of teams or, saved
without ``--team``, as the fallback when no token is saved for a team. Which teams
Studio accepts it for is set in Studio when the token is created. Logins live in a
``credentials`` file next to the config: in the project's ``.datachain/``
(``--local``) or in the global config dir. The file is written ``0600`` and
ignored if other users can read it. A login without ``url`` is for the default
Studio.

A command finds its token from the Studio URL and team it acts on:
``DATACHAIN_STUDIO_TOKEN`` first, then the project's logins, then the user's. At
each level a token for the team wins over a token without teams.
"""

import logging
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import tomlkit

from datachain.config import Config, ConfigLevel
from datachain.error import DataChainError
from datachain.utils import STUDIO_URL, DataChainDir

logger = logging.getLogger("datachain")

CREDENTIALS = DataChainDir.CREDENTIALS
TOKEN_ENV = "DATACHAIN_STUDIO_TOKEN"  # noqa: S105

# Levels with a credentials file, in lookup order: a project's logins win.
LEVELS = (ConfigLevel.LOCAL, ConfigLevel.GLOBAL)

_warned: set[str] = set()


@dataclass
class Login:
    url: str
    token: str
    # None: saved without --team; the fallback when no token is saved for a team.
    teams: list[str] | None = None
    name: str | None = None
    expires: date | None = None
    level: ConfigLevel = ConfigLevel.GLOBAL

    def __post_init__(self) -> None:
        self.url = normalize_url(self.url)
        if self.teams is not None:
            self.teams = list(dict.fromkeys(self.teams))

    @property
    def path(self) -> str:
        return credentials_path(self.level)

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
        host_flag = f" -H {self.url}" if host else ""
        team_flag = f" --team {team or self.teams[0]}" if self.teams else ""
        local_flag = " --local" if self.level == ConfigLevel.LOCAL else ""
        return f"datachain auth logout{host_flag}{team_flag}{local_flag}"


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def credentials_path(level: ConfigLevel) -> str:
    return os.path.join(Config.get_dir(level), CREDENTIALS)


def _open_to_others(path: str) -> bool:
    if os.name == "nt":
        return False
    return bool(os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO))


def load_logins(level: ConfigLevel, *, for_update: bool = False) -> list[Login]:
    """The logins saved at this level.

    A file other users can access is skipped with a warning; updating it fails,
    so its tokens are not lost by saving over it.
    """
    path = credentials_path(level)
    try:
        if _open_to_others(path):
            # A token committed to a repo comes back 0644 from a clone.
            if for_update:
                raise DataChainError(
                    f"Other users can access {path}, so DataChain does not use the "
                    f"Studio tokens in it. Run `chmod 600 {path}` if you trust it."
                )
            if path not in _warned:
                _warned.add(path)
                logger.warning(
                    "Ignoring Studio logins in %s: other users can access it. "
                    "Run `chmod 600 %s` if you trust it.",
                    path,
                    path,
                )
            return []
        with open(path, encoding="utf-8") as f:
            doc = tomlkit.load(f).unwrap()
    except FileNotFoundError:
        return []

    return [
        Login(
            url=item.get("url") or STUDIO_URL,
            token=item["token"],
            teams=item.get("teams"),
            name=item.get("name"),
            expires=item.get("expires"),
            level=level,
        )
        for item in doc.get("login", [])
        if item.get("token")
    ]


def _save_logins(level: ConfigLevel, logins: list[Login]) -> str:
    path = credentials_path(level)
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
    doc["login"] = entries

    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    if level == ConfigLevel.LOCAL:
        _ignore_in_git(directory)

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


def _ignore_in_git(directory: str) -> None:
    pattern = f"/{CREDENTIALS}"
    gitignore = Path(directory) / ".gitignore"
    try:
        content = gitignore.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = ""
    if pattern in content.splitlines():
        return
    prefix = "\n" if content and not content.endswith("\n") else ""
    with open(gitignore, "a", encoding="utf-8") as f:
        f.write(f"{prefix}{pattern}\n")


def all_logins() -> list[Login]:
    """Every saved login, in lookup order."""
    return [login for level in LEVELS for login in load_logins(level)]


def select_login(logins: list[Login], url: str, team: str | None) -> Login | None:
    """The login to use for this Studio URL and team, from the given ones."""
    matching = [lg for lg in logins if lg.url == url and lg.covers(team)]
    # A project's login wins; then a login for the team wins over one without teams.
    return min(
        matching,
        key=lambda lg: (LEVELS.index(lg.level), lg.teams is None),
        default=None,
    )


def resolve_token(url: str, team: str | None) -> str | None:
    if token := os.environ.get(TOKEN_ENV):
        return token
    if not (login := select_login(all_logins(), url, team)):
        return None
    if login.expired():
        # Falling back to another token could switch to another account.
        which = f" for team '{team}'" if login.teams else ""
        raise DataChainError(
            f"The Studio token{which} in {login.path} expired on "
            f"{login.expires}. Run `{login.logout_command(team)}` and log in again."
        )
    return login.token


def conflicting_login(
    level: ConfigLevel, url: str, teams: list[str] | None
) -> Login | None:
    """A login at this level that a new login for these teams would overlap.

    Per level and URL there is at most one all-teams login, and each team is in
    at most one team login. Fails like an update would, so `auth login` stops
    before the browser if the login can't be saved.
    """
    for login in load_logins(level, for_update=True):
        if login.url != url:
            continue
        if teams is None and login.teams is None:
            return login
        if teams and login.teams and set(teams) & set(login.teams):
            return login
    return None


def add_login(login: Login) -> str:
    logins = load_logins(login.level, for_update=True)
    return _save_logins(login.level, [*logins, login])


def remove_login(login: Login) -> None:
    logins = load_logins(login.level, for_update=True)
    _save_logins(login.level, [lg for lg in logins if lg != login])
