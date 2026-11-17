"""Studio logins: tokens kept outside the config, each bound to one Studio URL.

A login holds a token for one Studio URL, scoped either to a list of teams or to
all teams the token can reach. Logins live in a ``credentials`` file next to the
config: in the project's ``.datachain/`` (``--local``) or in the global config dir.
The file is written ``0600`` and ignored if other users can read it.

A command finds its token from the Studio URL and team it acts on:
``DATACHAIN_STUDIO_TOKEN`` first, then the project's logins, then the user's. At
each level a login scoped to the team wins over an all-teams login.
"""

import logging
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import tomlkit

from datachain.config import Config, ConfigLevel
from datachain.error import DataChainError
from datachain.utils import STUDIO_URL, DataChainDir, interprocess_file_lock

logger = logging.getLogger("datachain")

CREDENTIALS = DataChainDir.CREDENTIALS
TOKEN_ENV = "DATACHAIN_STUDIO_TOKEN"  # noqa: S105

# Levels with a credentials file. A project's logins win over the user's.
LOGIN_LEVELS = (ConfigLevel.LOCAL, ConfigLevel.GLOBAL)
# Lookup order. A token an older version saved as `[studio] token` can also sit in
# the system config.
LOOKUP_LEVELS = (*LOGIN_LEVELS, ConfigLevel.SYSTEM)

_warned: set[str] = set()


@dataclass
class Login:
    url: str
    token: str
    # None: every team the token can reach.
    teams: list[str] | None = None
    name: str | None = None
    expires: str | None = None
    level: ConfigLevel = ConfigLevel.GLOBAL
    # Path of the file the login was read from or is saved to.
    path: str = field(default="", compare=False)
    # Saved by an older version as `[studio] token` in the config file.
    legacy: bool = False

    def __post_init__(self) -> None:
        self.url = normalize_url(self.url)
        if self.teams is not None:
            self.teams = list(dict.fromkeys(self.teams))

    def covers(self, team: str | None) -> bool:
        return self.teams is None or (team is not None and team in self.teams)

    def describe_teams(self) -> str:
        return "all teams" if self.teams is None else ", ".join(self.teams)


class LoginExistsError(DataChainError):
    def __init__(self, login: Login) -> None:
        self.login = login
        super().__init__(
            f"Token already exists for {login.url} ({login.describe_teams()}) "
            f"in {login.path}"
        )


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def credentials_path(level: ConfigLevel) -> str:
    if level not in LOGIN_LEVELS:
        raise ValueError(f"Logins are not stored at the {level.value} level")
    return os.path.join(Config.get_dir(level), CREDENTIALS)


def _warn_once(path: str, msg: str, *args) -> None:
    if path not in _warned:
        _warned.add(path)
        logger.warning(msg, *args)


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
            _warn_once(
                path,
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

    logins = []
    for item in doc.get("login", []):
        if not item.get("url") or not item.get("token"):
            continue
        logins.append(
            Login(
                url=item["url"],
                token=item["token"],
                teams=item.get("teams"),
                name=item.get("name"),
                expires=item.get("expires"),
                level=level,
                path=path,
            )
        )
    return logins


@contextmanager
def _updating(level: ConfigLevel) -> Iterator[list[Login]]:
    """Read this level's logins under a lock, and save the list when done."""
    path = credentials_path(level)
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    with interprocess_file_lock(f"{path}.lock"):
        logins = load_logins(level, for_update=True)
        yield logins
        _save_logins(level, path, logins)


def _save_logins(level: ConfigLevel, path: str, logins: list[Login]) -> None:
    if not logins:
        Path(path).unlink(missing_ok=True)
        return

    doc = tomlkit.document()
    entries = tomlkit.aot()
    for login in logins:
        entry = tomlkit.table()
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


def _ignore_in_git(directory: str) -> None:
    # Covers the lock file next to the credentials as well.
    pattern = f"/{CREDENTIALS}*"
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


def legacy_login(level: ConfigLevel) -> Login | None:
    """The token an older version saved in this level's config, if any.

    It is bound to the URL in the same file, so a URL set at another level (say,
    a repo's config) never receives it.
    """
    path = os.path.join(Config.get_dir(level), DataChainDir.CONFIG)
    try:
        with open(path, encoding="utf-8") as f:
            studio = tomlkit.load(f).unwrap().get("studio", {}) or {}
    except FileNotFoundError:
        return None
    if not (token := studio.get("token")):
        return None
    return Login(
        url=studio.get("url") or STUDIO_URL,
        token=token,
        level=level,
        path=path,
        legacy=True,
    )


def _logins_at(level: ConfigLevel) -> list[Login]:
    logins = load_logins(level) if level in LOGIN_LEVELS else []
    if legacy := legacy_login(level):
        logins.append(legacy)
    return logins


def all_logins() -> list[Login]:
    """Every stored login, in lookup order."""
    return [login for level in LOOKUP_LEVELS for login in _logins_at(level)]


def select_login(logins: list[Login], url: str, team: str | None) -> Login | None:
    """The login to use for this Studio URL and team, from the given ones."""
    url = normalize_url(url)
    for level in LOOKUP_LEVELS:
        matching = [
            lg
            for lg in logins
            if lg.level == level and lg.url == url and lg.covers(team)
        ]
        scoped = next((lg for lg in matching if lg.teams is not None), None)
        if found := scoped or next(iter(matching), None):
            return found
    return None


def find_login(url: str, team: str | None) -> Login | None:
    """The stored login to use for this Studio URL and team, if any."""
    for level in LOOKUP_LEVELS:
        if found := select_login(_logins_at(level), url, team):
            if found.legacy:
                _warn_legacy(found)
            return found
    return None


def _warn_legacy(login: Login) -> None:
    if login.level == ConfigLevel.SYSTEM:
        # `auth login` can't write the system config, so it can't move this one.
        _warn_once(
            login.path,
            "The Studio token is stored in the system config file %s, which other "
            "users and tools may read. Run `datachain auth login` to save your own "
            "token in a private credentials file, and remove `studio.token` from %s.",
            login.path,
            login.path,
        )
        return
    flag = " --local" if login.level == ConfigLevel.LOCAL else ""
    _warn_once(
        login.path,
        "The Studio token is stored in the config file %s, which other users and "
        "tools may read. Run `datachain auth login%s` to move it to a private "
        "credentials file.",
        login.path,
        flag,
    )


def resolve_token(url: str, team: str | None) -> str | None:
    if token := os.environ.get(TOKEN_ENV):
        return token
    login = find_login(url, team)
    return login.token if login else None


def _conflict(logins: list[Login], url: str, teams: list[str] | None) -> Login | None:
    for login in logins:
        if login.url != normalize_url(url):
            continue
        if teams is None and login.teams is None:
            return login
        if (
            teams is not None
            and login.teams is not None
            and set(teams) & set(login.teams)
        ):
            return login
    return None


def conflicting_login(
    level: ConfigLevel, url: str, teams: list[str] | None
) -> Login | None:
    """A login at this level that a new login for these teams would overlap.

    Per level and URL there is at most one all-teams login, and each team is in
    at most one scoped login. Fails like an update would, so `auth login` stops
    before the browser if the login can't be saved.
    """
    return _conflict(load_logins(level, for_update=True), url, teams)


def add_login(login: Login) -> str:
    """Save a login. Raises LoginExistsError if it overlaps a saved one."""
    with _updating(login.level) as logins:
        if existing := _conflict(logins, login.url, login.teams):
            raise LoginExistsError(existing)
        logins.append(login)
    login.path = credentials_path(login.level)
    return login.path


def remove_login(login: Login) -> None:
    if login.legacy:
        with Config(login.level).edit() as conf:
            if studio := conf.get("studio"):
                studio.pop("token", None)
                if not studio:
                    del conf["studio"]
        return
    with _updating(login.level) as logins:
        logins[:] = [lg for lg in logins if lg != login]


def migrate_legacy(level: ConfigLevel) -> Login | None:
    """Move the token an older version kept in this level's config into logins."""
    if level not in LOGIN_LEVELS or not (legacy := legacy_login(level)):
        return None
    login = Login(url=legacy.url, token=legacy.token, level=level)
    try:
        add_login(login)
    except LoginExistsError:
        # Already has an all-teams login there; leave the old token to logout.
        return None
    remove_login(legacy)
    return login
