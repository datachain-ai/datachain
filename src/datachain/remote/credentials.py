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
from dataclasses import dataclass, field

import tomlkit

from datachain.config import Config, ConfigLevel
from datachain.utils import (
    STUDIO_URL,
    DataChainDir,
    global_config_dir,
    system_config_dir,
)

logger = logging.getLogger("datachain")

CREDENTIALS = DataChainDir.CREDENTIALS
TOKEN_ENV = "DATACHAIN_STUDIO_TOKEN"  # noqa: S105

# Levels with a credentials file. A project's logins win over the user's.
LOGIN_LEVELS = (ConfigLevel.LOCAL, ConfigLevel.GLOBAL)
# Lookup order. A token an older version saved as `[studio] token` can also sit in
# the system config.
LOOKUP_LEVELS = (*LOGIN_LEVELS, ConfigLevel.SYSTEM)

_warned_legacy: set[str] = set()


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

    def covers(self, team: str | None) -> bool:
        return self.teams is None or (team is not None and team in self.teams)

    def describe_teams(self) -> str:
        return "all teams" if self.teams is None else ", ".join(self.teams)


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def level_dir(level: ConfigLevel) -> str:
    """The directory of a config level, without creating it as Config() does."""
    if level == ConfigLevel.SYSTEM:
        return system_config_dir()
    if level == ConfigLevel.GLOBAL:
        return global_config_dir()
    return DataChainDir().root


def credentials_path(level: ConfigLevel) -> str:
    if level not in LOGIN_LEVELS:
        raise ValueError(f"Logins are not stored at the {level.value} level")
    return os.path.join(level_dir(level), CREDENTIALS)


def _open_to_others(path: str) -> bool:
    if os.name == "nt":
        return False
    return bool(os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO))


def load_logins(level: ConfigLevel) -> list[Login]:
    path = credentials_path(level)
    try:
        if _open_to_others(path):
            # A token committed to a repo comes back 0644 from a clone.
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

    logins = []
    for item in doc.get("login", []):
        if not item.get("url") or not item.get("token"):
            continue
        teams = item.get("teams")
        logins.append(
            Login(
                url=normalize_url(item["url"]),
                token=item["token"],
                teams=list(teams) if teams is not None else None,
                name=item.get("name"),
                expires=item.get("expires"),
                level=level,
                path=path,
            )
        )
    return logins


def save_logins(level: ConfigLevel, logins: list[Login]) -> str:
    path = credentials_path(level)
    directory = os.path.dirname(path)
    if not logins:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return path

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
    gitignore = os.path.join(directory, ".gitignore")
    try:
        with open(gitignore, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []
    if CREDENTIALS in lines or f"/{CREDENTIALS}" in lines:
        return
    with open(gitignore, "a", encoding="utf-8") as f:
        if lines and lines[-1]:
            f.write("\n")
        f.write(f"/{CREDENTIALS}\n")


def legacy_login(level: ConfigLevel) -> Login | None:
    """The token an older version saved in this level's config, if any.

    It is bound to the URL in the same file, so a URL set at another level (say,
    a repo's config) never receives it.
    """
    path = os.path.join(level_dir(level), DataChainDir.CONFIG)
    try:
        with open(path, encoding="utf-8") as f:
            studio = tomlkit.load(f).unwrap().get("studio", {}) or {}
    except FileNotFoundError:
        return None
    if not (token := studio.get("token")):
        return None
    return Login(
        url=normalize_url(studio.get("url") or STUDIO_URL),
        token=token,
        level=level,
        path=path,
        legacy=True,
    )


def all_logins() -> list[Login]:
    """Every stored login, in lookup order."""
    logins: list[Login] = []
    for level in LOOKUP_LEVELS:
        if level in LOGIN_LEVELS:
            logins.extend(load_logins(level))
        if legacy := legacy_login(level):
            logins.append(legacy)
    return logins


def find_login(url: str, team: str | None) -> Login | None:
    """The stored login to use for this Studio URL and team, if any."""
    url = normalize_url(url)
    by_level: dict[ConfigLevel, list[Login]] = {}
    for login in all_logins():
        if login.url == url:
            by_level.setdefault(login.level, []).append(login)

    for level in LOOKUP_LEVELS:
        logins = by_level.get(level, [])
        scoped = [lg for lg in logins if lg.teams is not None and lg.covers(team)]
        found = next(iter(scoped), None) or next(
            (lg for lg in logins if lg.teams is None), None
        )
        if found:
            if found.legacy:
                _warn_legacy(found)
            return found
    return None


def _warn_legacy(login: Login) -> None:
    if login.path in _warned_legacy:
        return
    _warned_legacy.add(login.path)
    flag = " --local" if login.level == ConfigLevel.LOCAL else ""
    logger.warning(
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


def has_login(url: str) -> bool:
    """True if a token is available for this Studio URL, for any team."""
    if os.environ.get(TOKEN_ENV):
        return True
    url = normalize_url(url)
    return any(login.url == url for login in all_logins())


def conflicting_login(
    level: ConfigLevel, url: str, teams: list[str] | None
) -> Login | None:
    """A login at this level that a new login for these teams would overlap."""
    url = normalize_url(url)
    for login in load_logins(level):
        if login.url != url:
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


def add_login(login: Login) -> str:
    logins = load_logins(login.level)
    logins.append(login)
    return save_logins(login.level, logins)


def remove_login(login: Login) -> None:
    if login.legacy:
        with Config(login.level).edit() as conf:
            conf.get("studio", {}).pop("token", None)
        return
    logins = [lg for lg in load_logins(login.level) if lg != login]
    save_logins(login.level, logins)


def migrate_legacy(level: ConfigLevel) -> Login | None:
    """Move the token an older version kept in this level's config into logins."""
    if level not in LOGIN_LEVELS or not (legacy := legacy_login(level)):
        return None
    if conflicting_login(level, legacy.url, None):
        # Already has an all-teams login there; leave the old token to logout.
        return None
    login = Login(url=legacy.url, token=legacy.token, level=level)
    login.path = add_login(login)
    remove_login(legacy)
    return login
