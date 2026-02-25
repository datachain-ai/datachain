import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from fsspec.implementations.local import LocalFileSystem

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


def is_win_local_path(uri: str) -> bool:
    """Return True if *uri* looks like a Windows local path (drive letter or UNC)."""
    if sys.platform == "win32":
        if len(uri) >= 1 and uri[0] in ("\\", "/"):
            return True
        if (
            len(uri) >= 3
            and uri[1] == ":"
            and (uri[2] == "/" or uri[2] == "\\")
            and uri[0].isalpha()
        ):
            return True
    return False


def path_to_uri(path: str) -> str:
    """Resolve a local path (absolute, relative, ``~/``, Windows) to a ``file://`` URI.

    If *path* already carries a recognized URI scheme (and is not a Windows
    drive-letter path that ``urlparse`` misidentifies) it is returned as-is.

    Examples::

        ./animals          -> file:///home/user/cwd/animals
        ~/animals          -> file:///home/user/animals
        /home/user/animals -> file:///home/user/animals
        /data/dir/         -> file:///data/dir/
        C:\\windows\\animals -> file:///C:/windows/animals
    """
    # Preserve explicit URIs / URLs.
    parsed = urlparse(path)
    if parsed.scheme and not is_win_local_path(path):
        return path

    # Construct a file:// urlpath without percent-encoding.
    # Note: This is *not* a strict RFC-compliant URI when it contains
    # reserved characters or spaces, but aligns with the fsspec urlpath model.
    abs_path = Path(path).expanduser().absolute().resolve().as_posix()
    uri = f"file:///{abs_path.lstrip('/')}"
    if path and path[-1] in (os.sep, "/"):
        uri += "/"
    return uri


def is_subpath(parent: str, child: str) -> bool:
    """True iff *child* is strictly inside *parent* (path-traversal guard).

    Both paths must be absolute OS paths. ``..`` segments are resolved before
    comparison so that traversal tricks like ``/out/../etc/passwd`` are caught.
    Comparison is case-insensitive on Windows.
    """
    assert os.path.isabs(parent), f"parent must be absolute: {parent!r}"
    assert os.path.isabs(child), f"child must be absolute: {child!r}"

    parent_normed = os.path.normcase(os.path.normpath(parent))
    child_normed = os.path.normcase(os.path.normpath(child))

    if child_normed == parent_normed:
        return False
    return child_normed.startswith(parent_normed + os.sep)


def _isdir(fs: "AbstractFileSystem", path: str) -> bool:
    info = fs.info(path)
    return info["type"] == "directory" or (
        info["size"] == 0 and info["type"] == "file" and info["name"].endswith("/")
    )


def isfile(fs: "AbstractFileSystem", path: str) -> bool:
    """
    Returns True if uri points to a file.

    Supports special directories on object storages, e.g.:
    Google creates a zero byte file with the same name as the directory with a trailing
    slash at the end.
    """
    if isinstance(fs, LocalFileSystem):
        return fs.isfile(path)

    try:
        return not _isdir(fs, path)
    except FileNotFoundError:
        return False
