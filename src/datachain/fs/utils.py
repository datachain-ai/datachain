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


def path_to_fsspec_uri(path: str) -> str:
    """Convert a local path or existing URI to a ``file://`` fsspec urlpath.

    OS paths (absolute, relative, ``~/``, Windows drive-letter) are resolved to
    ``file://`` URIs using ``os.path.abspath`` - no symlink dereferencing, no
    percent-encoding.  Existing URIs (anything with a recognized scheme, e.g.
    ``file://``, ``s3://``) are returned **unchanged** (opaque passthrough).

    **No percent-encoding:** characters such as spaces, ``#``, and ``%`` are
    kept literal. fsspec's ``LocalFileSystem`` strips the ``file://`` prefix
    as a plain string and passes the rest straight to the OS, so ``%20`` would
    reach the OS as the three literal characters ``%20`` - not a space. Do
    **not** pass ``Path.as_uri()`` output; that produces RFC-encoded URIs which
    the OS cannot resolve.

    Examples::

        ./animals            -> file:///home/user/cwd/animals
        ~/animals            -> file:///home/user/animals
        /home/user/animals   -> file:///home/user/animals
        /data/dir/           -> file:///data/dir/
        /my dir/f.txt        -> file:///my dir/f.txt   (space kept literal)
        C:\\windows\\animals -> file:///C:/windows/animals
        file:///already/uri  -> file:///already/uri    (unchanged)
    """
    # Pass through existing URIs unchanged (opaque passthrough).
    parsed = urlparse(path)
    if parsed.scheme and not is_win_local_path(path):
        return path

    # Use abspath rather than Path.resolve() to avoid dereferencing symlinks.
    abs_path = Path(os.path.abspath(os.path.expanduser(path))).as_posix()
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
