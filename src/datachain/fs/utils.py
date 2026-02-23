import os
from typing import TYPE_CHECKING

from fsspec.implementations.local import LocalFileSystem

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


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
