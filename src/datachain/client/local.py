import os
import re
import posixpath
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fsspec.implementations.local import LocalFileSystem

from datachain.fs.utils import path_to_fsspec_uri
from datachain.lib.file import File

from .fsspec import Client

if TYPE_CHECKING:
    from datachain.cache import Cache
    from datachain.client.writeconfig import WriteConfig
    from datachain.dataset import StorageURI


# Canonical form of ``float.hex()`` / ``st_mtime.hex()`` (e.g. ``0x1.2p+3``).
_MTIME_HEX_RE = re.compile(r"^-?0x[0-9a-f]+(?:\.[0-9a-f]+)?p[+-]?\d+$")


def _local_mtime_iso(etag: str) -> str | None:
    """Return an ISO timestamp if *etag* is exactly ``st_mtime.hex()``."""
    if not _MTIME_HEX_RE.fullmatch(etag):
        return None
    mtime = float.fromhex(etag)
    if mtime.hex() != etag:
        return None
    try:
        return datetime.fromtimestamp(mtime, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


class FileClient(Client):
    FS_CLASS = LocalFileSystem
    PREFIX = "file://"
    protocol = "file"

    @staticmethod
    def _format_etag(etag: str) -> str:
        """Show the stored etag, plus mtime when it is ``st_mtime.hex()``.

        Local listings store mtime as ``float.hex()``. Conversion is limited to
        that canonical shape so HTTP-like values such as ``0x123`` stay intact.
        """
        rendered = str(etag)
        iso = _local_mtime_iso(rendered)
        if iso is None:
            return rendered
        return f"{rendered} (mtime {iso})"

    def __init__(
        self,
        name: str,
        fs_kwargs: dict[str, Any],
        cache: "Cache",
        use_symlinks: bool = False,
    ) -> None:
        super().__init__(name, fs_kwargs, cache)
        self.use_symlinks = use_symlinks

    @staticmethod
    def _write_kwargs(cfg: "WriteConfig", *, streaming: bool) -> dict[str, Any]:
        # Local files carry no content type / metadata; write metadata is ignored.
        return {}
