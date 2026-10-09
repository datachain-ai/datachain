import os
import re
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

    def url(
        self, path: str, expires: int = 3600, version_id: str | None = None, **kwargs
    ) -> str:
        raise NotImplementedError("Not available for local file systems")

    @classmethod
    def storage_uri(cls, storage_name: str) -> "StorageURI":
        from datachain.dataset import StorageURI

        return StorageURI(cls.PREFIX + Path(storage_name).as_posix())

    @classmethod
    def ls_buckets(cls, **kwargs) -> Iterator[Any]:
        # Local filesystem has no concept of buckets
        yield from ()

    @classmethod
    def split_url(cls, url: str) -> tuple[str, str]:
        """Split a local path or file:// URI into (parent_dir, name).

        Always returns the parent directory as the first element and the
        final path component as the second, whether *url* points at a file
        or a directory.  This matches the cloud-client convention where the
        first element is the container (bucket) and the second is the key
        within it.
        """
        path = Path(url.removeprefix(cls.PREFIX))
        return str(path.parent), path.name

    @classmethod
    def from_name(cls, name: str, cache: "Cache", kwargs) -> "FileClient":
        return cls(name, kwargs, cache)

    @classmethod
    def from_source(
        cls,
        uri: str | os.PathLike[str],
        cache: "Cache",
        use_symlinks: bool = False,
        **kwargs,
    ) -> "FileClient":
        host, path = cls.split_url(os.fspath(uri))
        # Reconstruct absolute path for the client root
        name = str(Path(host, path))
        return cls(name, kwargs, cache, use_symlinks=use_symlinks)

    async def get_current_etag(self, file: "File") -> str:
        info = self.fs.info(self.get_full_path(file.path))
        return self.info_to_file(info, file.path).etag

    def validate_file_path(path: str) -> None:
        Client.validate_file_path(path)

    def get_full_path(self, rel_path: str) -> str:
        return str(Path(self.name, rel_path))

    def get_file_info(self, path: str, version_id: str | None = None) -> File:
        info = self.fs.info(self.get_full_path(path))
        return self.info_to_file(info, path)

    async def get_size(self, file: File) -> int:
        return file.size

    async def get_file(self, lpath, rpath, callback, version_id: str | None = None):
        # Not used for local client in the same way; download is via cache
        raise NotImplementedError

    async def ls_dir(self, path):
        return self.fs.ls(path, detail=True)

    def rel_path(self, path):
        return Path(path).relative_to(self.name).as_posix()

    def get_uri(self, rel_path):
        """Build a full file:// URI for *rel_path* within this client's storage."""
        joined = Path(self.name, rel_path).as_posix()
        if rel_path.endswith("/") or not rel_path:
            joined += "/"
        return path_to_fsspec_uri(joined)

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        return File(
            source=self.uri,
            path=path,
            size=v.get("size", ""),
            etag=v["mtime"].hex(),
            is_latest=True,
            last_modified=datetime.fromtimestamp(v["mtime"], timezone.utc),
        )

    def fetch_nodes(
        self,
        nodes,
        shared_progress_bar=None,
    ):
        return super().fetch_nodes(nodes, shared_progress_bar)

    def do_instantiate_object(self, file: File, dst: str) -> None:
        src = self.get_full_path(file.path)
        if self.use_symlinks:
            os.symlink(src, dst)
        else:
            import shutil

            shutil.copyfile(src, dst)
