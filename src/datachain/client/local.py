import os
import posixpath
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fsspec.implementations.local import LocalFileSystem

from datachain.fs.utils import path_to_uri
from datachain.lib.file import File, FileError

from .fsspec import Client

if TYPE_CHECKING:
    from datachain.cache import Cache
    from datachain.dataset import StorageURI


class FileClient(Client):
    FS_CLASS = LocalFileSystem
    PREFIX = "file://"
    protocol = "file"

    def __init__(
        self,
        name: str,
        fs_kwargs: dict[str, Any],
        cache: "Cache",
        use_symlinks: bool = False,
    ) -> None:
        super().__init__(name, fs_kwargs, cache)
        self.use_symlinks = use_symlinks

    def url(
        self,
        path: str,
        expires: int = 3600,
        version_id: str | None = None,
        **kwargs,
    ) -> str:
        raise TypeError("Signed urls are not implemented for local file system")

    @classmethod
    def storage_uri(cls, storage_name: str) -> "StorageURI":
        from datachain.dataset import StorageURI

        return StorageURI(path_to_uri(storage_name))

    @classmethod
    def ls_buckets(cls, **kwargs) -> Iterator[Any]:
        return iter(())

    @classmethod
    def split_url(cls, url: str) -> tuple[str, str]:
        if not url.startswith("file://"):
            url = path_to_uri(url)

        os_path = LocalFileSystem._strip_protocol(url)

        # Preserve "directory" semantics when a trailing slash is present.
        if url.endswith("/"):
            bucket = os_path.rstrip("/")
            path = ""
        else:
            bucket, path = os_path.rsplit("/", 1)

        if os.name == "nt":
            bucket = bucket.removeprefix("/")

        return bucket, path

    @classmethod
    def from_name(cls, name: str, cache: "Cache", kwargs) -> "FileClient":
        use_symlinks = kwargs.pop("use_symlinks", False)
        return cls(name, kwargs, cache, use_symlinks=use_symlinks)

    @classmethod
    def from_source(
        cls,
        uri: str,
        cache: "Cache",
        use_symlinks: bool = False,
        **kwargs,
    ) -> "FileClient":
        return cls(
            LocalFileSystem._strip_protocol(uri),
            kwargs,
            cache,
            use_symlinks=use_symlinks,
        )

    async def get_current_etag(self, file: "File") -> str:
        info = self.fs.info(file.get_fs_path())
        return self.info_to_file(info, "").etag

    @classmethod
    def rel_path_for_file(cls, file: "File") -> str:
        path = file.path
        try:
            cls.validate_file_relpath(path)
        except ValueError as exc:
            raise FileError(str(exc), file.source, path) from None

        normpath = os.path.normpath(path)
        normpath = Path(normpath).as_posix()

        if normpath == ".":
            raise FileError("path must not be a directory", file.source, path)

        if any(part == ".." for part in Path(normpath).parts):
            raise FileError("path must not contain '..'", file.source, path)

        return normpath

    @staticmethod
    def validate_file_relpath(path: str) -> None:
        """Ensure a file's relative path is safe to use on the local filesystem.

        Rejects absolute, traversal, and malformed paths that could escape the
        intended directory or require implicit normalization. On Windows,
        backslashes and drive-letter prefixes are handled as separators/absolute.
        """

        if not path:
            raise ValueError(f"unsafe file path {path!r}: must not be empty")

        # On Windows, backslash is a path separator — normalize so the
        # checks below work uniformly.  On Linux/macOS, backslash is a legal
        # filename character and must not be reinterpreted as a separator.
        if os.name == "nt":
            canonical = path.replace("\\", "/")
        else:
            canonical = path

        if canonical.endswith("/"):
            raise ValueError(f"unsafe file path {path!r}: must not be a directory")

        # Disallow absolute paths; local file paths are interpreted relative to
        # the source/output prefix.
        if canonical.startswith("/"):
            raise ValueError(f"unsafe file path {path!r}: must not be absolute")

        # On Windows, a drive-letter prefix like "C:/" is absolute even
        # without a leading "/".  On Unix, colons are legal in filenames,
        # so only enforce this on Windows.
        if (
            os.name == "nt"
            and len(canonical) >= 2
            and canonical[0].isalpha()
            and canonical[1] == ":"
        ):
            raise ValueError(f"unsafe file path {path!r}: must not be absolute")

        # Disallow empty segments (e.g. 'dir//file.txt') to avoid implicit
        # normalization.
        if "//" in canonical:
            raise ValueError(
                f"unsafe file path {path!r}: must not contain empty segments"
            )

        # Disallow dot segments like '.' or '..' (even if they could be
        # normalized away) because they can make I/O and exports unsafe.
        raw_parts = canonical.split("/")
        if any(part in (".", "..") for part in raw_parts):
            raise ValueError(f"unsafe file path {path!r}: must not contain '.' or '..'")

    @staticmethod
    def _has_drive_letter(source: str) -> bool:
        """Check whether a ``file://`` URI contains an explicit drive letter.

        On Windows, fsspec silently prepends the current drive to paths like
        ``file:///bucket`` (→ ``C:/bucket``).  We use this helper to detect
        and reject such ambiguous URIs early.
        """
        rest = source[len("file://") :].lstrip("/")
        return len(rest) >= 2 and rest[0].isalpha() and rest[1] == ":"

    def get_file_info(self, path: str, version_id: str | None = None) -> File:
        info = self.fs.info(self.get_uri(path))
        return self.info_to_file(info, path)

    async def get_size(self, file: File) -> int:
        full_path = file.get_fs_path()

        size = self.fs.size(full_path)
        if size is None:
            raise FileNotFoundError(full_path)
        return int(size)

    async def get_file(self, lpath, rpath, callback, version_id: str | None = None):
        return self.fs.get_file(
            lpath,
            rpath,
            callback=callback,
        )

    async def ls_dir(self, path):
        return self.fs.ls(path, detail=True)

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)

    def get_uri(self, rel_path):
        """Build a full file:// URI for *rel_path* within this client's storage."""
        joined = Path(self.name, rel_path).as_posix()
        if rel_path.endswith("/") or not rel_path:
            joined += "/"
        return path_to_uri(joined)

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
    ) -> None:
        if not self.use_symlinks:
            super().fetch_nodes(nodes, shared_progress_bar)

    def do_instantiate_object(self, file: File, dst: str) -> None:
        if self.use_symlinks:
            os.symlink(Path(self.name, file.path), dst)
        else:
            super().do_instantiate_object(file, dst)
