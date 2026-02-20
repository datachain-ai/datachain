import os
import posixpath
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fsspec.implementations.local import LocalFileSystem

from datachain.lib.file import File, FileError

from .fsspec import Client, is_win_local_path

if TYPE_CHECKING:
    from datachain.cache import Cache
    from datachain.dataset import StorageURI


class FileClient(Client):
    FS_CLASS = LocalFileSystem
    PREFIX = "file://"
    protocol = "file"

    @classmethod
    def is_path_in(
        cls,
        output: str | os.PathLike[str],
        dst: str,
    ) -> bool:
        """Return whether `dst` is safe to write under local `output`.

        Accepts both plain OS paths and `file://` urlpaths. For other schemes,
        returns False.
        """

        from fsspec.utils import stringify_path

        output_str = stringify_path(output)

        # Only handle local paths + file:// urlpaths.
        if "://" in dst and not dst.startswith(cls.PREFIX):
            return False
        if "://" in output_str and not output_str.startswith(cls.PREFIX):
            return False

        output_os = (
            LocalFileSystem._strip_protocol(output_str)
            if output_str.startswith(cls.PREFIX)
            else output_str
        )
        dst_os = (
            LocalFileSystem._strip_protocol(dst) if dst.startswith(cls.PREFIX) else dst
        )

        output_resolved = Path(output_os).resolve(strict=False)
        dst_resolved = Path(dst_os).resolve(strict=False)

        # Destination must be a file path under output, not the output dir itself.
        if dst_resolved == output_resolved:
            return False

        return dst_resolved.is_relative_to(output_resolved)

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
    def get_uri(cls, name: str) -> "StorageURI":
        from datachain.dataset import StorageURI

        return StorageURI(cls.path_to_uri(name))

    @classmethod
    def ls_buckets(cls, **kwargs) -> Iterator[Any]:
        return iter(())

    @classmethod
    def path_to_uri(cls, path: str) -> str:
        """
        Resolving path, that can be absolute or relative, to file URI which
        starts with file:/// prefix
        In unix like systems we support home shortcut as well.
        Examples:
            ./animals -> file:///home/user/working_dir/animals
            ~/animals -> file:///home/user/animals
            /home/user/animals -> file:///home/user/animals
            /home/user/animals/ -> file:///home/user/animals/
            C:\\windows\animals -> file:///C:/windows/animals
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

    @classmethod
    def split_url(cls, url: str) -> tuple[str, str]:
        if not url.startswith("file://"):
            url = cls.path_to_uri(url)

        os_path = LocalFileSystem._strip_protocol(url)

        # Preserve "directory" semantics when a trailing slash is present.
        if url.endswith("/"):
            bucket = os_path.rstrip("/")
            path = ""
        else:
            bucket, path = os.path.split(os_path)

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
        info = self.fs.info(self.full_path_for_file(file))
        return self.info_to_file(info, "").etag

    @classmethod
    def rel_path_for_file(cls, file: "File") -> str:
        path = file.path
        cls.validate_local_relpath(file.source, path)

        normpath = os.path.normpath(path)
        normpath = Path(normpath).as_posix()

        if normpath == ".":
            raise FileError("path must not be a directory", file.source, path)

        if any(part == ".." for part in Path(normpath).parts):
            raise FileError("path must not contain '..'", file.source, path)

        return normpath

    @classmethod
    def validate_local_relpath(cls, source: str, path: str) -> None:
        """Validate a local relative path string.

        Used both for local `File.path` values and for local export destination
        suffixes. Reject inputs that would require normalization (e.g. empty
        segments) or that could be unsafe on the local filesystem.
        """

        if not path:
            raise FileError("path must not be empty", source, path)

        if path.endswith("/"):
            raise FileError("path must not be a directory", source, path)

        raw_posix = path.replace("\\", "/")

        # Disallow absolute paths; local file paths are interpreted relative to
        # the source/output prefix.
        if raw_posix.startswith("/"):
            raise FileError("path must not be absolute", source, path)

        # Disallow empty segments (e.g. 'dir//file.txt') to avoid implicit
        # normalization.
        if "//" in raw_posix:
            raise FileError("path must not contain empty segments", source, path)

        # Disallow dot segments like '.' or '..' (even if they could be
        # normalized away) because they can make I/O and exports unsafe.
        raw_parts = raw_posix.split("/")
        if any(part in (".", "..") for part in raw_parts):
            raise FileError("path must not contain '.' or '..'", source, path)

    @classmethod
    def full_path_for_file(cls, file: "File") -> str:
        rel_path = cls.rel_path_for_file(file)

        base_path = LocalFileSystem._strip_protocol(file.source)
        if not rel_path:
            return base_path
        return os.fspath(Path(base_path, *PurePosixPath(rel_path).parts))

    def get_file_info(self, path: str, version_id: str | None = None) -> File:
        info = self.fs.info(self.get_full_path(path))
        return self.info_to_file(info, path)

    async def get_size(self, file: File) -> int:
        full_path = self.full_path_for_file(file)

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

    def get_full_path(self, rel_path):
        full_path = Path(self.name, rel_path).as_posix()
        if rel_path.endswith("/") or not rel_path:
            full_path += "/"
        return full_path

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
