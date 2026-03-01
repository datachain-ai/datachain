from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar, cast
from urllib.parse import quote, urlparse

from fsspec.implementations.http import HTTPFileSystem

from datachain.dataset import StorageURI
from datachain.lib.file import File

from .fsspec import Client

if TYPE_CHECKING:
    from datachain.cache import Cache


class HTTPClient(Client):
    FS_CLASS = HTTPFileSystem
    PREFIX: ClassVar[str] = "http://"
    protocol: ClassVar[str] = "http"

    @classmethod
    def create_fs(cls, **kwargs) -> HTTPFileSystem:
        # Configure HTTPFileSystem options
        kwargs.setdefault("simple_links", True)
        kwargs.setdefault("same_scheme", True)
        kwargs.setdefault("cache_type", "bytes")

        kwargs.pop("version_aware", None)

        fs = cls.FS_CLASS(**kwargs)
        fs.invalidate_cache()
        return cast("HTTPFileSystem", fs)

    @classmethod
    def from_name(
        cls,
        name: str,
        cache: "Cache",
        kwargs: dict[str, Any],
    ) -> "HTTPClient":
        parsed = urlparse(name)

        if parsed.scheme:
            name = parsed.netloc + parsed.path

        return cls(name, kwargs, cache)

    @classmethod
    def split_url(cls, url: str) -> tuple[str, str]:
        """Split HTTP/HTTPS URL into domain (bucket equivalent) and path."""
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path.lstrip("/")

        if parsed.query:
            path += f"?{parsed.query}"
        if parsed.fragment:
            path += f"#{parsed.fragment}"

        return domain, path

    @classmethod
    def storage_uri(cls, storage_name: str) -> "StorageURI":
        if not storage_name.startswith(("http://", "https://")):
            return StorageURI(f"{cls.PREFIX}{storage_name}")
        return StorageURI(storage_name)

    @classmethod
    def is_root_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.path in ("", "/") and not parsed.query and not parsed.fragment

    def get_uri(self, rel_path: str) -> str:
        if self.name.startswith(("http://", "https://")):
            base_url = self.name
        else:
            if rel_path and "/" in rel_path:
                first_part = rel_path.split("/", maxsplit=1)[0]
                if "." in first_part and not first_part.startswith("."):
                    return f"{self.protocol}://{rel_path}"

            base_url = f"{self.protocol}://{self.name}"

        if rel_path:
            # `rel_path` may include query/fragment (see split_url). Only
            # percent-encode the path portion; keep query/fragment intact.
            path_part = rel_path
            query_part = ""
            fragment_part = ""
            if "#" in path_part:
                path_part, fragment_part = path_part.split("#", 1)
                fragment_part = f"#{fragment_part}"
            if "?" in path_part:
                path_part, query_part = path_part.split("?", 1)
                query_part = f"?{query_part}"

            # Preserve '/' delimiters and existing percent-escapes.
            path_part = quote(path_part, safe="/%")
            rel_path = f"{path_part}{query_part}{fragment_part}"

            if not base_url.endswith("/") and not rel_path.startswith("/"):
                base_url += "/"
            full_url = base_url + rel_path
        else:
            full_url = base_url

        return full_url

    def url(
        self,
        path: str,
        expires: int = 3600,
        version_id: str | None = None,
        **kwargs,
    ) -> str:
        """
        Generate URL for the given path.
        Note: HTTP URLs don't support signed/expiring URLs.
        """
        return self.get_uri(path)

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        etag = v.get("ETag", "").strip('"')
        last_modified = v.get("last_modified")
        if last_modified:
            if isinstance(last_modified, str):
                try:
                    from email.utils import parsedate_to_datetime

                    last_modified = parsedate_to_datetime(last_modified)
                except (ValueError, TypeError):
                    last_modified = datetime.now(timezone.utc)
            elif isinstance(last_modified, (int, float)):
                last_modified = datetime.fromtimestamp(last_modified, timezone.utc)
        else:
            last_modified = datetime.now(timezone.utc)

        return File(
            source=self.uri,
            path=path,
            size=v.get("size", 0),
            etag=etag,
            version="",
            is_latest=True,
            last_modified=last_modified,
        )

    def upload(self, data: bytes, path: str) -> "File":
        raise NotImplementedError(
            "HTTP/HTTPS client is read-only. Upload operations are not supported."
        )

    def get_file_info(self, path: str, version_id: str | None = None) -> "File":
        info = self.fs.info(self.get_uri(path))
        return self.info_to_file(info, path)

    def open_object(self, file: "File", use_cache: bool = True, cb=None):
        from datachain.client.fileslice import FileWrapper

        if use_cache and (cache_path := self.cache.get_path(file)):
            return open(cache_path, mode="rb")

        assert not file.location
        return FileWrapper(
            self.fs.open(file.get_fs_path()),
            cb or (lambda x: None),
        )

    async def get_file(self, lpath, rpath, callback, version_id: str | None = None):
        return await self.fs._get_file(lpath, rpath, callback=callback)

    async def _fetch_dir(self, prefix: str, pbar, result_queue) -> set[str]:
        full_url = self.get_uri(prefix)
        raise NotImplementedError(f"Cannot download file from {full_url}")


class HTTPSClient(HTTPClient):
    protocol = "https"
    PREFIX = "https://"
