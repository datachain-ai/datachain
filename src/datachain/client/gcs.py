import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, BinaryIO, cast
from urllib.parse import quote

from fsspec.asyn import get_loop, sync
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from gcsfs import GCSFileSystem
from gcsfs.checkers import get_consistency_checker
from gcsfs.core import GCSFile
from gcsfs.retry import HttpError

from datachain.client.fileslice import FileWrapper
from datachain.lib.file import File, FileRow

from .fsspec import BucketStatus, RangeListedClient
from .ranges import Keep, Page

if TYPE_CHECKING:
    from datachain.client.writeconfig import WriteConfig

# Patch gcsfs for consistency with s3fs
GCSFileSystem.set_session = GCSFileSystem._set_session
# Skip the GCE metadata check — it adds latency and hangs outside GCE.
os.environ.setdefault("NO_GCE_CHECK", "true")
SMALL_FILE_SIZE = 5 * 1024 * 1024


def _check_etag(file: File, etag: str) -> None:
    if not file.version and file.etag and file.etag != etag.strip('"'):
        raise FileNotFoundError(
            f"{file.source}/{file.path} changed on the source since the "
            f"catalog was created (etag was {file.etag}, now {etag}). "
            "Re-run the original dc.read_storage(...) call with "
            "update=True to refresh the catalog."
        )


class _ListedGCSFile(GCSFile):
    def __init__(self, fs, file: File, path: str, block_size: int | None):
        path = fs._strip_protocol(path)
        self._file = file
        self._details = {"name": path, "size": file.size, "type": "file"}
        super().__init__(
            fs,
            path,
            mode="rb",
            size=file.size,
            block_size=fs.default_block_size if block_size is None else block_size,
            cache_type="readahead",
            consistency=fs.consistency,
        )

    def _fetch_range(self, start=None, end=None):
        return sync(self.fs.loop, self._fetch_listed_range, start, end)

    async def _fetch_listed_range(self, start, end):
        if start is not None and end is not None and start >= end:
            return b""
        headers = {}
        if start is not None or end is not None:
            headers["Range"] = await self.fs._process_limits(self.path, start, end)
        headers, data = await self.fs._call(
            "GET", self.fs.url(self.path), headers=headers
        )
        _check_etag(self._file, headers.get("ETag", ""))
        return data


class GCSClient(RangeListedClient):
    FS_CLASS = GCSFileSystem
    PREFIX = "gs://"
    LISTS_LEVELS = True
    protocol = "gs"
    CREDENTIAL_KEYS = frozenset({"token"})
    LIST_PAGE_SIZE = 5000
    # The version id is the object's generation. Metadata updates, such as a storage
    # class change, change the etag but keep the generation. Appends to an object in
    # a zonal bucket also keep the generation, but they change the size.
    PINNED_VERSION_IS_IMMUTABLE = True
    # the fields a listing asks for: GCS sends every field otherwise
    LIST_FIELDS = (
        "items(name,size,generation,etag,timeDeleted,updated),nextPageToken,prefixes"
    )

    @classmethod
    def create_fs(cls, **kwargs) -> GCSFileSystem:
        if os.environ.get("DATACHAIN_GCP_CREDENTIALS"):
            kwargs["token"] = json.loads(os.environ["DATACHAIN_GCP_CREDENTIALS"])
        if kwargs.pop("anon", False):
            kwargs["token"] = "anon"  # noqa: S105
        # Read gzip-encoded objects without decompressing them, so the bytes match
        # File.size and range reads work. GCS decompresses unless the request accepts
        # gzip, and aiohttp decompresses unless told not to.
        # https://github.com/fsspec/gcsfs/issues/233#issuecomment-4465900826
        session_kwargs = kwargs.get("session_kwargs") or {}
        headers = {
            name: value
            for name, value in dict(session_kwargs.get("headers") or ()).items()
            if name.lower() != "accept-encoding"
        }
        headers["Accept-Encoding"] = "gzip"
        kwargs["session_kwargs"] = {
            **session_kwargs,
            "auto_decompress": False,
            "headers": headers,
        }

        return cast("GCSFileSystem", super().create_fs(**kwargs))

    @classmethod
    def bucket_status(cls, name: str, **kwargs) -> BucketStatus:  # noqa: PLR0911
        from google.api_core import exceptions as google_exceptions

        # Step 1: Anonymous probe.
        # Use _call with maxResults=1 (objects.list API) not _info (buckets.get API)
        # or _ls: GCS does not grant storage.buckets.get anonymously even for public
        # buckets, and _ls lists the whole bucket root and triggers unnecessary
        # gRPC calls.
        anon_kwargs = {k: v for k, v in kwargs.items() if k != "anon"}
        anon_kwargs["anon"] = True
        anon_fs = cls.create_fs(**anon_kwargs)
        try:
            sync(
                get_loop(),
                anon_fs._call,
                "GET",
                "b/{}/o",
                name,
                maxResults=1,
                json_out=True,
            )
            return BucketStatus(exists=True, access="anonymous")
        except FileNotFoundError:
            return BucketStatus(
                exists=False, access="denied", error=f"GCS bucket '{name}' not found"
            )
        except (PermissionError, HttpError, OSError) as e:
            if isinstance(e, HttpError) and e.code == 404:
                return BucketStatus(
                    exists=False,
                    access="denied",
                    error=f"GCS bucket '{name}' not found",
                )

        # Step 2: Authenticated probe — create_fs resolves credentials from
        # kwargs, environment, or application-default credentials.
        auth_fs = cls.create_fs(**kwargs)
        try:
            sync(get_loop(), auth_fs._info, name)
            return BucketStatus(exists=True, access="authenticated")
        except FileNotFoundError:
            return BucketStatus(
                exists=False, access="denied", error=f"GCS bucket '{name}' not found"
            )
        except (google_exceptions.Forbidden, google_exceptions.PermissionDenied) as e:
            return BucketStatus(exists=True, access="denied", error=str(e))
        except (PermissionError, HttpError, OSError) as e:
            if isinstance(e, HttpError) and e.code == 404:
                return BucketStatus(
                    exists=False,
                    access="denied",
                    error=f"GCS bucket '{name}' not found",
                )
            return BucketStatus(
                exists=True,
                access="denied",
                error=f"Access denied to GCS bucket '{name}'"
                " — check credentials/permissions",
            )

    @staticmethod
    def _write_kwargs(cfg: "WriteConfig", *, streaming: bool) -> dict[str, Any]:
        cfg.reject_write_options("GCS")
        kw: dict[str, Any] = {}
        if cfg.content_type:
            kw["content_type"] = cfg.content_type
        if cfg.metadata:
            kw["metadata"] = dict(cfg.metadata)
        fixed: dict[str, Any] = {}
        if cfg.content_disposition:
            fixed["content_disposition"] = cfg.content_disposition
        if cfg.cache_control:
            fixed["cache_control"] = cfg.cache_control
        if cfg.content_encoding:
            fixed["content_encoding"] = cfg.content_encoding
        if fixed:
            kw["fixed_key_metadata"] = fixed
        return kw

    def url(
        self,
        path: str,
        expires: int = 3600,
        version_id: str | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a signed URL for the given path.
        If the client is anonymous, a public URL is returned instead
        (see https://cloud.google.com/storage/docs/access-public-data#api-link).
        """
        content_disposition = kwargs.pop("content_disposition", None)
        if self.fs.storage_options.get("token") == "anon":
            query = f"?generation={version_id}" if version_id else ""
            # Public URL must be URI-encoded. Preserve '/' so object keys that
            # use it as a delimiter stay readable.
            encoded_path = quote(path, safe="/")
            return f"https://storage.googleapis.com/{self.name}/{encoded_path}{query}"
        full_path = self.get_uri(path)
        full_path = self._path_with_generation(full_path, version_id)
        return self.fs.sign(
            full_path,
            expiration=expires,
            response_disposition=content_disposition,
            **kwargs,
        )

    def _version_kwargs(self, version_id: str | None) -> dict[str, Any]:
        if version_id:
            return {"generation": version_id}
        return {}

    @staticmethod
    def _path_with_generation(path: str, generation: str | None) -> str:
        if generation:
            for char in ("#", "?"):
                if char in path:
                    raise ValueError(
                        f"Versioned access is not supported for GCS keys "
                        f"containing {char!r}: {path!r}"
                    )
            return f"{path}#{generation}"
        return path

    async def get_file(
        self,
        lpath: str,
        rpath: str,
        callback,
        version_id: str | None = None,
    ) -> None:
        # Workaround: gcsfs._get_file() silently ignores the generation= kwarg.
        # Embed it in the path as `path#generation` instead.
        # Remove the whole override once gcsfs supports generation in _get_file()
        path = self._path_with_generation(lpath, version_id)
        await self.fs._get_file(path, rpath, callback=callback)

    async def get_current_etag(self, file: File) -> str:
        path = self._path_with_generation(file.get_fs_path(), file.version)
        info = await self.fs._info(path)
        return self.info_to_file(info, file.path).etag

    @staticmethod
    def _can_download_listed(file: File) -> bool:
        return 0 <= file.size <= SMALL_FILE_SIZE and bool(file.etag or file.version)

    async def _put_in_cache(self, file: File, *, callback: Callback = None) -> None:
        assert not file.location
        if self._can_download_listed(file):
            await self.cache.download(file, self, callback=callback)
        else:
            await super()._put_in_cache(file, callback=callback)

    async def _download_file(self, file: File, path: str, callback: Callback):
        if not self._can_download_listed(file):
            return await super()._download_file(file, path, callback)
        remote = self._path_with_generation(file.get_fs_path(), file.version)
        headers, data = await self.fs._call("GET", self.fs.url(remote))
        _check_etag(file, headers.get("ETag", ""))
        checker = get_consistency_checker(self.fs.consistency)
        checker.update(data)
        checker.validate_headers(headers)
        if len(data) != file.size:
            raise FileNotFoundError(
                f"{file.source}/{file.path} changed on the source since the "
                f"catalog was created (size was {file.size}, now {len(data)}). "
                "Re-run the original dc.read_storage(...) call with "
                "update=True to refresh the catalog."
            )
        callback.set_size(len(data))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as stream:  # noqa: ASYNC230
            stream.write(data)
        callback.relative_update(len(data))

    def get_file_info(self, path: str, version_id: str | None = None) -> File:
        self.validate_file_path(path)
        fs_path = self._path_with_generation(self.get_uri(path), version_id)
        info = sync(get_loop(), self.fs._info, fs_path)
        return self.info_to_file(info, path)

    async def get_size(self, file: File) -> int:
        path = self._path_with_generation(file.get_fs_path(), file.version)
        info = await self.fs._info(path)
        size = info.get("size")
        if size is None:
            raise FileNotFoundError(file.get_fs_path())
        return int(size)

    def open_object(
        self,
        file: File,
        use_cache: bool = True,
        cb: Callback = DEFAULT_CALLBACK,
        block_size: int | None = None,
    ) -> BinaryIO:
        if use_cache and (cache_path := self.cache.get_path(file)):
            return open(cache_path, mode="rb")
        assert not file.location
        full_path = self._path_with_generation(
            file.get_fs_path(),
            file.version,
        )
        return FileWrapper(
            (
                _ListedGCSFile(self.fs, file, full_path, block_size)
                if file.size > 0 and self._can_download_listed(file)
                else self.fs.open(full_path, **self._block_size_kwargs(block_size))
            ),
            cb,
        )  # type: ignore[return-value]

    @staticmethod
    def parse_timestamp(timestamp: str) -> datetime:
        """
        Parse timestamp string returned by GCSFileSystem.

        This ensures that the passed timestamp is timezone aware.
        """
        dt = datetime.fromisoformat(timestamp)
        assert dt.tzinfo is not None
        return dt

    async def _list_page(
        self,
        prefix: str,
        start_after: str,
        token: Any,
        hi: str | None = None,
        page_size: int | None = None,
        delimiter: str = "",
    ) -> Page:
        page = await self.fs._call(
            "GET",
            "b/{}/o",
            self.name,
            delimiter=delimiter,
            prefix=prefix,
            startOffset=start_after or None,
            endOffset=None if hi is None else f"{hi}\0",  # exclusive
            maxResults=page_size or self.LIST_PAGE_SIZE,
            pageToken=token,
            fields=self.LIST_FIELDS,
            json_out=True,
            versions="true" if self._is_version_aware() else "false",
        )
        items = page.get("items", [])
        keep = Keep(self, start_after, hi)
        # folders past a delimiter: listed past, though they make no rows
        keys = [d["name"] for d in items] + page.get("prefixes", [])
        return Page(
            [self._file_row(d) for d in items if keep(d["name"])],
            items[0]["name"] if items else None,
            max(keys, default=None),
            page.get("nextPageToken"),
            keep.past_lo,
            keep.past_hi,
            keep.in_order,
        )

    def _file_row(self, d: dict[str, Any]) -> tuple:
        return FileRow(
            self.uri,
            d["name"],
            int(d.get("size", 0)),
            d.get("generation", "") if self._is_version_aware() else "",
            d.get("etag", ""),
            not d.get("timeDeleted"),
            self.parse_timestamp(d["updated"]),
        )

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        return File(
            source=self.uri,
            path=path,
            etag=v.get("etag", ""),
            version=v.get("generation", "") if self._is_version_aware() else "",
            is_latest=not v.get("timeDeleted"),
            last_modified=self.parse_timestamp(v["updated"]),
            size=v.get("size", ""),
        )
