import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, BinaryIO, cast
from urllib.parse import quote

from fsspec.asyn import get_loop, sync
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from gcsfs import GCSFileSystem
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


class GCSClient(RangeListedClient):
    FS_CLASS = GCSFileSystem
    PREFIX = "gs://"
    protocol = "gs"
    CREDENTIAL_KEYS = frozenset({"token"})
    LIST_PAGE_SIZE = 5000
    # the fields a listing asks for: GCS sends every field otherwise
    LIST_FIELDS = "items(name,size,generation,etag,timeDeleted,updated),nextPageToken"

    @classmethod
    def create_fs(cls, **kwargs) -> GCSFileSystem:
        if os.environ.get("DATACHAIN_GCP_CREDENTIALS"):
            kwargs["token"] = json.loads(os.environ["DATACHAIN_GCP_CREDENTIALS"])
        if kwargs.pop("anon", False):
            kwargs["token"] = "anon"  # noqa: S105

        return cast("GCSFileSystem", super().create_fs(**kwargs))

    @classmethod
    def bucket_status(cls, name: str, **kwargs) -> BucketStatus:  # noqa: PLR0911
        from google.api_core import exceptions as google_exceptions

        # Step 1: Anonymous probe.
        # Use _ls (objects.list API) not _info (buckets.get API): GCS does not
        # grant storage.buckets.get anonymously even for public buckets.
        anon_kwargs = {k: v for k, v in kwargs.items() if k != "anon"}
        anon_kwargs["anon"] = True
        anon_fs = cls.create_fs(**anon_kwargs)
        try:
            sync(get_loop(), anon_fs._ls, name)
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
            self.fs.open(full_path, block_size=block_size),
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
    ) -> Page:
        page_token, held = (
            token if self.list_deleted and token is not None else (token, None)
        )
        # a generation's delete time reads the next generation's creation time
        fields = self.LIST_FIELDS
        if self.list_deleted:
            fields = fields.replace("updated)", "updated,timeCreated)")
        page = await self.fs._call(
            "GET",
            "b/{}/o",
            self.name,
            delimiter="",
            prefix=prefix,
            startOffset=start_after or None,
            endOffset=None if hi is None else f"{hi}\0",  # exclusive
            maxResults=page_size or self.LIST_PAGE_SIZE,
            pageToken=page_token,
            fields=fields,
            json_out=True,
            versions="true" if self._is_version_aware() else "false",
        )
        items = page.get("items", [])
        keep = Keep(self, start_after, hi)
        first = items[0]["name"] if items else None
        last = items[-1]["name"] if items else None
        next_token = page.get("nextPageToken")
        if self.list_deleted:
            # a generation's delete time depends on the next generation of its key
            items = ([held] if held else []) + items
            held = items[-1] if next_token is not None and items else None
            rows = self._rows([d for d in items if keep(d["name"])], held)
            next_token = None if next_token is None else (next_token, held)
        else:
            rows = [self._file_row(d) for d in items if keep(d["name"])]
        return Page(
            rows,
            first,
            last,
            next_token,
            keep.past_lo,
            keep.past_hi,
            keep.in_order,
        )

    def _rows(
        self, items: list[dict[str, Any]], held: dict[str, Any] | None
    ) -> list[tuple]:
        """Rows of ``items`` with their delete rows; ``held`` waits for the next
        page."""
        items = sorted(items, key=lambda d: (d["name"], int(d["generation"])))
        rows = []
        for i, d in enumerate(items):
            if d is held:
                continue
            rows.append(self._file_row(d))
            after = items[i + 1] if i + 1 < len(items) else None
            if deleted_at := self._deleted_at(d, after):
                version = f"deleted:{d['generation']}"
                rows.append(self._deleted_row(d["name"], version, deleted_at))
        return rows

    def _deleted_at(
        self, d: dict[str, Any], after: dict[str, Any] | None
    ) -> datetime | None:
        """When generation ``d`` was deleted; None while it is live or after the
        next generation replaced it (an overwrite)."""
        if not (deleted := d.get("timeDeleted")):
            return None
        deleted_at = self.parse_timestamp(deleted)
        replaced = (
            after is not None
            and after["name"] == d["name"]
            and self.parse_timestamp(after["timeCreated"]) <= deleted_at
        )
        return None if replaced else deleted_at

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
