import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING, Any

from adlfs import AzureBlobFileSystem
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.storage.blob import BlobServiceClient
from fsspec.asyn import get_loop, sync

from datachain.lib.file import File
from datachain.progress import tqdm

from .fsspec import DELIMITER, BucketStatus, Client, ResultQueue

if TYPE_CHECKING:
    from datachain.client.writeconfig import WriteConfig

# Streams larger than this spill to disk while being buffered for upload_blob;
# smaller ones stay in memory.
_SPOOL_MAX_SIZE = 128 * 1024 * 1024
# Copy chunk size when spooling a source stream, matching the fsspec upload path.
_COPY_BUFFER_SIZE = 8 * 1024 * 1024


class _AzureWriteBuffer(SpooledTemporaryFile):
    """Spooled buffer for a metadata-carrying ``File.open`` write to Azure.

    ``upload_blob`` needs the payload up front, so a ``open("wb", content_type=…)``
    write is buffered and uploaded on close. It spills to disk beyond
    ``_SPOOL_MAX_SIZE`` so large writes don't sit in memory; the resulting version
    is exposed on ``version_id``.
    """

    version_id: str | None = None


class AzureClient(Client):
    FS_CLASS = AzureBlobFileSystem
    PREFIX = "az://"
    protocol = "az"
    CREDENTIAL_KEYS = frozenset(
        {
            "account_key",
            "sas_token",
            "connection_string",
            "credential",
            "client_id",
            "client_secret",
            "tenant_id",
        }
    )

    @classmethod
    def bucket_status(cls, name: str, **kwargs) -> BucketStatus:
        # Step 1: Anonymous probe — uses BlobServiceClient directly (not adlfs)
        # to avoid picking up credentials from environment variables like
        # AZURE_STORAGE_CONNECTION_STRING.
        account_name = kwargs.get("account_name")
        if account_name:
            try:
                url = f"https://{account_name}.blob.core.windows.net"
                anon_client = BlobServiceClient(account_url=url)
                anon_client.get_container_client(name).get_container_properties()
                return BucketStatus(exists=True, access="anonymous")
            except ClientAuthenticationError:
                pass
            except ResourceNotFoundError:
                return BucketStatus(
                    exists=False,
                    access="denied",
                    error=f"Azure container '{name}' not found",
                )
            except HttpResponseError as e:
                if e.status_code not in (401, 403):
                    raise

        # Step 2: Authenticated probe.
        try:
            auth_fs = cls.create_fs(**kwargs)
            sync(get_loop(), auth_fs._info, name)
            return BucketStatus(exists=True, access="authenticated")
        except (PermissionError, ClientAuthenticationError):
            return BucketStatus(
                exists=True,
                access="denied",
                error=f"Access denied to Azure container '{name}'"
                " — check credentials/configuration",
            )
        except FileNotFoundError:
            return BucketStatus(
                exists=False,
                access="denied",
                error=f"Azure container '{name}' not found",
            )
        except ValueError as e:
            return BucketStatus(exists=False, access="denied", error=str(e))

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        version_id = v.get("version_id") if self._is_version_aware() else None
        return File(
            source=self.uri,
            path=path,
            etag=v.get("etag", "").strip('"'),
            version=version_id or "",
            is_latest=version_id is None or bool(v.get("is_current_version")),
            last_modified=v["last_modified"],
            size=v.get("size", ""),
        )

    def _write_object(
        self,
        full_path: str,
        data: "bytes | bytearray | memoryview | Any",
        cfg: "WriteConfig",
        *,
        overwrite: bool = True,
    ) -> str | None:
        if isinstance(data, (bytes, bytearray, memoryview)):
            return self._upload_payload(full_path, data, cfg, overwrite=overwrite)
        # upload_blob runs on the fsspec event loop; reading a loop-backed source
        # stream there would re-enter sync(). Buffer it to a plain file (main
        # thread, spilling to disk) first so the loop reads a non-fsspec object.
        with SpooledTemporaryFile(max_size=_SPOOL_MAX_SIZE) as spool:
            shutil.copyfileobj(data, spool, length=_COPY_BUFFER_SIZE)
            spool.seek(0)
            return self._upload_payload(full_path, spool, cfg, overwrite=overwrite)

    def _upload_payload(
        self, full_path: str, payload: Any, cfg: "WriteConfig", *, overwrite: bool
    ) -> str | None:
        # adlfs does not forward content settings on write, so write via the
        # azure-storage-blob SDK, which sets content settings and metadata inline
        # in a single atomic upload (payload is bytes or a plain file object).
        # TODO: once adlfs forwards content settings on write in a released
        # version (https://github.com/fsspec/adlfs/pull/554), this can go through
        # fs.open()/pipe_file() with a _write_kwargs mapping like the other
        # backends, dropping this SDK path.
        cfg.reject_write_options("Azure")
        container, blob, _ = self.fs.split_path(full_path)
        content_settings = None
        if cfg.has_content_settings():
            from azure.storage.blob import ContentSettings

            content_settings = ContentSettings(
                content_type=cfg.content_type,
                content_disposition=cfg.content_disposition,
                cache_control=cfg.cache_control,
                content_encoding=cfg.content_encoding,
            )
        # Match adlfs: custom metadata replaces the default directory marker.
        metadata = dict(cfg.metadata) if cfg.metadata else {"is_directory": "false"}
        version_id = sync(
            get_loop(),
            self._upload_blob,
            container,
            blob,
            payload,
            content_settings,
            metadata,
            overwrite,
        )
        self.fs.invalidate_cache(full_path)
        return version_id

    async def _upload_blob(
        self, container, blob, data, content_settings, metadata, overwrite
    ) -> str | None:
        async with self.fs.service_client.get_blob_client(container, blob) as bc:
            try:
                resp = await bc.upload_blob(
                    data,
                    overwrite=overwrite,
                    content_settings=content_settings,
                    metadata=metadata,
                    max_concurrency=self.fs.max_concurrency,
                )
            except ResourceExistsError as e:  # exclusive ("x") write to existing blob
                raise FileExistsError(blob) from e
        return resp.get("version_id")

    @contextmanager
    def open_for_write(
        self, full_path: str, fs_mode: str, cfg: "WriteConfig", binary_kwargs: dict
    ) -> Iterator[Any]:
        if cfg.is_empty():
            # No write metadata: use adlfs's native streaming, which bounds
            # memory and handles append/exclusive/update modes correctly.
            with self.fs.open(full_path, fs_mode, **binary_kwargs) as handle:
                yield handle
            return
        # Content settings/metadata: adlfs can't set them on the streaming handle,
        # so buffer (spilling to disk) and write the whole object via upload_blob.
        # Only whole-object writes are supported this way.
        if "a" in fs_mode or "+" in fs_mode:
            raise NotImplementedError(
                "write metadata is not supported for Azure append/update writes"
            )
        with _AzureWriteBuffer(max_size=_SPOOL_MAX_SIZE) as buf:
            yield buf
            buf.seek(0)
            buf.version_id = self._upload_payload(
                full_path, buf, cfg, overwrite="x" not in fs_mode
            )

    def url(
        self,
        path: str,
        expires: int = 3600,
        version_id: str | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a signed URL for the given path.
        """
        content_disposition = kwargs.pop("content_disposition", None)
        full_path = self.get_uri(path)
        if version_id:
            # adlfs.split_path() reads version_id from the urlpath.
            full_path = f"{full_path}?versionid={version_id}"

        result = self.fs.sign(
            full_path,
            expiration=expires,
            content_disposition=content_disposition,
            **kwargs,
        )

        if version_id:
            # The Azure SDK does not embed versionid in the SAS token, so we
            # append it explicitly to route the request to the correct version.
            result += f"&versionid={version_id}"
        return result

    async def get_file(
        self,
        lpath: str,
        rpath: str,
        callback,
        version_id: str | None = None,
    ) -> None:
        if version_id:
            # adlfs._get_file() only reads version_id from split_path(rpath);
            # it does not accept version_id as a kwarg.  Embed it in the path
            # so split_path can recover it on the adlfs side.
            lpath = f"{lpath}?versionid={version_id}"
        await self.fs._get_file(lpath, rpath, callback=callback)

    async def _fetch_flat(self, start_prefix: str, result_queue: ResultQueue) -> None:
        prefix = start_prefix
        if prefix:
            prefix = prefix.lstrip(DELIMITER) + DELIMITER
        found = False
        try:
            with tqdm(desc=f"Listing {self.uri}", unit=" objects", leave=False) as pbar:
                async with self.fs.service_client.get_container_client(
                    container=self.name
                ) as container_client:
                    async for page in container_client.list_blobs(
                        include=["metadata", "versions"], name_starts_with=prefix
                    ).by_page():
                        entries = []
                        async for b in page:
                            found = True
                            if not self._is_valid_key(b["name"]):
                                continue
                            info = (await self.fs._details([b]))[0]
                            entries.append(
                                self.info_to_file(info, self.rel_path(info["name"]))
                            )
                        if entries:
                            await result_queue.put(entries)
                            pbar.update(len(entries))
                    if not found and prefix:
                        raise FileNotFoundError(
                            f"Unable to resolve remote path: {prefix}"
                        )
        finally:
            result_queue.put_nowait(None)

    _fetch_default = _fetch_flat
