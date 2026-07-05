from typing import TYPE_CHECKING, Any

from adlfs import AzureBlobFileSystem
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
)
from azure.storage.blob import BlobServiceClient
from fsspec.asyn import get_loop, sync

from datachain.lib.file import File
from datachain.progress import tqdm

from .fsspec import DELIMITER, BucketStatus, Client, ResultQueue

if TYPE_CHECKING:
    from datachain.client.writeconfig import WriteConfig


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

    def _can_pipe_upload(self) -> bool:
        # adlfs.pipe_file hardcodes the blob metadata and cannot set content
        # settings, so always stream: metadata is passed inline and content
        # settings are applied by a post-write header update.
        return False

    def _write_kwargs(self, cfg: "WriteConfig", *, streaming: bool) -> dict[str, Any]:
        # adlfs does not apply arbitrary write kwargs (they are stored on the
        # file object and never sent), so reject write_options rather than
        # silently drop it. Content settings are applied in _finalize_write.
        if cfg.extra:
            raise NotImplementedError(
                "write_options is not supported on Azure; use content_type, "
                "content_disposition, cache_control, content_encoding or "
                "metadata instead."
            )
        kw: dict[str, Any] = {}
        if cfg.metadata:
            # adlfs defaults blob metadata to {"is_directory": "false"};
            # keep it so directory detection during listing still works.
            kw["metadata"] = {"is_directory": "false", **dict(cfg.metadata)}
        return kw

    def _finalize_write(
        self, cfg: "WriteConfig", full_path: str, *, streaming: bool
    ) -> str | None:
        if not cfg.has_content_settings():
            return None
        from azure.storage.blob import ContentSettings

        settings = ContentSettings(
            content_type=cfg.content_type,
            content_disposition=cfg.content_disposition,
            cache_control=cfg.cache_control,
            content_encoding=cfg.content_encoding,
        )
        container, blob, _ = self.fs.split_path(full_path)
        # On a versioned account, set_http_headers creates a new current
        # version carrying the content settings; return it so the write path
        # refreshes to that version instead of the pre-settings one.
        version_id = sync(
            get_loop(), self._set_content_settings, container, blob, settings
        )
        self.fs.invalidate_cache(full_path)
        return version_id

    async def _set_content_settings(self, container, blob, settings) -> str | None:
        async with self.fs.service_client.get_blob_client(container, blob) as bc:
            resp = await bc.set_http_headers(content_settings=settings)
        return resp.get("version_id")

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
