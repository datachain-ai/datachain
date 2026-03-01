from typing import Any

from adlfs import AzureBlobFileSystem
from tqdm.auto import tqdm

from datachain.lib.file import File

from .fsspec import DELIMITER, Client, ResultQueue


class AzureClient(Client):
    FS_CLASS = AzureBlobFileSystem
    PREFIX = "az://"
    protocol = "az"

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
