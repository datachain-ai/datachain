import asyncio
import json
import os
from collections.abc import Iterable
from datetime import datetime
from typing import Any, cast

from dateutil.parser import isoparse
from gcsfs import GCSFileSystem
from tqdm.auto import tqdm

from datachain.lib.file import File

from .fsspec import DELIMITER, Client, ResultQueue

# Patch gcsfs for consistency with s3fs
GCSFileSystem.set_session = GCSFileSystem._set_session
PageQueue = asyncio.Queue[Iterable[dict[str, Any]] | None]


class GCSClient(Client):
    FS_CLASS = GCSFileSystem
    PREFIX = "gs://"
    protocol = "gs"

    @classmethod
    def create_fs(cls, **kwargs) -> GCSFileSystem:
        if os.environ.get("DATACHAIN_GCP_CREDENTIALS"):
            kwargs["token"] = json.loads(os.environ["DATACHAIN_GCP_CREDENTIALS"])
        if kwargs.pop("anon", False):
            kwargs["token"] = "anon"  # noqa: S105

        return cast("GCSFileSystem", super().create_fs(**kwargs))

    def url(self, path: str, expires: int = 3600, **kwargs) -> str:
        """
        Generate a signed URL for the given path.
        If the client is anonymous, a public URL is returned instead
        (see https://cloud.google.com/storage/docs/access-public-data#api-link).
        """
        version_id = kwargs.pop("version_id", None)
        content_disposition = kwargs.pop("content_disposition", None)
        if self.fs.storage_options.get("token") == "anon":
            query = f"?generation={version_id}" if version_id else ""
            return f"https://storage.googleapis.com/{self.name}/{path}{query}"
        return self.fs.sign(
            self.get_full_path(path, version_id),
            expiration=expires,
            response_disposition=content_disposition,
            **kwargs,
        )

    @staticmethod
    def parse_timestamp(timestamp: str) -> datetime:
        """
        Parse timestamp string returned by GCSFileSystem.

        This ensures that the passed timestamp is timezone aware.
        """
        dt = isoparse(timestamp)
        assert dt.tzinfo is not None
        return dt

    async def _fetch_flat(self, start_prefix: str, result_queue: ResultQueue) -> None:
        prefix = start_prefix
        if prefix:
            prefix = prefix.lstrip(DELIMITER) + DELIMITER
        found = False
        try:
            page_queue: PageQueue = asyncio.Queue(2)
            consumer = asyncio.create_task(
                self._process_pages(page_queue, result_queue)
            )
            try:
                await self._get_pages(prefix, page_queue)
                found = await consumer
                if not found and prefix:
                    raise FileNotFoundError(f"Unable to resolve remote path: {prefix}")
            finally:
                consumer.cancel()  # In case _get_pages() raised
        finally:
            result_queue.put_nowait(None)

    _fetch_default = _fetch_flat

    async def _process_pages(
        self, page_queue: PageQueue, result_queue: ResultQueue
    ) -> bool:
        found = False
        with tqdm(desc=f"Listing {self.uri}", unit=" objects", leave=False) as pbar:
            while (page := await page_queue.get()) is not None:
                if page:
                    found = True
                entries = [
                    self._entry_from_dict(d)
                    for d in page
                    if self._is_valid_key(d["name"])
                ]
                if entries:
                    await result_queue.put(entries)
                    pbar.update(len(entries))
        return found

    async def _get_pages(self, path: str, page_queue: PageQueue) -> None:
        page_size = 5000
        try:
            next_page_token = None
            while True:
                page = await self.fs._call(
                    "GET",
                    "b/{}/o",
                    self.name,
                    delimiter="",
                    prefix=path,
                    maxResults=page_size,
                    pageToken=next_page_token,
                    json_out=True,
                    versions="true" if self._is_version_aware() else "false",
                )
                assert page["kind"] == "storage#objects"
                await page_queue.put(page.get("items", []))
                next_page_token = page.get("nextPageToken")
                if next_page_token is None:
                    break
        finally:
            await page_queue.put(None)

    def _entry_from_dict(self, d: dict[str, Any]) -> File:
        info = self.fs._process_object(self.name, d)
        return self.info_to_file(info, self.rel_path(info["name"]))

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

    @classmethod
    def version_path(cls, path: str, version_id: str | None) -> str:
        return f"{path}#{version_id}" if version_id else path
