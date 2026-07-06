"""Bridge until adlfs forwards content settings on write.

adlfs only sends ``metadata`` when committing a blob, so content settings
(content type/disposition, cache control, ...) cannot be set inline on write —
unlike s3fs/gcsfs. This teaches ``AzureBlobFile`` to accept ``content_settings``
and pass it to the commit, mirroring https://github.com/fsspec/adlfs/pull/554.

Remove this module once that lands in a released adlfs and the floor is bumped.
"""

import asyncio
import inspect

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobBlock, BlobType

_state: dict = {}


def _patched_init(self, *args, content_settings=None, **kwargs):
    _state["orig_init"](self, *args, **kwargs)
    self.content_settings = content_settings


async def _patched_initiate_upload(self, **kwargs):
    self._block_list = []
    if self.mode == "ab" and not await self.fs._exists(self.path):
        async with self.container_client.get_blob_client(blob=self.blob) as bc:
            await bc.create_append_blob(
                metadata=self.metadata, content_settings=self.content_settings
            )


async def _patched_upload_chunk(self, final: bool = False, **kwargs):
    data = self.buffer.getvalue()
    length = len(data)
    block_id = self._get_block_id()
    commit_kw = {}
    if self.mode == "xb":
        commit_kw["headers"] = {"If-None-Match": "*"}
    if self.mode in {"wb", "xb"}:
        try:
            max_concurrency = self.fs.max_concurrency or 1
            semaphore = asyncio.Semaphore(max_concurrency)
            tasks = []
            for start, end in self._get_chunks(data):
                tasks.append(self._stage_block(data, start, end, block_id, semaphore))
                block_id = self._get_block_id()
            ids = await asyncio.gather(*tasks)
            self._block_list.extend(ids)

            if final:
                block_list = [BlobBlock(_id) for _id in self._block_list]
                async with self.container_client.get_blob_client(blob=self.blob) as bc:
                    response = await bc.commit_block_list(
                        block_list=block_list,
                        metadata=self.metadata,
                        content_settings=self.content_settings,
                        **commit_kw,
                    )
                    if self.fs.version_aware:
                        self.version_id = response.get("version_id")
        except ResourceExistsError as e:
            raise FileExistsError(self.path) from e
        except Exception as e:
            if not self._block_list and length == 0 and final:
                async with self.container_client.get_blob_client(blob=self.blob) as bc:
                    response = await bc.upload_blob(
                        data=data,
                        metadata=self.metadata,
                        content_settings=self.content_settings,
                        overwrite=(self.mode == "wb"),
                    )
                    if self.fs.version_aware:
                        self.version_id = response.get("version_id")
            elif length == 0 and final:
                block_list = [BlobBlock(_id) for _id in self._block_list]
                async with self.container_client.get_blob_client(blob=self.blob) as bc:
                    try:
                        response = await bc.commit_block_list(
                            block_list=block_list,
                            metadata=self.metadata,
                            content_settings=self.content_settings,
                            **commit_kw,
                        )
                        if self.fs.version_aware:
                            self.version_id = response.get("version_id")
                    except ResourceExistsError:
                        raise FileExistsError(self.path) from None
            else:
                raise RuntimeError(f"Failed to upload block: {e}!") from e
    elif self.mode == "ab":
        async with self.container_client.get_blob_client(blob=self.blob) as bc:
            await bc.upload_blob(
                data=data,
                length=length,
                blob_type=BlobType.AppendBlob,
                metadata=self.metadata,
            )
    else:
        raise ValueError(
            "File operation modes other than wb, xb or ab are not supported "
            "for upload_chunk"
        )


def apply() -> None:
    """Patch adlfs (idempotent, no-op once adlfs supports it natively)."""
    from adlfs.spec import AzureBlobFile
    from fsspec.asyn import sync_wrapper

    if "content_settings" in inspect.signature(AzureBlobFile.__init__).parameters:
        return

    _state["orig_init"] = AzureBlobFile.__init__
    AzureBlobFile.__init__ = _patched_init
    AzureBlobFile._async_initiate_upload = _patched_initiate_upload
    AzureBlobFile._initiate_upload = sync_wrapper(_patched_initiate_upload)
    AzureBlobFile._async_upload_chunk = _patched_upload_chunk
    AzureBlobFile._upload_chunk = sync_wrapper(_patched_upload_chunk)
