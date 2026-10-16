import asyncio
import os
from typing import TYPE_CHECKING, Any, cast

from botocore.exceptions import ClientError, NoCredentialsError
from fsspec.asyn import get_loop, sync
from s3fs import S3FileSystem
from s3fs.core import _error_wrapper

from datachain.lib.file import File
from datachain.progress import tqdm

from .fsspec import DELIMITER, BucketStatus, Client, ResultQueue

if TYPE_CHECKING:
    from datachain.cache import Cache
    from datachain.client.writeconfig import WriteConfig

UPDATE_CHUNKSIZE = 1000
_BUCKET_REGIONS: dict[tuple, str] = {}


def _region_key(name: str, kwargs: dict[str, Any]) -> tuple | None:
    """The cache key of a bucket's region: the same name under another profile,
    region or endpoint may be another bucket. None for custom endpoints and
    sessions."""
    client_kwargs = kwargs.get("client_kwargs") or {}
    config_kwargs = kwargs.get("config_kwargs") or {}
    if (
        "session" in kwargs
        or kwargs.get("endpoint_url")
        or kwargs.get("aws_endpoint_url")
        or client_kwargs.get("endpoint_url")
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT_URL_S3")
    ):
        return None
    profile = (
        kwargs.get("profile")
        or os.environ.get("AWS_DEFAULT_PROFILE")
        or os.environ.get("AWS_PROFILE")
    )
    region = (
        kwargs.get("region_name")
        or client_kwargs.get("region_name")
        or config_kwargs.get("region_name")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    return name, profile, region


async def _head_bucket(fs: S3FileSystem, name: str) -> tuple[int, str | None]:
    await fs.set_session()
    s3 = await fs.get_s3()
    params = fs._get_s3_method_kwargs(s3.head_bucket, Bucket=name)
    try:  # retried as s3fs retries its own requests
        out = await _error_wrapper(s3.head_bucket, kwargs=params, retries=fs.retries)
        meta = out["ResponseMetadata"]
    except OSError as e:
        if not isinstance(e.__cause__, ClientError):
            raise
        meta = e.__cause__.response["ResponseMetadata"]
        if meta["HTTPStatusCode"] not in (301, 403, 404):
            raise
    return meta["HTTPStatusCode"], meta.get("HTTPHeaders", {}).get(
        "x-amz-bucket-region"
    )


class ClientS3(Client):
    FS_CLASS = S3FileSystem
    PREFIX = "s3://"
    protocol = "s3"
    CREDENTIAL_KEYS = frozenset(
        {"key", "secret", "token", "aws_key", "aws_secret", "aws_token"}
    )

    @staticmethod
    def _normalize_s3_kwargs(kwargs: dict) -> dict:
        if "aws_endpoint_url" in kwargs:
            kwargs.setdefault("client_kwargs", {}).setdefault(
                "endpoint_url", kwargs.pop("aws_endpoint_url")
            )
        if "aws_key" in kwargs:
            kwargs.setdefault("key", kwargs.pop("aws_key"))
        if "aws_secret" in kwargs:
            kwargs.setdefault("secret", kwargs.pop("aws_secret"))
        if "aws_token" in kwargs:
            kwargs.setdefault("token", kwargs.pop("aws_token"))
        return kwargs

    @classmethod
    def create_fs(cls, **kwargs) -> S3FileSystem:
        kwargs = cls._normalize_s3_kwargs(kwargs)

        # We want to use newer v4 signature version since regions added after
        # 2014 are not going to support v2 which is the older one.
        # All regions support v4.
        kwargs["config_kwargs"] = {"signature_version": "s3v4"} | (
            kwargs.get("config_kwargs") or {}
        )

        if "region_name" in kwargs:
            kwargs["config_kwargs"].setdefault("region_name", kwargs.pop("region_name"))

        # remove this `if` when https://github.com/fsspec/s3fs/pull/929 lands
        if (
            not os.environ.get("AWS_REGION")
            and not os.environ.get("AWS_ENDPOINT_URL")
            and "region_name" not in kwargs["config_kwargs"]
        ):
            # caching bucket regions to use the right one in signed urls, otherwise
            # it tries to randomly guess and creates wrong signature
            kwargs.setdefault("cache_regions", True)

        if not kwargs.get("anon"):
            try:
                # Run an inexpensive check to see if credentials are available
                super().create_fs(**kwargs).sign("s3://bucket/object")
            except NoCredentialsError:
                kwargs["anon"] = True
            except NotImplementedError:
                pass

        return cast("S3FileSystem", super().create_fs(**kwargs))

    @classmethod
    def bucket_status(cls, name: str, **kwargs) -> BucketStatus:
        not_found = BucketStatus(
            exists=False, access="denied", error=f"S3 bucket '{name}' not found"
        )
        strip = cls.CREDENTIAL_KEYS | {"anon"}
        anon_kwargs = cls._normalize_s3_kwargs(
            {k: v for k, v in kwargs.items() if k not in strip}
        )
        anon_kwargs["anon"] = True
        status, region = sync(
            get_loop(), _head_bucket, S3FileSystem(**anon_kwargs), name
        )
        if region and (key := _region_key(name, kwargs)):
            _BUCKET_REGIONS[key] = region
        if status == 200:
            return BucketStatus(exists=True, access="anonymous")
        if status in (301, 404):
            return not_found

        # Use raw S3FileSystem to bypass ClientS3.create_fs()'s auto-anon fallback.
        auth_kwargs = cls._normalize_s3_kwargs(
            {k: v for k, v in kwargs.items() if k != "anon"}
        )
        if region:
            auth_kwargs["config_kwargs"] = {"region_name": region} | (
                auth_kwargs.get("config_kwargs") or {}
            )
        try:
            status, _ = sync(
                get_loop(), _head_bucket, S3FileSystem(**auth_kwargs), name
            )
        except NoCredentialsError:
            status = 403
        if status == 200:
            return BucketStatus(exists=True, access="authenticated")
        if status in (301, 404):
            return not_found
        return BucketStatus(
            exists=True,
            access="denied",
            error=f"Access denied to S3 bucket '{name}'"
            " — check AWS credentials/permissions",
        )

    @classmethod
    def from_name(cls, name: str, cache: "Cache", kwargs: dict[str, Any]) -> "Client":
        key = _region_key(name, kwargs)
        region = _BUCKET_REGIONS.get(key) if key else None
        if region and "region_name" not in kwargs:
            kwargs = {**kwargs, "region_name": region}
        return cls(name, kwargs, cache)

    @staticmethod
    def _write_kwargs(cfg: "WriteConfig", *, streaming: bool) -> dict[str, Any]:
        # Both pipe_file and fs.open forward extra kwargs into
        # s3_additional_kwargs, which become boto3 put_object / multipart args.
        kw: dict[str, Any] = {}
        if cfg.content_type:
            kw["ContentType"] = cfg.content_type
        if cfg.content_disposition:
            kw["ContentDisposition"] = cfg.content_disposition
        if cfg.cache_control:
            kw["CacheControl"] = cfg.cache_control
        if cfg.content_encoding:
            kw["ContentEncoding"] = cfg.content_encoding
        if cfg.metadata:
            kw["Metadata"] = dict(cfg.metadata)
        kw.update(cfg.write_options or {})
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
        """
        content_disposition = kwargs.pop("content_disposition", None)
        if content_disposition:
            kwargs["ResponseContentDisposition"] = content_disposition

        if version_id:
            # botocore expects VersionId for GetObject presign
            kwargs["VersionId"] = version_id

        return self.fs.sign(self.get_uri(path), expiration=expires, **kwargs)

    async def _fetch_flat(self, start_prefix: str, result_queue: ResultQueue) -> None:
        async def get_pages(it, page_queue):
            try:
                async for page in it:
                    await page_queue.put(page.get(contents_key, []))
            finally:
                await page_queue.put(None)

        async def process_pages(page_queue, result_queue, prefix):
            found = False
            with tqdm(desc=f"Listing {self.uri}", unit=" objects", leave=False) as pbar:
                while (res := await page_queue.get()) is not None:
                    if res:
                        found = True
                    entries = [
                        self._entry_from_boto(d, self.name, versions)
                        for d in res
                        if self._is_valid_key(d["Key"])
                    ]
                    if entries:
                        await result_queue.put(entries)
                        pbar.update(len(entries))
            if not found and prefix:
                raise FileNotFoundError(f"Unable to resolve remote path: {prefix}")

        try:
            prefix = start_prefix
            if prefix:
                prefix = prefix.lstrip(DELIMITER) + DELIMITER
            versions = self._is_version_aware()
            fs = self.fs
            await fs.set_session()
            s3 = await fs.get_s3(self.name)
            if versions:
                method = "list_object_versions"
                contents_key = "Versions"
            else:
                method = "list_objects_v2"
                contents_key = "Contents"
            pag = s3.get_paginator(method)
            it = pag.paginate(
                Bucket=self.name,
                Prefix=prefix,
                Delimiter="",
            )
            page_queue: asyncio.Queue[list] = asyncio.Queue(2)
            consumer = asyncio.create_task(
                process_pages(page_queue, result_queue, prefix)
            )
            try:
                await get_pages(it, page_queue)
                await consumer
            finally:
                consumer.cancel()  # In case get_pages() raised
        finally:
            result_queue.put_nowait(None)

    async def _fetch_default(
        self, start_prefix: str, result_queue: ResultQueue
    ) -> None:
        await self._fetch_flat(start_prefix, result_queue)

    def _entry_from_boto(self, v, bucket, versions=False) -> File:
        return File(
            source=self.uri,
            path=v["Key"],
            etag=v.get("ETag", "").strip('"'),
            version=(
                ClientS3.clean_s3_version(v.get("VersionId", "")) if versions else ""
            ),
            is_latest=v.get("IsLatest", True),
            last_modified=v.get("LastModified", ""),
            size=v["Size"],
        )

    async def _fetch_dir(
        self,
        prefix,
        pbar,
        result_queue: ResultQueue,
    ) -> set[str]:
        if prefix:
            prefix = prefix.lstrip(DELIMITER) + DELIMITER
        files = []
        subdirs = set()
        found = False
        async for info in self.fs._iterdir(self.name, prefix=prefix, versions=True):
            full_path = info["name"]
            _, subprefix, _ = self.fs.split_path(full_path)
            if prefix.strip(DELIMITER) == subprefix.strip(DELIMITER):
                found = True
                continue
            if info["type"] == "directory":
                subdirs.add(subprefix)
            else:
                files.append(self.info_to_file(info, subprefix))
                pbar.update()
            found = True
        if not found:
            raise FileNotFoundError(f"Unable to resolve remote path: {prefix}")
        if files:
            await result_queue.put(files)
        pbar.update(len(subdirs))
        return subdirs

    @staticmethod
    def clean_s3_version(ver: str | None) -> str:
        return ver if (ver is not None and ver != "null") else ""

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        return File(
            source=self.uri,
            path=path,
            size=v["size"],
            version=(
                ClientS3.clean_s3_version(v.get("VersionId", ""))
                if self._is_version_aware()
                else ""
            ),
            etag=v.get("ETag", "").strip('"'),
            is_latest=v.get("IsLatest", True),
            last_modified=v.get("LastModified", ""),
        )
