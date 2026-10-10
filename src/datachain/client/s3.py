import functools
import os
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from xml.etree.ElementTree import Element, fromstring, tostring

from aiobotocore.parsers import AioResponseParserFactory, AioRestXMLParser
from aiobotocore.session import AioSession
from botocore.exceptions import NoCredentialsError
from botocore.model import Shape
from botocore.utils import parse_timestamp
from fsspec.asyn import get_loop, sync
from s3fs import S3FileSystem

from datachain.lib.file import File

from .fsspec import DELIMITER, BucketStatus, Client, Page, ResultQueue, iso_timestamp

if TYPE_CHECKING:
    from datachain.client.writeconfig import WriteConfig


def _parse_timestamp(value: Any) -> datetime:
    """botocore's timestamp parser, fast for ISO 8601."""
    parsed = iso_timestamp(value) if isinstance(value, str) else None
    return parsed or parse_timestamp(value)


_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_LISTINGS = ("ListObjectsV2Output", "ListObjectVersionsOutput")


_SCALARS: dict[str, Callable[[str], Any]] = {
    "integer": int,
    "long": int,
    "boolean": lambda text: text == "true",
    "timestamp": _parse_timestamp,
}
Fields = dict[str, tuple[str, Callable[[Element], Any], bool]]


@functools.cache
def _fields(shape: Shape) -> Fields:
    """Readers of a structure's child elements by XML tag."""
    fields = {}
    for name, member in shape.members.items():
        repeated = member.type_name == "list"
        fields[member.serialization.get("name", name)] = (
            name,
            _reader(member.member if repeated else member),
            repeated,
        )
    return fields


def _reader(shape: Shape) -> Callable[[Element], Any]:
    if shape.type_name == "structure":
        return lambda node: _read(node, _fields(shape))
    convert = _SCALARS.get(shape.type_name, str)
    return lambda node: convert(node.text or "")


def _read(node: Element, fields: Fields) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for child in node:
        if field := fields.get(child.tag.removeprefix(_NS)):
            key, read, repeated = field
            if repeated:
                out.setdefault(key, []).append(read(child))
            else:
                out[key] = read(child)
    return out


class _ListingParser(AioRestXMLParser):
    """Reads object listing entries straight from the XML, skipping botocore's
    model walk."""

    async def parse(self, response, shape):
        if (
            response["status_code"] != 200
            or getattr(shape, "name", "") not in _LISTINGS
        ):
            return await super().parse(response, shape)
        fields = {t: f for t, f in _fields(shape).items() if f[2]}
        root = fromstring(response["body"])  # noqa: S314
        entries = _read(root, fields)
        for node in [n for n in root if n.tag.removeprefix(_NS) in fields]:
            root.remove(node)
        parsed = await super().parse({**response, "body": tostring(root)}, shape)
        return parsed | entries


class _ParserFactory(AioResponseParserFactory):
    def create_parser(self, protocol_name):
        if protocol_name == "rest-xml":
            return _ListingParser(**self._defaults)
        return super().create_parser(protocol_name)


@functools.cache
def _session(profile: str | None) -> AioSession:
    """One per profile, so fsspec reuses filesystem instances."""
    session = AioSession(profile=profile)
    factory = _ParserFactory()
    factory.set_parser_defaults(timestamp_parser=_parse_timestamp)
    session.register_component("response_parser_factory", factory)
    return session


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
        kwargs.setdefault("config_kwargs", {}).setdefault("signature_version", "s3v4")

        if "region_name" in kwargs:
            kwargs["config_kwargs"].setdefault("region_name", kwargs.pop("region_name"))

        if "session" not in kwargs:
            kwargs["session"] = _session(kwargs.pop("profile", None))

        # remove this `if` when https://github.com/fsspec/s3fs/pull/929 lands
        if not os.environ.get("AWS_REGION") and not os.environ.get("AWS_ENDPOINT_URL"):
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
        # Step 1: Anonymous probe. cache_regions=True handles PermanentRedirect
        # (bucket-in-wrong-region) transparently so the bucket is found → exists=True.
        # Preserve endpoint/region settings from caller kwargs; strip credentials.
        strip = cls.CREDENTIAL_KEYS | {"anon"}
        anon_kwargs = cls._normalize_s3_kwargs(
            {k: v for k, v in kwargs.items() if k not in strip}
        )
        anon_kwargs["anon"] = True
        anon_kwargs.setdefault("cache_regions", True)
        anon_fs = S3FileSystem(**anon_kwargs)
        try:
            sync(get_loop(), anon_fs._info, name)
            return BucketStatus(exists=True, access="anonymous")
        except PermissionError:
            pass
        except FileNotFoundError:
            return BucketStatus(
                exists=False, access="denied", error=f"S3 bucket '{name}' not found"
            )

        # Step 2: Authenticated probe.
        # Use raw S3FileSystem to bypass ClientS3.create_fs()'s auto-anon fallback.
        auth_kwargs = cls._normalize_s3_kwargs(
            {k: v for k, v in kwargs.items() if k != "anon"}
        )
        auth_kwargs.setdefault("cache_regions", True)

        auth_fs = S3FileSystem(**auth_kwargs)
        try:
            sync(get_loop(), auth_fs._info, name)
            return BucketStatus(exists=True, access="authenticated")
        except (NoCredentialsError, PermissionError):
            return BucketStatus(
                exists=True,
                access="denied",
                error=f"Access denied to S3 bucket '{name}'"
                " — check AWS credentials/permissions",
            )
        except FileNotFoundError:
            return BucketStatus(
                exists=False, access="denied", error=f"S3 bucket '{name}' not found"
            )

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

    _fetch_default = Client._fetch_ranges

    async def _pages_after(
        self, prefix: str, start_after: str
    ) -> AsyncGenerator[Page, None]:
        versions = self._is_version_aware()
        method, key, start = (
            ("list_object_versions", "Versions", "KeyMarker")
            if versions
            else ("list_objects_v2", "Contents", "StartAfter")
        )
        await self.fs.set_session()
        s3 = await self.fs.get_s3(self.name)
        kwargs = {start: start_after} if start_after else {}
        async for page in s3.get_paginator(method).paginate(
            Bucket=self.name,
            Prefix=prefix,
            PaginationConfig={"PageSize": self.LIST_PAGE_SIZE},
            **kwargs,
        ):
            entries = page.get(key, [])
            keys = [d["Key"] for d in entries + page.get("DeleteMarkers", [])]
            yield (
                [
                    self._entry_from_boto(d, self.name, versions)
                    for d in entries
                    if self._is_valid_key(d["Key"])
                ],
                max(keys, default=None),
            )

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
