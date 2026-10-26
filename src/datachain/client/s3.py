import errno
import os
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from xml.etree.ElementTree import Element, ParseError, fromstring

from aiobotocore.parsers import AioResponseParserFactory, AioRestXMLParser
from aiobotocore.session import AioSession
from botocore import handlers
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.model import Shape
from botocore.utils import parse_timestamp
from fsspec.asyn import get_loop, sync, sync_wrapper
from s3fs import S3FileSystem

from datachain.lib.file import File, FileRow

from .fsspec import BucketStatus, RangeListedClient
from .ranges import PROCESS_LIST_WORKERS, Keep, Page

if TYPE_CHECKING:
    from datachain.client.writeconfig import WriteConfig


def _parse_timestamp(value: Any) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):  # header dates and epoch seconds
        return parse_timestamp(value)


_LISTINGS = ("ListObjectsV2Output", "ListObjectVersionsOutput")
_SCALARS: dict[str, Callable[[str], Any]] = {
    "integer": int,
    "long": int,
    "boolean": lambda text: text == "true",
    "timestamp": _parse_timestamp,
}
Fields = dict[str, tuple[str, Callable[[Element], Any], bool]]
_FIELDS: dict[tuple[str, str], Fields] = {}  # every client builds its own shapes


def _fields(shape: Shape, ns: str) -> Fields:
    """Readers of a structure's child elements by tag, in namespace ``ns``."""
    if (shape.name, ns) not in _FIELDS:
        _FIELDS[shape.name, ns] = {
            ns + member.serialization.get("name", name): (
                name,
                _reader(member.member if member.type_name == "list" else member, ns),
                member.type_name == "list",
            )
            for name, member in shape.members.items()
        }
    return _FIELDS[shape.name, ns]


def _reader(shape: Shape, ns: str) -> Callable[[Element], Any]:
    if shape.type_name == "structure":
        return lambda node: _read(node, _fields(shape, ns))
    convert = _SCALARS.get(shape.type_name, str)
    return lambda node: convert(node.text or "")


def _read(node: Element, fields: Fields) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for child in node:
        if field := fields.get(child.tag):
            key, read, repeated = field
            if repeated:
                out.setdefault(key, []).append(read(child))
            else:
                out[key] = read(child)
    return out


class _ListingParser(AioRestXMLParser):
    """Reads listing pages straight from the XML, without botocore's model walk.
    A page that does not parse, or holds an error, is a 500 that botocore retries."""

    def _do_parse(self, response, shape):
        if getattr(shape, "name", "") not in _LISTINGS or not response["body"]:
            return super()._do_parse(response, shape)
        try:
            root = fromstring(response["body"])  # noqa: S314
            ns = root.tag[: root.tag.find("}") + 1]  # the page's namespace, or ""
        except ParseError:
            root, ns = None, ""
        if root is None or root.tag == f"{ns}Error":
            response["status_code"] = 500
            return self._do_error_parse(response, shape)
        parsed = {"ResponseMetadata": self._populate_response_metadata(response)}
        self._parse_non_payload_attrs(response, shape, shape.members, parsed)
        return parsed | _read(root, _fields(shape, ns))


class _ParserFactory(AioResponseParserFactory):
    def create_parser(self, protocol_name):
        if protocol_name == "rest-xml":
            return _ListingParser(**self._defaults)
        return super().create_parser(protocol_name)


def _handle_200_error(operation_model, response_dict, **kwargs) -> None:
    """botocore's 200-error check; a listing page is checked by its parser."""
    if getattr(operation_model.output_shape, "name", "") not in _LISTINGS:
        handlers._handle_200_error(operation_model, response_dict, **kwargs)


class _ListingSession(AioSession):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        factory = _ParserFactory()
        factory.set_parser_defaults(timestamp_parser=_parse_timestamp)
        self.register_component("response_parser_factory", factory)
        self.unregister("before-parse.s3.*", handlers._handle_200_error)
        self.register("before-parse.s3.*", _handle_200_error)


class _S3FileSystem(S3FileSystem):
    """Every session it makes parses listings with ``_ListingParser``."""

    async def set_session(self, refresh=False, kwargs={}):  # noqa: B006
        mine = isinstance(self.session, _ListingSession)
        if self.session is None or (refresh and mine):
            self.session = _ListingSession(**self.kwargs)
        self._session_is_owned = False  # s3fs keeps the session, even on refresh
        return await super().set_session(refresh, kwargs)

    _connect = set_session
    connect = sync_wrapper(set_session)


class ClientS3(RangeListedClient):
    FS_CLASS = _S3FileSystem
    PREFIX = "s3://"
    LISTS_LEVELS = True
    protocol = "s3"
    CREDENTIAL_KEYS = frozenset(
        {"key", "secret", "token", "aws_key", "aws_secret", "aws_token"}
    )
    _unversioned = False  # the server can't list versions

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
        # a connection for each listing request in flight
        kwargs["config_kwargs"].setdefault("max_pool_connections", PROCESS_LIST_WORKERS)

        if "region_name" in kwargs:
            kwargs["config_kwargs"].setdefault("region_name", kwargs.pop("region_name"))

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
        except FileNotFoundError:
            return BucketStatus(
                exists=False, access="denied", error=f"S3 bucket '{name}' not found"
            )
        except PermissionError:
            pass
        except OSError as e:
            # s3fs maps R2's anonymous HTTP 400 response to EINVAL.
            if e.errno != errno.EINVAL:
                raise

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

    def _lists_in_key_order(self) -> bool:
        # directory buckets and R2 don't list keys in code point order
        s3 = self._fs and self._fs._s3  # a listing has made it
        endpoint = s3.meta.endpoint_url if s3 else ""
        return not (self.name.endswith("--x-s3") or "r2.cloudflarestorage" in endpoint)

    async def _list_page(
        self,
        prefix: str,
        start_after: str,
        token: Any,
        hi: str | None = None,
        page_size: int | None = None,
        delimiter: str = "",
    ) -> Page:
        versions = self._is_version_aware() and not self._unversioned
        kwargs: dict[str, Any] = {"Delimiter": delimiter} if delimiter else {}
        if versions:
            if token is not None:
                kwargs["KeyMarker"], version_marker = token
                if version_marker:
                    kwargs["VersionIdMarker"] = version_marker
            elif start_after:
                kwargs["KeyMarker"] = start_after
        elif token is not None:
            kwargs["ContinuationToken"] = token
        elif start_after:
            kwargs["StartAfter"] = start_after
        await self.fs.set_session()
        s3 = await self.fs.get_s3(self.name)
        list_page = s3.list_object_versions if versions else s3.list_objects_v2
        try:
            page = await list_page(
                Bucket=self.name,
                Prefix=prefix,
                MaxKeys=page_size or self.LIST_PAGE_SIZE,
                **kwargs,
                **self.fs.req_kw,  # requester_pays
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if not versions or token is not None or code != "NotImplemented":
                raise
            # stores without version support, such as R2, hold only current objects
            self._unversioned = True
            return await self._list_page(
                prefix, start_after, None, hi, page_size, delimiter
            )
        entries = page.get("Versions" if versions else "Contents", [])
        keys = [d["Key"] for d in entries + page.get("DeleteMarkers", [])]
        # folders past a delimiter: listed past, though they make no rows
        keys += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        if not page.get("IsTruncated"):
            token = None
        elif versions:
            token = page["NextKeyMarker"], page.get("NextVersionIdMarker")
        else:
            token = page["NextContinuationToken"]
        keep = Keep(self, start_after, hi)
        return Page(
            [self._file_row(d, versions) for d in entries if keep(d["Key"])],
            entries[0]["Key"] if entries else None,
            max(keys, default=None),
            token,
            keep.past_lo,
            keep.past_hi,
            keep.in_order,
        )

    def _file_row(self, v, versions: bool) -> tuple:
        return FileRow(
            self.uri,
            v["Key"],
            v["Size"],
            ClientS3.clean_s3_version(v.get("VersionId", "")) if versions else "",
            v.get("ETag", "").strip('"'),
            v.get("IsLatest", True),
            v["LastModified"],
        )

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
