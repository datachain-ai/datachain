import os
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast
from xml.etree.ElementTree import Element, ParseError, fromstring

from aiobotocore.parsers import AioResponseParserFactory, AioRestXMLParser
from aiobotocore.session import AioSession
from botocore import handlers
from botocore.configloader import load_config
from botocore.exceptions import ClientError, ConfigNotFound, NoCredentialsError
from botocore.model import Shape
from botocore.utils import parse_timestamp
from fsspec.asyn import get_loop, sync, sync_wrapper
from s3fs import S3FileSystem

from datachain.lib.file import File, FileRow

from .fsspec import BucketStatus, RangeListedClient
from .ranges import PROCESS_LIST_WORKERS, Keep, Page

if TYPE_CHECKING:
    from datachain.cache import Cache
    from datachain.client.writeconfig import WriteConfig

_BUCKET_REGIONS: dict[tuple, str] = {}


@lru_cache(maxsize=16)
def _shared_config(path: str, _stamp: tuple[int, int, int]) -> dict[str, Any]:
    """Reuse parsed AWS config until its file changes."""
    try:
        return load_config(path)
    except ConfigNotFound:
        return {}


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
    path = os.path.expanduser(os.environ.get("AWS_CONFIG_FILE", "~/.aws/config"))
    try:
        stat = os.stat(path)
    except OSError:
        config = {}
    else:
        config = _shared_config(
            path, (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        )
    scoped = config.get("profiles", {}).get(profile or "default", {})
    services = config.get("services", {}).get(scoped.get("services"), {})
    if scoped.get("endpoint_url") or services.get("s3", {}).get("endpoint_url"):
        return None
    region = (
        kwargs.get("region_name")
        or client_kwargs.get("region_name")
        or config_kwargs.get("region_name")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    return name, profile, region


def _head_bucket(fs: S3FileSystem, name: str) -> tuple[int, str | None]:
    try:
        # An asynchronous filesystem has no fs.loop for its sync wrappers.
        meta = sync(get_loop(), fs._call_s3, "head_bucket", Bucket=name)[
            "ResponseMetadata"
        ]
    except OSError as e:
        if not isinstance(e.__cause__, ClientError):
            raise
        meta = e.__cause__.response["ResponseMetadata"]
        status = meta["HTTPStatusCode"]
        # R2 refuses anonymous requests with 400; other service errors must
        # keep their original exception, including authenticated 400 and 429.
        if status not in (301, 401, 403, 404) and not (status == 400 and fs.anon):
            raise
    headers = meta.get("HTTPHeaders", {})
    return meta["HTTPStatusCode"], headers.get("x-amz-bucket-region")


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
    # A version id always names the same bytes, except "null". S3 gives that id to
    # objects written while versioning is off or suspended, and another write in
    # that state replaces them, so those files are checked by etag.
    PINNED_VERSION_IS_IMMUTABLE = True
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
        if "region_name" in kwargs:
            kwargs["config_kwargs"] = {"region_name": kwargs.pop("region_name")} | (
                kwargs.get("config_kwargs") or {}
            )
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
        kwargs["config_kwargs"].setdefault("max_pool_connections", PROCESS_LIST_WORKERS)

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
        # HeadBucket returns the region; do not discover it with another probe.
        kwargs["cache_regions"] = False
        not_found = BucketStatus(
            exists=False, access="denied", error=f"S3 bucket '{name}' not found"
        )
        strip = cls.CREDENTIAL_KEYS | {"anon"}
        anon_kwargs = cls._normalize_s3_kwargs(
            {k: v for k, v in kwargs.items() if k not in strip}
        )
        anon_kwargs["anon"] = True
        status, region = _head_bucket(S3FileSystem(**anon_kwargs), name)
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
            status, auth_region = _head_bucket(S3FileSystem(**auth_kwargs), name)
            if auth_region and (key := _region_key(name, kwargs)):
                _BUCKET_REGIONS[key] = auth_region
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
    def from_name(cls, name: str, cache: "Cache", kwargs: dict[str, Any]) -> "ClientS3":
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
