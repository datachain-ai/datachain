import asyncio
import logging
import os
import re
import time
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from datetime import datetime
from time import monotonic
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote_plus, urlsplit
from xml.etree.ElementTree import Element, ParseError, fromstring

from aiobotocore.parsers import AioResponseParserFactory, AioRestXMLParser
from aiobotocore.session import AioSession
from botocore import handlers
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.model import Shape
from botocore.session import Session
from botocore.utils import parse_timestamp
from fsspec.asyn import sync_wrapper
from fsspec.utils import tokenize
from s3fs import S3FileSystem

from datachain.lib.file import File, FileRow, as_file_row

from .fsspec import BucketStatus, RangeListedClient
from .ranges import PROCESS_LIST_WORKERS, Keep, Page
from .s3_pages import (
    LeanS3,
    LeanSpec,
    clean_version,
    lean_verify,
    page_of,
    read_entries,
    split_page,
)

if TYPE_CHECKING:
    from datachain.client.writeconfig import WriteConfig

_CREDENTIAL_CHECKS: dict[tuple, tuple[float, bool]] = {}


def _has_explicit_credentials(kwargs: dict[str, Any]) -> bool:
    client_kwargs = kwargs.get("client_kwargs") or {}
    return any(
        kwargs.get(key) is not None for key in ClientS3.CREDENTIAL_KEYS - {"profile"}
    ) or any(
        client_kwargs.get(key) is not None
        for key in (
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
        )
    )


def _credential_key(kwargs: dict[str, Any]) -> tuple:
    env = frozenset(
        (k, v)
        for k, v in os.environ.items()
        if k.startswith("AWS_") or k == "BOTO_CONFIG"
    )
    files = []
    for path in (
        os.environ.get("AWS_CONFIG_FILE", "~/.aws/config"),
        os.environ.get("AWS_SHARED_CREDENTIALS_FILE", "~/.aws/credentials"),
        os.environ.get("BOTO_CONFIG", "~/.boto"),
    ):
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        try:
            stat = os.stat(path)
        except OSError:
            stamp = None
        else:
            stamp = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        files.append((path, stamp))
    return kwargs.get("profile"), env, tuple(files)


def _has_credentials(kwargs: dict[str, Any]) -> bool:
    return _has_explicit_credentials(kwargs) or _finds_credentials(
        *_credential_key(kwargs)
    )


def _finds_credentials(profile: str | None, env: frozenset, files: tuple) -> bool:
    key = profile, env, files
    if key in _CREDENTIAL_CHECKS:
        checked, found = _CREDENTIAL_CHECKS[key]
        if found or monotonic() - checked < 60:
            return found
    found = Session(profile=profile).get_credentials() is not None
    if len(_CREDENTIAL_CHECKS) >= 16:
        _CREDENTIAL_CHECKS.pop(next(iter(_CREDENTIAL_CHECKS), ()), None)
    _CREDENTIAL_CHECKS[key] = monotonic(), found
    return found


def _head_bucket(fs: S3FileSystem, name: str) -> tuple[int, str | None]:
    try:
        meta = fs.call_s3("head_bucket", Bucket=name)["ResponseMetadata"]
    except OSError as e:
        if not isinstance(e.__cause__, ClientError):
            raise
        meta = e.__cause__.response["ResponseMetadata"]
        status = meta["HTTPStatusCode"]
        # R2 refuses anonymous requests with 400; other service errors must
        # keep their original exception, including authenticated 400 and 429.
        if status == 301:
            raise e.__cause__ from None
        if status not in (401, 403, 404) and not (status == 400 and fs.anon):
            raise
    headers = meta.get("HTTPHeaders", {})
    return meta["HTTPStatusCode"], headers.get("x-amz-bucket-region")


logger = logging.getLogger("datachain")


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


def _read(node: Iterable[Element], fields: Fields) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for child in node:
        if field := fields.get(child.tag):
            key, read, repeated = field
            if repeated:
                out.setdefault(key, []).append(read(child))
            else:
                out[key] = read(child)
    return out


# While ClientS3._list_page waits for a page, the source its rows are listed from
_ROWS_OF: ContextVar[str | None] = ContextVar("_ROWS_OF", default=None)


def _rows_page(body: bytes, shape: Shape, source: str) -> dict[str, Any] | None:
    """The page as ``_list_page`` reads it, or ``None`` when ElementTree must
    read it. Its objects are ``FileRows`` with keys as written; delete markers
    and folders are dicts of their key alone."""
    versions = shape.name == "ListObjectVersionsOutput"
    if (split := split_page(body, versions)) is None:
        return None
    others, ns, entries = split
    try:
        rows, markers, folders = read_entries(
            entries, versions, source, as_file_row, _parse_timestamp
        )
    except ValueError:
        return None
    return _read(others, _fields(shape, ns)) | {
        "FileRows": rows,
        "DeleteMarkers": [{"Key": key} for key in markers],
        "CommonPrefixes": [{"Prefix": prefix} for prefix in folders],
    }


_PASSED_HEADERS = frozenset({"x-amz-request-payer", "x-amz-expected-bucket-owner"})
_SIGNING_HEADERS = frozenset(
    {
        "x-amz-date",
        "x-amz-content-sha256",
        "x-amz-security-token",
        "user-agent",
        "amz-sdk-invocation-id",
        "amz-sdk-request",
    }
)
_SCOPE = re.compile(r"AWS4-HMAC-SHA256 Credential=[^/]+/\d{8}/([^/]+)/s3/aws4_request,")


def _plain_request(
    sent: Any, config: dict
) -> tuple[str, dict[str, str], str | None] | None:
    """The bucket URL, the headers a lean child sends too, and the signing
    region (None: unsigned) of the last listing request in ``sent``; None
    unless it was a plain ``GET``, signed with SigV4 or not at all."""
    off = os.environ.get("DATACHAIN_LEAN_LISTING", "1") == "0"
    if not isinstance(sent, list) or not sent or off or config.get("proxies"):
        return None
    url, method, headers = sent[-1]
    # botocore holds header values as bytes
    headers = {
        k.lower(): v.decode() if isinstance(v, bytes) else v for k, v in headers.items()
    }
    auth = headers.pop("authorization", None)
    scope = _SCOPE.match(auth) if auth is not None else None
    parts = urlsplit(url)
    # a header no lean child sends, such as a directory bucket's session token
    unknown = headers.keys() - _PASSED_HEADERS - _SIGNING_HEADERS
    if (
        method != "GET"
        or unknown
        or (auth is not None and not scope)
        or parts.scheme not in ("http", "https")
    ):
        return None
    passed = {k: v for k, v in headers.items() if k in _PASSED_HEADERS}
    region = scope[1] if scope else None
    return f"{parts.scheme}://{parts.netloc}{parts.path}", passed, region


# While ClientS3._list_page waits for a page it learns from, the requests sent
_SENT: ContextVar[list | None] = ContextVar("_SENT", default=None)


def _remember(request, **kwargs) -> None:
    if (sent := _SENT.get()) is not None:
        sent.append((request.url, request.method, dict(request.headers)))


def _remember_requests(s3) -> list:
    """A list of the listing requests ``s3`` sends in this task from now on, as
    (url, method, headers): other listings on the same client leave it alone."""
    if not getattr(s3, "_datachain_remembers", False):
        for operation in ("ListObjectsV2", "ListObjectVersions"):
            s3.meta.events.register(f"before-send.s3.{operation}", _remember)
        s3._datachain_remembers = True
    sent: list = []
    _SENT.set(sent)
    return sent


def _decode_rows(parsed, context, **kwargs) -> None:
    """URL-decodes the keys of ``FileRows`` when botocore decodes the others:
    when it asked for URL encoding itself, and the page is URL encoded."""
    rows = parsed.get("FileRows")
    if (
        rows
        and parsed.get("EncodingType") == "url"
        and context.get("encoding_type_auto_set")
    ):
        parsed["FileRows"] = [
            row._replace(path=unquote_plus(row.path))
            if "%" in row.path or "+" in row.path
            else row
            for row in rows
        ]


class _ListingParser(AioRestXMLParser):
    """Reads listing pages straight from the XML, without botocore's model walk,
    and for ``ClientS3._list_page`` with patterns when it can (``_rows_page``).
    A page that does not parse, or holds an error, is a 500 that botocore retries."""

    def _do_parse(self, response, shape):
        if getattr(shape, "name", "") not in _LISTINGS or not response["body"]:
            return super()._do_parse(response, shape)
        source = _ROWS_OF.get()
        page = _rows_page(response["body"], shape, source) if source else None
        if page is None:
            try:
                root = fromstring(response["body"])  # noqa: S314
                ns = root.tag[: root.tag.find("}") + 1]  # the page's namespace, or ""
            except ParseError:
                root, ns = None, ""
            if root is None or root.tag == f"{ns}Error":
                response["status_code"] = 500
                return self._do_error_parse(response, shape)
            page = _read(root, _fields(shape, ns))
        parsed = {"ResponseMetadata": self._populate_response_metadata(response)}
        self._parse_non_payload_attrs(response, shape, shape.members, parsed)
        return parsed | page


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
        self.register("after-call.s3.ListObjectsV2", _decode_rows)
        self.register("after-call.s3.ListObjectVersions", _decode_rows)


class _S3FileSystem(S3FileSystem):
    """Every session it makes parses listings with ``_ListingParser``."""

    def __init__(self, *args, credential_token=None, **kwargs):
        # The token partitions fsspec's instance cache when provider inputs change.
        self._credential_retry_at = 0.0
        super().__init__(*args, **kwargs)

    async def set_session(self, refresh=False, kwargs={}):  # noqa: B006
        mine = isinstance(self.session, _ListingSession)
        auto_anon = self.storage_options.get("anon") is None
        retry_credentials = False
        if (
            auto_anon
            and self._s3 is not None
            and self.anon
            and mine
            and monotonic() >= self._credential_retry_at
        ):
            checked, found = _CREDENTIAL_CHECKS.get(
                _credential_key(self.storage_options), (0.0, False)
            )
            self._credential_retry_at = checked + 60
            retry_credentials = found or monotonic() >= self._credential_retry_at
        if (self._s3 is None or refresh or retry_credentials) and auto_anon:
            was_anonymous = self.anon
            if self.session is None or mine:
                if refresh:
                    _CREDENTIAL_CHECKS.pop(_credential_key(self.storage_options), None)
                self.anon = not await asyncio.to_thread(
                    _has_credentials, self.storage_options
                )
                if self.anon:
                    checked, _ = _CREDENTIAL_CHECKS.get(
                        _credential_key(self.storage_options), (monotonic(), False)
                    )
                    self._credential_retry_at = checked + 60
            elif not _has_explicit_credentials(self.storage_options):
                self.anon = await self.session.get_credentials() is None
            refresh = refresh or (self._s3 is not None and was_anonymous != self.anon)
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
    NAMES_FOLDERS = True
    protocol = "s3"
    CREDENTIAL_KEYS = frozenset(
        {
            "key",
            "secret",
            "token",
            "aws_key",
            "aws_secret",
            "aws_token",
            "username",
            "password",
        }
    )
    # A version id always names the same bytes, except "null". S3 gives that id to
    # objects written while versioning is off or suspended, and another write in
    # that state replaces them, so those files are checked by etag.
    PINNED_VERSION_IS_IMMUTABLE = True
    _unversioned = False  # the server can't list versions
    # how listing children list without botocore; False: they can't; None: unknown
    _lean: "LeanSpec | bool | None" = None

    @staticmethod
    def _normalize_s3_kwargs(kwargs: dict) -> dict:
        if "aws_endpoint_url" in kwargs:
            kwargs["client_kwargs"] = {
                "endpoint_url": kwargs.pop("aws_endpoint_url")
            } | (kwargs.get("client_kwargs") or {})
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

        if kwargs.get("anon") is None and not _has_explicit_credentials(kwargs):
            kwargs["credential_token"] = tokenize(_credential_key(kwargs))

        return cast("S3FileSystem", super().create_fs(**kwargs))

    @classmethod
    def bucket_status(cls, name: str, **kwargs) -> BucketStatus:
        # HeadBucket returns the region; do not discover it with another probe.
        kwargs["cache_regions"] = False
        kwargs["skip_instance_cache"] = True
        kwargs.pop("asynchronous", None)
        kwargs.pop("loop", None)
        not_found = BucketStatus(
            exists=False, access="denied", error=f"S3 bucket '{name}' not found"
        )
        strip = (cls.CREDENTIAL_KEYS - {"profile"}) | {"anon"}
        anon_kwargs = cls._normalize_s3_kwargs(
            {k: v for k, v in kwargs.items() if k not in strip}
        )
        anon_kwargs["anon"] = True
        status, region = _head_bucket(S3FileSystem(**anon_kwargs), name)
        if status == 200:
            return BucketStatus(exists=True, access="anonymous")
        if status == 404:
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
            status, _ = _head_bucket(S3FileSystem(**auth_kwargs), name)
        except NoCredentialsError:
            status = 403
        if status == 200:
            return BucketStatus(exists=True, access="authenticated")
        if status == 404:
            return not_found
        return BucketStatus(
            exists=True,
            access="denied",
            error=f"Access denied to S3 bucket '{name}'"
            " — check AWS credentials/permissions",
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
        learn = not delimiter and self._lean is None
        sent = _remember_requests(s3) if learn else None
        list_page = s3.list_object_versions if versions else s3.list_objects_v2
        rows_of = _ROWS_OF.set(self.uri)
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
            page = None
        finally:
            _ROWS_OF.reset(rows_of)
            _SENT.set(None)
        if page is None:
            # stores without version support, such as R2, hold only current objects
            self._unversioned = True
            return await self._list_page(
                prefix, start_after, None, hi, page_size, delimiter
            )
        if sent is not None:
            await self._learn_lean(s3, sent, versions)
        keep = Keep(self, start_after, hi)
        if (rows := page.get("FileRows")) is None:  # ElementTree read the page
            entries = page.get("Versions" if versions else "Contents", [])
            keys = [d["Key"] for d in entries]
            rows = [self._file_row(d, versions) for d in entries if keep(d["Key"])]
        else:
            keys = [row[1] for row in rows]
            rows = keep.take(rows, keys)
        markers = [d["Key"] for d in page.get("DeleteMarkers", [])]
        folders = [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        if not page.get("IsTruncated"):
            token = None
        elif versions:
            token = page["NextKeyMarker"], page.get("NextVersionIdMarker")
        else:
            token = page["NextContinuationToken"]
        return page_of(rows, keys, markers, folders, token, keep)

    async def _learn_lean(self, s3, sent: list, versions: bool) -> None:
        try:
            self._lean = await self._lean_spec(s3, sent, versions)
        except Exception:
            logger.debug("Listing children use botocore", exc_info=True)
            self._lean = False

    async def _lean_spec(self, s3, sent: list, versions: bool) -> LeanSpec | bool:
        """How a listing child can list as this client just did, or False when
        the request botocore sent was not a plain one, signed by SigV4 or not
        signed: a child then lists with botocore."""
        request = _plain_request(sent, self.fs.config_kwargs or {})
        if request is None:
            return False
        url, headers, region = request
        credentials = None
        if region is not None:
            signer = s3._request_signer._credentials
            expiry = getattr(signer, "_expiry_time", None)
            # none, or ones that would expire while children list
            if signer is None or (expiry and expiry.timestamp() < time.time() + 900):
                return False
            frozen = await signer.get_frozen_credentials()
            credentials = (frozen.access_key, frozen.secret_key, frozen.token)
        return LeanSpec(
            uri=self.uri,
            url=url,
            region=region,
            credentials=credentials,
            headers=headers,
            versions=versions,
            verify=lean_verify(self.fs.client_kwargs or {}),
            in_key_order=self._lists_in_key_order(),
        )

    async def _fetch_prefix(self, start_prefix: str, result_queue) -> None:
        self._lean = None  # learnt again: credentials and versions may differ
        await super()._fetch_prefix(start_prefix, result_queue)

    def _child_spec(self) -> tuple[type, tuple] | None:
        if isinstance(self._lean, LeanSpec):
            return LeanS3, (self._lean,)
        return None

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

    clean_s3_version = staticmethod(clean_version)

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
