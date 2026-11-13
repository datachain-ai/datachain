import errno
import os
import re
from collections import defaultdict
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote_plus
from xml.etree.ElementTree import Element, ParseError, fromstring

from aiobotocore.parsers import AioResponseParserFactory, AioRestXMLParser
from aiobotocore.session import AioSession
from botocore import handlers
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.model import Shape
from botocore.utils import parse_timestamp
from fsspec.asyn import get_loop, sync, sync_wrapper
from s3fs import S3FileSystem

from datachain.lib.file import File, FileRow, as_file_row

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


# While ClientS3._list_page waits for a page, the source its rows are listed from
_ROWS_OF: ContextVar[str | None] = ContextVar("_ROWS_OF", default=None)
_DECLARATION = re.compile(r"""<\?xml(?:[^>]*?\sencoding=["']([^"']*)["'])?[^>]*\?>""")
_ROOT = re.compile(r"[ \t\n]*<([A-Za-z]\w*)(?:[ \t\n][^<>]*)?>")
_FORBIDDEN = bytes(set(range(32)) - {9, 10, 13})  # control characters XML forbids
_QUOTES = {"&#34;": '"', "&apos;": "'", "&#39;": "'"}
_SPACE = "[ \t\n]*"
_GUARD = re.compile(r"\(\?\(@(\w+)@\)\(\?!\)\)")  # (?(@group@)(?!)), see _entry
# the fields _list_page reads or ElementTree would convert, per list of entries
_READS = {
    "Contents": ("Key", "Size", "ETag", "LastModified"),
    "Versions": ("Key", "Size", "ETag", "LastModified", "VersionId", "IsLatest"),
    "DeleteMarkers": ("Key", "LastModified"),
    "CommonPrefixes": ("Prefix",),
}
_ORDER = ("Key", "VersionId", "IsLatest", "LastModified", "ETag", "Size", "Owner")
_CHILDREN: dict[str, tuple[re.Pattern, dict[str, list[str]]]] = {}


def _unescape(text: str) -> str:
    """Text with its quote entities replaced; any other entity is a ValueError."""
    if "&" not in text:
        return text
    text = text.replace("&quot;", '"')
    if "&" in text:
        for escaped, char in _QUOTES.items():
            text = text.replace(escaped, char)
        if "&" in text:
            raise ValueError(f"an entity in {text!r}")
    return text


def _children(shape: Shape) -> tuple[re.Pattern, dict[str, list[str]]]:
    """A pattern of one child of a listing's root element, with any space before
    it, and the groups of each list's fields. An entry is a group named for its
    list, with a group for each field of ``_READS``; any other child must hold
    text only."""
    if shape.name not in _CHILDREN:
        lists = {n: m for n, m in shape.members.items() if n in _READS}
        entries = [
            _entry(name, member.serialization.get("name", name), member.member)
            for name, member in lists.items()
        ]
        other = r"(?P<other><(?P<tag>[A-Za-z]\w*)(?:>[^<]*</(?P=tag)>|/>))"
        source = rf"{_SPACE}(?:{'|'.join(entries)}|{other})"
        # re takes a group defined later in a condition by number only
        numbers = re.compile(_GUARD.sub("", source)).groupindex
        pattern = re.compile(_GUARD.sub(lambda g: f"(?({numbers[g[1]]})(?!))", source))
        groups = {name: [f"{name}_{f}" for f in _READS[name]] for name in lists}
        _CHILDREN[shape.name] = pattern, groups
    return _CHILDREN[shape.name]


def _entry(name: str, tag: str, entry: Shape) -> str:
    fields = {}
    for field, member in entry.members.items():
        child = member.serialization.get("name", field)
        if field in _READS[name]:
            group = f"{name}_{field}"
            # a field read twice is left to ElementTree, which checks both
            fields[field] = f"<{child}>(?(@{group}@)(?!))(?P<{group}>[^<]*)</{child}>"
        elif member.type_name in ("string", "boolean") or (
            member.type_name == "list" and member.member.type_name == "string"
        ):
            fields[field] = _unread(child)
        elif member.type_name == "structure" and all(
            m.type_name == "string" for m in member.members.values()
        ):
            inner = "|".join(
                _unread(m.serialization.get("name", n))
                for n, m in member.members.items()
            )
            fields[field] = (
                rf"<{child}(?:/>|>(?:{_SPACE}(?:{inner}))*{_SPACE}</{child}>)"
            )
    # tried in the order S3 writes them, as each miss costs a try
    order = sorted(fields, key=lambda f: _ORDER.index(f) if f in _ORDER else 99)
    alternatives = "|".join(fields[f] for f in order)
    return rf"(?P<{name}><{tag}>(?:{_SPACE}(?:{alternatives}))*{_SPACE}</{tag}>)"


def _unread(tag: str) -> str:
    # no entity: one ElementTree rejects would go unseen
    return f"<{tag}(?:/>|>[^<&]*</{tag}>)"


def _row(
    source: str,
    key: str | None,
    size: str | None,
    etag: str | None,
    modified: str | None,
    version: str | None = None,
    latest: str | None = None,
) -> FileRow:
    if key is None or size is None or modified is None:
        raise ValueError("an entry without a key, size or modification time")
    return as_file_row(
        (
            source,
            _unescape(key),
            int(size),
            "" if version is None else ClientS3.clean_s3_version(_unescape(version)),
            "" if etag is None else _unescape(etag).strip('"'),
            True if latest is None else _unescape(latest) == "true",
            _parse_timestamp(_unescape(modified)),
            None,
        )
    )


def _plain_text(body: bytes) -> str | None:
    """The page as text, when all its text reads as it is written, but for
    entities: XML rewrites carriage returns and forbids control characters,
    U+FFFE, U+FFFF and "]]>"."""
    try:
        text = body.decode()
    except UnicodeDecodeError:
        return None
    declaration = _DECLARATION.match(text)
    if declaration and (declaration[1] or "utf-8").lower() != "utf-8":
        return None
    if (
        b"\r" in body
        or (b"]" in body and b"]]>" in body)
        or len(body.translate(None, _FORBIDDEN)) != len(body)
        or (not body.isascii() and (b"\xef\xbf\xbe" in body or b"\xef\xbf\xbf" in body))
    ):
        return None
    return text[declaration.end() :] if declaration else text


def _split(text: str, shape: Shape) -> tuple[str, dict[str, list]] | None:
    """The root element without its entries, and the fields read of each list
    of entries, when patterns read every child of the root."""
    if not (root := _ROOT.match(text)):
        return None
    (pattern, groups), end = _children(shape), root.end()
    found: dict[str, list] = defaultdict(list)
    others = []
    for child in pattern.finditer(text, end):
        if child.start() != end:
            return None  # something no pattern reads, such as a comment
        end = child.end()
        if (name := child.lastgroup or "") == "other":
            others.append(child[0])
        else:
            found[name].append(child.group(*groups[name]))
    close = f"</{root[1]}>"
    if text[end:].strip(" \t\n") != close:
        return None
    return root[0] + "".join(others) + close, found


def _rows_page(body: bytes, shape: Shape, source: str) -> dict[str, Any] | None:
    """The page as ``_list_page`` reads it, or ``None`` when ElementTree must
    read it. Its objects are ``FileRows`` with keys as written; delete markers
    and folders are dicts of their key alone. Patterns read the root's children
    one after another, and ElementTree reads the root with its entries left out."""
    if (text := _plain_text(body)) is None or (split := _split(text, shape)) is None:
        return None
    try:
        root = fromstring(split[0])  # noqa: S314
    except ParseError:
        return None
    ns = root.tag[: root.tag.find("}") + 1]
    if root.tag == f"{ns}Error":
        return None
    page = _read(root, _fields(shape, ns))
    if page.keys() & _READS.keys():
        return None  # an entry the patterns did not match
    found = split[1]
    try:
        page["FileRows"] = [
            _row(source, *e) for e in found["Contents"] + found["Versions"]
        ]
        page["DeleteMarkers"] = []
        for key, modified in found["DeleteMarkers"]:
            if modified is not None:  # ElementTree converts it, so it must be valid
                _parse_timestamp(_unescape(modified))
            page["DeleteMarkers"].append({"Key": _unescape(key)})
        page["CommonPrefixes"] = [
            {"Prefix": _unescape(p)} for p in found["CommonPrefixes"]
        ]
    except ValueError:
        return None
    return page


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
        if page is None:
            # stores without version support, such as R2, hold only current objects
            self._unversioned = True
            return await self._list_page(
                prefix, start_after, None, hi, page_size, delimiter
            )
        keep = Keep(self, start_after, hi)
        if (rows := page.get("FileRows")) is None:  # ElementTree read the page
            entries = page.get("Versions" if versions else "Contents", [])
            keys = [d["Key"] for d in entries]
            rows = [self._file_row(d, versions) for d in entries if keep(d["Key"])]
        else:
            keys = [row[1] for row in rows]
            rows = [row for row in rows if keep(row[1])]
        listed = [d["Key"] for d in page.get("DeleteMarkers", [])]
        # folders past a delimiter: listed past, though they make no rows
        listed += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        if not page.get("IsTruncated"):
            token = None
        elif versions:
            token = page["NextKeyMarker"], page.get("NextVersionIdMarker")
        else:
            token = page["NextContinuationToken"]
        return Page(
            rows,
            keys[0] if keys else None,
            max(keys + listed, default=None),
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
