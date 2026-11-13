"""S3 listing pages without botocore: how ``ClientS3`` reads a page's entries,
and ``LeanS3``, which listing child processes use to request and read pages
themselves. A child imports this module and not botocore, so that it starts in
a fraction of the time."""

import asyncio
import hashlib
import hmac
import os
import random
import re
import ssl
import time
from collections.abc import Callable
from datetime import datetime
from functools import partial
from typing import Any, NamedTuple
from urllib.parse import quote, unquote_plus, urlsplit
from xml.etree.ElementTree import Element, ParseError, fromstring

from .ranges import Keep, Page, valid_key

# Patterns read a page as S3 writes it; ElementTree reads any other page.
_DECLARATION = b'<?xml version="1.0" encoding="UTF-8"?>'
_ROOT = re.compile(r"\s*<List(?:Bucket|Versions)Result[\s>]")
# what the patterns do not read: comments, CDATA, instructions, carriage
# returns, and characters XML forbids
_ODD = ((b"!", b"<!"), (b"?", b"<?"), (b"]", b"]]>"))  # a byte, then the text
_CONTROL = bytes(set(range(32)) - {9, 10})  # characters XML forbids, and CR
_NONCHARACTERS = (b"\xef\xbf\xbe", b"\xef\xbf\xbf")  # U+FFFE, U+FFFF
_ENTITY = re.compile(rb"&(?!(?:amp|lt|gt|quot|apos|#34|#39);)")  # any other
_ENTITIES = (
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&#34;", '"'),
    ("&apos;", "'"),
    ("&#39;", "'"),
    ("&amp;", "&"),  # last: "&amp;lt;" is "&lt;"
)
_TEXT = "([^<]*)"
_SUMS = r"(?:<Checksum\w*>[^<&]*</Checksum\w*>)*"
# the rest of an entry: fields that are not read, and no entry
_REST = (
    r"(?:[^<]+|<(?!/?(?:Key|Size|ETag|LastModified|VersionId|IsLatest"
    r"|Contents|Version|DeleteMarker|CommonPrefixes)[\s/>]))*"
)
_OBJECT = (
    rf"<LastModified>{_TEXT}</LastModified><ETag>{_TEXT}</ETag>{_SUMS}"
    rf"<Size>(\d+)</Size>{_REST}"
)
_FOLDER = rf"<CommonPrefixes><Prefix>{_TEXT}</Prefix></CommonPrefixes>"
# groups: the entry, then key, modified, etag, size | folder
_CONTENTS = rf"(<Contents><Key>{_TEXT}</Key>{_OBJECT}</Contents>|{_FOLDER})"
# groups: the entry, then key, version, latest, modified, etag, size | marker:
# tag, key, version, modified | folder
_VERSIONS = (
    rf"(<Version><Key>{_TEXT}</Key><VersionId>{_TEXT}</VersionId>"
    rf"<IsLatest>(true|false)</IsLatest>{_OBJECT}</Version>"
    rf"|(<DeleteMarker>)<Key>{_TEXT}</Key><VersionId>{_TEXT}</VersionId>"
    rf"<IsLatest>(?:true|false)</IsLatest><LastModified>{_TEXT}</LastModified>"
    rf"{_REST}</DeleteMarker>|{_FOLDER})"
)
_ENTRIES = {False: re.compile(_CONTENTS), True: re.compile(_VERSIONS)}
_FIRST = re.compile(r"<(?:Contents|Version|DeleteMarker|CommonPrefixes)[\s/>]")
_MARK = "<_entries/>"  # where the entries were, to check they are the root's


class Row(NamedTuple):
    """A listed object as ``FileRow``'s columns, for a process without it."""

    source: str
    path: str
    size: int
    version: str
    etag: str
    is_latest: bool
    last_modified: datetime
    location: None


def unescape(text: str) -> str:
    if "&" in text:
        text = text.replace("&quot;", '"')  # in every ETag
        if "&" in text:
            for entity, char in _ENTITIES:
                text = text.replace(entity, char)
    return text


def clean_version(version: str | None) -> str:
    return version if (version is not None and version != "null") else ""


def plain_text(body: bytes) -> str | None:
    """The page as text, when it holds nothing the patterns do not read."""
    if body.startswith(b"<?xml"):
        if body[: len(_DECLARATION)].lower() != _DECLARATION.lower():
            return None
        body = body[len(_DECLARATION) :]
    if (
        any(byte in body and odd in body for byte, odd in _ODD)
        or len(body.translate(None, _CONTROL)) != len(body)
        or (not body.isascii() and any(n in body for n in _NONCHARACTERS))
        or _ENTITY.search(body)
    ):
        return None
    try:
        text = body.decode()
    except UnicodeDecodeError:
        return None
    return text if _ROOT.match(text) else None


def split_page(body: bytes, versions: bool) -> tuple[list[Element], str, list] | None:
    """The root's children but entries, its namespace, and the fields of every
    entry, of a page as S3 writes it: all entries one after another, each with
    its fields in S3's order. ``None`` for any other page."""
    if (text := plain_text(body)) is None:
        return None
    entries = []
    if first := _FIRST.search(text):
        start = first.start()
        if not (entries := _ENTRIES[versions].findall(text, start)):
            return None
        # where the last entry of its kind ends: any entry past it not read
        # is the root's child, which the check below turns away
        close = entries[-1][0][entries[-1][0].rindex("</") :]
        end = text.rfind(close) + len(close)
        # nothing between entries: they cover all from the first to the last
        if sum(len(entry[0]) for entry in entries) != end - start:
            return None
        text = text[:start] + _MARK + text[end:]
    try:
        root = fromstring(text)  # noqa: S314
    except ParseError:
        return None
    ns = root.tag[: root.tag.find("}") + 1]
    others = [c for c in root if c.tag != f"{ns}_entries"]
    if (first and len(others) == len(root)) or any(
        c.tag[len(ns) :] in _ENTRY_TAGS for c in others
    ):
        return None  # entries inside another element, or ones not read
    return others, ns, entries


def tree_page(body: bytes, versions: bool) -> tuple[list[Element], str, list] | None:
    """What ``split_page`` gives, read by ElementTree: for a page in another
    store's layout. ``None`` for an error, or an entry missing a field."""
    try:
        root = fromstring(body)  # noqa: S314
    except ParseError:
        return None
    ns = root.tag[: root.tag.find("}") + 1]
    if root.tag[len(ns) :] not in ("ListBucketResult", "ListVersionsResult"):
        return None
    others, entries = [], []
    for child in root:
        tag = child.tag[len(ns) :]
        if tag not in _ENTRY_TAGS:
            others.append(child)
            continue
        fields = {c.tag[len(ns) :]: c.text or "" for c in child}
        if len(fields) != len(child):
            return None  # a field written twice: botocore converts each
        try:
            entries.append(_tree_entry(tag, fields, versions))
        except KeyError:
            return None
    return others, ns, entries


def _tree_entry(tag: str, f: dict[str, str], versions: bool) -> tuple:
    """An entry as ``split_page``'s groups hold it; a KeyError if a field is
    missing."""
    if tag == "CommonPrefixes":
        return ("",) * (11 if versions else 5) + (f["Prefix"],)
    if not f.get("Size", "x"):
        raise KeyError("Size")  # an empty size: a size the parent fails
    if tag == "Contents" and not versions:
        key, etag, size, modified = (
            f["Key"],
            f.get("ETag", ""),
            f["Size"],
            f["LastModified"],
        )
        return ("", key, modified, etag, size, "")
    if tag == "Version" and versions:
        object_ = (f["Key"], f["VersionId"], f["IsLatest"], f["LastModified"])
        return ("", *object_, f.get("ETag", ""), f["Size"], "", "", "", "", "")
    if tag == "DeleteMarker" and versions:
        marker = (f["Key"], f.get("VersionId", ""), f["LastModified"])
        return ("",) * 7 + ("marker", *marker, "")
    raise KeyError(tag)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def read_entries(
    entries: list,
    versions: bool,
    source: str,
    make: Callable[[tuple], Any],
    parse_timestamp: Callable[[str], datetime] = parse_time,
    unescape: Callable[[str], str] = unescape,
) -> tuple[list, list[str], list[str]]:
    """The rows of objects, made with ``make``, and the keys of delete markers
    and folders, of the entries ``split_page`` or ``tree_page`` read; a field
    that does not convert is a ValueError."""
    rows, markers, folders = [], [], []
    for entry in entries:
        if versions:
            _, key, version, latest, modified, etag, size, marker, *rest = entry
            marker_key, _, marker_modified, folder = rest
        else:
            _, key, modified, etag, size, folder = entry
            version, latest, marker = "", "true", ""
        if size:
            row = (
                source,
                unescape(key),
                int(size),
                clean_version(unescape(version)),
                unescape(etag).strip('"'),
                latest == "true",
                parse_timestamp(unescape(modified)),
                None,
            )
            rows.append(make(row))
        elif marker:  # its date checked, as botocore converts it
            parse_timestamp(unescape(marker_modified))
            markers.append(unescape(marker_key))
        else:
            folders.append(unescape(folder))
    return rows, markers, folders


def page_of(
    rows: list[Any],
    keys: list[str],
    markers: list[str],
    folders: list[str],
    token: Any,
    keep: Keep,
) -> Page:
    """A listing page from its object rows and keys, the keys of its delete
    markers, its folders, and the token of the next page."""
    return Page(
        rows,
        keys[0] if keys else None,
        # folders past a delimiter: listed past, though they make no rows
        max(keys + markers + folders, default=None),
        token,
        keep.past_lo,
        keep.past_hi,
        keep.in_order,
        tuple(folders),
    )


class LeanSpec(NamedTuple):
    """What a listing child needs to list as the parent's S3 client does."""

    uri: str  # the rows' source
    url: str  # the bucket's endpoint, as botocore addressed it: no query
    region: str | None  # None: requests go unsigned
    credentials: tuple[str, str, str | None] | None  # key, secret, token
    headers: dict[str, str]  # other x-amz- headers botocore sent, such as a payer
    versions: bool
    verify: bool | str  # TLS: False, True, or a CA bundle
    in_key_order: bool


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_RETRIES = 4  # as botocore's legacy mode: five attempts in all
_RETRIED = frozenset({500, 502, 503, 504})
_ERROR = re.compile(rb"(?:<\?xml[^>]*\?>)?\s*<Error[\s>]")


class LeanError(Exception):
    """A page a lean child does not read as botocore would: the parent reads the
    rest of its range with botocore."""


class LeanS3:
    """An S3 client for listing child processes: signed ``GET``s with aiohttp,
    pages read by ``split_page``, nothing from botocore. It takes what
    ``ClientS3._child_spec`` saw botocore send for the listing, so it lists only
    where that request was plain, and any page it reads differently fails."""

    LIST_PAGE_SIZE = 1000
    key_order = staticmethod(str)  # code points, as S3 lists them
    _is_valid_key = staticmethod(valid_key)

    def __init__(self, spec: LeanSpec) -> None:
        self.spec = spec
        self.uri = spec.uri
        parts = urlsplit(spec.url)
        # as aiohttp sends the host, and botocore signs it: no default port
        default = {"http": 80, "https": 443}[parts.scheme]
        host = (parts.hostname or "").lower()
        if ":" in host:
            host = f"[{host}]"
        port = parts.port
        self.host = host if port in (None, default) else f"{host}:{port}"
        self.path = parts.path or "/"  # encoded already, as botocore signs it
        self.session: Any = None
        self.keys: dict[str, bytes] = {}  # signing keys by date

    def _lists_in_key_order(self) -> bool:
        return self.spec.in_key_order

    async def _list_page(
        self,
        prefix: str,
        start_after: str,
        token: Any,
        hi: str | None = None,
        page_size: int | None = None,
    ) -> Page:
        query = {"prefix": prefix, "max-keys": str(page_size or self.LIST_PAGE_SIZE)}
        query["encoding-type"] = "url"  # as botocore asks: keys are XML-safe
        if self.spec.versions:
            query["versions"] = ""
            if token is not None:
                query["key-marker"], marker = token
                if marker:
                    query["version-id-marker"] = marker
            elif start_after:
                query["key-marker"] = start_after
        else:
            query["list-type"] = "2"
            if token is not None:
                query["continuation-token"] = token
            elif start_after:
                query["start-after"] = start_after
        body = await self._get(query)
        return self._page(body, start_after, hi)

    async def _get(self, query: dict[str, str]) -> bytes:
        import aiohttp
        from yarl import URL

        if self.session is None:
            verify = self.spec.verify
            context: ssl.SSLContext | bool = bool(verify)
            if isinstance(verify, str):
                context = ssl.create_default_context(cafile=verify)
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, ssl=context),
                timeout=aiohttp.ClientTimeout(sock_connect=60, sock_read=60),
                trust_env=True,  # proxies from the environment, as botocore
            )
        canonical = "&".join(
            f"{_quote(k)}={_quote(v)}" for k, v in sorted(query.items())
        )
        url = f"{self.spec.url}?{canonical}"
        for attempt in range(_RETRIES + 1):
            headers = self._signed(canonical)
            try:
                # encoded: aiohttp must send the query as signed
                target = URL(url, encoded=True)
                async with self.session.get(target, headers=headers) as response:
                    body = await response.read()
                    status = response.status
            except (TimeoutError, aiohttp.ClientError):
                if attempt == _RETRIES:
                    raise
                status, body = 0, b""
            if status == 200 and not _ERROR.match(body):
                return body
            # a 200 with an error, as botocore takes it: a 500
            retried = status in (0, 200, *_RETRIED)
            if not retried or attempt == _RETRIES:
                raise LeanError(f"HTTP {status}: {body[:300]!r}")
            # botocore's legacy backoff: random, doubling, at most 20 s
            await asyncio.sleep(random.random() * min(20, 2**attempt))  # noqa: S311
        raise AssertionError("unreachable")

    def _signed(self, canonical_query: str) -> dict[str, str]:
        """The request's headers, signed with AWS Signature Version 4 as botocore
        signs S3 requests; none to sign when the parent's went unsigned."""
        headers = {"host": self.host, **self.spec.headers}
        if self.spec.region is None or self.spec.credentials is None:
            del headers["host"]  # aiohttp sends it
            return headers
        key, secret, token = self.spec.credentials
        now = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        day = now[:8]
        headers["x-amz-content-sha256"] = _EMPTY_SHA256
        headers["x-amz-date"] = now
        if token:
            headers["x-amz-security-token"] = token
        names = sorted(headers)
        request = "\n".join(
            [
                "GET",
                self.path,
                canonical_query,
                "".join(f"{n}:{' '.join(headers[n].split())}\n" for n in names),
                ";".join(names),
                _EMPTY_SHA256,
            ]
        )
        scope = f"{day}/{self.spec.region}/s3/aws4_request"
        to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                now,
                scope,
                hashlib.sha256(request.encode()).hexdigest(),
            ]
        )
        if day not in self.keys:
            signing = f"AWS4{secret}".encode()
            for part in (day, self.spec.region, "s3", "aws4_request"):
                signing = hmac.new(signing, part.encode(), hashlib.sha256).digest()
            self.keys = {day: signing}
        signature = hmac.new(self.keys[day], to_sign.encode(), hashlib.sha256)
        headers["authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={key}/{scope}, "
            f"SignedHeaders={';'.join(names)}, Signature={signature.hexdigest()}"
        )
        del headers["host"]  # aiohttp sends it, the same
        return headers

    def _page(self, body: bytes, start_after: str, hi: str | None) -> Page:
        versions = self.spec.versions
        text: Callable[[str], str] = unescape
        if (split := split_page(body, versions)) is None:
            # another store's layout: ElementTree reads it, entities and all
            split, text = tree_page(body, versions), str
        if split is None:
            raise LeanError("a page neither patterns nor ElementTree read")
        others, ns, entries = split
        fields = {child.tag[len(ns) :]: child.text or "" for child in others}
        try:
            rows, markers, folders = read_entries(
                entries, versions, self.uri, _ROW, unescape=text
            )
        except ValueError as e:
            raise LeanError(str(e)) from e
        encoded = fields.get("EncodingType") == "url"
        decode = unquote_plus if encoded else str
        if encoded:
            rows = [
                r._replace(path=unquote_plus(r.path))
                if "%" in r.path or "+" in r.path
                else r
                for r in rows
            ]
        keys = [r.path for r in rows]
        keep = Keep(self, start_after, hi)
        rows = keep.take(rows, keys)
        markers = [decode(key) for key in markers]
        folders = [decode(prefix) for prefix in folders]
        token: Any = None
        if fields.get("IsTruncated") == "true":
            if self.spec.versions:
                marker = fields.get("NextVersionIdMarker")
                token = decode(fields["NextKeyMarker"]), marker
            else:
                token = fields["NextContinuationToken"]
        return page_of(rows, keys, markers, folders, token, keep)

    async def _probe(self, prefix: str, start_after: str) -> str | None:
        raise NotImplementedError  # the parent probes


# the tags of entries in either listing; botocore names Version and DeleteMarker
# lists by their members
_ENTRY_TAGS = frozenset(
    {"Contents", "Version", "DeleteMarker", "CommonPrefixes", "Versions"}
)
_ROW = partial(tuple.__new__, Row)


def _quote(text: str) -> str:
    return quote(text, safe="-_.~")


def lean_verify(client_kwargs: dict) -> bool | str:
    """TLS verification as botocore would do it for these client kwargs."""
    verify = client_kwargs.get("verify")
    if verify is None:
        verify = os.environ.get("AWS_CA_BUNDLE") or True
    return verify
