import functools
import glob
import logging
import os
import pickle
import re
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import blake2b
from typing import TYPE_CHECKING, TypeVar, cast

from fsspec.asyn import get_loop, sync
from sqlalchemy import and_, bindparam, case, false, or_, true
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement

import datachain.fs.utils as fsutils
from datachain.asyn import iter_over_async
from datachain.client import Client
from datachain.error import (
    ClientError,
    ConcurrentDatasetModificationError,
    DatasetNotFoundError,
)
from datachain.lib.convert.flatten import flatten
from datachain.lib.file import File, FileRow
from datachain.query.schema import Column
from datachain.sql.functions import path as pathfunc
from datachain.sql.types import Boolean
from datachain.utils import parent_path, safe_closing, uses_glob

if TYPE_CHECKING:
    from sqlalchemy import BindParameter, ColumnElement

    from datachain.catalog import Catalog
    from datachain.client.fsspec import RangeListedClient
    from datachain.data_storage.codec import RowCodec
    from datachain.dataset import DatasetRecord, DatasetVersion
    from datachain.lib.dc import DataChain
    from datachain.project import Project
    from datachain.query.session import Session

LISTING_TTL = 4 * 60 * 60  # cached listing lasts 4 hours
LISTING_PREFIX = "lst__"  # listing datasets start with this name
LISTING_BATCH_SIZE = 50_000  # rows per insert: listing rows are small
RESERVATION_TOUCH_INTERVAL = 300
# a listing version's content_hash: this prefix, its row count and rows_digest sum
LISTING_DIGEST = "listing-v1"
# listing of current objects only, read_storage(versions=False)
CURRENT_LISTING_PREFIX = f"{LISTING_PREFIX}current__"

D = TypeVar("D", bound="DataChain")

# Disable warnings for remote errors in clients
logging.getLogger("aiobotocore.credentials").setLevel(logging.CRITICAL)
logging.getLogger("gcsfs").setLevel(logging.CRITICAL)


def listing_dataset_expired(lst_ds) -> bool:
    """Function that checks if listing dataset is expired or not"""
    lst_version = lst_ds.versions[-1]
    if not lst_version.finished_at:
        return False

    expires = lst_version.finished_at + timedelta(seconds=LISTING_TTL)
    return datetime.now(UTC) > expires


def list_pages(
    uri: str, cache, client_config=None, encode: "PageEncoder | None" = None
) -> Iterator:
    """Pages of ``FileRow`` rows of every object under the ``uri`` prefix, or
    with ``encode``, ``StoredPage``s: listing children encode the pages they
    parse, and this process the rest."""
    config = client_config or {}
    client = Client.get_client(uri, cache, **config)  # type: ignore[arg-type]
    client.encode_page = encode
    _, path = Client.parse_url(uri)
    # a child's pages are StoredPages already
    with safe_closing(
        iter_over_async(client.scandir(path.removesuffix("/")), get_loop())
    ) as pages:
        for page in pages:
            if isinstance(page, StoredPage):
                yield page
                continue
            rows = (
                [*map(file_row, page)] if page and isinstance(page[0], File) else page
            )
            yield encode(rows) if encode else rows


def list_scope_pages(
    uri: str, scope: "ListScope", cache, client_config=None
) -> Iterator[Sequence[tuple]]:
    """Pages of ``FileRow`` rows of the keys ``scope`` holds in the storage of
    ``uri``, whose client lists parts of a folder (S3, GCS, Azure)."""
    config = client_config or {}
    client = cast("RangeListedClient", Client.get_client(uri, cache, **config))  # type: ignore[arg-type]
    method = "level" if scope.one_level and client.LISTS_LEVELS else "prefix"
    return _scope_pages(client, scope, method)


def _scope_pages(client, scope: "ListScope", method: str) -> Iterator[Sequence[tuple]]:
    pages = client.scandir(scope.prefix, method=method)
    try:
        yield from iter_over_async(pages, get_loop())
    except FileNotFoundError:  # nothing under the prefix: fine in a folder
        _check_folder(client, scope.folder)


def _check_folder(client, folder: str) -> None:
    """Raise, as a listing of ``folder`` would, when it holds no keys. The probe
    also raises the backend's error for a missing bucket, when ``folder`` is
    its root."""
    has_keys = sync(get_loop(), client.has_keys, folder)  # a missing bucket raises
    if folder and not has_keys:
        raise FileNotFoundError(f"Unable to resolve remote path: {folder}")


@dataclass(frozen=True)
class ListScope:
    """What a URI's listing must read of its ``folder``: the keys that start
    with ``prefix``, only those with no delimiter past it if ``one_level``."""

    folder: str
    prefix: str
    one_level: bool


def file_row(file: File) -> FileRow:
    return FileRow(*flatten(file))


def get_file_info(uri: str, cache, client_config=None) -> File:
    """
    Wrapper to return File object by its URI
    """
    client = Client.get_client(uri, cache, **(client_config or {}))  # type: ignore[arg-type]
    _, path = Client.parse_url(uri)
    return client.get_file_info(path)


def ls(
    dc: D,
    path: str,
    recursive: bool | None = True,
) -> D:
    """
    Return files by some path from DataChain instance which contains bucket listing.
    Path can have globs.
    If recursive is set to False, only first level children will be returned by
    specified path
    """
    return select_paths(dc, PathFilter.ls(path, recursive))


def select_paths(dc: D, paths: "PathFilter") -> D:
    """The latest files of a listing chain whose paths ``paths`` selects."""
    dc = dc.filter(Column("file.is_latest") == true())
    for condition in paths.sql(Column("file.path")):
        dc = dc.filter(condition)
    return dc


def list_folder(path: str) -> str:
    return path.lstrip("/").removesuffix("/")


@dataclass(frozen=True)
class PathFilter:
    """The paths of a listing that a URI selects: those that pass one of
    ``any_of`` and, if there are ``alternatives``, all conditions of one of
    them. A condition is an ``(op, value)``: op ``eq`` is a path, ``parent``
    its folder, ``glob`` SQLite's GLOB, whose ``*`` crosses ``/``, and
    ``pattern`` a glob matched folder by folder, whose ``**`` is any folders.
    ``read_storage`` applies it in SQL, ``dc.list_storage`` in Python."""

    any_of: tuple[tuple[str, str], ...] = ()
    alternatives: tuple[tuple[tuple[str, str], ...], ...] = ()

    @classmethod
    def ls(cls, path: str, recursive: bool | None) -> "PathFilter":
        if not recursive:
            return cls(any_of=(("parent", list_folder(path)),))
        if not path or path == "/":
            return cls()
        if uses_glob(path):
            return cls(any_of=(("glob", path),))
        return cls(any_of=(("eq", path), ("glob", path.removesuffix("/") + "/*")))

    def sql(self, path: Column) -> "list[ColumnElement]":
        """One condition per filter step: ``any_of`` joined, then those of the
        one alternative, or one for all of them. The steps feed chain hashes,
        so their split is kept."""

        def cond(op: str, value: str) -> "ColumnElement":
            if op == "eq":
                return path == value
            if op == "glob":
                return path.glob(_inline(value))
            if op == "pattern":
                regex = _pattern_regex(value, atomic=False)
                if regex is None:
                    return false()
                atomic = _pattern_regex(value, atomic=True)
                return _PatternMatch(path, f"^{regex}$", _inline(f"^{atomic}\\Z"))
            return pathfunc.parent(path) == value

        steps = [or_(*(cond(*c) for c in self.any_of))] if self.any_of else []
        if len(self.alternatives) == 1:
            return steps + [cond(*c) for c in self.alternatives[0]]
        if self.alternatives:
            # A CASE stays flat: SQLite parses a chain of ORs into a tree as
            # deep as it is long, and refuses one deeper than 1000.
            whens = [(and_(*(cond(*c) for c in a)), true()) for a in self.alternatives]
            steps.append(case(*whens, else_=false()))
        return steps

    def matcher(self) -> "Callable[[str], bool]":
        """The filter as a function of a path, built once for many rows."""

        def check(op: str, value: str) -> "Callable[[str], object]":
            if op == "eq":
                return value.__eq__
            if op == "glob":
                return _glob_regex(value).fullmatch
            if op == "pattern":
                return _pattern_matcher(value).fullmatch
            return lambda path: parent_path(path) == value

        any_of = [check(*c) for c in self.any_of]
        alternatives = [[check(*c) for c in a] for a in self.alternatives]
        return lambda path: (
            (not any_of or any(c(path) for c in any_of))
            and (not alternatives or any(all(c(path) for c in a) for a in alternatives))
        )


_NOTHING = "(?!)"  # a regex that matches nothing


@functools.lru_cache(maxsize=256)
def _glob_parts(pattern: str, one_folder: bool = False) -> "list[str] | None":
    """SQLite's GLOB split at ``*`` into regexes; None matches nothing. With
    ``one_folder``, ``?`` and ``[...]`` never match ``/``."""
    parts: list[list[str]] = []
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        i += 1
        if c == "*":
            parts.append(out)
            out = []
        elif c == "?":
            out.append("[^/]" if one_folder else ".")
        elif c == "[":
            negate = i < n and pattern[i] == "^"
            j = i + negate
            members, prior = [], None
            if j < n and pattern[j] == "]":
                members.append("\\]")
                j += 1
            while j < n and pattern[j] != "]":
                c = pattern[j]
                if c == "-" and prior and j + 1 < n and pattern[j + 1] != "]":
                    ranges = [(prior, pattern[j + 1])]
                    if one_folder and not negate and prior <= "/" <= pattern[j + 1]:
                        ranges = [(prior, "."), ("0", pattern[j + 1])]
                    members += [
                        f"{re.escape(a)}-{re.escape(b)}" for a, b in ranges if a <= b
                    ]
                    prior = None
                    j += 2
                else:
                    members.append(re.escape(c))
                    prior = c
                    j += 1
            if j >= n:
                return None
            if one_folder and negate:
                members.append("/")
            out.append(f"[{'^' if negate else ''}{''.join(members)}]")
            i = j + 1
        else:
            out.append(re.escape(c))
    parts.append(out)
    return ["".join(p) for p in parts]


def _atomic_join(parts: list[str], star: str) -> str:
    """``parts`` joined by ``star``, each part between stars kept at its
    leftmost match by ``(?>...)``, so that matching stays linear."""
    head, *rest = parts
    if not rest:
        return head
    *middle, tail = rest
    return head + "".join(f"(?>{star}?{m})" for m in middle) + f"{star}{tail}"


@functools.lru_cache(maxsize=256)
def _glob_regex(pattern: str) -> re.Pattern:
    """SQLite's GLOB as a regex."""
    parts = _glob_parts(pattern)
    return re.compile(
        _NOTHING if parts is None else _atomic_join(parts, ".*"), re.DOTALL
    )


@functools.lru_cache(maxsize=256)
def _pattern_regex(pattern: str, atomic: bool) -> "str | None":
    """A glob ``pattern`` matched folder by folder, as a regex; None matches
    nothing. ``*``, ``?`` and ``[...]`` stay in one folder and ``**`` is any
    folders. The plain form is RE2's too. The ``atomic`` one, for Python's
    ``re``, keeps each part between stars, and each run of folders between
    two ``**``, at its leftmost match, so that matching stays linear."""
    runs: list[list[str]] = [[]]
    for segment in pattern.split("/"):
        if segment == "**":
            runs.append([])
            continue
        parts = _glob_parts(segment, one_folder=True)
        if parts is None:
            return None
        runs[-1].append(_atomic_join(parts, "[^/]*") if atomic else "[^/]*".join(parts))
    head, *rest = runs
    if not rest:
        return "/".join(head)
    *middle, tail = rest
    regex = "".join(f"{s}/" for s in head)
    for run in middle:
        folders = "".join(f"{s}/" for s in run)
        regex += f"(?>(?:[^/]*/)*?{folders})" if atomic else f"(?:[^/]*/)*{folders}"
    return f"{regex}(?:[^/]*/)*" + ("/".join(tail) if tail else "[^/]*")


@functools.lru_cache(maxsize=256)
def _pattern_matcher(pattern: str) -> re.Pattern:
    regex = _pattern_regex(pattern, atomic=True)
    return re.compile(_NOTHING if regex is None else regex)


def _inline(value: str) -> "BindParameter":
    """``value`` written into SQLite's SQL rather than bound: a URI can expand
    to more patterns than old SQLite binds (999). ClickHouse binds GLOBs anew,
    and gets only the RE2 regex, which stays bound."""
    return bindparam(None, value, literal_execute=True)


class _PatternMatch(FunctionElement):
    """Whether a path matches a regex, given in RE2's form and in the one that
    Python's ``re`` matches in linear time: SQLite's REGEXP is Python's."""

    type = Boolean()
    inherit_cache = True


@compiles(_PatternMatch)
def _compile_pattern_match(element, compiler, **kwargs):
    path, regex, _ = element.clauses.clauses
    return compiler.process(path.op("REGEXP", is_comparison=True)(regex), **kwargs)


@compiles(_PatternMatch, "sqlite")
def _compile_pattern_match_sqlite(element, compiler, **kwargs):
    path, _, regex = element.clauses.clauses
    return compiler.process(path.op("REGEXP", is_comparison=True)(regex), **kwargs)


def glob_match(path: str, pattern: str) -> bool:
    """SQLite's GLOB."""
    return _glob_regex(pattern).fullmatch(path) is not None


_DS_ENCODE_RE = re.compile(r"[^a-zA-Z0-9_/:\-]")


def _sanitize_ds_name(raw: str) -> str:
    """Make *raw* safe for use as a SQL table-name component.

    The encoding is injective (collision-free) and handles the full Unicode
    range via a variable-length hex scheme:

    1. Pre-escape literals ``_x``, ``_y``, ``_z`` that appear in the raw string
       (``_x`` → ``_x_x``, ``_y`` → ``_y_y``, ``_z`` → ``_z_z``).  All three
       replacements must run in this order *before* the regex step, because the
       regex introduces new ``_x``/``_y``/``_z`` tokens that must never be
       re-escaped.  Order also matters within the three: e.g. ``"_x_y"`` must
       become ``"_x_x_y_y"``, not ``"_x_x_y"`` (which the decoder can't
       round-trip).
    2. Encode every char outside ``[a-zA-Z0-9_/:-]``:
       - U+0000-U+00FF  →  ``_xHH``     (2 hex digits, e.g. ``.`` → ``_x2e``)
       - U+0100-U+FFFF  →  ``_yHHHH``   (4 hex digits, e.g. ``é`` → ``_y00e9``)
       - U+10000-U+10FFFF → ``_zHHHHHH`` (6 hex digits, e.g. 🎵 → ``_z01f3b5``)

    ``_`` is not a hex digit, so ``_y_y`` / ``_z_z`` escape tokens are
    unambiguous with ``_yHHHH`` / ``_zHHHHHH`` encoding tokens.
    """
    raw = raw.replace("_x", "_x_x")
    raw = raw.replace("_y", "_y_y")
    raw = raw.replace("_z", "_z_z")

    def _encode_char(m: re.Match) -> str:
        o = ord(m.group())
        if o <= 0xFF:
            return f"_x{o:02x}"
        if o <= 0xFFFF:
            return f"_y{o:04x}"
        return f"_z{o:06x}"

    return _DS_ENCODE_RE.sub(_encode_char, raw)


# _y(4 hex | _y) →  BMP char or literal _y
# _z(6 hex | _z) →  supplementary char or literal _z
# _x(_x | 2 hex) →  literal _x or low-ISO char
_DS_DECODE_RE = re.compile(r"_y([0-9a-f]{4}|_y)|_z([0-9a-f]{6}|_z)|_x(_x|[0-9a-f]{2})")


def _desanitize_ds_name(encoded: str) -> str:
    """Reverse :func:`_sanitize_ds_name`.

    ``_xHH`` → char, ``_yHHHH`` → BMP char,
    ``_zHHHHHH`` → supplementary char,
    ``_x_x`` → ``_x``, ``_y_y`` → ``_y``, ``_z_z`` → ``_z``.
    """

    def _repl(m: re.Match) -> str:
        g1, g2, g3 = m.group(1), m.group(2), m.group(3)
        if g1 is not None:
            return "_y" if g1 == "_y" else chr(int(g1, 16))
        if g2 is not None:
            return "_z" if g2 == "_z" else chr(int(g2, 16))
        return "_x" if g3 == "_x" else chr(int(g3, 16))

    return _DS_DECODE_RE.sub(_repl, encoded)


def parse_listing_uri(uri: str, versions: bool = True) -> tuple[str, str, str]:
    """
    Parsing uri and returns listing dataset name, listing uri and listing path.
    A listing of current objects only (``versions=False``) gets its own name.
    """
    storage_uri, path = Client.parse_url(uri)
    if uses_glob(path):
        # up to the last "/": "d//*.txt" is listed under "d//"
        lst_uri_path = path[: path.rfind("/") + 1]
    else:
        if path:
            storage_uri, path = Client.parse_url(
                uri if uri.endswith("/") else f"{uri}/"
            )
        lst_uri_path = path

    lst_uri = f"{storage_uri}/{lst_uri_path.lstrip('/')}"
    ds_name = _sanitize_ds_name(f"{LISTING_PREFIX}{lst_uri}")
    if not versions:
        ds_name = CURRENT_LISTING_PREFIX + ds_name.removeprefix(LISTING_PREFIX)

    return ds_name, lst_uri, path


def listing_name_prefixes(prefix: str) -> list[str]:
    """Listing-name prefixes that match a storage URI prefix, one per mode."""
    if prefix.startswith(LISTING_PREFIX):
        return [prefix]
    return [LISTING_PREFIX + prefix, CURRENT_LISTING_PREFIX + prefix]


def is_listing_dataset(name: str) -> bool:
    """Returns True if it's special listing dataset"""
    return name.startswith(LISTING_PREFIX)


def listing_uri_from_name(dataset_name: str) -> str:
    """Returns clean storage URI from listing dataset name.

    Strips the ``lst__`` prefix and reverses the ``_xHH`` encoding
    applied by :func:`_sanitize_ds_name`.
    """
    if not is_listing_dataset(dataset_name):
        raise ValueError(f"Dataset {dataset_name} is not a listing")
    name = dataset_name.removeprefix(CURRENT_LISTING_PREFIX)
    return _desanitize_ds_name(name.removeprefix(LISTING_PREFIX))


@contextmanager
def _reraise_as_client_error() -> Iterator[None]:
    try:
        yield
    except Exception as e:
        raise ClientError(message=str(e), error_code=getattr(e, "code", None)) from e


def names_object(client: Client, uri: str) -> bool:
    """True if ``uri`` names a single object rather than a prefix or a glob."""
    isfile = _reraise_as_client_error()(fsutils.isfile)
    return not glob.has_magic(uri) and not uri.endswith("/") and isfile(client.fs, uri)


def describe_listing(
    client: Client, uri: str, versions: bool = True
) -> tuple[str, str, str] | None:
    """The listing dataset name, listing URI and listing path of ``uri``, or
    None when it names a single object."""
    if names_object(client, uri):
        return None
    # local, HF and HTTP storages have no versions: one kind of listing
    versions = versions or not hasattr(client.fs, "version_aware")
    return parse_listing_uri(uri, versions=versions)


def get_listing(
    uri: str | os.PathLike[str],
    session: "Session",
    update: bool = False,
    versions: bool = True,
) -> tuple[str | None, str, str, bool]:
    """Returns correct listing dataset name that must be used for saving listing
    operation. It takes into account existing listings and reusability of those.
    It also returns boolean saying if returned dataset name is reused / already
    exists or not (on update it always returns False - just because there was no
    reason to complicate it so far). And it returns correct listing path that should
    be used to find rows based on uri. With ``versions=False`` on a storage
    that keeps object versions, the listing holds current objects only and
    neither reuses nor is reused by a listing of all versions.
    """
    from datachain.client.local import FileClient
    from datachain.telemetry import telemetry

    catalog = session.catalog
    cache = catalog.cache
    client_config = catalog.client_config

    client = Client.get_client(uri, cache, **client_config)
    telemetry.log_param("client", client.PREFIX)
    if not isinstance(uri, str):
        uri = os.fspath(uri)

    # we don't want to use cached dataset (e.g. for a single file listing)
    described = describe_listing(client, uri, versions)
    if described is None:
        _, path = Client.parse_url(uri)
        return None, uri, path, False

    ds_name, list_uri, list_path = described
    listing = None
    listings = [
        ls for ls in catalog.listings() if not ls.is_expired and ls.contains(ds_name)
    ]

    # if no need to update - choosing the most recent one;
    # otherwise, we'll using the exact original `ds_name`` in this case:
    # - if a "bigger" listing exists, we don't want to update it, it's better
    #   to create a new "smaller" one on "update=True"
    # - if an exact listing exists it will have the same name as `ds_name`
    #   anyway below
    if listings and not update:
        listing = max(listings, key=lambda ls: ls.created_at)

    # `list_path` filters the listing's rows, so it has to be written the way
    # those rows store their paths - and that differs by backend. Cloud rows
    # keep the full bucket-relative key even when the listing covers only a
    # prefix (a listing of "droid/" stores "droid/a.txt"), and `list_path` is
    # bucket-relative too, so the two line up whichever listing is reused.
    # Local rows drop the listed directory instead (a listing of "/a/b/" stores
    # "c.txt") and `list_path` comes back empty, so reusing a listing rooted
    # higher up would filter by nothing and match everything in it. Rebuild the
    # missing prefix from the two names, which both spell out the full URI:
    #   lst__file:///a/b/c/  minus  lst__file:///a/  ->  b/c/
    if isinstance(client, FileClient) and listing and listing.name != ds_name:
        # names are sanitized (`.` -> `_x2e` etc), so decode back into a path
        relpath = _desanitize_ds_name(ds_name.removeprefix(listing.name)).strip("/")
        list_path = f"{relpath}/{list_path}"

    ds_name = listing.name if listing else ds_name
    return ds_name, list_uri, list_path, bool(listing)


def ensure_listing(
    name: str, catalog: "Catalog", write: Callable | None, *, update: bool = False
) -> "DatasetRecord":
    """Create or refresh the selected snapshot only when needed."""
    dataset = None
    if not update:
        try:
            dataset = catalog.get_dataset(name, versions=None, include_incomplete=False)
        except DatasetNotFoundError:
            pass
    if dataset is None or listing_dataset_expired(dataset):
        assert write is not None
        dataset = write() or catalog.get_dataset(
            name, versions=None, include_incomplete=False
        )
    return dataset


def rows_digest(rows: Iterable[tuple], with_etag: bool) -> int:
    """The sum of a 128-bit hash of each ``FileRow`` row's identity: its source,
    path, version id (or md5, or etag if it has neither), size, whether it is the
    latest version, and its etag when ``with_etag``. A sum, not XOR, so that a
    file listed twice counts twice; the order of rows does not matter."""
    total = 0
    for source, path, size, version, etag, is_latest, _, _, md5 in rows:
        key = f"{source}\0{path}\0{version or md5 or etag}\0{size}\0{is_latest:d}"
        if with_etag:
            key += f"\0{etag}"
        digest = blake2b(key.encode(errors="surrogatepass"), digest_size=16)
        total += int.from_bytes(digest.digest())
    return total


class StoredPage:
    """A page of listing rows in the warehouse's stored form, and their
    ``rows_digest``: made where the page was parsed, inserted as it is."""

    def __init__(self, rows: list[tuple], digest: int) -> None:
        self._rows: list[tuple] | bytes = rows
        self.digest = digest
        self.size = len(rows)

    @property
    def rows(self) -> list[tuple]:
        if isinstance(rows := self._rows, bytes):
            self._rows = rows = pickle.loads(rows)  # noqa: S301
        return rows

    def __len__(self) -> int:
        return self.size

    def __getstate__(self) -> tuple[bytes, int, int]:
        return pickle.dumps(self.rows, protocol=5), self.digest, self.size

    def __setstate__(self, state: tuple[bytes, int, int]) -> None:
        self._rows, self.digest, self.size = state


@dataclass(frozen=True)
class PageEncoder:
    """``FileRow`` rows to a ``StoredPage``; listing children run it."""

    codec: "RowCodec"
    with_etag: bool

    def __call__(self, rows: Sequence[tuple]) -> StoredPage:
        stored = list(map(self.codec.encode, rows))
        return StoredPage(stored, rows_digest(rows, self.with_etag))


def write_listing(
    uri: str, ds_name: str, session: "Session", client_config: dict
) -> "DatasetRecord | None":
    """Store the listing of the prefix ``uri`` as the next version of ``ds_name``.
    An unchanged listing drops that version and restarts the latest one's TTL."""
    from datachain import semver
    from datachain.dataset import DEFAULT_DATASET_VERSION, DatasetStatus
    from datachain.lib.signal_schema import SignalSchema

    catalog = session.catalog
    metastore, warehouse = catalog.metastore, catalog.warehouse
    project = metastore.get_project(
        metastore.listing_project_name, metastore.system_namespace_name, create=True
    )
    columns = warehouse.dataset_row_cls.listing_columns()
    with_etag = not Client.get_implementation(uri).PINNED_VERSION_IS_IMMUTABLE
    encode = PageEncoder(warehouse.row_codec(columns), with_etag)
    job = session.get_job()
    job_id = job.id if job else None
    dataset = catalog.create_dataset(
        ds_name,
        project,
        columns=columns,
        feature_schema=SignalSchema({"file": File}).serialize(),
        listing=True,
        update_version="major",
        job_id=job_id,
        query_script=job.query if job else "",
    )
    version = dataset.versions[0].version
    count = total = 0

    def rows(pages: Iterator[StoredPage]) -> Iterator[tuple]:
        nonlocal count, total
        touched = time.monotonic()
        for page in pages:
            count += len(page)
            total += page.digest
            yield from page.rows
            if time.monotonic() - touched > RESERVATION_TOUCH_INTERVAL:
                if (
                    metastore.update_dataset_version(
                        dataset,
                        version,
                        expected_status=DatasetStatus.CREATED,
                        expected_job_id=job_id,
                        created_at=datetime.now(UTC),
                    )
                    is None
                ):
                    raise ConcurrentDatasetModificationError(
                        f"Listing {dataset.name}@{version} was removed while running"
                    )
                touched = time.monotonic()

    table = None
    try:
        if job:
            metastore.link_dataset_version_to_job(
                dataset.versions[0].id, job.id, is_creator=True
            )
        table_name = warehouse.dataset_table_name(dataset, version)
        table = warehouse.create_dataset_rows_table(
            f"{table_name}__{dataset.versions[0].uuid[:8]}", columns=columns
        )
        with safe_closing(
            list_pages(uri, catalog.cache, client_config, encode)
        ) as pages:
            try:
                warehouse.insert_stored_rows(
                    table, [c.name for c in columns], rows(pages), LISTING_BATCH_SIZE
                )
            except Exception as e:
                if not warehouse.db.has_table(table.name):
                    raise ConcurrentDatasetModificationError(
                        f"Listing {dataset.name}@{version} was removed while running"
                    ) from e
                raise
        content_hash = f"{LISTING_DIGEST}:{count}:{total % (1 << 128):032x}"
        latest = _latest_version(catalog, ds_name, project)
        if latest:
            prev_ds, prev_ver = latest
            ver_val = dataset.versions[0].version_value
            fence = (
                max(prev_ver.version_value, semver.value(prev_ver.sources))
                if prev_ver.sources
                else prev_ver.version_value
            )
            older_publishing = any(
                prev_ver.version_value < v.version_value < ver_val
                and v.status == DatasetStatus.CREATED
                and v.content_hash is not None
                for v in prev_ds.all_versions
            )
            if (
                fence > ver_val
                or (prev_ver.content_hash == content_hash and not older_publishing)
            ) and catalog.release_reservation(dataset, version, job_id):
                if prev_ver.content_hash == content_hash and ver_val >= fence:
                    metastore.update_dataset_version(
                        prev_ds,
                        prev_ver.version,
                        expected_status=DatasetStatus.COMPLETE,
                        finished_at=datetime.now(UTC),
                        sources=version,
                    )
                return None
        if (
            metastore.update_dataset_version(
                dataset,
                version,
                expected_status=DatasetStatus.CREATED,
                expected_job_id=job_id,
                content_hash=content_hash,
            )
            is None
        ):
            raise ConcurrentDatasetModificationError(
                f"Could not save {dataset.name}@{version}: "
                "the version was removed or modified before save completed."
            )
        warehouse.rename_table(table, table_name)
        completed = catalog.complete_dataset_version(
            dataset, version, expected_job_id=job_id
        )
        return completed if version == DEFAULT_DATASET_VERSION else None
    except BaseException:
        with suppress(Exception):
            catalog.release_reservation(dataset, version, job_id)
        raise
    finally:
        if table is not None:
            warehouse.db.drop_table(table, if_exists=True)


def _latest_version(
    catalog: "Catalog", name: str, project: "Project"
) -> "tuple[DatasetRecord, DatasetVersion] | None":
    """The listing ``name`` and its latest complete version, if it has one."""
    try:
        dataset = catalog.get_dataset(
            name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            versions=None,
        )
    except DatasetNotFoundError:
        return None
    version = dataset.latest_complete_version
    return (dataset, dataset.get_version(version)) if version else None
