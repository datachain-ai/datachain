import functools
import glob
import logging
import os
import posixpath
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, TypeVar, cast

from fsspec.asyn import get_loop, sync
from sqlalchemy.sql.expression import or_, true

import datachain.fs.utils as fsutils
from datachain.asyn import iter_over_async
from datachain.client import Client
from datachain.error import ClientError, DatasetNotFoundError
from datachain.lib.convert.flatten import flatten
from datachain.lib.file import File, FileRow
from datachain.query.schema import C, Column
from datachain.sql.functions import path as pathfunc
from datachain.utils import uses_glob

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement

    from datachain.catalog import Catalog
    from datachain.client.fsspec import RangeListedClient
    from datachain.dataset import DatasetRecord
    from datachain.lib.dc import DataChain
    from datachain.query.session import Session

LISTING_TTL = 4 * 60 * 60  # cached listing lasts 4 hours
LISTING_PREFIX = "lst__"  # listing datasets start with this name
LISTING_BATCH_SIZE = 50_000  # rows per insert: listing rows are small
# listing of current objects only, read_storage(versions=False)
CURRENT_LISTING_PREFIX = f"{LISTING_PREFIX}current__"

D = TypeVar("D", bound="DataChain")

# Disable warnings for remote errors in clients
logging.getLogger("aiobotocore.credentials").setLevel(logging.CRITICAL)
logging.getLogger("gcsfs").setLevel(logging.CRITICAL)


def calc_fingerprint(session: "Session", dataset: "DatasetRecord", version: str) -> int:
    """Compute an order-independent content fingerprint for a listing version.

    Uses file path + version (if available) or etag as the content identifier.
    Version is preferred over etag because etags can be unstable on some cloud
    providers (GCS, Azure) due to internal operations like storage class changes.
    """
    # Inline imports: lib.listing sits below lib.dc / query.dataset / func in the
    # import graph (catalog.catalog imports this module during init), so hoisting
    # any of these triggers a circular import.
    from datachain.func.aggregate import xor_agg
    from datachain.func.conditional import case
    from datachain.func.string import string_hash
    from datachain.lib.dc.datachain import DataChain
    from datachain.lib.dc.utils import Sys
    from datachain.lib.settings import Settings
    from datachain.lib.signal_schema import SignalSchema
    from datachain.query.dataset import DatasetQuery

    query = DatasetQuery(
        name=dataset.name,
        version=version,
        session=session,
        include_incomplete=True,
    )
    signal_schema = SignalSchema({"sys": Sys, "file": File})
    chain = DataChain(query, Settings(), signal_schema)

    identifier = case(
        (C("file.version") != "", C("file.version")),
        else_=C("file.etag"),
    )
    return chain.group_by(  # type: ignore[return-value]
        fp=xor_agg(string_hash(C("file.path"), identifier)),
    ).to_values("fp")[0]


def listing_dataset_expired(lst_ds) -> bool:
    """Function that checks if listing dataset is expired or not"""
    lst_version = lst_ds.versions[-1]
    if not lst_version.finished_at:
        return False

    expires = lst_version.finished_at + timedelta(seconds=LISTING_TTL)
    return datetime.now(UTC) > expires


def list_pages(uri: str, cache, client_config=None) -> Iterator[Sequence[tuple]]:
    """Pages of ``FileRow`` rows of every object under the ``uri`` prefix."""
    config = client_config or {}
    client = Client.get_client(uri, cache, **config)  # type: ignore[arg-type]
    _, path = Client.parse_url(uri)
    pages = client.scandir(path.rstrip("/"))
    for page in iter_over_async(pages, get_loop()):
        yield [*map(file_row, page)] if page and isinstance(page[0], File) else page


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


def list_bucket(uri: str, cache, client_config=None) -> Callable:
    """A generator function, for ``gen()``, of the rows of ``list_pages``."""

    def list_func() -> Iterator[File]:  # FileRow rows
        for page in list_pages(uri, cache, client_config):
            yield from page  # type: ignore[misc]

    list_func.__name__ = "read_storage"

    return list_func


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
    column="file",
) -> D:
    """
    Return files by some path from DataChain instance which contains bucket listing.
    Path can have globs.
    If recursive is set to False, only first level children will be returned by
    specified path
    """
    return select_paths(dc, PathFilter.ls(path, recursive), column)


def select_paths(dc: D, paths: "PathFilter", column: str = "file") -> D:
    """The latest files of a listing chain whose paths ``paths`` selects."""
    dc = dc.filter(Column(f"{column}.is_latest") == true())
    for condition in paths.sql(Column(f"{column}.path")):
        dc = dc.filter(condition)
    return dc


@dataclass(frozen=True)
class PathFilter:
    """The paths of a listing that a URI selects: any of ``any_of`` and all of
    ``all_of``, each an ``(op, value)`` with op ``eq``, ``glob`` or ``parent``.
    ``read_storage`` applies it in SQL, ``dc.list_storage`` in Python."""

    any_of: tuple[tuple[str, str], ...] = ()
    all_of: tuple[tuple[str, str], ...] = ()

    @classmethod
    def ls(cls, path: str, recursive: bool | None) -> "PathFilter":
        if not recursive:
            return cls(any_of=(("parent", path.lstrip("/").rstrip("/*")),))
        if not path or path == "/":
            return cls()
        if uses_glob(path):
            return cls(any_of=(("glob", path),))
        return cls(any_of=(("eq", path), ("glob", path.rstrip("/") + "/*")))

    def glob(self, pattern: str) -> "PathFilter":
        return PathFilter(self.any_of, (*self.all_of, ("glob", pattern)))

    def sql(self, path: Column) -> "list[ColumnElement]":
        """One condition per filter step: ``any_of`` combined, then each of
        ``all_of``. The steps feed chain hashes, so their split is kept."""

        def cond(op: str, value: str) -> "ColumnElement":
            if op == "eq":
                return path == value
            if op == "glob":
                return path.glob(value)
            return pathfunc.parent(path) == value

        steps = [or_(*(cond(*c) for c in self.any_of))] if self.any_of else []
        return steps + [cond(*c) for c in self.all_of]

    def matcher(self) -> "Callable[[str], bool]":
        """The filter as a function of a path, built once for many rows."""

        def check(op: str, value: str) -> "Callable[[str], bool]":
            if op == "eq":
                return value.__eq__
            if op == "glob":
                regex = _glob_regex(value)
                return lambda path: regex is not None and bool(regex.fullmatch(path))
            return lambda path: _parent(path) == value

        any_of = [check(*c) for c in self.any_of]
        all_of = [check(*c) for c in self.all_of]
        return lambda path: (
            (not any_of or any(c(path) for c in any_of))
            and all(c(path) for c in all_of)
        )


def _parent(path: str) -> str:
    """SQLite's ``path_parent``: the path without its name and the slashes
    before it."""
    return path.rstrip(path.replace("/", "")).rstrip("/")


@functools.lru_cache(maxsize=256)
def _glob_parts(pattern: str) -> "list[str] | None":
    """SQLite's GLOB split at ``*`` into regexes; None matches nothing."""
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
            out.append(".")
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
                    if prior <= pattern[j + 1]:
                        members.append(
                            f"{re.escape(prior)}-{re.escape(pattern[j + 1])}"
                        )
                    prior = None
                    j += 2
                else:
                    members.append(re.escape(c))
                    prior = c
                    j += 1
            if j >= n:
                return None
            out.append(f"[{'^' if negate else ''}{''.join(members)}]")
            i = j + 1
        else:
            out.append(re.escape(c))
    parts.append(out)
    return ["".join(p) for p in parts]


@functools.lru_cache(maxsize=256)
def _glob_regex(pattern: str) -> "re.Pattern | None":
    """SQLite's GLOB as a regex; None matches nothing. ``(?>...)`` keeps the
    leftmost match of each part between stars, so matching stays linear."""
    parts = _glob_parts(pattern)
    if parts is None:
        return None
    head, *rest = parts
    if rest:
        *middle, tail = rest
        head += "".join(f"(?>.*?{m})" for m in middle) + f".*{tail}"
    return re.compile(head, re.DOTALL)


def glob_match(path: str, pattern: str) -> bool:
    """SQLite's GLOB."""
    regex = _glob_regex(pattern)
    return regex is not None and regex.fullmatch(path) is not None


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
        lst_uri_path = posixpath.dirname(path)
    else:
        if path:
            storage_uri, path = Client.parse_url(f"{uri.rstrip('/')}/")
        lst_uri_path = path

    lst_uri = f"{storage_uri}/{lst_uri_path.lstrip('/')}"
    ds_name = (
        f"{LISTING_PREFIX}{storage_uri}/{posixpath.join(lst_uri_path, '').lstrip('/')}"
    )

    ds_name = _sanitize_ds_name(ds_name)
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
    try:
        dataset = catalog.get_dataset(name, versions=None, include_incomplete=False)
    except DatasetNotFoundError:
        dataset = None
    if dataset is None or update or listing_dataset_expired(dataset):
        assert write is not None
        write()
        dataset = catalog.get_dataset(name, versions=None, include_incomplete=False)
    return dataset


def write_listing(
    uri: str,
    ds_name: str,
    session: "Session",
    client_config: dict,
    *,
    column: str = "file",
    file_type: type[File] = File,
    settings: dict | None = None,
    in_memory: bool = False,
) -> None:
    """Store the listing of the prefix ``uri``, whole, as the listing dataset
    ``ds_name``: the snapshot ``read_storage`` reads and reuses.
    """
    from datachain.query.session import Session

    from .dc.records import create_records_dataset

    namespace = session.catalog.metastore.system_namespace_name
    project = session.catalog.metastore.listing_project_name
    list_func = list_bucket(uri, session.catalog.cache, client_config=client_config)
    # content_hash=None because hash_callable doesn't capture list_func's
    # closure (which holds the URI and the client config) -- auto-hashing
    # the seed would let the UDF checkpoint cache return a stale listing
    # across URIs/runs.
    # The seed is needed only while writing this snapshot. Clean it up now rather
    # than keeping its catalog alive until the caller's (possibly global) exit.
    writer = Session("listing", catalog=session.catalog)
    try:
        (
            create_records_dataset(
                [{"seed": 0}],
                schema={"seed": int},
                content_hash=None,
                session=writer,
                settings=settings,
                in_memory=in_memory,
            )
            .settings(
                prefetch=0,
                batch_size=LISTING_BATCH_SIZE,
                namespace=namespace,
                project=project,
            )
            .gen(list_func, output={column: file_type})
            # for internal listing datasets, we always bump major version
            .save(ds_name, listing=True, update_version="major")
        )
    finally:
        writer._close()
