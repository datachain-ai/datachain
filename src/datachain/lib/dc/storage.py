import itertools
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING
from uuid import uuid4

from datachain.client import Client
from datachain.lib.dc.storage_pattern import (
    list_scope,
    path_filter,
    storage_sources,
    validate_cloud_bucket_name,
)
from datachain.lib.file import File, FileRow, FileType, get_file_type
from datachain.lib.listing import (
    LISTING_BATCH_SIZE,
    PathFilter,
    describe_listing,
    file_row,
    get_file_info,
    get_listing,
    list_bucket,
    list_pages,
    list_scope_pages,
    listing_uri_from_name,
    select_paths,
)
from datachain.query import Session

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    from datachain.catalog import Catalog
    from datachain.lib.listing import ListScope

    from .datachain import DataChain


def _backends_have_credentials(uris, client_config: dict | None) -> bool:
    """True if any backend behind ``uris`` sees credentials in client_config."""
    if not client_config:
        return False
    seen: set[type[Client]] = set()
    for uri in uris:
        try:
            seen.add(Client.get_implementation(str(uri)))
        except NotImplementedError:
            return True
    return any(c.has_explicit_credentials(client_config) for c in seen)


def _all_buckets_anonymous(uris, client_config: dict | None) -> bool:
    """Probe each unique bucket; True iff all probe as anonymous."""
    to_probe: set[tuple[type[Client], str]] = set()
    for uri in uris:
        try:
            client_cls = Client.get_implementation(str(uri))
        except NotImplementedError:
            return False
        name, _ = client_cls.split_url(str(uri))
        if not name:
            return False
        to_probe.add((client_cls, name))

    if not to_probe:
        return False

    for client_cls, name in to_probe:
        try:
            status = client_cls.bucket_status(name, **(client_config or {}))
        except NotImplementedError:
            # Backend doesn't support bucket_status (e.g. local files).
            return False
        if status.access != "anonymous":
            return False
    return True


def read_storage(  # noqa: PLR0913
    uri: str | os.PathLike[str] | list[str] | list[os.PathLike[str]],
    *,
    type: FileType = "binary",
    session: Session | None = None,
    settings: dict | None = None,
    in_memory: bool = False,
    recursive: bool | None = True,
    column: str = "file",
    update: bool = False,
    anon: bool | None = None,
    versions: bool = True,
    delta: bool | None = False,
    delta_on: str | Sequence[str] | None = (
        "file.path",
        "file.etag",
        "file.version",
    ),
    delta_result_on: str | Sequence[str] | None = None,
    delta_compare: str | Sequence[str] | None = None,
    delta_retry: bool | str | None = None,
    delta_unsafe: bool = False,
    client_config: dict | None = None,
) -> "DataChain":
    """A chain of the files of storage(s), over a saved snapshot of a listing.

    A saved listing of the URI, or of a prefix above it, younger than 4 hours
    is reused; otherwise the storage is listed and saved, as
    `dc.list_storage(uri).save()` does. The chain holds current objects. To list a
    storage without saving it, use `dc.list_storage()`.

    Parameters:
        uri: Storage path(s) or URI(s). Can be a local path or start with a
            storage prefix like `s3://`, `gs://`, `az://`, `hf://` or "file:///".
            Supports glob patterns:
              - `*` : wildcard
              - `**` : recursive wildcard
              - `?` : single character
              - `{a,b}` : brace expansion list
              - `{1..9}` : brace numeric or alphabetic range
        type: read file as "binary", "text", or "image" data. Default is "binary".
        recursive: search recursively for the given path.
        column: Column name that will contain File objects. Default is "file".
        update: list the storage again instead of reusing a saved listing.
            Default is False.
        anon: If True, we will treat cloud bucket as public one.
        versions: The chain holds current files either way: old versions are
            not included. If True, the default, on a versioned bucket each file
            carries its version id, and the saved listing keeps old versions
            too. If False, no version ids, and the listing is faster.
        client_config: Optional client configuration for the storage client.
            Replaces the session's.
        delta: If True, only process new or changed files instead of reprocessing
            everything. This saves time by skipping files that were already processed in
            previous versions. The optimization is working when a new version of the
            dataset is created.
            Default is False.
        delta_on: Field(s) that uniquely identify each record in the source data.
            Used to detect which records are new or changed.
            Default is ("file.path", "file.etag", "file.version").
        delta_result_on: Field(s) in the result dataset that match `delta_on` fields.
            Only needed if you rename the identifying fields during processing.
            Default is None.
        delta_compare: Field(s) used to detect if a record has changed.
            If not specified, all fields except `delta_on` fields are used.
            Default is None.
        delta_retry: Controls retry behavior for failed records:
            - String (field name): Reprocess records where this field is not empty
              (error mode)
            - True: Reprocess records missing from the result dataset (missing mode)
            - None: No retry processing (default)
        delta_unsafe: Allow restricted ops in delta: merge, union, subtract,
            diff, file_diff, agg, group_by, distinct. When multiple delta
            sources participate in one composed query, this must be enabled on
            every participating delta source. Caller must ensure datasets are
            consistent and not partially updated.

    Returns:
        DataChain: A DataChain object containing the file information.

    Examples:
        Simple call from s3:
        ```python
        import datachain as dc
        dc.read_storage("s3://my-bucket/my-dir")
        ```

        Match all .json files recursively using glob pattern
        ```py
        dc.read_storage("gs://bucket/meta/**/*.json")
        ```

        Match image file extensions for directories with pattern
        ```py
        dc.read_storage("s3://bucket/202?/**/*.{jpg,jpeg,png}")
        ```

        By ranges in filenames:
        ```py
        dc.read_storage("s3://bucket/202{1..4}/**/*.{jpg,jpeg,png}")
        ```

        Multiple URIs:
        ```python
        dc.read_storage(["s3://my-bkt/dir1", "s3://bucket2/dir2/dir3"])
        ```

        Current objects only, without the history of a versioned bucket:
        ```python
        dc.read_storage("s3://my-bucket/my-dir", versions=False)
        ```

        With AWS S3-compatible storage:
        ```python
        dc.read_storage(
            "s3://my-bucket/my-dir",
            client_config = {"aws_endpoint_url": "<minio-endpoint-url>"}
        )
        ```
    """
    uris = uri if isinstance(uri, (list, tuple)) else [uri]
    if not uris:
        raise ValueError("No URIs provided")
    session_config, client_config = _client_configs(
        uris, session, anon, client_config, versions
    )
    session = _session_for(session, session_config, in_memory)
    return _read_storage(
        uris,
        session,
        client_config,
        type=type,
        settings=settings,
        in_memory=in_memory,
        recursive=recursive,
        column=column,
        update=update,
        versions=versions,
        delta=delta,
        delta_on=delta_on,
        delta_result_on=delta_result_on,
        delta_compare=delta_compare,
        delta_retry=delta_retry,
        delta_unsafe=delta_unsafe,
    )


def _client_configs(
    uris: Sequence,
    session: Session | None,
    anon: bool | None,
    client_config: dict | None,
    versions: bool,
) -> tuple[dict | None, dict]:
    """The client config a session is made with, if not the active one's, and
    the client config to list ``uris`` with.

    ``client_config`` if given, else the config of ``session`` or the active
    session. ``anon`` is set
    on it, True when None and every bucket is public and has no credentials,
    and ``version_aware=False`` without ``versions``. An explicit session is
    never changed.
    """
    for single_uri in uris:
        validate_cloud_bucket_name(str(single_uri))
    config = _session_client_config(session, client_config)
    if (
        anon is None
        and not _backends_have_credentials(uris, config)
        and _all_buckets_anonymous(uris, config)
    ):
        anon = True
    if anon is not None:
        config = config | {"anon": anon}
    session_config = config if client_config or anon is not None else None
    if not versions:
        config = config | {"version_aware": False}
    return session_config, config


def _session_for(
    session: Session | None, client_config: dict | None, in_memory: bool
) -> Session:
    """``Session.get(session)`` with ``client_config``, if any: an explicit
    session is never changed, so a session over a copy of its catalog.
    """
    session = Session.get(session, client_config=client_config, in_memory=in_memory)
    if client_config is None or session.catalog.client_config == client_config:
        return session
    catalog = session.catalog.copy()
    catalog.client_config = client_config
    return Session("session" + uuid4().hex[:4], catalog, in_memory=in_memory)


def _session_client_config(session: Session | None, client_config: dict | None) -> dict:
    """``client_config`` if given, else the session's, without creating one."""
    if client_config:
        return client_config
    session = session or Session.active()
    return session.catalog.client_config if session else {}


def _read_storage(
    uris: Sequence,
    session: Session,
    client_config: dict,
    *,
    type: FileType = "binary",
    settings: dict | None = None,
    in_memory: bool = False,
    recursive: bool | None = True,
    column: str = "file",
    update: bool = False,
    versions: bool = True,
    delta: bool | None = False,
    delta_on: str | Sequence[str] | None = None,
    delta_result_on: str | Sequence[str] | None = None,
    delta_compare: str | Sequence[str] | None = None,
    delta_retry: bool | str | None = None,
    delta_unsafe: bool = False,
) -> "DataChain":
    from .datasets import read_dataset
    from .records import read_records

    file_type = get_file_type(type)
    catalog = session.catalog
    cache = catalog.cache
    listing_namespace_name = catalog.metastore.system_namespace_name
    listing_project_name = catalog.metastore.listing_project_name

    chains = []
    file_values = []
    updated_listings: set[str] = set()

    for list_uri_to_use, glob_pattern in storage_sources(uris):
        list_ds_name, list_uri, list_path, _ = get_listing(
            list_uri_to_use, session, update=update, versions=versions
        )

        # list_ds_name is None if object is a file, we don't want to use cache
        # or do listing in that case - just read that single object
        if not list_ds_name:
            file_values.append(
                get_file_info(list_uri, cache, client_config=client_config)
            )
            continue

        dc = read_dataset(
            list_ds_name,
            namespace=listing_namespace_name,
            project=listing_project_name,
            session=session,
            settings=settings,
            delta=delta,
            delta_on=delta_on,
            delta_result_on=delta_result_on,
            delta_compare=delta_compare,
            delta_retry=delta_retry,
            delta_unsafe=delta_unsafe,
        )
        # a listing shared by several URIs is refreshed once, so all read one snapshot
        dc._query.update = update and list_ds_name not in updated_listings
        updated_listings.add(list_ds_name)
        dc.signals_schema = dc.signals_schema.mutate({f"{column}": file_type})

        # Always attach listing_fn so resolve_listing can refresh stale listings;
        # a reused snapshot may cover a parent of list_uri: refresh all of it
        dc._query.set_listing_fn(
            lambda ds_name=list_ds_name: _write_listing(
                listing_uri_from_name(ds_name),
                ds_name,
                session,
                client_config,
                column=column,
                file_type=file_type,
                settings=settings,
                in_memory=in_memory,
            )
        )

        chains.append(
            select_paths(dc, path_filter(list_path, glob_pattern, recursive), column)
        )

    storage_chain = None if not chains else reduce(lambda x, y: x.union(y), chains)

    if file_values:
        # Use read_records directly (not read_values) so the chain hash is
        # derived from the flattened File records (deterministic across runs) —
        # needed for checkpoint reuse on single-file read_storage /
        # read_csv / read_parquet.
        file_chain = read_records(
            [{"file": f} for f in file_values],
            schema={"file": file_type},
            session=session,
            settings=settings,
            in_memory=in_memory,
        )
        file_chain.signals_schema = file_chain.signals_schema.mutate(
            {f"{column}": file_type}
        )
        storage_chain = storage_chain.union(file_chain) if storage_chain else file_chain

    assert storage_chain is not None

    return storage_chain


def list_storage(
    uri: str | os.PathLike[str] | list[str] | list[os.PathLike[str]],
    *,
    recursive: bool | None = True,
    versions: bool = True,
    anon: bool | None = None,
    client_config: dict | None = None,
    session: Session | None = None,
    in_memory: bool = False,
) -> "StorageListing":
    """List the files in a storage and get them as rows in Python.

    Unlike `read_storage`, this returns no chain and saves nothing: it lists
    the storage each time you use the result. Use it to count, sum or check
    files; use `read_storage` to process them.

    Each row is a `dc.FileRow` with the columns of `File` (`path`, `size`,
    `etag`, `last_modified`, ...) and its path helpers (`name`, `parent`,
    `get_file_ext()`). Rows don't read files: `to_objects()` gives `File`
    objects that do.

    Parameters:
        uri: Storage URI(s), as in `read_storage`: buckets, folders or files,
            with glob patterns (`*`, `**`, `?`, `{a,b}`, `{1..9}`).
        recursive: If False, list only the files directly in the folder.
        versions: As in `read_storage`, the rows are current files. If True,
            the default, on a versioned bucket they carry their version id. If
            False, they don't, and the listing is faster.
        anon: If True, read the bucket as public, without credentials. If
            None, the default, a public bucket is detected.
        client_config: Client configuration, such as credentials or an
            endpoint. Replaces the session's.
        session: The session to take the client configuration from and to
            save into with `save()`.
        in_memory: Make the session that `save()` creates, if any, in memory.

    Returns:
        StorageListing: The files of `uri`.

    Example:
        ```py
        import datachain as dc

        uri = "s3://my-bucket/dir/"
        total = sum(f.size for f in dc.list_storage(uri))
        jpgs = [f.path for f in dc.list_storage(f"{uri}*.jpg")]
        # ImageFile objects, which read the images
        images = dc.list_storage(uri).to_objects(type="image")
        table = dc.list_storage(uri).to_arrow()
        ```
    """
    uris = [os.fspath(u) for u in (uri if isinstance(uri, (list, tuple)) else [uri])]
    if not uris:
        raise ValueError("No URIs provided")
    session_config, config = _client_configs(
        uris, session, anon, client_config, versions
    )
    return StorageListing(
        uris,
        recursive=recursive,
        versions=versions,
        client_config=config,
        session=session,
        session_config=session_config,
        in_memory=in_memory,
    )


@dataclass(frozen=True)
class _Object:
    """A URI of a single object: read its info, not a listing."""

    uri: str


@dataclass(frozen=True)
class _Scan:
    """One read of the listing of ``uri``: of the part ``scope`` holds, or all
    of it if None. A row comes once per filter of ``filters`` it passes;
    ``PathFilter()`` passes every row.
    """

    uri: str
    filters: "tuple[PathFilter, ...]"
    scope: "ListScope | None" = None


class StorageListing:
    """The objects of storage URIs, from `dc.list_storage()`: iterate over it
    for `dc.FileRow` rows, or call `to_list()`, `to_objects()`, `to_arrow()`,
    `to_pandas()` or `save()`. Each of them lists the storage again.
    """

    def __init__(
        self,
        uris: list[str],
        *,
        recursive: bool | None = True,
        versions: bool,
        client_config: dict,
        session: Session | None = None,
        session_config: dict | None = None,
        in_memory: bool = False,
    ) -> None:
        """Use `dc.list_storage()` to create a listing."""
        self.uris = uris
        self.recursive = recursive
        self.versions = versions
        self.client_config = client_config
        self._session = session
        self._session_config = session_config
        self._save_session: Session | None = None
        self._in_memory = in_memory

    @property
    def session(self) -> Session:
        """The session `save()` writes with, created on first use."""
        if self._save_session is None:
            self._save_session = _session_for(
                self._session, self._session_config, self._in_memory
            )
        return self._save_session

    def __repr__(self) -> str:  # noqa: D105
        return f"StorageListing({self.uris!r}, versions={self.versions})"

    def __iter__(self) -> Iterator[FileRow]:
        """List the storage, yielding `dc.FileRow` rows."""
        return itertools.chain.from_iterable(self._pages())  # type: ignore[arg-type]

    def _pages(self) -> Iterator[Sequence[tuple]]:
        session = self._session or Session.active()
        cache = session.catalog.cache if session else None
        for source in self._sources(cache):
            if isinstance(source, _Object):
                info = get_file_info(source.uri, cache, self.client_config)
                yield [file_row(info)]
                continue
            if source.scope is None:
                pages = list_pages(source.uri, cache, self.client_config)
            else:
                pages = list_scope_pages(
                    source.uri, source.scope, cache, self.client_config
                )
            # current objects, as read_storage's chain: a listing with versions
            # holds old ones too, listed for the version ids of current ones
            latest = self.versions
            if source.filters == (PathFilter(),) and not latest:
                yield from pages
                continue
            matchers = [f.matcher() for f in source.filters]
            for page in pages:
                yield [
                    r
                    for r in page
                    if not latest or r[_IS_LATEST]
                    for match in matchers
                    if match(r[_PATH])
                ]

    def _sources(self, cache) -> "list[_Object | _Scan]":
        """What to list: one scan per part of a listing that a client can list
        alone (S3, GCS, Azure), or one whole scan when a URI needs all of it.
        URIs of single objects come last, as ``read_storage`` unions them.
        """
        from datachain.client.fsspec import RangeListedClient

        scans: dict[str, dict[ListScope | None, list[PathFilter]]] = {}
        objects: list[_Object | _Scan] = []
        described: dict[str, tuple[str, str, str] | None] = {}
        for uri, pattern in storage_sources(self.uris):
            client = Client.get_client(uri, cache, **self.client_config)  # type: ignore[arg-type]
            if uri not in described:  # braces expand to URIs with one base
                described[uri] = describe_listing(client, uri, self.versions)
            if (listing := described[uri]) is None:
                objects.append(_Object(uri))
                continue
            _, list_uri, list_path = listing
            # the listing of a prefix holds only paths under it, so a recursive
            # filter by that same prefix selects them all
            whole = list_path == Client.parse_url(list_uri)[1]
            paths = (
                PathFilter()
                if pattern is None and self.recursive and whole
                else path_filter(list_path, pattern, self.recursive)
            )
            scope = (
                list_scope(list_path, pattern, self.recursive)
                if isinstance(client, RangeListedClient)
                else None
            )
            scans.setdefault(list_uri, {}).setdefault(scope, []).append(paths)
        sources: list[_Object | _Scan] = []
        for list_uri, by_scope in scans.items():
            if None in by_scope:  # one of them needs it whole: read it once
                by_scope = {None: [f for fs in by_scope.values() for f in fs]}
            sources += [_Scan(list_uri, tuple(fs), s) for s, fs in by_scope.items()]
        return sources + objects

    def to_list(self) -> list[FileRow]:
        """List the storage into a list of `dc.FileRow` rows."""
        return list(self)

    def to_objects(self, type: FileType = "binary") -> list[File]:
        """List the storage into `File` objects that read files with this
        listing's client config.

        Parameters:
            type: "binary", "text", "image", "video" or "audio", as in
                `read_storage`, for `File`, `TextFile`, `ImageFile`, ...
        """
        file_type = get_file_type(type)
        catalog = self._file_catalog()
        files = []
        for row in self:
            file = file_type(**row._asdict())
            file._set_stream(catalog)
            files.append(file)
        return files

    def _file_catalog(self) -> "Catalog":
        from datachain.catalog import Catalog

        session = self._session or Session.active()
        if session is None:
            catalog = Catalog(None, None)  # type: ignore[arg-type]
        else:
            catalog = session.catalog.copy(db=False)
        catalog.client_config = self.client_config
        return catalog

    def to_arrow(self) -> "pa.Table":
        """List the storage into a `pyarrow.Table` with `File`'s columns.

        `location` holds JSON text.
        """
        import pyarrow as pa

        schema = _arrow_schema()
        batches = (_record_batch(page, schema) for page in self._pages() if page)
        return pa.Table.from_batches(batches, schema=schema)

    def to_pandas(self) -> "pd.DataFrame":
        """List the storage into a `pandas.DataFrame` with `File`'s columns."""
        return self.to_arrow().to_pandas()

    def save(self, settings: dict | None = None) -> "DataChain":
        """List the storage again and save it, as `read_storage(uri,
        update=True)` does, for later `read_storage(uri)` calls to reuse.

        Parameters:
            settings: Keyword arguments of `DataChain.settings()` for the job
                that lists and saves, such as `{"cache": True}`.

        Returns:
            DataChain: The chain `read_storage(uri)` returns.
        """
        chain = _read_storage(
            self.uris,
            self.session,
            self.client_config,
            recursive=self.recursive,
            settings=settings,
            in_memory=self._in_memory,
            update=True,
            versions=self.versions,
        )
        chain._query.resolve_all_listings()
        return chain


def _write_listing(
    uri: str,
    ds_name: str,
    session: Session,
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
    from .records import create_records_dataset

    namespace = session.catalog.metastore.system_namespace_name
    project = session.catalog.metastore.listing_project_name
    list_func = list_bucket(uri, session.catalog.cache, client_config=client_config)
    # content_hash=None because hash_callable doesn't capture list_func's
    # closure (which holds the URI and the client config) -- auto-hashing
    # the seed would let the UDF checkpoint cache return a stale listing
    # across URIs/runs.
    (
        create_records_dataset(
            [{"seed": 0}],
            schema={"seed": int},
            content_hash=None,
            session=session,
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


_PATH = FileRow._fields.index("path")
_IS_LATEST = FileRow._fields.index("is_latest")


def _arrow_schema() -> "pa.Schema":
    import pyarrow as pa

    types = {
        "source": pa.string(),
        "path": pa.string(),
        "size": pa.int64(),
        "version": pa.string(),
        "etag": pa.string(),
        "is_latest": pa.bool_(),
        "last_modified": pa.timestamp("us", tz="UTC"),
        "location": pa.string(),
    }
    return pa.schema([(name, types[name]) for name in FileRow._fields])


def _record_batch(page: Sequence[tuple], schema: "pa.Schema") -> "pa.RecordBatch":
    import pyarrow as pa

    columns = list(zip(*page, strict=True))
    loc = schema.get_field_index("location")
    columns[loc] = tuple(None if v is None else json.dumps(v) for v in columns[loc])
    return pa.RecordBatch.from_arrays(
        [pa.array(col, type=t) for col, t in zip(columns, schema.types, strict=True)],
        schema=schema,
    )
