import os
from collections.abc import Sequence
from functools import reduce
from typing import TYPE_CHECKING

from datachain.client import Client
from datachain.lib.dc.storage_pattern import (
    apply_glob_filter,
    expand_brace_pattern,
    should_use_recursion,
    split_uri_pattern,
    validate_cloud_bucket_name,
)
from datachain.lib.file import FileType, get_file_type
from datachain.lib.listing import (
    LISTING_BATCH_SIZE,
    Versions,
    get_file_info,
    get_listing,
    list_bucket,
    ls,
)
from datachain.query import Session

if TYPE_CHECKING:
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


def _validate_versions(versions: Versions, delta: bool | None) -> None:
    if versions not in (True, False, None, "all"):
        raise ValueError(f"versions must be True, False or 'all', got {versions!r}")
    if versions == "all" and delta:
        raise ValueError("delta is not supported with versions='all'")


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
    versions: Versions = True,
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
    """Get data from storage(s) as a list of file with all file attributes.
    It returns the chain itself as usual.

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
        update: force storage reindexing. Default is False.
        anon: If True, we will treat cloud bucket as public one.
        versions: If False, list only current objects, without versions.
            If True, list every version and return the latest ones. If "all",
            return every version and a delete row for each delete the storage
            records (see Notes).
        client_config: Optional client configuration for the storage client.
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

    Notes:
        With ``versions="all"``, a delete row is a ``File`` whose
        ``is_deleted`` is True: ``last_modified`` is the time of the delete,
        ``version`` the id of the delete (the delete marker id on S3,
        ``deleted:<deleted version>`` on GCS and Azure), ``size`` 0, ``etag`` empty and
        ``is_latest`` False. Opening it raises an error.
        ``func.is_deleted("file")`` selects them in queries. A key is deleted
        now when none of its rows has ``is_latest`` True.

        - The history is what the storage still keeps: permanently deleted
          versions and expired delete markers leave no row.
        - Rows of one key can share ``last_modified`` (S3 keeps whole
          seconds, and a multipart upload is stamped with its start time),
          so ordering by time may not match the order of the writes.
        - Azure keeps neither past deletes nor the time of a delete: a key
          that is deleted now gets one delete row, with ``last_modified``
          set to 1970-01-01 (epoch 0).
        - S3 records a delete marker for every delete, also of a key that is
          already deleted or never existed; GCS records only the delete of a
          live object.
        - A URI that names a single object returns its current version only;
          one that names a key deleted now returns no rows. Its prefix lists
          the key's history.
        - ``delta`` is not supported.

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

        Every version and every delete of a versioned bucket:
        ```python
        history = dc.read_storage("s3://my-bucket/my-dir", versions="all")
        deleted = history.filter(dc.func.is_deleted("file"))
        ```

        With AWS S3-compatible storage:
        ```python
        dc.read_storage(
            "s3://my-bucket/my-dir",
            client_config = {"aws_endpoint_url": "<minio-endpoint-url>"}
        )
        ```
    """
    from .datasets import read_dataset
    from .records import create_records_dataset, read_records

    file_type = get_file_type(type)

    uris = uri if isinstance(uri, (list, tuple)) else [uri]

    if not uris:
        raise ValueError("No URIs provided")

    _validate_versions(versions, delta)

    for single_uri in uris:
        validate_cloud_bucket_name(str(single_uri))

    probe_config = client_config or (
        session.catalog.client_config if session is not None else None
    )

    if (
        anon is None
        and not _backends_have_credentials(uris, probe_config)
        and _all_buckets_anonymous(uris, probe_config)
    ):
        anon = True

    if anon is not None:
        client_config = (client_config or {}) | {"anon": anon}
    session = Session.get(session, client_config=client_config, in_memory=in_memory)
    catalog = session.catalog
    cache = catalog.cache
    client_config = session.catalog.client_config
    if anon is not None:
        # Session.get discards our client_config when an existing session is
        # passed. Re-apply anon locally for the listing path without mutating
        # the caller's session.
        client_config = client_config | {"anon": anon}
    if not versions:
        client_config = client_config | {"version_aware": False}
    with_deleted = versions == "all"
    listing_namespace_name = catalog.metastore.system_namespace_name
    listing_project_name = catalog.metastore.listing_project_name

    # Then expand all URIs that contain brace patterns
    expanded_uris = []
    for single_uri in uris:
        uri_str = str(single_uri)
        expanded_uris.extend(expand_brace_pattern(uri_str))

    # Now process each expanded URI
    chains = []
    listed_ds_name = set()
    file_values = []

    updated_uris = set()

    for single_uri in expanded_uris:
        # Check if URI contains glob patterns and split them
        base_uri, glob_pattern = split_uri_pattern(single_uri)

        # If a pattern is found, use the base_uri for listing
        # The pattern will be used for filtering later
        list_uri_to_use = base_uri if glob_pattern else single_uri

        # Avoid double updates for the same URI
        update_single_uri = False
        if update and (list_uri_to_use not in updated_uris):
            updated_uris.add(list_uri_to_use)
            update_single_uri = True

        list_ds_name, list_uri, list_path, _ = get_listing(
            list_uri_to_use, session, update=update_single_uri, versions=versions
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
        dc._query.update = update
        dc.signals_schema = dc.signals_schema.mutate({f"{column}": file_type})

        def lst_fn(ds_name, lst_uri):
            # Seed for .gen() iteration. content_hash=None because hash_callable
            # doesn't capture list_func's closure (which holds `lst_uri`, `cache`,
            # `client_config`) -- auto-hashing the seed would let the UDF
            # checkpoint cache return a stale listing across URIs/runs.
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
                    namespace=listing_namespace_name,
                    project=listing_project_name,
                )
                .gen(
                    list_bucket(
                        lst_uri,
                        cache,
                        client_config=client_config,
                        deleted=with_deleted,
                    ),
                    output={f"{column}": file_type},
                )
                # for internal listing datasets, we always bump major version
                .save(ds_name, listing=True, update_version="major")
            )

        # Always attach listing_fn so resolve_listing can refresh stale listings.
        dc._query.set_listing_fn(
            lambda ds_name=list_ds_name, lst_uri=list_uri: lst_fn(ds_name, lst_uri)
        )

        # If a glob pattern was detected, use it for filtering
        # Otherwise, use the original list_path from get_listing
        if glob_pattern:
            # Determine if we should use recursive listing based on the pattern
            use_recursive = should_use_recursion(glob_pattern, recursive or False)

            # Apply glob filter - no need for brace expansion here as it's done above
            chain = apply_glob_filter(
                dc, glob_pattern, list_path, use_recursive, column, not with_deleted
            )
            chains.append(chain)
        else:
            # No glob pattern detected, use normal ls behavior
            chains.append(
                ls(
                    dc,
                    list_path,
                    recursive=recursive,
                    column=column,
                    latest=not with_deleted,
                )
            )

        listed_ds_name.add(list_ds_name)

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
