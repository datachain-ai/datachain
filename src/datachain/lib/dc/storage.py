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
from datachain.lib.listing import get_file_info, get_listing, list_bucket, ls
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


def _prepare_uris(uri: str | os.PathLike | list | tuple) -> list[str]:
    uris = uri if isinstance(uri, (list, tuple)) else [uri]
    if not uris:
        raise ValueError("No URIs provided")
    for single_uri in uris:
        validate_cloud_bucket_name(str(single_uri))
    return uris


def _resolve_session_and_config(uris, session, client_config, anon, in_memory):
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
        client_config = client_config | {"anon": anon}
    listing_namespace_name = catalog.metastore.system_namespace_name
    listing_project_name = catalog.metastore.listing_project_name
    return session, catalog, cache, client_config, listing_namespace_name, listing_project_name


def _expand_brace_patterns(uris):
    expanded_uris = []
    for single_uri in uris:
        uri_str = str(single_uri)
        expanded_uris.extend(expand_brace_pattern(uri_str))
    return expanded_uris


def _process_expanded_uri(
    single_uri,
    session,
    catalog,
    cache,
    client_config,
    listing_namespace_name,
    listing_project_name,
    file_type,
    column,
    update,
    recursive,
    settings,
    in_memory,
    delta,
    delta_on,
    delta_result_on,
    delta_compare,
    delta_retry,
    delta_unsafe,
    updated_uris,
):
    from .datasets import read_dataset
    from .records import create_records_dataset

    base_uri, glob_pattern = split_uri_pattern(single_uri)
    list_uri_to_use = base_uri if glob_pattern else single_uri

    update_single_uri = False
    if update and (list_uri_to_use not in updated_uris):
        updated_uris.add(list_uri_to_use)
        update_single_uri = True

    list_ds_name, list_uri, list_path, _ = get_listing(
        list_uri_to_use, session, update=update_single_uri
    )

    if not list_ds_name:
        return None, get_file_info(list_uri, cache, client_config=client_config)

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
                namespace=listing_namespace_name,
                project=listing_project_name,
            )
            .gen(
                list_bucket(lst_uri, cache, client_config=client_config),
                output={f"{column}": file_type},
            )
            .save(ds_name, listing=True, update_version="major")
        )

    dc._query.set_listing_fn(
        lambda ds_name=list_ds_name, lst_uri=list_uri: lst_fn(ds_name, lst_uri)
    )

    if glob_pattern:
        use_recursive = should_use_recursion(glob_pattern, recursive or False)
        return apply_glob_filter(dc, glob_pattern, list_path, use_recursive, column), None
    return ls(dc, list_path, recursive=recursive, column=column), None


def read_storage(
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

        With AWS S3-compatible storage:
        ```python
        dc.read_storage(
            "s3://my-bucket/my-dir",
            client_config = {"aws_endpoint_url": "<minio-endpoint-url>"}
        )
        ```
    """
    from .records import read_records

    file_type = get_file_type(type)
    uris = _prepare_uris(uri)

    session, catalog, cache, client_config, listing_ns, listing_proj = (
        _resolve_session_and_config(uris, session, client_config, anon, in_memory)
    )

    expanded_uris = _expand_brace_patterns(uris)

    chains = []
    file_values = []
    updated_uris = set()

    for single_uri in expanded_uris:
        chain, file_value = _process_expanded_uri(
            single_uri, session, catalog, cache, client_config,
            listing_ns, listing_proj, file_type, column,
            update, recursive, settings, in_memory,
            delta, delta_on, delta_result_on, delta_compare,
            delta_retry, delta_unsafe, updated_uris,
        )
        if chain is not None:
            chains.append(chain)
        if file_value is not None:
            file_values.append(file_value)

    storage_chain = None if not chains else reduce(lambda x, y: x.union(y), chains)

    if file_values:
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
