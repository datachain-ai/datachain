import io
import logging
import os
import os.path
import posixpath
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from copy import copy
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy import Column
from tqdm.auto import tqdm

from datachain.cache import Cache
from datachain.client import Client
from datachain.dataset import (
    DATASET_PREFIX,
    DEFAULT_DATASET_VERSION,
    QUERY_DATASET_PREFIX,
    DatasetDependency,
    DatasetListRecord,
    DatasetRecord,
    DatasetStatus,
    StorageURI,
    create_dataset_uri,
    parse_dataset_name,
    parse_dataset_uri,
    parse_dataset_with_version,
    parse_schema,
)
from datachain.error import (
    DataChainError,
    DatasetInvalidVersionError,
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
    NamespaceNotFoundError,
    ProjectNotFoundError,
)
from datachain.lib.listing import get_listing
from datachain.node import DirType, Node, NodeWithPath
from datachain.nodes_thread_pool import NodesThreadPool
from datachain.project import Project
from datachain.sql.types import DateTime, SQLType
from datachain.utils import DataChainDir, DatasetIdentifier

from .datasource import DataSource
from .dependency import build_dependency_hierarchy, populate_nested_dependencies

if TYPE_CHECKING:
    from datachain.data_storage import AbstractMetastore, AbstractWarehouse
    from datachain.dataset import DatasetListVersion
    from datachain.job import Job
    from datachain.lib.listing_info import ListingInfo
    from datachain.listing import Listing

logger = logging.getLogger("datachain")

DEFAULT_DATASET_DIR = "dataset"

TTL_INT = 4 * 60 * 60

INDEX_INTERNAL_ERROR_MESSAGE = "Internal error on indexing"
DATASET_INTERNAL_ERROR_MESSAGE = "Internal error on creating dataset"
# exit code we use if query script was canceled
QUERY_SCRIPT_CANCELED_EXIT_CODE = 11
QUERY_SCRIPT_SIGTERM_EXIT_CODE = -15  # if query script was terminated by SIGTERM

# dataset pull
PULL_DATASET_MAX_THREADS = 5
PULL_DATASET_CHUNK_TIMEOUT = 3600
PULL_DATASET_SLEEP_INTERVAL = 0.1  # sleep time while waiting for chunk to be available
PULL_DATASET_CHECK_STATUS_INTERVAL = 20  # interval to check export status in Studio


def is_namespace_local(namespace_name) -> bool:
    """Checks if namespace is from local environment, i.e. is `local`"""
    return namespace_name == "local"


class DatasetRowsFetcher(NodesThreadPool):
    def __init__(
        self,
        metastore: "AbstractMetastore",
        warehouse: "AbstractWarehouse",
        remote_ds: DatasetRecord,
        remote_ds_version: str,
        export_id: int,
        local_ds: DatasetRecord,
        local_ds_version: str,
        schema: dict[str, SQLType | type[SQLType]],
        max_threads: int = PULL_DATASET_MAX_THREADS,
        progress_bar=None,
    ):
        from datachain.remote.studio import StudioClient

        super().__init__(max_threads)
        self._check_dependencies()
        self.metastore = metastore
        self.warehouse = warehouse
        self.remote_ds = remote_ds
        self.remote_ds_version = remote_ds_version
        self.export_id = export_id
        self.local_ds = local_ds
        self.local_ds_version = local_ds_version
        self.schema = schema
        self.last_status_check: float | None = None
        self.studio_client = StudioClient()
        self.progress_bar = progress_bar
        self._last_export_status: str | None = None
        self._last_export_files_done: int | None = None
        self._last_export_num_files: int | None = None

    def done_task(self, done):
        for task in done:
            task.result()

    def _check_dependencies(self) -> None:
        try:
            import lz4.frame  # noqa: F401
            import numpy as np  # noqa: F401
            import pandas as pd  # noqa: F401
            import pyarrow as pa  # noqa: F401
        except ImportError as exc:
            raise Exception(
                f"Missing dependency: {exc.name}\n"
                "To install run:\n"
                "\tpip install 'datachain[remote]'"
            ) from None

    def should_check_for_status(self) -> bool:
        if not self.last_status_check:
            return True
        return time.time() - self.last_status_check > PULL_DATASET_CHECK_STATUS_INTERVAL

    def check_for_status(self) -> None:
        """
        Method that checks export status in Studio and raises Exception if export
        failed or was removed.
        Checks are done every PULL_DATASET_CHECK_STATUS_INTERVAL seconds
        """
        export_status_response = self.studio_client.dataset_export_status(
            self.export_id
        )
        if not export_status_response.ok:
            raise DataChainError(export_status_response.message)

        data = export_status_response.data
        export_status = data["status"]

        # Surface Studio-side export progress (if available) while we're pulling.
        files_done = data.get("files_done")
        num_files = data.get("num_files")

        if (
            files_done is not None
            and num_files is not None
            and (
                export_status != self._last_export_status
                or files_done != self._last_export_files_done
                or num_files != self._last_export_num_files
            )
        ):
            self._last_export_status = export_status
            self._last_export_files_done = files_done
            self._last_export_num_files = num_files

            if self.progress_bar is not None:
                # Keep the main bar semantics (rows) and just add a postfix.
                self.progress_bar.set_postfix_str(
                    f"studio_export={files_done}/{num_files} ({export_status})"
                )

        if export_status == "failed":
            raise DataChainError("Dataset export failed in Studio")
        if export_status == "removed":
            raise DataChainError("Dataset export removed in Studio")

        self.last_status_check = time.time()

    def fix_columns(self, df) -> None:
        import pandas as pd

        """
        Method that does various column decoding or parsing, depending on a type
        before inserting into DB
        """
        # we get dataframe from parquet export files where datetimes are serialized
        # as timestamps so we need to parse it back to datetime objects
        for c in [c for c, t in self.schema.items() if t == DateTime]:
            df[c] = pd.to_datetime(df[c], unit="s")

        # id will be autogenerated in DB
        return df.drop("sys__id", axis=1)

    def get_parquet_content(self, url: str):
        import requests

        while True:
            if self.should_check_for_status():
                self.check_for_status()
            r = requests.get(url, timeout=PULL_DATASET_CHUNK_TIMEOUT)
            if r.status_code == 404:
                time.sleep(PULL_DATASET_SLEEP_INTERVAL)
                continue
            r.raise_for_status()
            return r.content

    def do_task(self, urls):
        import lz4.frame
        import pandas as pd

        # metastore and warehouse are not thread safe
        with self.warehouse.clone() as warehouse:
            urls = list(urls)

            for url in urls:
                if self.should_check_for_status():
                    self.check_for_status()

                df = pd.read_parquet(
                    io.BytesIO(lz4.frame.decompress(self.get_parquet_content(url)))
                )
                df = self.fix_columns(df)

                inserted = warehouse.insert_dataset_rows(
                    df, self.local_ds, self.local_ds_version
                )
                self.increase_counter(inserted)  # type: ignore [arg-type]
                # sometimes progress bar doesn't get updated so manually updating it
                self.update_progress_bar(self.progress_bar)


@dataclass
class NodeGroup:
    """Class for a group of nodes from the same source"""

    listing: "Listing | None"
    client: Client
    sources: list[DataSource]

    # The source path within the bucket
    # (not including the bucket name or s3:// prefix)
    source_path: str = ""
    dataset_name: str | None = None
    dataset_version: str | None = None
    instantiated_nodes: list[NodeWithPath] | None = None

    @property
    def is_dataset(self) -> bool:
        return bool(self.dataset_name)

    def iternodes(self, recursive: bool = False):
        for src in self.sources:
            if recursive and src.is_container():
                for nwp in src.find():
                    yield nwp.n
            else:
                yield src.node

    def download(self, recursive: bool = False, pbar=None) -> None:
        """
        Download this node group to cache.
        """
        if self.sources:
            self.client.fetch_nodes(self.iternodes(recursive), shared_progress_bar=pbar)

    def close(self) -> None:
        if self.listing:
            self.listing.close()

    def __enter__(self) -> "NodeGroup":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def prepare_output_for_cp(
    node_groups: list[NodeGroup],
    output: str,
    force: bool = False,
    no_cp: bool = False,
) -> tuple[bool, str | None]:
    total_node_count = 0
    for node_group in node_groups:
        if not node_group.sources:
            raise FileNotFoundError(
                f"No such file or directory: {node_group.source_path}"
            )
        total_node_count += len(node_group.sources)

    always_copy_dir_contents = False
    copy_to_filename = None

    if no_cp:
        return always_copy_dir_contents, copy_to_filename

    if not os.path.isdir(output):
        if all(n.is_dataset for n in node_groups):
            os.mkdir(output)
        elif total_node_count == 1:
            first_source = node_groups[0].sources[0]
            if first_source.is_container():
                if os.path.exists(output):
                    if force:
                        os.remove(output)
                    else:
                        raise FileExistsError(f"Path already exists: {output}")
                always_copy_dir_contents = True
                os.mkdir(output)
            else:  # Is a File
                if os.path.exists(output):
                    if force:
                        os.remove(output)
                    else:
                        raise FileExistsError(f"Path already exists: {output}")
                copy_to_filename = output
        else:
            raise FileNotFoundError(f"Is not a directory: {output}")
    return always_copy_dir_contents, copy_to_filename


def collect_nodes_for_cp(
    node_groups: Iterable[NodeGroup],
    recursive: bool = False,
) -> tuple[int, int]:
    total_size: int = 0
    total_files: int = 0

    # Collect all sources to process
    for node_group in node_groups:
        listing: Listing | None = node_group.listing
        valid_sources: list[DataSource] = []
        for dsrc in node_group.sources:
            if dsrc.is_single_object():
                total_size += dsrc.node.size
                total_files += 1
                valid_sources.append(dsrc)
            else:
                assert listing
                node = dsrc.node
                if not recursive:
                    print(f"{node.full_path} is a directory (not copied).")
                    continue
                add_size, add_files = listing.du(node, count_files=True)
                total_size += add_size
                total_files += add_files
                valid_sources.append(dsrc)

        node_group.sources = valid_sources

    return total_size, total_files


def get_download_bar(bar_format: str, total_size: int):
    return tqdm(
        desc="Downloading files: ",
        unit="B",
        bar_format=bar_format,
        unit_scale=True,
        unit_divisor=1000,
        total=total_size,
        leave=False,
    )


def instantiate_node_groups(
    node_groups: Iterable[NodeGroup],
    output: str,
    bar_format: str,
    total_files: int,
    force: bool = False,
    recursive: bool = False,
    virtual_only: bool = False,
    always_copy_dir_contents: bool = False,
    copy_to_filename: str | None = None,
) -> None:
    instantiate_progress_bar = (
        None
        if virtual_only
        else tqdm(
            desc=f"Instantiating {output}: ",
            unit=" f",
            bar_format=bar_format,
            unit_scale=True,
            unit_divisor=1000,
            total=total_files,
            leave=False,
        )
    )

    output_dir = output
    output_file = None
    if copy_to_filename:
        output_dir = os.path.dirname(output)
        if not output_dir:
            output_dir = "."
        output_file = os.path.basename(output)

    # Instantiate these nodes
    for node_group in node_groups:
        if not node_group.sources:
            continue
        listing: Listing | None = node_group.listing
        source_path: str = node_group.source_path

        copy_dir_contents = always_copy_dir_contents or source_path.endswith("/")
        if not listing:
            source = node_group.sources[0]
            client = source.client
            node = NodeWithPath(source.node, [output_file or source.node.path])
            instantiated_nodes = [node]
            if not virtual_only:
                node.instantiate(
                    client, output_dir, instantiate_progress_bar, force=force
                )
        else:
            instantiated_nodes = listing.collect_nodes_to_instantiate(
                node_group.sources,
                copy_to_filename,
                recursive,
                copy_dir_contents,
                node_group.is_dataset,
            )
            if not virtual_only:
                listing.instantiate_nodes(
                    instantiated_nodes,
                    output_dir,
                    total_files,
                    force=force,
                    shared_progress_bar=instantiate_progress_bar,
                )

        node_group.instantiated_nodes = instantiated_nodes

    if instantiate_progress_bar:
        instantiate_progress_bar.close()


def find_column_to_str(  # noqa: PLR0911
    row: tuple[Any, ...], field_lookup: dict[str, int], src: DataSource, column: str
) -> str:
    if column == "du":
        return str(
            src.listing.du(
                {f: row[field_lookup[f]] for f in ["dir_type", "size", "path"]}
            )[0]
        )
    if column == "name":
        return posixpath.basename(row[field_lookup["path"]]) or ""
    if column == "path":
        is_dir = row[field_lookup["dir_type"]] == DirType.DIR
        path = row[field_lookup["path"]]
        if is_dir and path:
            full_path = path + "/"
        else:
            full_path = path
        return src.get_node_full_path_from_path(full_path)
    if column == "size":
        return str(row[field_lookup["size"]])
    if column == "type":
        dt = row[field_lookup["dir_type"]]
        if dt == DirType.DIR:
            return "d"
        if dt == DirType.FILE:
            return "f"
        if dt == DirType.TAR_ARCHIVE:
            return "t"
        # Unknown - this only happens if a type was added elsewhere but not here
        return "u"
    return ""


def clone_catalog_with_cache(catalog: "Catalog", cache: "Cache") -> "Catalog":
    clone = catalog.copy()
    clone.cache = cache
    return clone


class Catalog:
    def __init__(
        self,
        metastore: "AbstractMetastore",
        warehouse: "AbstractWarehouse",
        cache_dir=None,
        tmp_dir=None,
        client_config: dict[str, Any] | None = None,
        warehouse_ready_callback: Callable[["AbstractWarehouse"], None] | None = None,
        in_memory: bool = False,
    ):
        datachain_dir = DataChainDir(cache=cache_dir, tmp=tmp_dir)
        datachain_dir.init()
        self.metastore = metastore
        self._warehouse = warehouse
        self.cache = Cache(datachain_dir.cache, datachain_dir.tmp)
        self.client_config = client_config if client_config is not None else {}
        self._init_params = {
            "cache_dir": cache_dir,
            "tmp_dir": tmp_dir,
        }
        self._warehouse_ready_callback = warehouse_ready_callback
        self.in_memory = in_memory
        self._owns_connections = True  # False for copies, prevents double-close

    @cached_property
    def warehouse(self) -> "AbstractWarehouse":
        if self._warehouse_ready_callback:
            self._warehouse_ready_callback(self._warehouse)

        return self._warehouse

    @cached_property
    def session(self):
        from datachain.query.session import Session

        return Session.get(catalog=self)

    def get_init_params(self) -> dict[str, Any]:
        return {
            **self._init_params,
            "client_config": self.client_config,
        }

    def copy(self, cache=True, db=True):
        """
        Create a shallow copy of this catalog.

        The copy shares metastore and warehouse with the original but will not
        close them - only the original catalog owns the connections.
        """
        result = copy(self)
        result._owns_connections = False
        if not db:
            result.metastore = None
            result._warehouse = None
            result.warehouse = None
        return result

    def close(self) -> None:
        if not self._owns_connections:
            return
        if self.metastore is not None:
            with suppress(Exception):
                self.metastore.close_on_exit()
        if self._warehouse is not None:
            with suppress(Exception):
                self._warehouse.close_on_exit()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @classmethod
    def generate_query_dataset_name(cls) -> str:
        return f"{QUERY_DATASET_PREFIX}_{uuid4().hex}"

    def get_client(self, uri: str, **config: Any) -> Client:
        """
        Return the client corresponding to the given source `uri`.
        """
        config = config or self.client_config
        cls = Client.get_implementation(uri)
        return cls.from_source(StorageURI(uri), self.cache, **config)

    def enlist_source(
        self,
        source: str,
        update=False,
        client_config=None,
        column="file",
        skip_indexing=False,
    ) -> tuple["Listing | None", Client, str]:
        from datachain import read_storage
        from datachain.listing import Listing

        read_storage(source, session=self.session, update=update, column=column).exec()

        list_ds_name, list_uri, list_path, _ = get_listing(
            source, self.session, update=update
        )
        lst = None
        client = Client.get_client(list_uri, self.cache, **self.client_config)

        if list_ds_name:
            lst = Listing(
                self.metastore.clone(),
                self.warehouse.clone(),
                client,
                dataset_name=list_ds_name,
                column=column,
            )

        return lst, client, list_path

    def _remove_dataset_rows_and_warehouse_info(
        self, dataset: DatasetRecord, version: str, **kwargs
    ):
        self.warehouse.drop_dataset_rows_table(dataset, version)
        self.update_dataset_version_with_warehouse_info(
            dataset,
            version,
            rows_dropped=True,
            **kwargs,
        )

    @contextmanager
    def enlist_sources(
        self,
        sources: list[str],
        update: bool,
        skip_indexing=False,
        client_config=None,
        only_index=False,
    ) -> Iterator[list["DataSource"] | None]:
        enlisted_sources: list[tuple[Listing | None, Client, str]] = []
        try:
            for src in sources:  # Opt: parallel
                listing, client, file_path = self.enlist_source(
                    src,
                    update,
                    client_config=client_config or self.client_config,
                    skip_indexing=skip_indexing,
                )
                enlisted_sources.append((listing, client, file_path))

            if only_index:
                # sometimes we don't really need listing result (e.g. on indexing
                # process) so this is to improve performance
                yield None
                return

            dsrc_all: list[DataSource] = []
            for listing, client, file_path in enlisted_sources:
                if not listing:
                    nodes = [Node.from_file(client.get_file_info(file_path))]
                    dir_only = False
                else:
                    nodes = listing.expand_path(file_path)
                    dir_only = file_path.endswith("/")
                dsrc_all.extend(
                    DataSource(listing, client, node, dir_only) for node in nodes
                )
            yield dsrc_all
        finally:
            for listing, _, _ in enlisted_sources:
                if listing:
                    with suppress(Exception):
                        listing.close()

    def enlist_sources_grouped(
        self,
        sources: list[str],
        update: bool,
        no_glob: bool = False,
        client_config=None,
    ) -> list[NodeGroup]:
        from datachain.listing import Listing
        from datachain.query.dataset import DatasetQuery

        def _row_to_node(d: dict[str, Any]) -> Node:
            del d["file__source"]
            return Node.from_row(d)

        enlisted_sources: list[tuple[bool, bool, Any]] = []
        client_config = client_config or self.client_config
        for src in sources:  # Opt: parallel
            listing: Listing | None
            if src.startswith("ds://"):
                ds_name, ds_version = parse_dataset_uri(src)
                ds_namespace, ds_project, ds_name = parse_dataset_name(ds_name)
                assert ds_namespace
                assert ds_project
                dataset = self.get_dataset(
                    ds_name,
                    namespace_name=ds_namespace,
                    project_name=ds_project,
                    include_incomplete=False,
                )
                if not ds_version:
                    ds_version = dataset.latest_version
                dataset_sources = self.warehouse.get_dataset_sources(
                    dataset,
                    ds_version,
                )
                indexed_sources = []
                for source in dataset_sources:
                    client = self.get_client(source, **client_config)
                    uri = client.uri
                    dataset_name, _, _, _ = get_listing(uri, self.session)
                    assert dataset_name
                    listing = Listing(
                        self.metastore.clone(),
                        self.warehouse.clone(),
                        client,
                        dataset_name=dataset_name,
                    )
                    rows = DatasetQuery(
                        name=dataset.name,
                        namespace_name=dataset.project.namespace.name,
                        project_name=dataset.project.name,
                        version=ds_version,
                        catalog=self,
                    ).to_db_records()
                    indexed_sources.append(
                        (
                            listing,
                            client,
                            source,
                            [_row_to_node(r) for r in rows],
                            ds_name,
                            ds_version,
                        )  # type: ignore [arg-type]
                    )

                enlisted_sources.append((False, True, indexed_sources))
            else:
                listing, client, source_path = self.enlist_source(
                    src, update, client_config=client_config
                )
                enlisted_sources.append((False, False, (listing, client, source_path)))

        node_groups = []
        for is_datachain, is_dataset, payload in enlisted_sources:  # Opt: parallel
            if is_dataset:
                for (
                    listing,
                    client,
                    source_path,
                    nodes,
                    dataset_name,
                    dataset_version,
                ) in payload:
                    assert listing
                    dsrc = [DataSource(listing, client, node) for node in nodes]
                    node_groups.append(
                        NodeGroup(
                            listing,
                            client,
                            dsrc,
                            source_path,
                            dataset_name=dataset_name,
                            dataset_version=dataset_version,
                        )
                    )
            elif is_datachain:
                for listing, source_path, paths in payload:
                    assert listing
                    dsrc = [
                        DataSource(listing, listing.client, listing.resolve_path(p))
                        for p in paths
                    ]
                    node_groups.append(
                        NodeGroup(
                            listing,
                            listing.client,
                            dsrc,
                            source_path,
                        )
                    )
            else:
                listing, client, source_path = payload
                if not listing:
                    nodes = [Node.from_file(client.get_file_info(source_path))]
                    as_container = False
                else:
                    as_container = source_path.endswith("/")
                    nodes = listing.expand_path(source_path, use_glob=not no_glob)
                dsrc = [DataSource(listing, client, n, as_container) for n in nodes]
                node_groups.append(NodeGroup(listing, client, dsrc, source_path))

        return node_groups

    def create_dataset(
        self,
        name: str,
        project: Project | None = None,
        version: str | None = None,
        *,
        columns: Sequence[Column],
        feature_schema: dict | None = None,
        query_script: str = "",
        create_rows: bool | None = True,
        validate_version: bool | None = True,
        listing: bool | None = False,
        uuid: str | None = None,
        description: str | None = None,
        attrs: list[str] | None = None,
        update_version: str | None = "patch",
        job_id: str | None = None,
    ) -> "DatasetRecord":
        """
        Creates new dataset of a specific version.
        If dataset is not yet created, it will create it with version 1
        If version is None, then next unused version is created.
        If version is given, then it must be an unused version.
        """
        DatasetRecord.validate_name(name)
        assert [c.name for c in columns if c.name != "sys__id"], f"got {columns=}"
        if not listing and Client.is_data_source_uri(name):
            raise RuntimeError(
                "Cannot create dataset that starts with source prefix, e.g s3://"
            )
        default_version = DEFAULT_DATASET_VERSION
        try:
            dataset = self.get_dataset(
                name,
                namespace_name=project.namespace.name if project else None,
                project_name=project.name if project else None,
            )
            default_version = dataset.next_version_patch
            if update_version == "major":
                default_version = dataset.next_version_major
            if update_version == "minor":
                default_version = dataset.next_version_minor

            if (description or attrs) and (
                dataset.description != description or dataset.attrs != attrs
            ):
                description = description or dataset.description
                attrs = attrs or dataset.attrs

                self.update_dataset(
                    dataset,
                    description=description,
                    attrs=attrs,
                )

        except DatasetNotFoundError:
            schema = {
                c.name: c.type.to_dict() for c in columns if isinstance(c.type, SQLType)
            }
            dataset = self.metastore.create_dataset(
                name,
                project.id if project else None,
                feature_schema=feature_schema,
                query_script=query_script,
                schema=schema,
                ignore_if_exists=True,
                description=description,
                attrs=attrs,
            )

        version = version or default_version

        if dataset.has_version(version):
            raise DatasetInvalidVersionError(
                f"Version {version} already exists in dataset {name}"
            )

        if validate_version and not dataset.is_valid_next_version(version):
            raise DatasetInvalidVersionError(
                f"Version {version} must be higher than the current latest one"
            )

        return self.create_dataset_version(
            dataset,
            version,
            feature_schema=feature_schema,
            query_script=query_script,
            create_rows_table=create_rows,
            columns=columns,
            uuid=uuid,
            job_id=job_id,
        )

    def create_dataset_version(
        self,
        dataset: DatasetRecord,
        version: str,
        *,
        columns: Sequence[Column],
        sources="",
        feature_schema=None,
        query_script="",
        error_message="",
        error_stack="",
        script_output="",
        create_rows_table=True,
        job_id: str | None = None,
        uuid: str | None = None,
    ) -> DatasetRecord:
        """
        Creates dataset version if it doesn't exist.
        If create_rows is False, dataset rows table will not be created
        """
        assert [c.name for c in columns if c.name != "sys__id"], f"got {columns=}"
        schema = {
            c.name: c.type.to_dict() for c in columns if isinstance(c.type, SQLType)
        }

        dataset = self.metastore.create_dataset_version(
            dataset,
            version,
            status=DatasetStatus.CREATED,
            sources=sources,
            feature_schema=feature_schema,
            query_script=query_script,
            error_message=error_message,
            error_stack=error_stack,
            script_output=script_output,
            schema=schema,
            job_id=job_id,
            ignore_if_exists=True,
            uuid=uuid,
        )

        if create_rows_table:
            table_name = self.warehouse.dataset_table_name(dataset, version)
            self.warehouse.create_dataset_rows_table(table_name, columns=columns)
            self.update_dataset_version_with_warehouse_info(dataset, version)

        return dataset

    def update_dataset_version_with_warehouse_info(
        self, dataset: DatasetRecord, version: str, rows_dropped=False, **kwargs
    ) -> None:
        from datachain.query.dataset import DatasetQuery

        dataset_version = dataset.get_version(version)

        values = {**kwargs}

        if rows_dropped:
            values["num_objects"] = None
            values["size"] = None
            values["preview"] = None
            self.metastore.update_dataset_version(dataset, version, **values)
            return

        if not dataset_version.num_objects:
            num_objects, size = self.warehouse.dataset_stats(dataset, version)
            if num_objects != dataset_version.num_objects:
                values["num_objects"] = num_objects
            if size != dataset_version.size:
                values["size"] = size

        if not dataset_version.preview:
            values["preview"] = (
                DatasetQuery(
                    name=dataset.name,
                    namespace_name=dataset.project.namespace.name,
                    project_name=dataset.project.name,
                    version=version,
                    catalog=self,
                    include_incomplete=True,  # Allow reading CREATED version
                )
                .limit(20)
                .to_db_records()
            )

        if not values:
            return

        self.metastore.update_dataset_version(dataset, version, **values)

    def update_dataset(
        self, dataset: DatasetRecord, conn=None, **kwargs
    ) -> DatasetRecord:
        """Updates dataset fields."""
        dataset_updated = self.metastore.update_dataset(dataset, conn=conn, **kwargs)
        self.warehouse.rename_dataset_tables(dataset, dataset_updated)
        return dataset_updated

    def remove_dataset_version(
        self, dataset: DatasetRecord, version: str, drop_rows: bool | None = True
    ) -> None:
        """
        Deletes one single dataset version.
        If it was last version, it removes dataset completely
        """
        if not dataset.has_version(version):
            return
        dataset = self.metastore.remove_dataset_version(dataset, version)
        if drop_rows:
            self.warehouse.drop_dataset_rows_table(dataset, version)

    def get_temp_table_names(self) -> list[str]:
        return self.warehouse.get_temp_table_names()

    def cleanup_tables(self, names: Iterable[str]) -> None:
        """
        Drop tables passed.

        This should be implemented to ensure that the provided tables
        are cleaned up as soon as they are no longer needed.
        """
        self.warehouse.cleanup_tables(names)

    def cleanup_failed_dataset_versions(self, job_id: str | None = None) -> int:
        """
        Clean up failed/incomplete dataset versions.

        Removes dataset versions that:
        - Have status CREATED or FAILED
        - Belong to completed/failed/canceled jobs (not running)

        Returns:
            Number of removed versions
        """
        versions_to_clean = self.metastore.get_incomplete_dataset_versions(
            job_id=job_id
        )

        num_removed = 0
        for dataset, version in versions_to_clean:
            try:
                # Remove dataset version (drops warehouse table and metastore record)
                self.remove_dataset_version(dataset, version)
                num_removed += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to clean dataset %s version %s: %s",
                    dataset.name,
                    version,
                    e,
                )

        return num_removed

    def create_dataset_from_sources(
        self,
        name: str,
        sources: list[str],
        project: Project | None = None,
        client_config=None,
        recursive=False,
    ) -> DatasetRecord:
        if not sources:
            raise ValueError("Sources needs to be non empty list")

        from datachain import read_dataset, read_storage

        project = project or self.metastore.default_project

        chains = []
        for source in sources:
            if source.startswith(DATASET_PREFIX):
                dc = read_dataset(source[len(DATASET_PREFIX) :], session=self.session)
            else:
                dc = read_storage(source, session=self.session, recursive=recursive)

            chains.append(dc)

        # create union of all dataset queries created from sources
        dc = reduce(lambda dc1, dc2: dc1.union(dc2), chains)
        try:
            dc = dc.settings(project=project.name, namespace=project.namespace.name)
            dc.save(name)
        except Exception as e:  # noqa: BLE001
            try:
                ds = self.get_dataset(
                    name,
                    namespace_name=project.namespace.name,
                    project_name=project.name,
                )
                self.metastore.update_dataset_status(
                    ds,
                    DatasetStatus.FAILED,
                    version=ds.latest_version,
                    error_message=DATASET_INTERNAL_ERROR_MESSAGE,
                    error_stack=traceback.format_exc(),
                )
                self._remove_dataset_rows_and_warehouse_info(
                    ds,
                    ds.latest_version,
                    sources="\n".join(sources),
                )
                raise
            except DatasetNotFoundError:
                raise e from None

        ds = self.get_dataset(
            name,
            namespace_name=project.namespace.name,
            project_name=project.name,
        )

        self.update_dataset_version_with_warehouse_info(
            ds,
            ds.latest_version,
            sources="\n".join(sources),
        )

        return self.get_dataset(
            name,
            namespace_name=project.namespace.name,
            project_name=project.name,
        )

    def get_full_dataset_name(
        self,
        name: str,
        project_name: str | None = None,
        namespace_name: str | None = None,
    ) -> tuple[str, str, str]:
        """
        Returns dataset name together with separated namespace and project name.
        It takes into account all the ways namespace and project can be added.
        """
        parsed_namespace_name, parsed_project_name, name = parse_dataset_name(name)

        namespace_env = os.environ.get("DATACHAIN_NAMESPACE")
        project_env = os.environ.get("DATACHAIN_PROJECT")
        if project_env and len(project_env.split(".")) == 2:
            # we allow setting both namespace and project in DATACHAIN_PROJECT
            namespace_env, project_env = project_env.split(".")

        namespace_name = (
            parsed_namespace_name
            or namespace_name
            or namespace_env
            or self.metastore.default_namespace_name
        )
        project_name = (
            parsed_project_name
            or project_name
            or project_env
            or self.metastore.default_project_name
        )

        return namespace_name, project_name, name

    def parse_dataset_name(
        self,
        dataset_name: str,
        namespace_name: str | None = None,
        project_name: str | None = None,
        version: str | None = None,
    ) -> DatasetIdentifier:
        if not version:
            dataset_name, version = parse_dataset_with_version(dataset_name)

        namespace, project, name = self.get_full_dataset_name(
            dataset_name,
            namespace_name=namespace_name,
            project_name=project_name,
        )

        return DatasetIdentifier(
            namespace=namespace,
            project=project,
            name=name,
            version=version,
        )

    def get_dataset(
        self,
        name: str,
        namespace_name: str | None = None,
        project_name: str | None = None,
        include_incomplete: bool = True,
    ) -> DatasetRecord:
        from datachain.lib.listing import is_listing_dataset

        namespace_name = namespace_name or self.metastore.default_namespace_name
        project_name = project_name or self.metastore.default_project_name

        if is_listing_dataset(name):
            namespace_name = self.metastore.system_namespace_name
            project_name = self.metastore.listing_project_name

        return self.metastore.get_dataset(
            name,
            namespace_name=namespace_name,
            project_name=project_name,
            include_incomplete=include_incomplete,
        )

    def get_dataset_with_remote_fallback(
        self,
        name: str,
        namespace_name: str,
        project_name: str,
        version: str | None = None,
        pull_dataset: bool = False,
        update: bool = False,
        include_incomplete: bool = True,
    ) -> DatasetRecord:
        from datachain.lib.dc.utils import is_studio

        # Intentionally ignore update flag is version is provided. Here only exact
        # version can be provided and update then doesn't make sense.
        # It corresponds to a query like this for example:
        #
        #    dc.read_dataset("some.remote.dataset", version="1.0.0", update=True)
        if version:
            update = False

        # we don't do Studio fallback is script is already ran in Studio, or if we try
        # to fetch dataset with local namespace as that one cannot
        # exist in Studio in the first place
        no_fallback = is_studio() or is_namespace_local(namespace_name)

        if no_fallback or not update:
            try:
                ds = self.get_dataset(
                    name,
                    namespace_name=namespace_name,
                    project_name=project_name,
                    include_incomplete=include_incomplete,
                )
                if not version or ds.has_version(version):
                    return ds
            except (NamespaceNotFoundError, ProjectNotFoundError, DatasetNotFoundError):
                pass

        if no_fallback:
            raise DatasetNotFoundError(
                f"Dataset {name}"
                + (f" version {version} " if version else " ")
                + f"not found in namespace {namespace_name} and project {project_name}"
            )

        if pull_dataset:
            print("Dataset not found in local catalog, trying to get from studio")
            remote_ds_uri = create_dataset_uri(
                name, namespace_name, project_name, version
            )

            self.pull_dataset(
                remote_ds_uri=remote_ds_uri,
                local_ds_name=name,
                local_ds_version=version,
            )
            return self.get_dataset(
                name,
                namespace_name=namespace_name,
                project_name=project_name,
                include_incomplete=include_incomplete,
            )

        return self.get_remote_dataset(namespace_name, project_name, name)

    def get_dataset_with_version_uuid(self, uuid: str) -> DatasetRecord:
        """Returns dataset that contains version with specific uuid"""
        for dataset in self.ls_datasets():
            if dataset.has_version_with_uuid(uuid):
                return self.get_dataset(
                    dataset.name,
                    namespace_name=dataset.project.namespace.name,
                    project_name=dataset.project.name,
                    include_incomplete=False,
                )
        raise DatasetNotFoundError(f"Dataset with version uuid {uuid} not found.")

    def get_remote_dataset(
        self, namespace: str, project: str, name: str
    ) -> DatasetRecord:
        from datachain.remote.studio import StudioClient

        studio_client = StudioClient()

        info_response = studio_client.dataset_info(namespace, project, name)
        if not info_response.ok:
            if info_response.status == 404:
                raise DatasetNotFoundError(
                    f"Dataset {namespace}.{project}.{name} not found"
                )
            raise DataChainError(info_response.message)

        dataset_info = info_response.data
        assert isinstance(dataset_info, dict)
        return DatasetRecord.from_dict(dataset_info)

    def get_dataset_dependencies_by_ids(
        self,
        dataset_id: int,
        version_id: int,
        indirect: bool = True,
    ) -> list[DatasetDependency | None]:
        dependency_nodes = self.metastore.get_dataset_dependency_nodes(
            dataset_id=dataset_id,
            version_id=version_id,
        )

        if not dependency_nodes:
            return []

        dependency_map, children_map = build_dependency_hierarchy(dependency_nodes)

        root_key = (dataset_id, version_id)
        if root_key not in children_map:
            return []

        root_dependency_ids = children_map[root_key]
        root_dependencies = [dependency_map[dep_id] for dep_id in root_dependency_ids]

        if indirect:
            for dependency in root_dependencies:
                if dependency is not None:
                    populate_nested_dependencies(
                        dependency, dependency_nodes, dependency_map, children_map
                    )

        return root_dependencies

    def get_dataset_dependencies(
        self,
        name: str,
        version: str,
        namespace_name: str | None = None,
        project_name: str | None = None,
        indirect=False,
    ) -> list[DatasetDependency | None]:
        dataset = self.get_dataset(
            name,
            namespace_name=namespace_name,
            project_name=project_name,
            include_incomplete=False,
        )
        dataset_version = dataset.get_version(version)
        dataset_id = dataset.id
        dataset_version_id = dataset_version.id

        if not indirect:
            return self.metastore.get_direct_dataset_dependencies(
                dataset,
                version,
            )

        return self.get_dataset_dependencies_by_ids(
            dataset_id,
            dataset_version_id,
            indirect,
        )

    def ls_datasets(
        self,
        prefix: str | None = None,
        include_listing: bool = False,
        studio: bool = False,
        project: Project | None = None,
    ) -> Iterator[DatasetListRecord]:
        from datachain.query.session import Session
        from datachain.remote.studio import StudioClient

        project_id = project.id if project else None

        if studio:
            client = StudioClient()
            response = client.ls_datasets(prefix=prefix)
            if not response.ok:
                raise DataChainError(response.message)
            if not response.data:
                return

            datasets: Iterator[DatasetListRecord] = (
                DatasetListRecord.from_dict(d)
                for d in response.data
                if not d.get("name", "").startswith(QUERY_DATASET_PREFIX)
            )
        elif prefix:
            datasets = self.metastore.list_datasets_by_prefix(
                prefix, project_id=project_id
            )
        else:
            datasets = self.metastore.list_datasets(project_id=project_id)

        for d in datasets:
            if Session.is_temp_dataset(d.name):
                continue
            if not d.is_bucket_listing or include_listing:
                yield d

    def list_datasets_versions(
        self,
        prefix: str | None = None,
        include_listing: bool = False,
        with_job: bool = True,
        studio: bool = False,
        project: Project | None = None,
    ) -> Iterator[tuple[DatasetListRecord, "DatasetListVersion", "Job | None"]]:
        """Iterate over all dataset versions with related jobs."""
        datasets = list(
            self.ls_datasets(
                prefix=prefix,
                include_listing=include_listing,
                studio=studio,
                project=project,
            )
        )

        # preselect dataset versions jobs from db to avoid multiple queries
        jobs: dict[str, Job] = {}
        if with_job:
            jobs_ids: set[str] = {
                v.job_id for ds in datasets for v in ds.versions if v.job_id
            }
            if jobs_ids:
                jobs = {
                    j.id: j for j in self.metastore.list_jobs_by_ids(list(jobs_ids))
                }

        for d in datasets:
            yield from (
                (d, v, jobs.get(str(v.job_id)) if with_job and v.job_id else None)
                for v in d.versions
            )

    def listings(self, prefix: str | None = None) -> list["ListingInfo"]:
        """
        Returns list of ListingInfo objects which are representing specific
        storage listing datasets
        """
        from datachain.lib.listing import LISTING_PREFIX, is_listing_dataset
        from datachain.lib.listing_info import ListingInfo

        if prefix and not prefix.startswith(LISTING_PREFIX):
            prefix = LISTING_PREFIX + prefix

        listing_datasets_versions = self.list_datasets_versions(
            prefix=prefix,
            include_listing=True,
            with_job=False,
            project=self.metastore.listing_project,
        )

        return [
            ListingInfo.from_models(d, v, j)
            for d, v, j in listing_datasets_versions
            if is_listing_dataset(d.name)
        ]

    def ls_dataset_rows(
        self,
        dataset: DatasetRecord,
        version: str,
        offset=None,
        limit=None,
    ) -> list[dict]:
        from datachain.query.dataset import DatasetQuery

        q = DatasetQuery(
            name=dataset.name,
            namespace_name=dataset.project.namespace.name,
            project_name=dataset.project.name,
            version=version,
            catalog=self,
        )
        if limit:
            q = q.limit(limit)
        if offset:
            q = q.offset(offset)

        return q.to_db_records()

    def signed_url(
        self,
        source: str,
        path: str,
        version_id: str | None = None,
        client_config=None,
        content_disposition: str | None = None,
        **kwargs,
    ) -> str:
        client_config = client_config or self.client_config
        if client_config.get("anon"):
            content_disposition = None
        client = Client.get_client(source, self.cache, **client_config)
        return client.url(
            path,
            version_id=version_id,
            content_disposition=content_disposition,
            **kwargs,
        )

    def export_dataset_table(
        self,
        bucket: str,
        name: str,
        version: str,
        project: Project | None = None,
        *,
        file_format: str | None = None,
        base_file_name: str,
        client_config=None,
    ) -> None:
        dataset = self.get_dataset(
            name,
            namespace_name=project.namespace.name if project else None,
            project_name=project.name if project else None,
        )

        self.warehouse.export_dataset_table(
            bucket,
            dataset,
            version,
            file_format=file_format,
            base_file_name=base_file_name,
            client_config=client_config,
        )

    def remove_dataset(
        self,
        name: str,
        project: Project | None = None,
        version: str | None = None,
        force: bool | None = False,
    ):
        dataset = self.get_dataset(
            name,
            namespace_name=project.namespace.name if project else None,
            project_name=project.name if project else None,
        )
        if not version and not force:
            raise ValueError(f"Missing dataset version from input for dataset {name}")
        if version and not dataset.has_version(version):
            raise DatasetInvalidVersionError(
                f"Dataset {name} doesn't have version {version}"
            )

        if version:
            self.remove_dataset_version(dataset, version)
            return

        while dataset.versions:
            version = dataset.versions[0].version
            self.remove_dataset_version(
                dataset,
                version,
            )

    def edit_dataset(
        self,
        name: str,
        project: Project | None = None,
        new_name: str | None = None,
        description: str | None = None,
        attrs: list[str] | None = None,
    ) -> DatasetRecord:
        update_data = {}
        if new_name:
            DatasetRecord.validate_name(new_name)
            update_data["name"] = new_name
        if description is not None:
            update_data["description"] = description
        if attrs is not None:
            update_data["attrs"] = attrs  # type: ignore[assignment]

        dataset = self.get_dataset(
            name,
            namespace_name=project.namespace.name if project else None,
            project_name=project.name if project else None,
        )
        return self.update_dataset(dataset, **update_data)

    def ls(
        self,
        sources: list[str],
        fields: Iterable[str],
        update=False,
        skip_indexing=False,
        *,
        client_config=None,
    ) -> Iterator[tuple[DataSource, Iterable[tuple]]]:
        with self.enlist_sources(
            sources,
            update,
            skip_indexing=skip_indexing,
            client_config=client_config or self.client_config,
        ) as data_sources:
            if data_sources is None:
                return

            for source in data_sources:
                yield source, source.ls(fields)

    def pull_dataset(  # noqa: C901, PLR0915
        self,
        remote_ds_uri: str,
        output: str | None = None,
        local_ds_name: str | None = None,
        local_ds_version: str | None = None,
        cp: bool = False,
        force: bool = False,
        *,
        client_config=None,
    ) -> None:
        def _instantiate(ds_uri: str) -> None:
            if not cp:
                return
            assert output
            self.cp(
                [ds_uri],
                output,
                force=force,
                client_config=client_config,
            )
            print(f"Dataset {ds_uri} instantiated locally to {output}")

        if cp and not output:
            raise ValueError("Please provide output directory for instantiation")

        from datachain.remote.studio import StudioClient

        studio_client = StudioClient()

        try:
            remote_ds_name, version = parse_dataset_uri(remote_ds_uri)
        except Exception as e:
            raise DataChainError("Error when parsing dataset uri") from e

        remote_namespace, remote_project, remote_ds_name = parse_dataset_name(
            remote_ds_name
        )
        if not remote_namespace or not remote_project:
            raise DataChainError(
                f"Invalid fully qualified dataset name {remote_ds_name}, namespace"
                f" or project missing"
            )

        if local_ds_name:
            local_namespace, local_project, local_ds_name = parse_dataset_name(
                local_ds_name
            )
            if local_namespace and local_namespace != remote_namespace:
                raise DataChainError(
                    "Local namespace must be the same to remote namespace"
                )
            if local_project and local_project != remote_project:
                raise DataChainError("Local project must be the same to remote project")

        remote_ds = self.get_remote_dataset(
            remote_namespace, remote_project, remote_ds_name
        )

        try:
            # if version is not specified in uri, take the latest one
            if not version:
                version = remote_ds.latest_version
                print(f"Version not specified, pulling the latest one (v{version})")
                # updating dataset uri with latest version
                remote_ds_uri = create_dataset_uri(
                    remote_ds.name,
                    remote_ds.project.namespace.name,
                    remote_ds.project.name,
                    version,
                )
            remote_ds_version = remote_ds.get_version(version)
        except (DatasetVersionNotFoundError, StopIteration) as exc:
            raise DataChainError(
                f"Dataset {remote_ds_name} doesn't have version {version} on server"
            ) from exc

        local_ds_name = local_ds_name or remote_ds.name
        local_ds_version = local_ds_version or remote_ds_version.version

        local_ds_uri = create_dataset_uri(
            local_ds_name,
            remote_ds.project.namespace.name,
            remote_ds.project.name,
            local_ds_version,
        )

        try:
            # try to find existing dataset with the same uuid to avoid pulling again
            existing_ds = self.get_dataset_with_version_uuid(remote_ds_version.uuid)
            existing_ds_version = existing_ds.get_version_by_uuid(
                remote_ds_version.uuid
            )
            existing_ds_uri = create_dataset_uri(
                existing_ds.name,
                existing_ds.project.namespace.name,
                existing_ds.project.name,
                existing_ds_version.version,
            )
            if existing_ds_uri == remote_ds_uri:
                print(f"Local copy of dataset {remote_ds_uri} already present")
            else:
                print(
                    f"Local copy of dataset {remote_ds_uri} already present as"
                    f" dataset {existing_ds_uri}"
                )
            _instantiate(existing_ds_uri)
            return
        except DatasetNotFoundError:
            pass

        # Create namespace and project if doesn't exist
        print(
            f"Creating namespace {remote_ds.project.namespace.name} and project"
            f" {remote_ds.project.name}"
        )

        namespace = self.metastore.create_namespace(
            remote_ds.project.namespace.name,
            description=remote_ds.project.namespace.descr,
            uuid=remote_ds.project.namespace.uuid,
            validate=False,
        )
        project = self.metastore.create_project(
            namespace.name,
            remote_ds.project.name,
            description=remote_ds.project.descr,
            uuid=remote_ds.project.uuid,
            validate=False,
        )

        try:
            local_dataset = self.get_dataset(
                local_ds_name,
                namespace_name=namespace.name,
                project_name=project.name,
                include_incomplete=True,
            )
            if local_dataset and local_dataset.has_version(local_ds_version):
                raise DataChainError(
                    f"Local dataset {local_ds_uri} already exists with different uuid,"
                    " please choose different local dataset name or version"
                )
        except DatasetNotFoundError:
            pass

        dataset_save_progress_bar = tqdm(
            desc=f"Saving dataset {remote_ds_uri} locally: ",
            unit=" rows",
            unit_scale=True,
            unit_divisor=1000,
            total=remote_ds_version.num_objects,  # type: ignore [union-attr]
            leave=False,
        )

        schema = parse_schema(remote_ds_version.schema)

        local_ds = self.create_dataset(
            local_ds_name,
            project,
            local_ds_version,
            query_script=remote_ds_version.query_script,
            create_rows=True,
            columns=tuple(sa.Column(n, t) for n, t in schema.items() if n != "sys__id"),
            feature_schema=remote_ds_version.feature_schema,
            validate_version=False,
            uuid=remote_ds_version.uuid,
        )

        # asking remote to export dataset rows table to s3 and to return signed
        # urls of exported parts, which are in parquet format
        export_response = studio_client.export_dataset_table(
            remote_ds, remote_ds_version.version
        )
        if not export_response.ok:
            raise DataChainError(export_response.message)

        export_data = export_response.data
        export_id = export_data["export_id"]
        signed_urls: list[str] = export_data["signed_urls"]

        if signed_urls:
            with (
                self.metastore.clone() as metastore,
                self.warehouse.clone() as warehouse,
            ):

                def batch(urls):
                    """
                    Batching urls in a way that fetching is most efficient as
                    urls with lower id will be created first. Because that, we
                    are making sure all threads are pulling most recent urls
                    from beginning
                    """
                    res = [[] for i in range(PULL_DATASET_MAX_THREADS)]
                    current_worker = 0
                    for url in urls:
                        res[current_worker].append(url)
                        current_worker = (current_worker + 1) % PULL_DATASET_MAX_THREADS

                    return res

                rows_fetcher = DatasetRowsFetcher(
                    metastore,
                    warehouse,
                    remote_ds,
                    remote_ds_version.version,
                    export_id,
                    local_ds,
                    local_ds_version,
                    schema,
                    progress_bar=dataset_save_progress_bar,
                )

                try:
                    rows_fetcher.run(
                        iter(batch(signed_urls)), dataset_save_progress_bar
                    )
                except:
                    self.remove_dataset(local_ds_name, project, local_ds_version)
                    raise

        local_ds = self.metastore.update_dataset_status(
            local_ds,
            DatasetStatus.COMPLETE,
            version=local_ds_version,
            error_message=remote_ds.error_message,
            error_stack=remote_ds.error_stack,
            script_output=remote_ds.error_stack,
        )
        self.update_dataset_version_with_warehouse_info(local_ds, local_ds_version)

        dataset_save_progress_bar.close()
        print(f"Dataset {remote_ds_uri} saved locally")

        _instantiate(local_ds_uri)

    def clone(
        self,
        sources: list[str],
        output: str,
        force: bool = False,
        update: bool = False,
        recursive: bool = False,
        no_glob: bool = False,
        no_cp: bool = False,
        *,
        client_config=None,
    ) -> None:
        """
        This command takes cloud path(s) and duplicates files and folders in
        them into the dataset folder.
        It also adds those files to a dataset in database, which is
        created if doesn't exist yet
        """
        if not no_cp:
            self.cp(
                sources,
                output,
                force=force,
                update=update,
                recursive=recursive,
                no_glob=no_glob,
                no_cp=no_cp,
                client_config=client_config,
            )
        else:
            # since we don't call cp command, which does listing implicitly,
            # it needs to be done here
            with self.enlist_sources(
                sources,
                update,
                client_config=client_config or self.client_config,
            ):
                pass

        self.create_dataset_from_sources(
            output,
            sources,
            self.metastore.default_project,
            client_config=client_config,
            recursive=recursive,
        )

    def cp(
        self,
        sources: list[str],
        output: str,
        force: bool = False,
        update: bool = False,
        recursive: bool = False,
        no_cp: bool = False,
        no_glob: bool = False,
        *,
        client_config: dict | None = None,
    ) -> None:
        """
        This function copies files from cloud sources to local destination directory
        If cloud source is not indexed, or has expired index, it runs indexing
        """
        client_config = client_config or self.client_config
        node_groups = self.enlist_sources_grouped(
            sources,
            update,
            no_glob,
            client_config=client_config,
        )
        try:
            always_copy_dir_contents, copy_to_filename = prepare_output_for_cp(
                node_groups, output, force, no_cp
            )
            total_size, total_files = collect_nodes_for_cp(node_groups, recursive)
            if not total_files:
                return

            desc_max_len = max(len(output) + 16, 19)
            bar_format = (
                "{desc:<"
                f"{desc_max_len}"
                "}{percentage:3.0f}%|{bar}| {n_fmt:>5}/{total_fmt:<5} "
                "[{elapsed}<{remaining}, {rate_fmt:>8}]"
            )

            if not no_cp:
                with get_download_bar(bar_format, total_size) as pbar:
                    for node_group in node_groups:
                        node_group.download(recursive=recursive, pbar=pbar)

            instantiate_node_groups(
                node_groups,
                output,
                bar_format,
                total_files,
                force,
                recursive,
                no_cp,
                always_copy_dir_contents,
                copy_to_filename,
            )
        finally:
            for node_group in node_groups:
                with suppress(Exception):
                    node_group.close()

    def du(
        self,
        sources,
        depth=0,
        update=False,
        *,
        client_config=None,
    ) -> Iterable[tuple[str, float]]:
        with self.enlist_sources(
            sources,
            update,
            client_config=client_config or self.client_config,
        ) as matched_sources:
            if matched_sources is None:
                return

            def du_dirs(src, node, subdepth):
                if subdepth > 0:
                    subdirs = src.listing.get_dirs_by_parent_path(node.path)
                    for sd in subdirs:
                        yield from du_dirs(src, sd, subdepth - 1)
                yield (
                    src.get_node_full_path(node),
                    src.listing.du(node)[0],
                )

            for src in matched_sources:
                yield from du_dirs(src, src.node, depth)

    def find(
        self,
        sources,
        update=False,
        names=None,
        inames=None,
        paths=None,
        ipaths=None,
        size=None,
        typ=None,
        columns=None,
        *,
        client_config=None,
    ) -> Iterator[str]:
        with self.enlist_sources(
            sources,
            update,
            client_config=client_config or self.client_config,
        ) as matched_sources:
            if matched_sources is None:
                return

            if not columns:
                columns = ["path"]
            field_set = set()
            for column in columns:
                if column == "du":
                    field_set.add("dir_type")
                    field_set.add("size")
                    field_set.add("path")
                elif column == "name":
                    field_set.add("path")
                elif column == "path":
                    field_set.add("dir_type")
                    field_set.add("path")
                elif column == "size":
                    field_set.add("size")
                elif column == "type":
                    field_set.add("dir_type")
            fields = list(field_set)
            field_lookup = {f: i for i, f in enumerate(fields)}
            for src in matched_sources:
                results = src.listing.find(
                    src.node, fields, names, inames, paths, ipaths, size, typ
                )
                for row in results:
                    yield "\t".join(
                        find_column_to_str(row, field_lookup, src, column)
                        for column in columns
                    )

    def index(
        self,
        sources,
        update=False,
        *,
        client_config=None,
    ) -> None:
        with self.enlist_sources(
            sources,
            update,
            client_config=client_config or self.client_config,
            only_index=True,
        ):
            pass
