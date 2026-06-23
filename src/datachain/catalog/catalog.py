import io
import logging
import os
import os.path
import posixpath
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import sqlalchemy as sa
from sqlalchemy import Column

from datachain.cache import Cache
from datachain.checkpoint import Checkpoint, CheckpointStatus
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
from datachain.progress import tqdm
from datachain.project import Project
from datachain.sql.types import DateTime, SQLType
from datachain.utils import DataChainDir, DatasetIdentifier, interprocess_file_lock

from .datasource import DataSource
from .dependency import build_dependency_hierarchy, populate_nested_dependencies

if TYPE_CHECKING:
    import pandas as pd

    from datachain.data_storage import AbstractMetastore, AbstractWarehouse
    from datachain.dataset import DatasetListVersion
    from datachain.job import Job
    from datachain.lib.dc.datachain import DataChain
    from datachain.lib.listing_info import ListingInfo
    from datachain.listing import Listing
    from datachain.remote.studio import StudioClient

logger = logging.getLogger("datachain")

DEFAULT_DATASET_DIR = "dataset"

CHECKPOINTS_TTL = 4 * 60 * 60

INDEX_INTERNAL_ERROR_MESSAGE = "Internal error on indexing"
# exit code we use if query script was canceled
QUERY_SCRIPT_CANCELED_EXIT_CODE = 11
# exit code we use if the job is already in a terminal state (failed/canceled elsewhere)
QUERY_SCRIPT_ABORTED_EXIT_CODE = 12
QUERY_SCRIPT_SIGTERM_EXIT_CODE = -15  # if query script was terminated by SIGTERM

# dataset pull
PULL_DATASET_MAX_THREADS = 5
PULL_DATASET_CHUNK_TIMEOUT = 3600
PULL_DATASET_SLEEP_INTERVAL = 0.1  # sleep time while waiting for chunk to be available
PULL_DATASET_CHECK_STATUS_INTERVAL = 20  # interval to check export status in Studio
_MAX_VERSION_CLAIM_RETRIES = 5

_COLUMN_FIELD_MAP = {
    "du": ["dir_type", "size", "path"],
    "name": ["path"],
    "path": ["dir_type", "path"],
    "size": ["size"],
    "type": ["dir_type"],
}


def _round_robin_batch(urls: list[str], num_workers: int) -> list[list[str]]:
    """Round-robin distribute urls across workers so each starts with low-index urls."""
    batches: list[list[str]] = [[] for _ in range(num_workers)]
    for i, url in enumerate(urls):
        batches[i % num_workers].append(url)
    return batches


def is_namespace_local(namespace_name) -> bool:
    """Checks if namespace is from local environment, i.e. is `local`"""
    return namespace_name == "local"


class DatasetRowsFetcher(NodesThreadPool):
    """
    Fetches dataset rows from Studio export and inserts them into a staging table.

    This class downloads parquet files from signed URLs and inserts the data
    into a temporary staging table.
    """

    def __init__(
        self,
        warehouse: "AbstractWarehouse",
        temp_table_name: str,
        export_id: int,
        schema: dict[str, SQLType | type[SQLType]],
        studio_client: "StudioClient",
        max_threads: int = PULL_DATASET_MAX_THREADS,
        progress_bar=None,
    ):
        super().__init__(max_threads)
        self._check_dependencies()
        self._set_attributes(
            warehouse, temp_table_name, export_id, schema, studio_client, progress_bar
        )
        self._init_tracking_state()

    def _set_attributes(
        self, warehouse, temp_table_name, export_id, schema, studio_client, progress_bar
    ):
        self.warehouse = warehouse
        self.temp_table_name = temp_table_name
        self.export_id = export_id
        self.schema = schema
        self.studio_client = studio_client
        self.progress_bar = progress_bar

    def _init_tracking_state(self):
        self.last_status_check: float | None = None
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

    def _update_progress_postfix(self, files_done, num_files, export_status):
        if self.progress_bar is not None:
            self.progress_bar.set_postfix_str(
                f"studio_export={files_done}/{num_files} ({export_status})"
            )

    def _check_export_errors(self, export_status):
        if export_status == "failed":
            raise DataChainError("Dataset export failed in Studio")
        if export_status == "removed":
            raise DataChainError("Dataset export removed in Studio")

    def _update_tracking(self, export_status, files_done, num_files):
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
            self._update_progress_postfix(files_done, num_files, export_status)

    def check_for_status(self) -> None:
        response = self.studio_client.dataset_export_status(self.export_id)
        if not response.ok:
            raise DataChainError(response.message)
        data = response.data
        self._update_tracking(
            data["status"], data.get("files_done"), data.get("num_files")
        )
        self._check_export_errors(data["status"])
        self.last_status_check = time.time()

    def fix_columns(self, df) -> "pd.DataFrame":
        import pandas as pd

        for c in [c for c, t in self.schema.items() if t == DateTime]:
            df[c] = pd.to_datetime(df[c], unit="s")
        return df.drop("sys__id", axis=1)

    def get_parquet_content(self, url: str):
        import requests

        return self._fetch_parquet_content(url, requests)

    def _fetch_parquet_content(self, url, requests):
        while True:
            if self.should_check_for_status():
                self.check_for_status()
            content = self._attempt_get_content(url, requests)
            if content is not None:
                return content

    @staticmethod
    def _attempt_get_content(url, requests):
        r = requests.get(url, timeout=PULL_DATASET_CHUNK_TIMEOUT)
        if r.status_code == 404:
            time.sleep(PULL_DATASET_SLEEP_INTERVAL)
            return None
        r.raise_for_status()
        return r.content

    def _process_single_url(self, url, warehouse):
        import lz4.frame
        import pandas as pd

        self._process_url_imports(url, warehouse, lz4.frame, pd)

    def _process_url_imports(self, url, warehouse, lz4_frame, pd):
        if self.should_check_for_status():
            self.check_for_status()
        df = pd.read_parquet(
            io.BytesIO(lz4_frame.decompress(self.get_parquet_content(url)))
        )
        df = self.fix_columns(df)
        inserted = warehouse.insert_dataframe_to_table(self.temp_table_name, df)
        self.increase_counter(inserted)
        self.update_progress_bar(self.progress_bar)

    def do_task(self, urls):
        with self.warehouse.clone() as warehouse:
            for url in list(urls):
                self._process_url(url, warehouse)


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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _handle_existing_output(output: str, force: bool) -> None:
    if os.path.exists(output):
        if force:
            os.remove(output)
        else:
            raise FileExistsError(f"Path already exists: {output}")


def _validate_node_groups(
    node_groups: list[NodeGroup],
) -> int:
    total_node_count = 0
    for node_group in node_groups:
        if not node_group.sources:
            raise FileNotFoundError(
                f"No such file or directory: {node_group.source_path}"
            )
        total_node_count += len(node_group.sources)
    return total_node_count


def _prepare_output_dir(
    node_groups: list[NodeGroup],
    output: str,
    total_node_count: int,
    force: bool,
) -> tuple[bool, str | None]:
    if os.path.isdir(output):
        return False, None
    if all(n.is_dataset for n in node_groups):
        os.mkdir(output)
        return False, None
    _ensure_single_source(total_node_count, output)
    return _handle_container_or_file(node_groups, output, force)


def _ensure_single_source(total_node_count, output):
    if total_node_count != 1:
        raise FileNotFoundError(f"Is not a directory: {output}")


def _handle_container_or_file(node_groups, output, force):
    first_source = node_groups[0].sources[0]
    if first_source.is_container():
        _handle_existing_output(output, force)
        os.mkdir(output)
        return True, None
    _handle_existing_output(output, force)
    return False, output


def prepare_output_for_cp(
    node_groups: list[NodeGroup],
    output: str,
    force: bool = False,
    no_cp: bool = False,
) -> tuple[bool, str | None]:
    total_node_count = _validate_node_groups(node_groups)
    if no_cp:
        return False, None
    return _prepare_output_dir(node_groups, output, total_node_count, force)


def _process_source_node(
    dsrc: DataSource,
    listing: "Listing | None",
    recursive: bool,
) -> tuple[int, int, bool]:
    if dsrc.is_single_object():
        return dsrc.node.size, 1, True
    assert listing
    node = dsrc.node
    if not recursive:
        print(f"{node.full_path} is a directory (not copied).")
        return 0, 0, False
    add_size, add_files = listing.du(node, count_files=True)
    return add_size, add_files, True


def _accumulate_source(
    valid_sources,
    total_size,
    total_files,
    dsrc,
    listing,
    recursive,
):
    add_size, add_files, keep = _process_source_node(dsrc, listing, recursive)
    if keep:
        valid_sources.append(dsrc)
        total_size += add_size
        total_files += add_files
    return valid_sources, total_size, total_files


def _collect_node_group_sources(
    node_group: NodeGroup,
    recursive: bool,
) -> tuple[list[DataSource], int, int]:
    listing = node_group.listing
    valid_sources: list[DataSource] = []
    total_size = 0
    total_files = 0
    for dsrc in node_group.sources:
        valid_sources, total_size, total_files = _accumulate_source(
            valid_sources,
            total_size,
            total_files,
            dsrc,
            listing,
            recursive,
        )
    return valid_sources, total_size, total_files


def collect_nodes_for_cp(
    node_groups: Iterable[NodeGroup],
    recursive: bool = False,
) -> tuple[int, int]:
    total_size: int = 0
    total_files: int = 0

    for node_group in node_groups:
        valid_sources, add_size, add_files = _collect_node_group_sources(
            node_group, recursive
        )
        total_size += add_size
        total_files += add_files
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


def _make_instantiate_progress_bar(output: str, bar_format: str, total_files: int):
    return tqdm(
        desc=f"Instantiating {output}: ",
        unit=" f",
        bar_format=bar_format,
        unit_scale=True,
        unit_divisor=1000,
        total=total_files,
        leave=False,
    )


def _resolve_output_path(
    output: str, copy_to_filename: str | None
) -> tuple[str, str | None]:
    if not copy_to_filename:
        return output, None
    output_dir = os.path.dirname(output)
    return output_dir or ".", os.path.basename(output)


def _instantiate_node_group_no_listing(
    node_group: NodeGroup,
    output_dir: str,
    output_file: str | None,
    force: bool,
    virtual_only: bool,
    progress_bar,
):
    source = node_group.sources[0]
    client = source.client
    node = NodeWithPath(source.node, [output_file or source.node.path])
    instantiated_nodes = [node]
    if not virtual_only:
        node.instantiate(client, output_dir, progress_bar, force=force)
    return instantiated_nodes


def _instantiate_node_group_with_listing(
    node_group: NodeGroup,
    output_dir: str,
    copy_to_filename: str | None,
    recursive: bool,
    copy_dir_contents: bool,
    total_files: int,
    force: bool,
    virtual_only: bool,
    progress_bar,
):
    listing = node_group.listing
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
            shared_progress_bar=progress_bar,
        )
    return instantiated_nodes


def _instantiate_single_node_group(
    node_group,
    output_dir,
    output_file,
    copy_to_filename,
    recursive,
    copy_dir_contents,
    total_files,
    force,
    virtual_only,
    progress_bar,
):
    if not node_group.listing:
        node_group.instantiated_nodes = _instantiate_node_group_no_listing(
            node_group, output_dir, output_file, force, virtual_only, progress_bar
        )
    else:
        node_group.instantiated_nodes = _instantiate_node_group_with_listing(
            node_group,
            output_dir,
            copy_to_filename,
            recursive,
            copy_dir_contents,
            total_files,
            force,
            virtual_only,
            progress_bar,
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
    progress_bar = (
        None
        if virtual_only
        else _make_instantiate_progress_bar(output, bar_format, total_files)
    )
    output_dir, output_file = _resolve_output_path(output, copy_to_filename)

    _process_all_node_groups(
        node_groups,
        output_dir,
        output_file,
        copy_to_filename,
        recursive,
        always_copy_dir_contents,
        total_files,
        force,
        virtual_only,
        progress_bar,
    )

    if progress_bar:
        progress_bar.close()


def _process_all_node_groups(
    node_groups,
    output_dir,
    output_file,
    copy_to_filename,
    recursive,
    always_copy_dir_contents,
    total_files,
    force,
    virtual_only,
    progress_bar,
):
    for node_group in node_groups:
        if not node_group.sources:
            continue
        copy_dir_contents = always_copy_dir_contents or node_group.source_path.endswith(
            "/"
        )
        _instantiate_single_node_group(
            node_group,
            output_dir,
            output_file,
            copy_to_filename,
            recursive,
            copy_dir_contents,
            total_files,
            force,
            virtual_only,
            progress_bar,
        )


def _column_du(row, field_lookup, src):
    return str(
        src.listing.du({f: row[field_lookup[f]] for f in ["dir_type", "size", "path"]})[
            0
        ]
    )


def _column_name(row, field_lookup):
    return posixpath.basename(row[field_lookup["path"]]) or ""


def _column_path(row, field_lookup, src):
    is_dir = row[field_lookup["dir_type"]] == DirType.DIR
    path = row[field_lookup["path"]]
    full_path = (path + "/") if (is_dir and path) else path
    return src.get_node_uri_from_path(full_path)


def _column_size(row, field_lookup):
    return str(row[field_lookup["size"]])


def _column_type(row, field_lookup):
    dt = row[field_lookup["dir_type"]]
    if dt == DirType.DIR:
        return "d"
    if dt == DirType.FILE:
        return "f"
    if dt == DirType.TAR_ARCHIVE:
        return "t"
    return "u"


def find_column_to_str(
    row: tuple[Any, ...], field_lookup: dict[str, int], src: DataSource, column: str
) -> str:
    if column == "du":
        return _column_du(row, field_lookup, src)
    if column == "name":
        return _column_name(row, field_lookup)
    if column == "path":
        return _column_path(row, field_lookup, src)
    if column == "size":
        return _column_size(row, field_lookup)
    if column == "type":
        return _column_type(row, field_lookup)
    return ""


def clone_catalog_with_cache(catalog: "Catalog", cache: "Cache") -> "Catalog":
    clone = catalog.copy()
    clone.cache = cache
    return clone


@dataclass
class CatalogConfig:
    client_config: dict[str, Any]
    _init_params: dict[str, Any]
    in_memory: bool
    _owns_connections: bool = True


class Catalog:
    def __init__(
        self,
        metastore: "AbstractMetastore",
        warehouse: "AbstractWarehouse",
        cache_dir=None,
        tmp_dir=None,
        client_config: dict[str, Any] | None = None,
        in_memory: bool = False,
    ):
        self.metastore = metastore
        self.warehouse = warehouse
        self.config = CatalogConfig(
            client_config=client_config if client_config is not None else {},
            _init_params={
                "cache_dir": cache_dir,
                "tmp_dir": tmp_dir,
            },
            in_memory=in_memory,
        )
        self._init_cache(cache_dir, tmp_dir)

    def _init_cache(self, cache_dir, tmp_dir):
        datachain_dir = DataChainDir(cache=cache_dir, tmp=tmp_dir)
        datachain_dir.init()
        self.cache = Cache(datachain_dir.cache, datachain_dir.tmp)

    @cached_property
    def session(self):
        from datachain.query.session import Session

        return Session.get(catalog=self)

    def get_init_params(self) -> dict[str, Any]:
        return {
            **self.config._init_params,
            "client_config": self.config.client_config,
        }

    def copy(self, cache=True, db=True):
        result = copy(self)
        result.config._owns_connections = False
        if not db:
            result.metastore = None
            result.warehouse = None
        return result

    def _close_safe(self, resource, method_name):
        if resource is not None:
            with suppress(Exception):
                getattr(resource, method_name)()

    def close(self) -> None:
        if not self.config._owns_connections:
            return
        self._close_safe(self.metastore, "close_on_exit")
        self._close_safe(self.warehouse, "close_on_exit")

    def __enter__(self) -> Self:
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
        config = config or self.config.client_config
        cls = Client.get_implementation(uri)
        return cls.from_source(StorageURI(uri), self.cache, **config)

    def _build_listing(self, list_ds_name, list_uri, client, column):
        if not list_ds_name:
            return None
        from datachain.listing import Listing

        return Listing(
            self.metastore.clone(),
            self.warehouse.clone(),
            client,
            dataset_name=list_ds_name,
            column=column,
        )

    def enlist_source(
        self,
        source: str,
        update=False,
        client_config=None,
        column="file",
        skip_indexing=False,
    ) -> tuple["Listing | None", Client, str]:
        from datachain import read_storage

        read_storage(source, session=self.session, update=update, column=column).exec()

        list_ds_name, list_uri, list_path, _ = get_listing(
            source, self.session, update=update
        )
        client = Client.get_client(list_uri, self.cache, **self.config.client_config)
        lst = self._build_listing(list_ds_name, list_uri, client, column)
        return lst, client, list_path

    def _enlist_all_sources(self, sources, update, skip_indexing, client_config):
        enlisted_sources: list[tuple[Listing | None, Client, str]] = []
        for src in sources:
            listing, client, file_path = self.enlist_source(
                src,
                update,
                client_config=client_config or self.config.client_config,
                skip_indexing=skip_indexing,
            )
            enlisted_sources.append((listing, client, file_path))
        return enlisted_sources

    def _build_data_source(self, listing, client, file_path):
        if not listing:
            nodes = [Node.from_file(client.get_file_info(file_path))]
            dir_only = False
        else:
            nodes = listing.expand_path(file_path)
            dir_only = file_path.endswith("/")
        return [DataSource(listing, client, node, dir_only) for node in nodes]

    def _build_data_sources(self, enlisted_sources):
        dsrc_all: list[DataSource] = []
        for listing, client, file_path in enlisted_sources:
            dsrc_all.extend(self._build_data_source(listing, client, file_path))
        return dsrc_all

    def _close_listings(self, enlisted_sources):
        for listing, _, _ in enlisted_sources:
            if listing:
                with suppress(Exception):
                    listing.close()

    @contextmanager
    def enlist_sources(
        self,
        sources: list[str],
        update: bool,
        skip_indexing=False,
        client_config=None,
        only_index=False,
    ) -> Iterator[list["DataSource"] | None]:
        enlisted_sources = self._enlist_all_sources(
            sources, update, skip_indexing, client_config
        )
        try:
            yield self._enlist_yield_data(enlisted_sources, only_index)
        finally:
            self._close_listings(enlisted_sources)

    def _enlist_yield_data(self, enlisted_sources, only_index):
        if only_index:
            return None
        return self._build_data_sources(enlisted_sources)

    def _enlist_plain_source(self, src, update, client_config):
        listing, client, source_path = self.enlist_source(
            src, update, client_config=client_config
        )
        return (False, False, (listing, client, source_path))

    def _build_listing_for_source(self, source, client_config):
        from datachain.listing import Listing

        client = self.get_client(source, **client_config)
        uri = client.uri
        dataset_name, _, _, _ = get_listing(uri, self.session)
        assert dataset_name
        return client, Listing(
            self.metastore.clone(),
            self.warehouse.clone(),
            client,
            dataset_name=dataset_name,
        )

    def _get_dataset_rows(self, dataset, ds_version):
        from datachain.query.dataset import DatasetQuery

        return DatasetQuery(
            name=dataset.name,
            namespace_name=dataset.project.namespace.name,
            project_name=dataset.project.name,
            version=ds_version,
            catalog=self,
        ).to_db_records()

    def _index_dataset_source(
        self, source, dataset, ds_version, ds_name, client_config
    ):
        client, listing = self._build_listing_for_source(source, client_config)
        rows = self._get_dataset_rows(dataset, ds_version)
        return (
            listing,
            client,
            source,
            [self._row_to_node(r) for r in rows],
            ds_name,
            ds_version,
        )

    @staticmethod
    def _row_to_node(d):
        d = dict(d)
        del d["file__source"]
        return Node.from_row(d)

    def _enlist_single_source_grouped(
        self,
        src: str,
        update: bool,
        client_config: dict,
    ) -> tuple[bool, bool, Any]:
        if not src.startswith("ds://"):
            return self._enlist_plain_source(src, update, client_config)

        (ds_namespace, ds_project, ds_name, ds_version) = parse_dataset_uri(src)
        dataset = self.get_dataset(
            ds_name,
            namespace_name=ds_namespace,
            project_name=ds_project,
            versions=[ds_version] if ds_version else None,
            include_incomplete=False,
        )
        if not ds_version:
            ds_version = dataset.latest_version
        dataset_sources = self.warehouse.get_dataset_sources(dataset, ds_version)
        indexed_sources = [
            self._index_dataset_source(s, dataset, ds_version, ds_name, client_config)
            for s in dataset_sources
        ]
        return (False, True, indexed_sources)

    def _build_node_group_dataset(self, payload):
        node_groups = []
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
        return node_groups

    def _build_node_group_datachain(self, payload):
        node_groups = []
        for listing, source_path, paths in payload:
            assert listing
            dsrc = [
                DataSource(listing, listing.client, listing.resolve_path(p))
                for p in paths
            ]
            node_groups.append(NodeGroup(listing, listing.client, dsrc, source_path))
        return node_groups

    def _resolve_nodes(self, listing, client, source_path, no_glob):
        if not listing:
            return [Node.from_file(client.get_file_info(source_path))], False
        return listing.expand_path(
            source_path, use_glob=not no_glob
        ), source_path.endswith("/")

    def _build_node_group_plain(self, payload, no_glob):
        listing, client, source_path = payload
        nodes, as_container = self._resolve_nodes(listing, client, source_path, no_glob)
        dsrc = [DataSource(listing, client, n, as_container) for n in nodes]
        return [NodeGroup(listing, client, dsrc, source_path)]

    def _build_node_groups_from_enlisted(
        self,
        enlisted_sources: list[tuple[bool, bool, Any]],
        no_glob: bool,
    ) -> list[NodeGroup]:
        node_groups = []
        for is_datachain, is_dataset, payload in enlisted_sources:
            if is_dataset:
                node_groups.extend(self._build_node_group_dataset(payload))
            elif is_datachain:
                node_groups.extend(self._build_node_group_datachain(payload))
            else:
                node_groups.extend(self._build_node_group_plain(payload, no_glob))
        return node_groups

    def enlist_sources_grouped(
        self,
        sources: list[str],
        update: bool,
        no_glob: bool = False,
        client_config=None,
    ) -> list[NodeGroup]:
        client_config = client_config or self.config.client_config
        enlisted_sources = [
            self._enlist_single_source_grouped(src, update, client_config)
            for src in sources
        ]
        return self._build_node_groups_from_enlisted(enlisted_sources, no_glob)

    def _get_or_create_dataset(
        self, name, project, feature_schema, query_script, columns, description, attrs
    ):
        try:
            dataset = self.get_dataset(
                name,
                namespace_name=project.namespace.name if project else None,
                project_name=project.name if project else None,
                versions=None,
            )
            if (description or attrs) and (
                dataset.description != description or dataset.attrs != attrs
            ):
                self.update_dataset(
                    dataset,
                    description=description or dataset.description,
                    attrs=attrs or dataset.attrs,
                )
            return dataset
        except DatasetNotFoundError:
            schema = {
                c.name: c.type.to_dict() for c in columns if isinstance(c.type, SQLType)
            }
            return self.metastore.create_dataset(
                name,
                project.id if project else None,
                feature_schema=feature_schema,
                query_script=query_script,
                schema=schema,
                ignore_if_exists=True,
                description=description,
                attrs=attrs,
            )

    def create_dataset(
        self,
        name: str,
        project: Project | None = None,
        version: str | None = None,
        *,
        columns: Sequence[Column],
        feature_schema: dict | None = None,
        query_script: str = "",
        sources: str = "",
        validate_version: bool | None = True,
        listing: bool | None = False,
        uuid: str | None = None,
        description: str | None = None,
        attrs: list[str] | None = None,
        update_version: str | None = "patch",
        job_id: str | None = None,
        content_hash: str | None = None,
    ) -> "DatasetRecord":
        DatasetRecord.validate_name(name)
        assert [c.name for c in columns if c.name != "sys__id"], f"got {columns=}"
        if not listing and Client.is_data_source_uri(name):
            raise RuntimeError(
                "Cannot create dataset that starts with source prefix, e.g s3://"
            )

        dataset = self._get_or_create_dataset(
            name, project, feature_schema, query_script, columns, description, attrs
        )
        return self._try_claim_version(
            dataset=dataset,
            name=name,
            version=version,
            project=project,
            feature_schema=feature_schema,
            query_script=query_script,
            sources=sources,
            columns=columns,
            uuid=uuid,
            job_id=job_id,
            validate_version=validate_version,
            update_version=update_version,
            content_hash=content_hash,
        )

    @staticmethod
    def _next_auto_version(dataset: "DatasetRecord", update_version: str | None) -> str:
        """Compute the next version for a dataset based on the update strategy."""
        if not dataset.versions:
            return DEFAULT_DATASET_VERSION
        if update_version == "major":
            return dataset.next_version_major
        if update_version == "minor":
            return dataset.next_version_minor
        return dataset.next_version_patch

    def _validate_claim_version(self, dataset, target_version, name, validate_version):
        if dataset.has_version(target_version):
            raise DatasetInvalidVersionError(
                f"Version {target_version} already exists in dataset {name}"
            )
        if validate_version and not dataset.is_valid_next_version(target_version):
            raise DatasetInvalidVersionError(
                f"Version {target_version} must be higher than the current latest one"
            )

    def _retry_claim_version(
        self, dataset, name, project, target_version, update_version
    ):
        logger.debug(
            "Version %s of dataset %s was claimed by another writer, retrying",
            target_version,
            name,
        )
        dataset = self.get_dataset(
            name,
            namespace_name=project.namespace.name if project else None,
            project_name=project.name if project else None,
            versions=None,
        )
        return dataset, self._next_auto_version(dataset, update_version)

    def _claim_one_attempt(
        self,
        dataset,
        target_version,
        name,
        validate_version,
        feature_schema,
        query_script,
        sources,
        columns,
        uuid,
        job_id,
        content_hash,
    ):
        self._validate_claim_version(dataset, target_version, name, validate_version)
        dataset, version_created = self.create_dataset_version(
            dataset,
            target_version,
            feature_schema=feature_schema,
            query_script=query_script,
            sources=sources,
            columns=columns,
            uuid=uuid,
            job_id=job_id,
            content_hash=content_hash,
        )
        if version_created:
            assert len(dataset.versions) == 1
            assert dataset.versions[0].version == target_version
            return dataset, True
        return dataset, False

    def _try_claim_version(
        self,
        dataset: "DatasetRecord",
        name: str,
        version: str | None,
        project: Project | None,
        feature_schema: dict | None,
        query_script: str,
        sources: str,
        columns: Sequence[Column],
        uuid: str | None,
        job_id: str | None,
        validate_version: bool | None,
        update_version: str | None,
        content_hash: str | None = None,
    ) -> "DatasetRecord":
        max_retries = 0 if version else _MAX_VERSION_CLAIM_RETRIES
        target_version = version or self._next_auto_version(dataset, update_version)

        result = self._claim_loop(
            dataset,
            target_version,
            name,
            validate_version,
            feature_schema,
            query_script,
            sources,
            columns,
            uuid,
            job_id,
            content_hash,
            max_retries,
            project,
            update_version,
        )
        if result is not None:
            return result

        msg = (
            f"Version {target_version} of dataset {name} was claimed by another writer"
            if version
            else (
                f"Failed to claim a version for dataset {name}"
                f" after {1 + max_retries} attempts"
            )
        )
        raise DatasetInvalidVersionError(msg)

    def _claim_loop(
        self,
        dataset,
        target_version,
        name,
        validate_version,
        feature_schema,
        query_script,
        sources,
        columns,
        uuid,
        job_id,
        content_hash,
        max_retries,
        project,
        update_version,
    ):
        for attempt in range(1 + max_retries):
            dataset, claimed = self._claim_one_attempt(
                dataset,
                target_version,
                name,
                validate_version,
                feature_schema,
                query_script,
                sources,
                columns,
                uuid,
                job_id,
                content_hash,
            )
            if claimed:
                return dataset
            if attempt >= max_retries:
                break
            dataset, target_version = self._retry_claim_version(
                dataset, name, project, target_version, update_version
            )
        return None

    @staticmethod
    def _build_version_schema(columns):
        return {
            c.name: c.type.to_dict() for c in columns if isinstance(c.type, SQLType)
        }

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
        job_id: str | None = None,
        uuid: str | None = None,
        content_hash: str | None = None,
    ) -> tuple[DatasetRecord, bool]:
        assert [c.name for c in columns if c.name != "sys__id"], f"got {columns=}"
        schema = self._build_version_schema(columns)

        return self.metastore.create_dataset_version(
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
            content_hash=content_hash,
        )

    def _fill_missing_stats(self, dataset, version, dataset_version, values):
        if dataset_version.stats.num_objects:
            return None, None
        num_objects, size = self.warehouse.dataset_stats(dataset, version)
        if num_objects != dataset_version.stats.num_objects:
            values["num_objects"] = num_objects
        if size != dataset_version.stats.size:
            values["size"] = size
        return num_objects, size

    def _generate_preview(self, dataset, version):
        from datachain.query.dataset import DatasetQuery

        return (
            DatasetQuery(
                name=dataset.name,
                namespace_name=dataset.project.namespace.name,
                project_name=dataset.project.name,
                version=version,
                catalog=self,
                include_incomplete=True,
            )
            .limit(20)
            .to_db_records()
        )

    def _log_stats_anomaly(
        self,
        dataset,
        version,
        dataset_version,
        stats_num_objects,
        stats_size,
        preview_rows,
    ):
        logger.warning(
            "Inconsistency detected for %s@%s: "
            "Initial state: num_objects=%s, size=%s, has_preview=%s. "
            "dataset_stats returned: num_objects=%s, size=%s. "
            "Preview generated: %s rows. "
            "This may indicate ClickHouse replication delay.",
            dataset.name,
            version,
            dataset_version.stats.num_objects,
            dataset_version.stats.size,
            False,
            stats_num_objects,
            stats_size,
            preview_rows,
        )

    def _check_preview_loaded(self, dataset_version):
        if dataset_version._preview_loaded:
            raise RuntimeError(
                "update_dataset_version_with_warehouse_info expects preview to be "
                "unloaded and regenerates it from warehouse rows"
            )

    def update_dataset_version_with_warehouse_info(
        self, dataset: DatasetRecord, version: str, **kwargs
    ) -> None:
        dataset_version = dataset.get_version(version)
        self._check_preview_loaded(dataset_version)
        values = {**kwargs}
        stats_num_objects, stats_size, preview = self._fill_version_info(
            dataset,
            version,
            dataset_version,
            values,
        )
        self._log_anomaly_if_needed(
            dataset,
            version,
            dataset_version,
            stats_num_objects,
            stats_size,
            preview,
        )
        if values:
            self.metastore.update_dataset_version(dataset, version, **values)

    def _fill_version_info(self, dataset, version, dataset_version, values):
        stats_num_objects, stats_size = self._fill_missing_stats(
            dataset,
            version,
            dataset_version,
            values,
        )
        preview = self._generate_preview(dataset, version)
        values["preview"] = preview
        return stats_num_objects, stats_size, preview

    def _log_anomaly_if_needed(
        self,
        dataset,
        version,
        dataset_version,
        stats_num_objects,
        stats_size,
        preview,
    ):
        if stats_num_objects == 0 and len(preview):
            self._log_stats_anomaly(
                dataset,
                version,
                dataset_version,
                stats_num_objects,
                stats_size,
                len(preview),
            )

    def complete_dataset_version(
        self,
        dataset: DatasetRecord,
        version: str,
        *,
        error_message: str = "",
        error_stack: str = "",
        script_output: str = "",
        **kwargs,
    ) -> None:
        """Finalize a dataset version after its rows table has been populated.

        This refreshes warehouse-derived metadata first, then marks the version
        as COMPLETE.
        """
        self.update_dataset_version_with_warehouse_info(dataset, version, **kwargs)
        self.metastore.update_dataset_status(
            dataset,
            DatasetStatus.COMPLETE,
            version=version,
            error_message=error_message,
            error_stack=error_stack,
            script_output=script_output,
        )

    def update_dataset(self, dataset: DatasetRecord, **kwargs) -> DatasetRecord:
        """Updates dataset fields."""
        dataset_updated = self.metastore.update_dataset(dataset, **kwargs)
        self.warehouse.rename_dataset_tables(dataset, dataset_updated)
        return dataset_updated

    def remove_dataset_version(
        self, dataset: DatasetRecord, version: str, drop_rows: bool | None = True
    ) -> None:
        """
        Deletes one single dataset version.
        If it was last version, it removes dataset completely.
        """
        if not dataset.has_version(version):
            return
        self.metastore.update_dataset_version(
            dataset, version, status=DatasetStatus.REMOVING
        )
        if drop_rows:
            self.warehouse.drop_dataset_rows_table(dataset, version)
        dataset = self.metastore.remove_dataset_version(dataset, version)

    def _remove_single_version(self, dataset, version):
        try:
            self.remove_dataset_version(dataset, version)
            return 1
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to remove dataset %s version %s: %s",
                dataset.name,
                version,
                e,
            )
            return 0

    def _remove_versions(self, pairs: Iterable[tuple[DatasetRecord, str]]) -> int:
        num_removed = 0
        for dataset, version in pairs:
            num_removed += self._remove_single_version(dataset, version)
        return num_removed

    def remove_dataset_versions(
        self, job_id: str | None = None, version_ids: list[int] | None = None
    ) -> int:
        versions_to_remove = self.metastore.get_dataset_versions(
            job_id=job_id,
            version_ids=version_ids,
        )
        return self._remove_versions(versions_to_remove)

    def get_temp_table_names(self) -> list[str]:
        return self.warehouse.get_temp_table_names()

    def cleanup_tables(self, names: Iterable[str]) -> None:
        """
        Drop tables passed.

        This should be implemented to ensure that the provided tables
        are cleaned up as soon as they are no longer needed.
        """
        self.warehouse.cleanup_tables(names)

    def cleanup_dataset_versions(self, job_id: str | None = None) -> int:
        """
        Clean up dataset versions that are no longer needed.

        Removes dataset versions that:
        - Have status CREATED, FAILED, STALE, or REMOVING
        - Belong to completed/failed/canceled jobs (not running)
        - Are session_* datasets from finished jobs (orphaned intermediates)

        Returns:
            Number of removed versions
        """
        versions_to_clean = self.metastore.get_dataset_versions_to_clean(job_id=job_id)
        return self._remove_versions(versions_to_clean)

    def _build_chain(self, source, recursive, client_config):
        from datachain import read_dataset, read_storage

        if source.startswith(DATASET_PREFIX):
            return read_dataset(source[len(DATASET_PREFIX) :], session=self.session)
        return read_storage(
            source,
            session=self.session,
            recursive=recursive,
            client_config=client_config or self.config.client_config,
        )

    def _build_chains_from_sources(self, sources, recursive, client_config):
        return [self._build_chain(s, recursive, client_config) for s in sources]

    def create_dataset_from_sources(
        self,
        name: str,
        sources: list[str],
        project: Project | None = None,
        client_config=None,
        recursive=False,
    ) -> "DataChain":
        if not sources:
            raise ValueError("Sources needs to be non empty list")

        project = project or self.metastore.default_project
        chains = self._build_chains_from_sources(sources, recursive, client_config)

        return (
            reduce(lambda dc1, dc2: dc1.union(dc2), chains)
            .settings(project=project.name, namespace=project.namespace.name)
            .save(name, sources="\n".join(sources), query_script="")
        )

    @staticmethod
    def _resolve_namespace_project_from_env():
        namespace_env = os.environ.get("DATACHAIN_NAMESPACE")
        project_env = os.environ.get("DATACHAIN_PROJECT")
        if project_env and "." in project_env:
            namespace_env, project_env = project_env.split(".", 1)
        return namespace_env, project_env

    def get_full_dataset_name(
        self,
        name: str,
        project_name: str | None = None,
        namespace_name: str | None = None,
    ) -> tuple[str, str, str]:
        parsed_namespace_name, parsed_project_name, name = parse_dataset_name(name)
        namespace_env, project_env = self._resolve_namespace_project_from_env()

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

    def _maybe_override_listing_namespace(self, name, namespace_name, project_name):
        from datachain.lib.listing import is_listing_dataset

        if is_listing_dataset(name):
            return (
                self.metastore.system_namespace_name,
                self.metastore.listing_project_name,
            )
        return namespace_name, project_name

    def get_dataset(
        self,
        name: str,
        namespace_name: str | None = None,
        project_name: str | None = None,
        *,
        versions: Sequence[str] | None = (),
        include_incomplete: bool = True,
        include_preview: bool = False,
    ) -> DatasetRecord:
        namespace_name = namespace_name or self.metastore.default_namespace_name
        project_name = project_name or self.metastore.default_project_name
        namespace_name, project_name = self._maybe_override_listing_namespace(
            name, namespace_name, project_name
        )
        return self.metastore.get_dataset(
            name,
            namespace_name=namespace_name,
            project_name=project_name,
            versions=versions,
            include_incomplete=include_incomplete,
            include_preview=include_preview,
        )

    def _try_get_local_dataset(
        self, name, namespace_name, project_name, version, include_incomplete
    ):
        try:
            ds = self.get_dataset(
                name,
                namespace_name=namespace_name,
                project_name=project_name,
                versions=None,
                include_incomplete=include_incomplete,
            )
            if not version or ds.has_version(version):
                return ds
        except (NamespaceNotFoundError, ProjectNotFoundError, DatasetNotFoundError):
            pass
        return None

    def _pull_and_get_dataset(
        self, name, namespace_name, project_name, version, include_incomplete
    ):
        remote_ds_uri = create_dataset_uri(name, namespace_name, project_name, version)
        self.pull_dataset(
            remote_ds_uri=remote_ds_uri,
            local_ds_name=name,
            local_ds_version=version,
        )
        return self.get_dataset(
            name,
            namespace_name=namespace_name,
            project_name=project_name,
            versions=None,
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
        if version:
            update = False
        no_fallback = is_namespace_local(namespace_name)

        ds = self._try_local_dataset_or_raise(
            name,
            namespace_name,
            project_name,
            version,
            include_incomplete,
            no_fallback,
            update,
        )
        if ds is not None:
            return ds

        return self._fallback_to_remote(
            name,
            namespace_name,
            project_name,
            version,
            include_incomplete,
            pull_dataset,
        )

    def _fallback_to_remote(
        self,
        name,
        namespace_name,
        project_name,
        version,
        include_incomplete,
        pull_dataset,
    ):
        if pull_dataset:
            return self._pull_and_get_dataset(
                name,
                namespace_name,
                project_name,
                version,
                include_incomplete,
            )
        return self.get_remote_dataset(namespace_name, project_name, name)

    def _try_local_dataset_or_raise(
        self,
        name,
        namespace_name,
        project_name,
        version,
        include_incomplete,
        no_fallback,
        update,
    ):
        if no_fallback or not update:
            ds = self._try_get_local_dataset(
                name,
                namespace_name,
                project_name,
                version,
                include_incomplete,
            )
            if ds is not None:
                return ds
        if no_fallback:
            raise DatasetNotFoundError(
                f"Dataset {name}"
                + (f" version {version} " if version else " ")
                + f"not found in namespace {namespace_name} and project {project_name}"
            )
        return None

    def get_remote_dataset(
        self, namespace: str, project: str, name: str
    ) -> DatasetRecord:
        from datachain.remote.studio import StudioClient

        studio_client = StudioClient()
        info_response = studio_client.dataset_info(namespace, project, name)
        self._validate_dataset_info_response(info_response, namespace, project, name)
        dataset_info = info_response.data
        assert isinstance(dataset_info, dict)
        return DatasetRecord.from_dict(dataset_info)

    @staticmethod
    def _validate_dataset_info_response(info_response, namespace, project, name):
        if not info_response.ok:
            if info_response.status == 404:
                raise DatasetNotFoundError(
                    f"Dataset {namespace}.{project}.{name} not found"
                )
            raise DataChainError(info_response.message)

    def _get_root_dependencies(
        self, dependency_map, children_map, dataset_id, version_id
    ):
        root_key = (dataset_id, version_id)
        if root_key not in children_map:
            return []
        root_dependency_ids = children_map[root_key]
        return [dependency_map[dep_id] for dep_id in root_dependency_ids]

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
        root_dependencies = self._get_root_dependencies(
            dependency_map, children_map, dataset_id, version_id
        )

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
            versions=[version],
            include_incomplete=False,
        )
        dataset_version = dataset.get_version(version)

        if not indirect:
            return self.metastore.get_direct_dataset_dependencies(dataset, version)

        return self.get_dataset_dependencies_by_ids(
            dataset.id,
            dataset_version.id,
            indirect,
        )

    def _ls_datasets_from_studio(self, prefix):
        from datachain.remote.studio import StudioClient

        client = StudioClient()
        response = client.ls_datasets(prefix=prefix)
        if not response.ok:
            raise DataChainError(response.message)
        if not response.data:
            return []
        return [
            d
            for d in (DatasetListRecord.from_dict(i) for i in response.data)
            if not d.name.startswith(QUERY_DATASET_PREFIX)
        ]

    def _filter_datasets(self, datasets, include_listing):
        from datachain.query.session import Session

        for d in datasets:
            if Session.is_temp_dataset(d.name):
                continue
            if not d.is_bucket_listing or include_listing:
                yield d

    def _fetch_datasets(self, prefix, studio, project_id):
        if studio:
            return self._ls_datasets_from_studio(prefix)
        if prefix:
            return self.metastore.list_datasets_by_prefix(prefix, project_id=project_id)
        return self.metastore.list_datasets(project_id=project_id)

    def ls_datasets(
        self,
        prefix: str | None = None,
        include_listing: bool = False,
        studio: bool = False,
        project: Project | None = None,
    ) -> Iterator[DatasetListRecord]:
        project_id = project.id if project else None
        datasets = self._fetch_datasets(prefix, studio, project_id)
        yield from self._filter_datasets(datasets, include_listing)

    def _preload_jobs(self, datasets):
        jobs_ids = {
            v.execution.job_id
            for ds in datasets
            for v in ds.versions
            if v.execution.job_id
        }
        if not jobs_ids:
            return {}
        return {j.id: j for j in self.metastore.list_jobs_by_ids(list(jobs_ids))}

    def list_datasets_versions(
        self,
        prefix: str | None = None,
        include_listing: bool = False,
        with_job: bool = True,
        studio: bool = False,
        project: Project | None = None,
    ) -> Iterator[tuple[DatasetListRecord, "DatasetListVersion", "Job | None"]]:
        datasets = list(
            self.ls_datasets(
                prefix=prefix,
                include_listing=include_listing,
                studio=studio,
                project=project,
            )
        )
        jobs = self._preload_jobs(datasets) if with_job else {}

        for d in datasets:
            yield from (
                (
                    d,
                    v,
                    jobs.get(str(v.execution.job_id))
                    if with_job and v.execution.job_id
                    else None,
                )
                for v in d.versions
            )

    def listings(self, prefix: str | None = None) -> list["ListingInfo"]:
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

    def _build_dataset_query(self, dataset, version):
        from datachain.query.dataset import DatasetQuery

        return DatasetQuery(
            name=dataset.name,
            namespace_name=dataset.project.namespace.name,
            project_name=dataset.project.name,
            version=version,
            catalog=self,
        )

    def ls_dataset_rows(
        self,
        dataset: DatasetRecord,
        version: str,
        offset=None,
        limit=None,
    ) -> list[dict]:
        q = self._build_dataset_query(dataset, version)
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
        client_config = client_config or self.config.client_config
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
            versions=[version],
        )

        self.warehouse.export_dataset_table(
            bucket,
            dataset,
            version,
            file_format=file_format,
            base_file_name=base_file_name,
            client_config=client_config,
        )

    def _remove_all_versions(self, dataset):
        for v in dataset.versions:
            self.remove_dataset_version(dataset, v.version)

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
            versions=None,
        )
        self._validate_remove_dataset(name, version, force, dataset)

        if version:
            self.remove_dataset_version(dataset, version)
        else:
            self._remove_all_versions(dataset)

    @staticmethod
    def _validate_remove_dataset(name, version, force, dataset):
        if not version and not force:
            raise ValueError(f"Missing dataset version from input for dataset {name}")
        if version and not dataset.has_version(version):
            raise DatasetInvalidVersionError(
                f"Dataset {name} doesn't have version {version}"
            )

    @staticmethod
    def _set_if_not_none(update_data, key, value):
        if value is not None:
            update_data[key] = value

    @staticmethod
    def _build_edit_update_data(new_name, description, attrs):
        update_data = {}
        if new_name:
            DatasetRecord.validate_name(new_name)
            update_data["name"] = new_name
        DatasetRecord._set_if_not_none(update_data, "description", description)
        DatasetRecord._set_if_not_none(update_data, "attrs", attrs)
        return update_data

    def edit_dataset(
        self,
        name: str,
        project: Project | None = None,
        new_name: str | None = None,
        description: str | None = None,
        attrs: list[str] | None = None,
    ) -> DatasetRecord:
        update_data = self._build_edit_update_data(new_name, description, attrs)
        dataset = self.get_dataset(
            name,
            namespace_name=project.namespace.name if project else None,
            project_name=project.name if project else None,
            versions=None,
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
            client_config=client_config or self.config.client_config,
        ) as data_sources:
            if data_sources is None:
                return

            for source in data_sources:
                yield source, source.ls(fields)

    def _instantiate_dataset(
        self,
        ds_uri: str,
        output: str,
        force: bool,
        client_config: dict | None,
    ) -> None:
        """Copy dataset files to *output* directory."""
        assert output, "output must be provided when instantiating a dataset"
        self.cp(
            [ds_uri],
            output,
            force=force,
            client_config=client_config,
        )
        print(f"Dataset {ds_uri} instantiated locally to {output}")

    def _parse_pull_uri(self, remote_ds_uri: str) -> tuple[str, str, str, str | None]:
        try:
            return parse_dataset_uri(remote_ds_uri)
        except ValueError as e:
            raise DataChainError("Error when parsing dataset uri") from e

    def _validate_pull_remote(
        self, namespace: str | None, project: str | None, name: str
    ) -> None:
        if not namespace or not project:
            raise DataChainError(
                f"Invalid fully qualified dataset name {name}, namespace"
                f" or project missing"
            )

    def _validate_pull_local_name(
        self,
        local_ds_name: str | None,
        remote_namespace: str,
        remote_project: str,
    ) -> None:
        if not local_ds_name:
            return
        local_ns, local_proj, _ = parse_dataset_name(local_ds_name)
        if local_ns and local_ns != remote_namespace:
            raise DataChainError("Local namespace must be the same to remote namespace")
        if local_proj and local_proj != remote_project:
            raise DataChainError("Local project must be the same to remote project")

    def _resolve_pull_version(
        self,
        remote_ds: Any,
        version: str | None,
    ) -> Any:
        try:
            if not version:
                version = remote_ds.latest_version
                print(f"Version not specified, pulling the latest one (v{version})")
            return remote_ds.get_version(version)
        except (DatasetVersionNotFoundError, StopIteration) as exc:
            raise DataChainError(
                f"Dataset {remote_ds.name} doesn't have version {version} on server"
            ) from exc

    def _reuse_complete_version(self, ds, ver, cp, output, force, client_config):
        ds_uri = create_dataset_uri(
            ds.name,
            ds.project.namespace.name,
            ds.project.name,
            ver.version,
        )
        print(f"Dataset already available locally as {ds_uri}")
        if cp:
            assert output is not None
            self._instantiate_dataset(ds_uri, output, force, client_config)
        return True

    def _reuse_or_cleanup_pull_version(
        self,
        version_uuid,
        cp,
        output,
        force,
        client_config,
    ) -> bool:
        try:
            return self._try_reuse_or_cleanup(
                version_uuid,
                cp,
                output,
                force,
                client_config,
            )
        except DatasetNotFoundError:
            pass
        return False

    def _try_reuse_or_cleanup(self, version_uuid, cp, output, force, client_config):
        ds = self.metastore.get_dataset_by_version_uuid(
            version_uuid, include_incomplete=True
        )
        ver = ds.get_version_by_uuid(version_uuid)
        if ver.status == DatasetStatus.COMPLETE:
            return self._reuse_complete_version(
                ds,
                ver,
                cp,
                output,
                force,
                client_config,
            )
        print("Cleaning up stale existing dataset version")
        self.remove_dataset_version(ds, ver.version)
        return False

    def _ensure_pull_namespace_project(self, remote_ds: Any) -> tuple[Any, Any]:
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
        return namespace, project

    def _handle_version_conflict(self, local_dataset, local_ds_version, local_ds_uri):
        local_ver = local_dataset.get_version(local_ds_version)
        if local_ver.status != DatasetStatus.COMPLETE:
            print(f"Cleaning up stale incomplete version (uuid={local_ver.uuid})")
            self.remove_dataset_version(local_dataset, local_ds_version)
        else:
            raise DataChainError(
                f"Local dataset {local_ds_uri} already exists with"
                " different uuid, please choose different local"
                " dataset name or version"
            )

    def _check_pull_name_conflict(
        self,
        local_ds_name,
        local_ds_version,
        local_ds_uri,
        namespace,
        project,
    ) -> None:
        try:
            local_dataset = self.get_dataset(
                local_ds_name,
                namespace_name=namespace.name,
                project_name=project.name,
                versions=None,
                include_incomplete=True,
            )
            if local_dataset.has_version(local_ds_version):
                self._handle_version_conflict(
                    local_dataset, local_ds_version, local_ds_uri
                )
        except DatasetNotFoundError:
            pass

    def _fetch_signed_urls(
        self, studio_client, remote_ds, remote_ds_version, temp_table_name
    ):
        export_response = studio_client.export_dataset_table(
            remote_ds, remote_ds_version.version
        )
        if not export_response.ok:
            with suppress(Exception):
                self.warehouse.cleanup_tables([temp_table_name])
            raise DataChainError(export_response.message)
        return export_response.data

    def _run_rows_fetcher(
        self,
        signed_urls,
        export_id,
        schema,
        studio_client,
        temp_table_name,
        progress_bar,
    ):
        with self.warehouse.clone() as warehouse:
            rows_fetcher = DatasetRowsFetcher(
                warehouse=warehouse,
                temp_table_name=temp_table_name,
                export_id=export_id,
                schema=schema,
                studio_client=studio_client,
                progress_bar=progress_bar,
            )
            self._run_fetcher_batches(
                rows_fetcher, signed_urls, temp_table_name, progress_bar
            )

    def _run_fetcher_batches(
        self, rows_fetcher, signed_urls, temp_table_name, progress_bar
    ):
        try:
            batches = _round_robin_batch(signed_urls, PULL_DATASET_MAX_THREADS)
            rows_fetcher.run(iter(batches), progress_bar)
        except Exception:
            with suppress(Exception):
                self.warehouse.cleanup_tables([temp_table_name])
            raise

    def _download_export_data(
        self,
        studio_client,
        remote_ds,
        remote_ds_version,
        schema,
        temp_table_name,
        progress_bar,
    ):
        export_data = self._fetch_signed_urls(
            studio_client, remote_ds, remote_ds_version, temp_table_name
        )
        signed_urls = export_data["signed_urls"]
        if signed_urls:
            self._run_rows_fetcher(
                signed_urls,
                export_data["export_id"],
                schema,
                studio_client,
                temp_table_name,
                progress_bar,
            )

    def _export_and_download_pull(
        self,
        studio_client,
        remote_ds,
        remote_ds_version,
        columns,
        schema,
        progress_bar,
    ) -> str:
        temp_table_name = self.warehouse.temp_table_name()
        self.warehouse.create_dataset_rows_table(temp_table_name, columns=columns)
        self._download_export_data(
            studio_client,
            remote_ds,
            remote_ds_version,
            schema,
            temp_table_name,
            progress_bar,
        )
        return temp_table_name

    def _rename_and_finalize(
        self,
        local_ds,
        local_ds_version,
        temp_table_name,
        remote_ds_version,
    ):
        final_table_name = self.warehouse.dataset_table_name(local_ds, local_ds_version)
        temp_table = self.warehouse.get_table(temp_table_name)
        self.warehouse.rename_table(temp_table, final_table_name)

        self.complete_dataset_version(
            local_ds,
            local_ds_version,
            error_message=remote_ds_version.execution.error_message,
            error_stack=remote_ds_version.execution.error_stack,
            script_output=remote_ds_version.execution.script_output,
        )

    def _finalize_pull_dataset(
        self,
        local_ds_name,
        local_ds_version,
        project,
        temp_table_name,
        remote_ds_version,
        columns,
    ) -> None:
        try:
            self._finalize_with_cleanup(
                local_ds_name,
                local_ds_version,
                project,
                temp_table_name,
                remote_ds_version,
                columns,
            )
        except Exception:
            with suppress(Exception):
                self.warehouse.cleanup_tables([temp_table_name])
            raise

    def _finalize_with_cleanup(
        self,
        local_ds_name,
        local_ds_version,
        project,
        temp_table_name,
        remote_ds_version,
        columns,
    ):
        local_ds = self.create_dataset(
            local_ds_name,
            project,
            local_ds_version,
            query_script=remote_ds_version.execution.query_script,
            columns=columns,
            feature_schema=remote_ds_version.schema_info.feature_schema,
            validate_version=False,
            uuid=remote_ds_version.uuid,
        )
        self._rename_and_finalize(
            local_ds,
            local_ds_version,
            temp_table_name,
            remote_ds_version,
        )

    def _prepare_pull(self, remote_ds_uri, local_ds_name, local_ds_version, cp, output):
        if cp and not output:
            raise ValueError("Please provide output directory for instantiation")

        from datachain.remote.studio import StudioClient

        studio_client = StudioClient()
        return self._prepare_pull_inner(
            studio_client,
            remote_ds_uri,
            local_ds_name,
            local_ds_version,
        )

    def _prepare_pull_inner(
        self, studio_client, remote_ds_uri, local_ds_name, local_ds_version
    ):
        ns, proj, ds_name, version = self._parse_pull_uri(remote_ds_uri)
        self._validate_pull_remote(ns, proj, ds_name)
        self._validate_pull_local_name(local_ds_name, ns, proj)
        remote_ds, remote_ds_version = self._fetch_and_resolve(
            ns, proj, ds_name, version
        )
        local_ds_name, local_ds_version, local_ds_uri = self._build_local_ds_info(
            local_ds_name,
            local_ds_version,
            remote_ds,
            remote_ds_version,
        )
        return (
            studio_client,
            remote_ds,
            remote_ds_version,
            local_ds_name,
            local_ds_version,
            local_ds_uri,
        )

    def _fetch_and_resolve(self, ns, proj, ds_name, version):
        remote_ds = self.get_remote_dataset(ns, proj, ds_name)
        remote_ds_version = self._resolve_pull_version(remote_ds, version)
        return remote_ds, remote_ds_version

    @staticmethod
    def _build_local_ds_info(
        local_ds_name, local_ds_version, remote_ds, remote_ds_version
    ):
        local_ds_name = local_ds_name or remote_ds.name
        local_ds_version = local_ds_version or remote_ds_version.version
        local_ds_uri = create_dataset_uri(
            local_ds_name,
            remote_ds.project.namespace.name,
            remote_ds.project.name,
            local_ds_version,
        )
        return local_ds_name, local_ds_version, local_ds_uri

    def _inside_pull_lock(
        self,
        studio_client,
        remote_ds,
        remote_ds_version,
        local_ds_name,
        local_ds_version,
        local_ds_uri,
        remote_ds_uri,
        cp,
        output,
        force,
        client_config,
    ):
        if self._reuse_or_cleanup_pull_version(
            remote_ds_version.uuid, cp, output, force, client_config
        ):
            return True

        namespace, project = self._ensure_pull_namespace_project(remote_ds)
        self._check_pull_name_conflict(
            local_ds_name, local_ds_version, local_ds_uri, namespace, project
        )

        schema = parse_schema(remote_ds_version.schema_info.schema)
        columns = tuple(sa.Column(n, t) for n, t in schema.items() if n != "sys__id")

        self._pull_with_progress(
            studio_client,
            remote_ds,
            remote_ds_version,
            local_ds_name,
            local_ds_version,
            project,
            columns,
            schema,
            remote_ds_uri,
            local_ds_uri,
        )
        return False

    def _execute_pull(
        self,
        studio_client,
        remote_ds,
        remote_ds_version,
        local_ds_name,
        local_ds_version,
        local_ds_uri,
        remote_ds_uri,
        cp,
        output,
        force,
        client_config,
    ):
        lock_dir = os.path.join(DataChainDir.find().tmp, "pull-locks")
        lock_path = os.path.join(lock_dir, f"{remote_ds_version.uuid}.lock")
        with interprocess_file_lock(
            lock_path,
            wait_message=(
                "Another pull for this dataset version is already"
                " in progress. Waiting..."
            ),
        ):
            self._inside_pull_lock(
                studio_client,
                remote_ds,
                remote_ds_version,
                local_ds_name,
                local_ds_version,
                local_ds_uri,
                remote_ds_uri,
                cp,
                output,
                force,
                client_config,
            )

    def _pull_with_progress(
        self,
        studio_client,
        remote_ds,
        remote_ds_version,
        local_ds_name,
        local_ds_version,
        project,
        columns,
        schema,
        remote_ds_uri,
        local_ds_uri,
    ):
        with tqdm(
            desc=f"Saving dataset {remote_ds_uri} locally: ",
            unit=" rows",
            unit_scale=True,
            unit_divisor=1000,
            total=remote_ds_version.stats.num_objects,
            leave=False,
        ) as pbar:
            temp_table_name = self._export_and_download_pull(
                studio_client,
                remote_ds,
                remote_ds_version,
                columns,
                schema,
                pbar,
            )
            self._finalize_pull_dataset(
                local_ds_name,
                local_ds_version,
                project,
                temp_table_name,
                remote_ds_version,
                columns,
            )

        print(f"Dataset {remote_ds_uri} saved locally as {local_ds_uri}")

    def _after_pull(self, cp, output, force, local_ds_uri, client_config):
        if cp:
            assert output is not None
            self._instantiate_dataset(local_ds_uri, output, force, client_config)

    def pull_dataset(
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
        result = self._prepare_pull(
            remote_ds_uri, local_ds_name, local_ds_version, cp, output
        )
        (
            studio_client,
            remote_ds,
            remote_ds_version,
            local_ds_name,
            local_ds_version,
            local_ds_uri,
        ) = result
        self._execute_pull(
            studio_client,
            remote_ds,
            remote_ds_version,
            local_ds_name,
            local_ds_version,
            local_ds_uri,
            remote_ds_uri,
            cp,
            output,
            force,
            client_config,
        )
        self._after_pull(cp, output, force, local_ds_uri, client_config)

    def _run_cp_and_listing(
        self, sources, output, force, update, recursive, no_glob, no_cp, client_config
    ):
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
            with self.enlist_sources(
                sources,
                update,
                client_config=client_config or self.config.client_config,
            ):
                pass

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
        self._run_cp_and_listing(
            sources, output, force, update, recursive, no_glob, no_cp, client_config
        )
        self.create_dataset_from_sources(
            output,
            sources,
            self.metastore.default_project,
            client_config=client_config,
            recursive=recursive,
        )

    def _build_bar_format(self, output):
        desc_max_len = max(len(output) + 16, 19)
        return (
            "{desc:<"
            f"{desc_max_len}"
            "}{percentage:3.0f}%|{bar}| {n_fmt:>5}/{total_fmt:<5} "
            "[{elapsed}<{remaining}, {rate_fmt:>8}]"
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
        client_config = client_config or self.config.client_config
        node_groups = self.enlist_sources_grouped(
            sources,
            update,
            no_glob,
            client_config=client_config,
        )
        try:
            self._execute_cp(node_groups, output, force, recursive, no_cp)
        finally:
            self._close_node_groups(node_groups)

    def _execute_cp(self, node_groups, output, force, recursive, no_cp):
        always_copy_dir_contents, copy_to_filename = prepare_output_for_cp(
            node_groups,
            output,
            force,
            no_cp,
        )
        total_size, total_files = collect_nodes_for_cp(node_groups, recursive)
        if not total_files:
            return
        bar_format = self._build_bar_format(output)
        self._download_node_groups(
            node_groups, no_cp, bar_format, total_size, recursive
        )
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

    def _download_node_groups(
        self, node_groups, no_cp, bar_format, total_size, recursive
    ):
        if not no_cp:
            with get_download_bar(bar_format, total_size) as pbar:
                for node_group in node_groups:
                    node_group.download(recursive=recursive, pbar=pbar)

    @staticmethod
    def _close_node_groups(node_groups):
        for node_group in node_groups:
            with suppress(Exception):
                node_group.close()

    def _du_dirs(self, src, node, subdepth):
        if subdepth > 0:
            subdirs = src.listing.get_dirs_by_parent_path(node.path)
            for sd in subdirs:
                yield from self._du_dirs(src, sd, subdepth - 1)
        yield (src.get_node_uri(node), src.listing.du(node)[0])

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
            client_config=client_config or self.config.client_config,
        ) as matched_sources:
            if matched_sources is None:
                return
            for src in matched_sources:
                yield from self._du_dirs(src, src.node, depth)

    def _columns_to_fields(self, columns):
        field_set = set()
        for column in columns:
            self._add_column_field(field_set, column)
        return list(field_set)

    @staticmethod
    def _add_column_field(field_set, column):
        if column in _COLUMN_FIELD_MAP:
            field_set.update(_COLUMN_FIELD_MAP[column])

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
            client_config=client_config or self.config.client_config,
        ) as matched_sources:
            if matched_sources is None:
                return
            yield from self._execute_find(
                matched_sources,
                names,
                inames,
                paths,
                ipaths,
                size,
                typ,
                columns,
            )

    def _execute_find(
        self,
        matched_sources,
        names,
        inames,
        paths,
        ipaths,
        size,
        typ,
        columns,
    ):
        if not columns:
            columns = ["path"]
        fields = self._columns_to_fields(columns)
        field_lookup = {f: i for i, f in enumerate(fields)}
        for src in matched_sources:
            results = src.listing.find(
                src.node,
                fields,
                names,
                inames,
                paths,
                ipaths,
                size,
                typ,
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
            client_config=client_config or self.config.client_config,
            only_index=True,
        ):
            pass

    def _cleanup_output_tables(self, checkpoints):
        output_tables = [ch.table_name for ch in checkpoints]
        if not output_tables:
            return
        logger.info(
            "Removing %d UDF output tables: %s", len(output_tables), output_tables
        )
        self.warehouse.cleanup_tables(output_tables)

    def _cleanup_partition_tables(self, checkpoints):
        expired_job_ids = {ch.job_id for ch in checkpoints}
        all_tables = self.warehouse.db.list_tables(
            pattern=Checkpoint.partition_table_pattern()
        )
        partition_tables = [
            t for t in all_tables if any(j_id in t for j_id in expired_job_ids)
        ]
        if not partition_tables:
            return
        logger.info(
            "Removing %d partition tables: %s", len(partition_tables), partition_tables
        )
        self.warehouse.cleanup_tables(partition_tables)

    def _cleanup_input_tables(self, inactive_group_ids):
        for group_id in inactive_group_ids:
            input_tables = self.warehouse.db.list_tables(
                pattern=Checkpoint.input_table_pattern(group_id)
            )
            if not input_tables:
                continue
            logger.info(
                "Removing %d shared input tables: %s", len(input_tables), input_tables
            )
            self.warehouse.cleanup_tables(input_tables)

    def cleanup_checkpoints(self, ttl_seconds: int | None = None) -> int:
        if ttl_seconds is None:
            ttl_seconds = CHECKPOINTS_TTL

        ttl_threshold = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)

        checkpoints, inactive_group_ids = self.metastore.expire_checkpoints(
            ttl_threshold,
        )
        if not checkpoints:
            return 0

        return self._perform_checkpoint_cleanup(checkpoints, inactive_group_ids)

    def _perform_checkpoint_cleanup(self, checkpoints, inactive_group_ids):
        logger.info(
            "Cleaning %d expired checkpoints across %d inactive run groups",
            len(checkpoints),
            len(inactive_group_ids),
        )
        self._cleanup_output_tables(checkpoints)
        self._cleanup_partition_tables(checkpoints)
        self._cleanup_input_tables(inactive_group_ids)
        self.metastore.update_checkpoints(
            [ch.id for ch in checkpoints],
            status=CheckpointStatus.DELETED,
        )
        logger.info(
            "Checkpoint cleanup complete: removed %d checkpoints",
            len(checkpoints),
        )
        return len(checkpoints)
