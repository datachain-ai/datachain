import glob
import os
from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import TYPE_CHECKING

from sqlalchemy import Column
from sqlalchemy.sql import func
from tqdm.auto import tqdm

from datachain.node import DirType, Node, NodeWithPath
from datachain.sql.functions import path as pathfunc
from datachain.utils import suffix_to_number

if TYPE_CHECKING:
    from datachain.catalog.datasource import DataSource
    from datachain.client import Client
    from datachain.data_storage import AbstractMetastore, AbstractWarehouse
    from datachain.dataset import DatasetRecord


class Listing:
    def __init__(
        self,
        metastore: "AbstractMetastore",
        warehouse: "AbstractWarehouse",
        client: "Client",
        dataset_name: str | None = None,
        dataset_version: str | None = None,
        column: str = "file",
    ):
        self.metastore = metastore
        self.warehouse = warehouse
        self.client = client
        self.dataset_name = dataset_name  # dataset representing bucket listing
        self.dataset_version = dataset_version  # dataset representing bucket listing
        self.column = column
        self._closed = False

    def clone(self) -> "Listing":
        return self.__class__(
            self.metastore.clone(),
            self.warehouse.clone(),
            self.client,
            self.dataset_name,
            self.dataset_version,
            self.column,
        )

    def __enter__(self) -> "Listing":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.warehouse.close_on_exit()
        finally:
            self.metastore.close_on_exit()

    @property
    def uri(self):
        from datachain.lib.listing import listing_uri_from_name

        assert self.dataset_name

        return listing_uri_from_name(self.dataset_name)

    @cached_property
    def dataset(self) -> "DatasetRecord":
        assert self.dataset_name
        project = self.metastore.listing_project
        return self.metastore.get_dataset(
            self.dataset_name,
            namespace_name=project.namespace.name,
            project_name=project.name,
        )

    @cached_property
    def dataset_rows(self):
        dataset = self.dataset
        return self.warehouse.dataset_rows(
            dataset,
            self.dataset_version or dataset.latest_version,
            column=self.column,
        )

    def expand_path(self, path, use_glob=True) -> list[Node]:
        if use_glob and glob.has_magic(path):
            return self.warehouse.expand_path(self.dataset_rows, path)
        return [self.resolve_path(path)]

    def resolve_path(self, path) -> Node:
        return self.warehouse.get_node_by_path(self.dataset_rows, path)

    def ls_path(self, node, fields):
        if node.location or node.dir_type == DirType.TAR_ARCHIVE:
            return self.warehouse.select_node_fields_by_parent_path_tar(
                self.dataset_rows, node.path, fields
            )
        return self.warehouse.select_node_fields_by_parent_path(
            self.dataset_rows, node.path, fields
        )

    def collect_nodes_to_instantiate(
        self,
        sources: Iterable["DataSource"],
        copy_to_filename: str | None,
        recursive=False,
        copy_dir_contents=False,
        from_dataset=False,
    ) -> list[NodeWithPath]:
        all_nodes: list[NodeWithPath] = []
        for src in sources:
            node = src.node
            if recursive and src.is_container():
                dir_path = []
                if not copy_dir_contents:
                    dir_path.append(node.name)
                subtree_nodes = src.find(sort=["path"])
                all_nodes.extend(
                    NodeWithPath(n.n, path=dir_path + n.path) for n in subtree_nodes
                )
            else:
                node_path = []
                if copy_to_filename:
                    node_path = [os.path.basename(copy_to_filename)]
                elif from_dataset:
                    node_path = [
                        src.listing.client.name,
                        node.path,
                    ]
                else:
                    node_path = [node.name]
                all_nodes.append(NodeWithPath(node, path=node_path))
        return all_nodes

    def instantiate_nodes(
        self,
        all_nodes: Iterable[NodeWithPath],
        output,
        total_files=None,
        force=False,
        shared_progress_bar=None,
    ) -> None:
        progress_bar = shared_progress_bar or tqdm(
            desc=f"Instantiating '{output}'",
            unit=" files",
            unit_scale=True,
            unit_divisor=1000,
            total=total_files,
            leave=False,
        )

        counter = 0
        for node in all_nodes:
            node.instantiate(self.client, output, progress_bar, force=force)
            counter += 1
            if counter > 1000:
                progress_bar.update(counter)
                counter = 0

        progress_bar.update(counter)

    def find(
        self,
        node,
        fields,
        names=None,
        inames=None,
        paths=None,
        ipaths=None,
        size=None,
        type=None,
        order_by=None,
    ):
        dr = self.dataset_rows
        conds = []
        if names:
            for name in names:
                conds.append(
                    pathfunc.name(Column(dr.col_name("path"))).op("GLOB")(name)
                )
        if inames:
            for iname in inames:
                conds.append(
                    func.lower(pathfunc.name(Column(dr.col_name("path")))).op("GLOB")(
                        iname.lower()
                    )
                )
        if paths:
            for path in paths:
                conds.append(Column(dr.col_name("path")).op("GLOB")(path))
        if ipaths:
            for ipath in ipaths:
                conds.append(
                    func.lower(Column(dr.col_name("path"))).op("GLOB")(ipath.lower())
                )

        if size is not None:
            size_limit = suffix_to_number(size)
            if size_limit >= 0:
                conds.append(Column(dr.col_name("size")) >= size_limit)
            else:
                conds.append(Column(dr.col_name("size")) <= -size_limit)

        return self.warehouse.find(
            dr,
            node,
            fields,
            type=type,
            conds=conds,
            order_by=order_by,
        )

    def du(self, node: Node, count_files: bool = False):
        return self.warehouse.size(self.dataset_rows, node, count_files)

    def subtree_files(self, node: Node, sort=None):
        if node.dir_type == DirType.TAR_ARCHIVE or node.location:
            include_subobjects = True
        else:
            include_subobjects = False

        return self.warehouse.get_subtree_files(
            self.dataset_rows,
            node,
            sort=sort,
            include_subobjects=include_subobjects,
        )

    def get_dirs_by_parent_path(
        self,
        parent_path: str,
    ) -> Iterator[Node]:
        return self.warehouse.get_dirs_by_parent_path(self.dataset_rows, parent_path)
