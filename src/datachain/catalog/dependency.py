import builtins
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from datachain.dataset import DatasetDependency

DDN = TypeVar("DDN", bound="DatasetDependencyNode")


@dataclass
class DatasetDependencyNode:
    namespace: str
    project: str
    id: int
    dataset_id: int | None
    dataset_version_id: int | None
    dataset_name: str | None
    dataset_version: str | None
    created_at: datetime
    source_dataset_id: int
    source_dataset_version_id: int | None
    depth: int
    dataset_version_status: int | None = None

    @classmethod
    def parse(
        cls: builtins.type[DDN],
        namespace: str,
        project: str,
        id: int,
        dataset_id: int | None,
        dataset_version_id: int | None,
        dataset_name: str | None,
        dataset_version: str | None,
        created_at: datetime,
        source_dataset_id: int,
        source_dataset_version_id: int | None,
        depth: int,
        dataset_version_status: int | None = None,
    ) -> "DatasetDependencyNode | None":
        return cls(
            namespace,
            project,
            id,
            dataset_id,
            dataset_version_id,
            dataset_name,
            dataset_version,
            created_at,
            source_dataset_id,
            source_dataset_version_id,
            depth,
            dataset_version_status,
        )

    def to_dependency(self) -> "DatasetDependency | None":
        return DatasetDependency.parse(
            namespace_name=self.namespace,
            project_name=self.project,
            id=self.id,
            dataset_id=self.dataset_id,
            dataset_version_id=self.dataset_version_id,
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            dataset_version_created_at=self.created_at,
            dataset_version_status=self.dataset_version_status,
        )


def build_dependency_hierarchy(
    dependency_nodes: list[DatasetDependencyNode | None],
    indirect: bool = False,
) -> tuple[
    dict[int, DatasetDependency | None], dict[tuple[int, int | None], list[int]]
]:
    """
    Build dependency hierarchy from dependency nodes.

    Args:
        dependency_nodes: List of DatasetDependencyNode objects from the database

    Returns:
        Tuple of (dependency_map, children_map) where:
        - dependency_map: Maps dependency_id -> DatasetDependency
        - children_map: Maps (source_dataset_id, source_version_id) ->
          list of dependency_ids
    """
    dependency_map: dict[int, DatasetDependency | None] = {}
    children_map: dict[tuple[int, int | None], list[int]] = {}

    nodes = {node.id: node for node in dependency_nodes if node is not None}
    for node in nodes.values():
        dependency = node.to_dependency()
        parent_key = (node.source_dataset_id, node.source_dataset_version_id)

        if dependency is not None:
            dependency_map[dependency.id] = dependency
            children_map.setdefault(parent_key, []).append(dependency.id)
        else:
            # Handle case where dependency creation failed (e.g., deleted dependency)
            dependency_map[node.id] = None
            children_map.setdefault(parent_key, []).append(node.id)

    if indirect:
        for node in nodes.values():
            dependency = dependency_map[node.id]
            if (
                dependency is not None
                and node.dataset_id is not None
                and node.dataset_version_id is not None
            ):
                dependency.dependencies = [
                    dependency_map[child_id]
                    for child_id in children_map.get(
                        (node.dataset_id, node.dataset_version_id), []
                    )
                ]

    return dependency_map, children_map
