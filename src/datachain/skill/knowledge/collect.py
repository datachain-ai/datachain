from typing import TYPE_CHECKING

from datachain.dataset import DatasetDependencyType
from datachain.error import DatasetVersionNotFoundError
from datachain.lib.dataset_metadata import DatasetMetadata
from datachain.skill.knowledge.scripts.utils import dep_entry
from datachain.skill.knowledge.snapshot import build_dataset_snapshot
from datachain.utils import DatasetIdentifier

if TYPE_CHECKING:
    from datachain.data_storage.metastore import AbstractMetastore
    from datachain.dataset import (
        DatasetDependency,
        DatasetVersion,
    )
    from datachain.skill.knowledge.types import DatasetSnapshot, DependencyEntry


def collect_dataset_snapshot(
    metastore: "AbstractMetastore",
    name: str,
    namespace: str | None = None,
    project: str | None = None,
    *,
    version: str,
    source: str = "studio",
) -> "DatasetSnapshot":
    """A dataset's snapshot describing `version`, with the history leading up to it.

    Raises ValueError when the dataset has no such completed version.
    """
    try:
        metadata = DatasetMetadata.from_metastore(
            metastore,
            name,
            namespace,
            project,
            version=version,
            include_preview=True,
            source=source,
        )
    except DatasetVersionNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return snapshot_from_metadata(metadata)


def snapshot_from_metadata(metadata: DatasetMetadata) -> "DatasetSnapshot":
    record = metadata.record

    def deps_provider(ver: "DatasetVersion") -> "list[DependencyEntry]":
        edges = metadata.dependencies(ver.version) or []
        return [
            dep_entry(_dep_name(e, metadata), e.version, e.type)
            if e is not None
            else dep_entry(None, None, None)
            for e in edges
        ]

    local_default = (
        metadata.source == "local"
        and metadata.metastore is not None
        and record.project is not None
        and metadata.metastore.is_default_project(
            record.project.name, record.project.namespace.name
        )
    )
    snapshot = build_dataset_snapshot(
        name=record.name if local_default else metadata.name,
        source=metadata.source,
        attrs=list(record.attrs or []),
        description=record.description or None,
        versions=metadata.history,
        deps_provider=deps_provider,
    )
    if metadata.metastore is None:
        snapshot.setdefault("warnings", []).append(
            "Dependencies are unavailable through the Studio API; "
            "dependency changes could not be determined."
        )
    return snapshot


def _dep_name(dep: "DatasetDependency", metadata: DatasetMetadata) -> str:
    if dep.type == DatasetDependencyType.STORAGE or (
        metadata.source == "local"
        and metadata.metastore is not None
        and metadata.metastore.is_default_project(dep.project, dep.namespace)
    ):
        return dep.name
    return str(DatasetIdentifier(dep.namespace, dep.project, dep.name, None))
