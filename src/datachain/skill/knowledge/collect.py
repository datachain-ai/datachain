from typing import TYPE_CHECKING

from datachain.dataset import DatasetDependencyType
from datachain.error import DatasetVersionNotFoundError
from datachain.lib.dataset_metadata import DatasetMetadata
from datachain.skill.knowledge.scripts.utils import dep_entry
from datachain.skill.knowledge.snapshot import build_dataset_snapshot

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
            dep_entry(_dep_name(e, metadata.source), e.version, e.type)
            if e is not None
            else dep_entry(None, None, None)
            for e in edges
        ]

    snapshot = build_dataset_snapshot(
        name=record.name if metadata.source == "local" else metadata.name,
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


def _dep_name(dep: "DatasetDependency", source: str) -> str:
    if dep.type == DatasetDependencyType.STORAGE or source == "local":
        return dep.name
    return f"{dep.namespace}.{dep.project}.{dep.name}"
