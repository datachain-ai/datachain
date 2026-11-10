from typing import TYPE_CHECKING

from datachain.dataset import DatasetDependencyType
from datachain.error import DatasetVersionNotFoundError
from datachain.lib.dataset_metadata import DatasetMetadata, read_direct_dependencies
from datachain.skill.knowledge.scripts.utils import dep_entry
from datachain.skill.knowledge.snapshot import (
    MAX_VERSION_ENTRIES,
    build_dataset_snapshot,
)
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
        record = metastore.get_dataset(
            name,
            namespace,
            project,
            versions=None,
            include_incomplete=True,
            include_preview=True,
        )
        metadata = DatasetMetadata.from_record(record, version, source)
    except DatasetVersionNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return snapshot_from_metadata(metadata, metastore)


def snapshot_from_metadata(
    metadata: DatasetMetadata,
    metastore: "AbstractMetastore | None",
    *,
    max_version_entries: int | None = MAX_VERSION_ENTRIES,
) -> "DatasetSnapshot":
    record = metadata.record

    def deps_provider(ver: "DatasetVersion") -> "list[DependencyEntry] | None":
        if metastore is None:
            return None
        edges = read_direct_dependencies(metastore, metadata, ver.version)
        if edges is None:
            return None
        return [
            dep_entry(_dep_name(e, metastore, metadata.source), e.version, e.type)
            if e is not None
            else dep_entry(None, None, None)
            for e in edges
        ]

    local_default = (
        metadata.source == "local"
        and metastore is not None
        and record.project is not None
        and metastore.is_default_project(
            record.project.name, record.project.namespace.name
        )
    )
    return build_dataset_snapshot(
        name=record.name if local_default else metadata.name,
        source=metadata.source,
        attrs=list(record.attrs or []),
        description=record.description or None,
        versions=metadata.history,
        deps_provider=deps_provider,
        max_version_entries=max_version_entries,
    )


def _dep_name(
    dep: "DatasetDependency", metastore: "AbstractMetastore", source: str
) -> str:
    if dep.type == DatasetDependencyType.STORAGE or (
        source == "local" and metastore.is_default_project(dep.project, dep.namespace)
    ):
        return dep.name
    return str(DatasetIdentifier(dep.namespace, dep.project, dep.name, None))
