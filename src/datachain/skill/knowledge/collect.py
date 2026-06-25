"""Build a dataset snapshot from any `AbstractMetastore` (local catalog, Studio's
PostgreSQL, a worker's API client), so the acquisition isn't re-implemented per side."""

from typing import TYPE_CHECKING

from datachain.dataset import DatasetDependencyType
from datachain.skill.knowledge.scripts.utils import dep_entry
from datachain.skill.knowledge.snapshot import build_dataset_snapshot

if TYPE_CHECKING:
    from datachain.data_storage.metastore import AbstractMetastore
    from datachain.dataset import DatasetDependency, DatasetRecord, DatasetVersion
    from datachain.skill.knowledge.types import DatasetSnapshot, DependencyEntry


def collect_dataset_snapshot(
    metastore: "AbstractMetastore",
    name: str,
    namespace: str | None = None,
    project: str | None = None,
    *,
    source: str = "studio",
) -> "DatasetSnapshot":
    """Build a dataset's snapshot from metastore reads alone (no warehouse access)."""
    record = metastore.get_dataset(
        name,
        namespace,
        project,
        versions=None,
        include_incomplete=False,
        include_preview=True,
    )

    def deps_provider(version: "DatasetVersion") -> "list[DependencyEntry]":
        edges = metastore.get_direct_dataset_dependencies(record, version.version) or []
        # A `None` edge is a deleted target — keep a name-less entry so it still warns.
        return [
            dep_entry(_dep_name(e), e.version, e.type)
            if e is not None
            else dep_entry(None, None, None)
            for e in edges
        ]

    return build_dataset_snapshot(
        name=_qualified_name(record),
        source=source,
        attrs=list(record.attrs or []),
        description=record.description or None,
        versions=record.versions,
        deps_provider=deps_provider,
    )


def _dep_name(dep: "DatasetDependency") -> str:
    # Storage edges carry the raw `lst__…` name (cleaned to a URI by `dep_entry`);
    # dataset edges store namespace/project/name separately, so re-qualify them.
    if dep.type == DatasetDependencyType.STORAGE:
        return dep.name
    return f"{dep.namespace}.{dep.project}.{dep.name}"


def _qualified_name(record: "DatasetRecord") -> str:
    project = record.project
    if project is None:
        return record.name
    return f"{project.namespace.name}.{project.name}.{record.name}"
