from typing import TYPE_CHECKING

from datachain.dataset import DatasetDependencyType, DatasetStatus
from datachain.skill.knowledge.scripts.utils import dep_entry, parse_semver
from datachain.skill.knowledge.snapshot import build_dataset_snapshot

if TYPE_CHECKING:
    from datachain.data_storage.metastore import AbstractMetastore
    from datachain.dataset import (
        DatasetDependency,
        DatasetRecord,
        DatasetVersion,
        Project,
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
    record = metastore.get_dataset(
        name,
        namespace,
        project,
        versions=None,
        include_incomplete=True,
        include_preview=True,
    )

    # Filtering here rather than through include_incomplete=False, which inner-joins
    # and raises DatasetNotFoundError for a dataset whose versions are all incomplete.
    completed = [v for v in record.versions if v.status == DatasetStatus.COMPLETE]
    ordered = sorted(completed, key=lambda v: parse_semver(v.version))
    target = next((i for i, v in enumerate(ordered) if v.version == version), None)
    if target is None:
        raise ValueError(f"{name} has no completed version {version}")
    history = ordered[: target + 1]

    def deps_provider(ver: "DatasetVersion") -> "list[DependencyEntry]":
        edges = metastore.get_direct_dataset_dependencies(record, ver.version) or []
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
        versions=history,
        deps_provider=deps_provider,
    )


def _dep_name(dep: "DatasetDependency") -> str:
    if dep.type == DatasetDependencyType.STORAGE:
        return dep.name
    return f"{dep.namespace}.{dep.project}.{dep.name}"


def _qualified_name(record: "DatasetRecord") -> str:
    project: Project | None = record.project
    if project is None:
        return record.name
    return f"{project.namespace.name}.{project.name}.{record.name}"
