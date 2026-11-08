"""Recorded dataset metadata for Python inspection and CLI rendering."""

from dataclasses import dataclass, fields
from datetime import UTC, datetime
from functools import cached_property
from typing import Any

from datachain.job import Job
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.listing_info import ListingInfo
from datachain.lib.signal_schema import SignalSchema
from datachain.utils import DatasetIdentifier


def _name(info: DatasetInfo) -> str:
    return str(DatasetIdentifier(info.namespace, info.project, info.name, info.version))


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _version_dict(info: DatasetInfo) -> dict[str, Any]:
    return {
        "name": _name(info),
        "dataset_uuid": info.dataset_uuid,
        "version_uuid": info.uuid,
        "num_objects": info.num_objects,
        "created_at": _timestamp(info.created_at),
        "finished_at": _timestamp(info.finished_at) if info.finished_at else None,
        "size": info.size,
        "status": info.status,
        "job_id": info.job_id,
    }


def _job_dict(job: Job) -> dict[str, Any]:
    result = {f.name: getattr(job, f.name) for f in fields(Job)}
    for name, value in result.items():
        if isinstance(value, datetime):
            result[name] = _timestamp(value)
    return result


@dataclass
class DatasetDependencyInfo:
    """A recorded direct dependency or dependent and its saved version metadata."""

    dataset: DatasetInfo
    removed: bool = False

    @property
    def name(self) -> str:
        """Canonical dataset name including the exact version."""
        return _name(self.dataset)

    @property
    def num_objects(self) -> int | None:
        """Recorded row count of the dependency, not its contribution to the result."""
        return self.dataset.num_objects

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible dependency metadata."""
        return {
            **_version_dict(self.dataset),
            "type": "storage" if isinstance(self.dataset, ListingInfo) else "dataset",
            "removed": self.removed,
        }


@dataclass
class DatasetDescription:
    """A selected completed version, available versions, and direct dependencies.

    ``selected_version`` and ``versions`` use the same DatasetInfo model as
    ``dc.datasets()``. Storage dependencies use ListingInfo, as in ``dc.listings()``.
    The schema is a SignalSchema; no dataset rows or current storage objects are read.
    """

    selected_version: DatasetInfo
    feature_schema: dict[str, Any]
    versions: list[DatasetInfo]
    description: str | None
    dependencies: list[DatasetDependencyInfo | None] | None
    dependents: list[DatasetDependencyInfo] | None
    jobs: dict[str, Job] | None

    @cached_property
    def schema(self) -> SignalSchema:
        """Typed view of the saved schema, following SignalSchema resolution rules."""
        return SignalSchema.deserialize(self.feature_schema)

    @property
    def job(self) -> Job | None:
        """Recorded producing job of the selected version, when its record exists."""
        return (self.jobs or {}).get(self.selected_version.job_id or "")

    @property
    def name(self) -> str:
        """Fully qualified name and exact selected version."""
        return _name(self.selected_version)

    @property
    def dataset_uuid(self) -> str | None:
        """Dataset identifier, stable across versions and renames within its catalog."""
        return self.selected_version.dataset_uuid

    @property
    def version_uuid(self) -> str:
        """Identifier of the selected saved version."""
        return self.selected_version.uuid

    @property
    def available_versions(self) -> list[str]:
        """Every available completed version, in ascending semantic-version order."""
        return [v.version for v in self.versions]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata with dataset and version UUIDs."""
        selected = self.selected_version
        return {
            **_version_dict(selected),
            "description": self.description,
            "schema": self.feature_schema,
            "versions": [_version_dict(v) for v in self.versions],
            "dependencies": [
                entry.to_dict() if entry else None for entry in self.dependencies
            ]
            if self.dependencies is not None
            else None,
            "dependents": [entry.to_dict() for entry in self.dependents]
            if self.dependents is not None
            else None,
            "query_script": selected.query_script,
            "jobs": [_job_dict(self.jobs[job_id]) for job_id in sorted(self.jobs)]
            if self.jobs is not None
            else None,
        }
