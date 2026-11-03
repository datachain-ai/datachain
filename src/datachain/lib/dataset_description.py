"""Recorded dataset metadata for Python inspection and CLI rendering."""

from dataclasses import dataclass, fields
from datetime import datetime
from functools import cached_property
from typing import Any

from datachain.job import Job
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.listing_info import ListingInfo
from datachain.lib.signal_schema import SignalSchema
from datachain.utils import DatasetIdentifier


def _name(info: DatasetInfo) -> str:
    return str(DatasetIdentifier(info.namespace, info.project, info.name, info.version))


def _version_dict(info: DatasetInfo) -> dict[str, Any]:
    return {
        "name": _name(info),
        "num_objects": info.num_objects,
        "created_at": info.created_at.isoformat(),
        "finished_at": info.finished_at.isoformat() if info.finished_at else None,
        "size": info.size,
        "status": info.status,
        "job_id": info.job_id,
    }


def _job_dict(job: Job) -> dict[str, Any]:
    result = {f.name: getattr(job, f.name) for f in fields(Job)}
    for name, value in result.items():
        if isinstance(value, datetime):
            result[name] = value.isoformat()
    return result


@dataclass
class DatasetDependencyInfo:
    """A recorded ancestor or dependent, with its entire version's row count."""

    dataset: DatasetInfo
    removed: bool = False
    depends_on: list[str | None] | None = None

    @property
    def name(self) -> str:
        """Canonical dataset name including the exact version."""
        return _name(self.dataset)

    @property
    def num_objects(self) -> int | None:
        """Recorded row count of the dependency, not its contribution to the result."""
        return self.dataset.num_objects

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dependency with one version identifier."""
        return {
            **_version_dict(self.dataset),
            "type": "storage" if isinstance(self.dataset, ListingInfo) else "dataset",
            "removed": self.removed,
            "depends_on": self.depends_on,
        }


@dataclass
class DatasetDescription:
    """A selected completed version, all available versions, and its lineage.

    ``selected_version`` and ``versions`` use the same DatasetInfo model as
    ``dc.datasets()``. Storage dependencies use ListingInfo, as in ``dc.listings()``.
    The schema is a SignalSchema; no dataset rows or current storage objects are read.
    """

    selected_version: DatasetInfo
    feature_schema: dict[str, Any]
    versions: list[DatasetInfo]
    description: str | None
    sources: list[str]
    depends_on: list[str | None] | None
    lineage: list[DatasetDependencyInfo] | None
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
    def available_versions(self) -> list[str]:
        """Every available completed version, in ascending semantic-version order."""
        return [v.version for v in self.versions]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata with one identifier per version."""
        selected = self.selected_version
        return {
            **_version_dict(selected),
            "description": self.description,
            "schema": self.feature_schema,
            "versions": [_version_dict(v) for v in self.versions],
            "sources": self.sources,
            "depends_on": self.depends_on,
            "lineage": [entry.to_dict() for entry in self.lineage]
            if self.lineage is not None
            else None,
            "dependents": [entry.to_dict() for entry in self.dependents]
            if self.dependents is not None
            else None,
            "query_script": selected.query_script,
            "jobs": [_job_dict(self.jobs[job_id]) for job_id in sorted(self.jobs)]
            if self.jobs is not None
            else None,
        }
