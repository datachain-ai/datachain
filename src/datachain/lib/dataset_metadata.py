from collections import defaultdict
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from datachain.data_storage import JobQueryType, JobStatus
from datachain.dataset import DatasetStatus, parse_dataset_with_version
from datachain.error import (
    DataChainError,
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
    NamespaceNotFoundError,
    ProjectNotFoundError,
)
from datachain.job import Job
from datachain.lib.dataset_description import DatasetDependencyInfo, DatasetDescription
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.listing import is_listing_dataset
from datachain.lib.listing_info import ListingInfo
from datachain.utils import DatasetIdentifier

if TYPE_CHECKING:
    from datachain.catalog import Catalog
    from datachain.data_storage.metastore import AbstractMetastore
    from datachain.dataset import (
        DatasetDependency,
        DatasetRecord,
        DatasetVersion,
        Project,
    )


@dataclass(frozen=True)
class DatasetMetadata:
    record: "DatasetRecord"
    versions: "list[DatasetVersion]"
    selected_version: "DatasetVersion"
    source: str

    @classmethod
    def from_record(
        cls,
        record: "DatasetRecord",
        version: str | None,
        source: str,
    ) -> "DatasetMetadata":
        versions = sorted(
            (v for v in record.versions if v.status == DatasetStatus.COMPLETE),
            key=lambda v: v.version_value,
        )
        selected = (
            next((v for v in versions if v.version == version), None)
            if version
            else (versions[-1] if versions else None)
        )
        if selected is None:
            raise DatasetVersionNotFoundError(
                f"{record.name} has no completed version {version or '(latest)'}"
            )
        return cls(record, versions, selected, source)

    @property
    def name(self) -> str:
        project: Project | None = self.record.project
        if project is None:
            return self.record.name
        return str(
            DatasetIdentifier(
                project.namespace.name, project.name, self.record.name, None
            )
        )

    @property
    def history(self) -> "list[DatasetVersion]":
        return self.versions[: self.versions.index(self.selected_version) + 1]


def read_dataset_metadata(
    catalog: "Catalog",
    name: str,
    namespace: str | None = None,
    project: str | None = None,
    version: str | None = None,
    *,
    include_preview: bool = False,
    studio: bool = False,
) -> DatasetMetadata:
    from datachain.lib.dc.utils import is_studio

    metastore = catalog.metastore
    bare_name, name_version = parse_dataset_with_version(name)
    listing_prefix = (
        f"{metastore.system_namespace_name}.{metastore.listing_project_name}."
    )
    listing_name = bare_name.removeprefix(listing_prefix)
    if is_listing_dataset(listing_name):
        identifier = DatasetIdentifier(
            metastore.system_namespace_name,
            metastore.listing_project_name,
            listing_name,
            name_version,
        )
    else:
        identifier = catalog.parse_dataset_name(name, namespace, project)
    version = version or identifier.version
    remote_fallback = not is_studio() and identifier.namespace not in {
        metastore.default_namespace_name,
        metastore.system_namespace_name,
    }
    if not studio or is_studio():
        try:
            record = metastore.get_dataset(
                identifier.name,
                identifier.namespace,
                identifier.project,
                versions=None,
                # Incomplete-only datasets must stay distinct from missing ones.
                include_incomplete=True,
                include_preview=include_preview,
            )
            return DatasetMetadata.from_record(
                record, version, "studio" if is_studio() else "local"
            )
        except (
            DatasetNotFoundError,
            DatasetVersionNotFoundError,
            NamespaceNotFoundError,
            ProjectNotFoundError,
        ):
            if not remote_fallback:
                raise
    record = catalog.get_remote_dataset(
        identifier.namespace, identifier.project, identifier.name
    )
    return DatasetMetadata.from_record(record, version, "studio")


def is_pulled_copy(metadata: DatasetMetadata, metastore: "AbstractMetastore") -> bool:
    project = metadata.record.project
    return (
        metadata.source == "local"
        and project is not None
        and project.namespace.name
        not in {metastore.default_namespace_name, metastore.system_namespace_name}
    )


def _read_related_versions(
    metastore: "AbstractMetastore",
    dependencies: "list[DatasetDependency | None] | None",
    dependents: "list[DatasetDependency | None] | None",
) -> "dict[DatasetIdentifier, tuple[DatasetRecord, DatasetVersion]]":
    inputs = [dep for dep in dependencies or [] if dep and dep.version]
    outputs = [dep for dep in dependents or [] if dep and dep.version]
    requested = {
        DatasetIdentifier.from_dataset_dependency(dep) for dep in [*inputs, *outputs]
    }
    if not requested:
        return {}
    try:
        pairs = metastore.get_dataset_related_versions(inputs, outputs)
    except NotImplementedError:
        grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for ref in requested:
            assert ref.version is not None
            grouped[ref.namespace, ref.project, ref.name].append(ref.version)
        pairs = []
        for (namespace, project, name), versions in grouped.items():
            try:
                record = metastore.get_dataset(
                    name, namespace, project, versions=versions, include_removed=True
                )
            except (
                DatasetNotFoundError,
                DatasetVersionNotFoundError,
                NamespaceNotFoundError,
                ProjectNotFoundError,
            ):
                continue
            pairs.extend((record, v.version) for v in record.all_versions)
    result = {}
    for record, version in pairs:
        key = DatasetIdentifier(
            record.project.namespace.name, record.project.name, record.name, version
        )
        if key in requested:
            result[key] = record, record.get_version(version)
    return result


def _read_studio_job(job_id: str) -> dict[str, Job] | None:
    from datachain.remote.studio import StudioClient

    response = StudioClient().get_jobs(job_id=job_id)
    if not response.ok:
        if response.status in (403, 429):
            return None
        raise DataChainError(response.message)
    adapter = TypeAdapter(Job)
    return {
        data["id"]: adapter.validate_python(
            {
                **data,
                # Public Studio params contain environment variables.
                "params": {},
                "status": JobStatus[data["status"]].value,
                "query_type": JobQueryType[data["query_type"]].value,
                "is_remote_execution": True,
            }
        )
        for data in response.data or []
        if data["id"] == job_id
    }


def read_dataset_description(
    catalog: "Catalog", metadata: DatasetMetadata
) -> DatasetDescription:
    from datachain.lib.dc.utils import is_studio
    from datachain.query.session import Session

    record = metadata.record
    selected = metadata.selected_version
    dependencies: list[DatasetDependency | None] | None = None
    dependents: list[DatasetDependency | None] | None = None
    related: dict[DatasetIdentifier, tuple[DatasetRecord, DatasetVersion]] = {}
    jobs: dict[str, Job] | None = {}
    pulled_copy = False
    if metadata.source == "studio" and not is_studio():
        if selected.job_id:
            jobs = _read_studio_job(selected.job_id)
    else:
        metastore = catalog.metastore
        pulled_copy = is_pulled_copy(metadata, metastore)
        dependencies = metastore.get_direct_dataset_dependencies(
            record, selected.version
        )
        if dependencies is not None:
            dependencies = [
                dep if dep and dep.version else None for dep in dependencies
            ]
        if not dependencies and pulled_copy:
            dependencies = None
        try:
            dependents = metastore.get_direct_dataset_dependents(
                record, selected.version
            )
        except NotImplementedError:
            dependents = None
        if dependents is not None:
            dependents = [
                dep
                for dep in dependents
                if dep and not Session.is_temp_dataset(dep.name)
            ]
        related = _read_related_versions(metastore, dependencies, dependents)
        if selected.job_id:
            jobs = {
                job.id: Job(**{f.name: getattr(job, f.name) for f in fields(Job)})
                for job in metastore.list_jobs_by_ids([selected.job_id])
            }

    def version_info(record: "DatasetRecord", version: "DatasetVersion") -> DatasetInfo:
        info_cls = ListingInfo if is_listing_dataset(record.name) else DatasetInfo
        return info_cls.from_models(
            record, version, (jobs or {}).get(version.job_id or "")
        )

    def dependency_info(
        dep: "DatasetDependency | None",
    ) -> DatasetDependencyInfo | None:
        if not dep or not dep.version:
            return None
        pair = related.get(DatasetIdentifier.from_dataset_dependency(dep))
        return (
            DatasetDependencyInfo(version_info(*pair), removed=dep.removed)
            if pair
            else None
        )

    dependent_info = (
        sorted(
            [info for dep in dependents if (info := dependency_info(dep)) is not None],
            key=lambda entry: entry.name,
        )
        if dependents is not None
        else None
    )
    if not dependent_info and pulled_copy:
        dependent_info = None
    versions = [version_info(record, version) for version in metadata.versions]
    return DatasetDescription(
        selected_version=next(v for v in versions if v.version == selected.version),
        feature_schema=selected.feature_schema,
        versions=versions,
        description=record.description,
        dependencies=sorted(
            [dependency_info(dep) for dep in dependencies],
            key=lambda entry: entry.name if entry else "",
        )
        if dependencies is not None
        else None,
        dependents=dependent_info,
        jobs=jobs,
        source=metadata.source,
    )
