from collections import defaultdict
from dataclasses import dataclass, field, fields
from functools import cached_property
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


@dataclass
class DatasetMetadata:
    record: "DatasetRecord"
    versions: "list[DatasetVersion]"
    selected_version: "DatasetVersion"
    metastore: "AbstractMetastore | None"
    source: str
    _dependencies: dict[str, "list[DatasetDependency | None] | None"] = field(
        default_factory=dict, init=False, repr=False
    )
    _dependents: dict[str, "list[DatasetDependency | None] | None"] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def from_record(
        cls,
        record: "DatasetRecord",
        version: str | None,
        metastore: "AbstractMetastore | None",
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
        return cls(record, versions, selected, metastore, source)

    @classmethod
    def from_metastore(
        cls,
        metastore: "AbstractMetastore",
        name: str,
        namespace: str | None = None,
        project: str | None = None,
        *,
        version: str | None = None,
        include_preview: bool = False,
        source: str = "studio",
    ) -> "DatasetMetadata":
        record = metastore.get_dataset(
            name,
            namespace,
            project,
            versions=None,
            # Incomplete-only datasets must remain distinguishable from missing ones.
            include_incomplete=True,
            include_preview=include_preview,
        )
        return cls.from_record(record, version, metastore, source)

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

    @property
    def _is_remote_copy(self) -> bool:
        project = self.record.project
        return (
            self.source == "local"
            and self.metastore is not None
            and project is not None
            and project.namespace.name
            not in {
                self.metastore.default_namespace_name,
                self.metastore.system_namespace_name,
            }
        )

    def dependencies(self, version: str) -> "list[DatasetDependency | None] | None":
        if version not in self._dependencies:
            dependencies = (
                self.metastore.get_direct_dataset_dependencies(self.record, version)
                if self.metastore is not None
                else None
            )
            if not dependencies and self._is_remote_copy:
                dependencies = None
            self._dependencies[version] = (
                [dep if dep and dep.version else None for dep in dependencies]
                if dependencies is not None
                else None
            )
        return self._dependencies[version]

    def dependents(self, version: str) -> "list[DatasetDependency | None] | None":
        from datachain.query.session import Session

        if version not in self._dependents:
            try:
                dependents = (
                    self.metastore.get_direct_dataset_dependents(self.record, version)
                    if self.metastore is not None
                    else None
                )
            except NotImplementedError:
                dependents = None
            if dependents is not None:
                dependents = [
                    dep
                    for dep in dependents
                    if dep and not Session.is_temp_dataset(dep.name)
                ]
            if not dependents and self._is_remote_copy:
                dependents = None
            self._dependents[version] = dependents
        return self._dependents[version]

    @cached_property
    def related_versions(
        self,
    ) -> "dict[DatasetIdentifier, tuple[DatasetRecord, DatasetVersion]]":
        version = self.selected_version.version
        dependencies = [
            dep for dep in self.dependencies(version) or [] if dep and dep.version
        ]
        dependents = [
            dep for dep in self.dependents(version) or [] if dep and dep.version
        ]
        refs = [*dependencies, *dependents]
        requested = {
            DatasetIdentifier.from_dataset_dependency(dep)
            for dep in refs
            if dep and dep.version
        }
        if not requested or self.metastore is None:
            return {}
        try:
            pairs = self.metastore.get_dataset_related_versions(
                dependencies, dependents
            )
        except NotImplementedError:
            grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
            for ref in requested:
                assert ref.version is not None
                grouped[ref.namespace, ref.project, ref.name].append(ref.version)
            pairs = []
            for (namespace, project, name), versions in grouped.items():
                try:
                    record = self.metastore.get_dataset(
                        name,
                        namespace,
                        project,
                        versions=versions,
                        include_removed=True,
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
        for record, ver in pairs:
            key = DatasetIdentifier(
                record.project.namespace.name, record.project.name, record.name, ver
            )
            if key in requested:
                result[key] = record, record.get_version(ver)
        return result

    @cached_property
    def jobs(self) -> dict[str, Job] | None:
        ids = [self.selected_version.job_id] if self.selected_version.job_id else []
        if not ids:
            return {}
        if self.metastore is not None:
            return {
                job.id: Job(**{f.name: getattr(job, f.name) for f in fields(Job)})
                for job in self.metastore.list_jobs_by_ids(ids)
            }

        from datachain.remote.studio import StudioClient

        client = StudioClient()
        jobs: dict[str, Job] = {}
        adapter = TypeAdapter(Job)
        for job_id in ids:
            response = client.get_jobs(job_id=job_id)
            if not response.ok:
                if response.status in (403, 429):
                    return jobs or None
                raise DataChainError(response.message)
            for data in response.data or []:
                if data["id"] == job_id:
                    jobs[job_id] = adapter.validate_python(
                        {
                            **data,
                            # Public Studio params contain environment variables.
                            "params": {},
                            "status": JobStatus[data["status"]].value,
                            "query_type": JobQueryType[data["query_type"]].value,
                            "is_remote_execution": True,
                        }
                    )
        return jobs

    def describe(self) -> DatasetDescription:
        def version_info(record: "DatasetRecord", ver: "DatasetVersion") -> DatasetInfo:
            info_cls = ListingInfo if is_listing_dataset(record.name) else DatasetInfo
            return info_cls.from_models(
                record, ver, (self.jobs or {}).get(ver.job_id or "")
            )

        def dependency_info(
            dep: "DatasetDependency | None",
        ) -> DatasetDependencyInfo | None:
            if not dep or not dep.version:
                return None
            pair = self.related_versions.get(
                DatasetIdentifier.from_dataset_dependency(dep)
            )
            return (
                DatasetDependencyInfo(version_info(*pair), removed=dep.removed)
                if pair
                else None
            )

        dependencies = self.dependencies(self.selected_version.version)
        dependents = self.dependents(self.selected_version.version)
        dependent_info = (
            sorted(
                [
                    info
                    for dep in dependents
                    if (info := dependency_info(dep)) is not None
                ],
                key=lambda entry: entry.name,
            )
            if dependents is not None
            else None
        )
        if not dependent_info and self._is_remote_copy:
            dependent_info = None
        versions = [version_info(self.record, v) for v in self.versions]
        return DatasetDescription(
            selected_version=next(
                v for v in versions if v.version == self.selected_version.version
            ),
            feature_schema=self.selected_version.feature_schema,
            versions=versions,
            description=self.record.description,
            dependencies=sorted(
                [dependency_info(dep) for dep in dependencies],
                key=lambda entry: entry.name if entry else "",
            )
            if dependencies is not None
            else None,
            dependents=dependent_info,
            jobs=self.jobs,
            source=self.source,
        )


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
            return DatasetMetadata.from_metastore(
                metastore,
                identifier.name,
                identifier.namespace,
                identifier.project,
                version=version,
                include_preview=include_preview,
                source="studio" if is_studio() else "local",
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
    return DatasetMetadata.from_record(record, version, None, "studio")
