from dataclasses import dataclass
from typing import TYPE_CHECKING

from packaging.version import Version

from datachain.dataset import DatasetStatus, parse_dataset_with_version
from datachain.error import DatasetVersionNotFoundError
from datachain.lib.listing import is_listing_dataset
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
            key=lambda v: Version(v.version),
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
        return f"{project.namespace.name}.{project.name}.{self.record.name}"

    @property
    def history(self) -> "list[DatasetVersion]":
        return self.versions[: self.versions.index(self.selected_version) + 1]

    def dependencies(self, version: str) -> "list[DatasetDependency | None] | None":
        if self.metastore is None:
            return None
        return self.metastore.get_direct_dataset_dependencies(self.record, version)


def read_dataset_metadata(
    catalog: "Catalog",
    name: str,
    namespace: str | None = None,
    project: str | None = None,
    version: str | None = None,
    *,
    include_preview: bool = False,
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
    if not is_studio() and identifier.namespace not in {
        metastore.default_namespace_name,
        metastore.system_namespace_name,
    }:
        record = catalog.get_remote_dataset(
            identifier.namespace, identifier.project, identifier.name
        )
        return DatasetMetadata.from_record(record, version, None, "studio")
    return DatasetMetadata.from_metastore(
        metastore,
        identifier.name,
        identifier.namespace,
        identifier.project,
        version=version,
        include_preview=include_preview,
        source="studio" if is_studio() else "local",
    )
