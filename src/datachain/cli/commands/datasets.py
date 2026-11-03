import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from tabulate import tabulate

from datachain import semver
from datachain.catalog import is_namespace_local
from datachain.cli.utils import determine_flavors
from datachain.config import Config
from datachain.data_storage import JobStatus
from datachain.error import (
    DataChainError,
    DatasetNotFoundError,
)
from datachain.lib.dataset_description import DatasetDescription
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.dc.datasets import describe_dataset
from datachain.lib.listing_info import ListingInfo
from datachain.query.session import Session
from datachain.studio import list_datasets as list_datasets_studio
from datachain.utils import DatasetIdentifier

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def group_dataset_versions(
    datasets: Iterable[tuple[str, str]], latest_only=True
) -> dict[str, str | list[str]]:
    grouped: dict[str, list[tuple[int, int, int]]] = {}

    # Sort to ensure groupby works as expected
    # (groupby expects consecutive items with the same key)
    for name, version in sorted(datasets):
        grouped.setdefault(name, []).append(semver.parse(version))

    if latest_only:
        # For each dataset name, pick the highest version.
        return {
            name: semver.create(*(max(versions))) for name, versions in grouped.items()
        }

    # For each dataset name, return a sorted list of unique versions.
    return {
        name: [semver.create(*v) for v in sorted(set(versions))]
        for name, versions in grouped.items()
    }


def list_datasets(
    catalog: "Catalog",
    studio: bool = False,
    local: bool = False,
    all: bool = False,
    team: str | None = None,
    latest_only: bool = True,
    name: str | None = None,
    include_removed: bool = False,
) -> None:
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = determine_flavors(studio, local, all, token)
    if name:
        latest_only = False

    local_pairs = (
        list(list_datasets_local(catalog, name, include_removed=include_removed))
        if all or local
        else []
    )
    local_datasets = {(n, v) for n, v, _ in local_pairs}
    local_removed = {(n, v) for n, v, r in local_pairs if r}
    studio_datasets = (
        set(list_datasets_studio(team=team, name=name, include_removed=include_removed))
        if all or studio
        else set()
    )

    # Group the datasets for both local and studio sources.
    local_grouped = group_dataset_versions(local_datasets, latest_only)
    studio_grouped = group_dataset_versions(studio_datasets, latest_only)

    # Merge all dataset names from both sources.
    all_dataset_names = sorted(set(local_grouped.keys()) | set(studio_grouped.keys()))

    datasets = []
    if latest_only:
        # For each dataset name, get the latest version from each source (if available).
        for n in all_dataset_names:
            datasets.append((n, (local_grouped.get(n), studio_grouped.get(n))))
    else:
        # For each dataset name, merge all versions from both sources.
        for n in all_dataset_names:
            local_versions = local_grouped.get(n, [])
            studio_versions = studio_grouped.get(n, [])

            # If neither source has any versions, record it as (None, None).
            if not local_versions and not studio_versions:
                datasets.append((n, (None, None)))
            else:
                # For each unique version from either source, record its presence.
                for version in sorted(set(local_versions) | set(studio_versions)):
                    datasets.append(
                        (
                            n,
                            (
                                version if version in local_versions else None,
                                version if version in studio_versions else None,
                            ),
                        )
                    )

    rows = [
        _datasets_tabulate_row(
            name=n,
            both=all or (local and studio),
            local_version=local_version,
            studio_version=studio_version,
            local_removed=local_removed,
        )
        for n, (local_version, studio_version) in datasets
    ]

    print(tabulate(rows, headers="keys"))


def list_datasets_local(
    catalog: "Catalog", name: str | None = None, include_removed: bool = False
) -> Iterator[tuple[str, str, bool]]:
    if name:
        yield from list_datasets_local_versions(
            catalog, name, include_removed=include_removed
        )
        return

    for d in catalog.ls_datasets(include_removed=include_removed):
        for v in d.all_versions:
            yield d.full_name, v.version, v.is_removed


def list_datasets_local_versions(
    catalog: "Catalog", name: str, include_removed: bool = False
) -> Iterator[tuple[str, str, bool]]:
    from datachain.dataset import DatasetStatus

    namespace_name, project_name, name = catalog.get_full_dataset_name(name)

    ds = catalog.get_dataset(
        name,
        namespace_name=namespace_name,
        project_name=project_name,
        versions=None,
        include_incomplete=False,
        include_removed=include_removed,
    )
    for v in ds.all_versions:
        if v.status == DatasetStatus.COMPLETE:
            yield name, v.version, False
        elif v.status == DatasetStatus.REMOVED:
            yield name, v.version, True


def _mark_local_version(version, name: str, local_removed: set[tuple[str, str]]) -> str:
    tag = " (removed)" if (name, str(version)) in local_removed else ""
    return f"v{version}{tag}"


def _datasets_tabulate_row(
    name,
    both,
    local_version,
    studio_version,
    local_removed: set[tuple[str, str]] | None = None,
) -> dict[str, str]:
    local_removed = local_removed or set()
    row = {
        "Name": name,
    }
    if both:
        row["Studio"] = f"v{studio_version}" if studio_version else "\u2716"
        row["Local"] = (
            _mark_local_version(local_version, name, local_removed)
            if local_version
            else "\u2716"
        )
    elif local_version:
        row["Latest Version"] = _mark_local_version(local_version, name, local_removed)
    elif studio_version:
        row["Latest Version"] = f"v{studio_version}"
    else:
        row["Latest Version"] = "\u2716"

    return row


def rm_dataset(
    catalog: "Catalog",
    name: str,
    version: str | None = None,
    force: bool | None = False,
    studio: bool | None = False,
    team: str | None = None,
) -> None:
    from datachain.dataset import parse_dataset_with_version

    name, name_version = parse_dataset_with_version(name)
    if version is None:
        version = name_version
    namespace_name, project_name, name = catalog.get_full_dataset_name(name)

    if studio:
        # removing Studio dataset from CLI
        from datachain.studio import remove_studio_dataset

        if Config().read().get("studio", {}).get("token"):
            remove_studio_dataset(
                team, name, namespace_name, project_name, version, force
            )
        else:
            raise DataChainError(
                "Not logged in to Studio. Log in with 'datachain auth login'."
            )
    else:
        try:
            project = catalog.metastore.get_project(project_name, namespace_name)
            catalog.remove_dataset(name, project, version=version, force=force)
        except DatasetNotFoundError:
            print("Dataset not found in local", file=sys.stderr)


def edit_dataset(
    catalog: "Catalog",
    name: str,
    new_name: str | None = None,
    description: str | None = None,
    attrs: list[str] | None = None,
    team: str | None = None,
) -> None:
    from datachain.lib.dc.utils import is_studio

    namespace_name, project_name, name = catalog.get_full_dataset_name(name)

    if is_studio() or is_namespace_local(namespace_name):
        try:
            catalog.edit_dataset(
                name, catalog.metastore.default_project, new_name, description, attrs
            )
        except DatasetNotFoundError:
            print("Dataset not found in local", file=sys.stderr)
    else:
        from datachain.studio import edit_studio_dataset

        if Config().read().get("studio", {}).get("token"):
            edit_studio_dataset(
                team, name, namespace_name, project_name, new_name, description, attrs
            )
        else:
            raise DataChainError(
                "Not logged in to Studio. Log in with 'datachain auth login'."
            )


def _timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S %Z").rstrip()


def _dataset_label(info: DatasetInfo) -> str:
    if isinstance(info, ListingInfo):
        return f"{info.uri}@{info.version}"
    return str(DatasetIdentifier(info.namespace, info.project, info.name, info.version))


def print_dataset_description(info: DatasetDescription) -> None:
    """Print a dataset description for the CLI."""
    selected = info.selected_version
    labels = {entry.name: _dataset_label(entry.dataset) for entry in info.lineage or []}
    counts = Counter(labels.values())
    for entry in info.lineage or []:
        label = labels[entry.name]
        if counts[label] > 1 and isinstance(entry.dataset, ListingInfo):
            kind = "all versions" if entry.dataset.versions else "current only"
            labels[entry.name] = f"{label} ({kind})"

    def names(values: Sequence[str | None] | None) -> str:
        if values is None:
            return "unavailable"
        return (
            ", ".join(
                labels.get(value, value) if value else "unavailable dependency"
                for value in values
            )
            or "none recorded"
        )

    print(f"Name: {_dataset_label(selected)}")
    print(f"Rows: {selected.num_objects}")
    print("Created: " + _timestamp(selected.created_at))
    if info.description:
        print(f"Description: {info.description}")
    print("Versions: " + ", ".join(info.available_versions))
    print("Schema:")
    info.schema.print_tree(start_at=2)
    print("Sources: " + names(info.sources))
    if info.lineage is None:
        print("Lineage: unavailable")
    elif info.lineage or info.depends_on:
        print("Lineage:")
        print(f"  {_dataset_label(selected)}")
        print(f"    depends_on: {names(info.depends_on)}")
    else:
        print("Lineage: none recorded")
    for entry in info.lineage or []:
        removed = " (removed)" if entry.removed else ""
        print(f"  {labels.get(entry.name, _dataset_label(entry.dataset))}{removed}")
        if entry.depends_on:
            print(f"    depends_on: {names(entry.depends_on)}")
    if info.dependents is None:
        print("Dependents: unavailable")
    else:
        print("Dependents:" + ("" if info.dependents else " none recorded"))
    for entry in info.dependents or []:
        removed = " (removed)" if entry.removed else ""
        print(f"  {labels.get(entry.name, _dataset_label(entry.dataset))}{removed}")
    if job := info.job:
        print(f"Job: {job.id}")
        print(f"Job_status: {JobStatus(job.status).name}")
        print("Job_created: " + _timestamp(job.created_at))
        print(
            "Job_finished: "
            + (_timestamp(job.finished_at) if job.finished_at else "not recorded")
        )
        print(f"Workers: {job.workers}")
        if job.python_version:
            print(f"Python: {job.python_version}")
    elif selected.job_id:
        print(f"Job: {selected.job_id} (unavailable)")
    print("Script:")
    print(selected.query_script or "No script recorded.")


def describe_dataset_cmd(
    catalog: "Catalog", name: str, json_output: bool = False
) -> None:
    info = describe_dataset(name, session=Session.get(catalog=catalog))
    if json_output:
        print(json.dumps(info.to_dict(), indent=2))
    else:
        print_dataset_description(info)
