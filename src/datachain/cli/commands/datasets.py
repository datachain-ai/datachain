import json
import sys
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from tabulate import tabulate

from datachain import semver
from datachain.catalog import is_namespace_local
from datachain.cli.utils import determine_flavors
from datachain.config import Config
from datachain.dataset import (
    DatasetDependencyType,
    create_dataset_full_name,
    parse_dataset_with_version,
)
from datachain.error import (
    DataChainError,
    DatasetNotFoundError,
)
from datachain.lib.listing import CURRENT_LISTING_PREFIX, listing_uri_from_name
from datachain.lib.signal_schema import SignalSchema
from datachain.studio import list_datasets as list_datasets_studio

if TYPE_CHECKING:
    from datachain.catalog import Catalog
    from datachain.dataset import DatasetDependency


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


def _dependency_name(dep: "DatasetDependency") -> str:
    if dep.is_dataset:
        return f"{dep.namespace}.{dep.project}.{dep.name}"
    return listing_uri_from_name(dep.name)


def _dependency_info(catalog: "Catalog", dep: "DatasetDependency") -> dict[str, Any]:
    rows = (
        catalog.get_dataset(
            dep.name,
            namespace_name=dep.namespace,
            project_name=dep.project,
            versions=[dep.version],
        )
        .get_version(dep.version)
        .num_objects
    )
    info = {
        "name": _dependency_name(dep),
        "version": dep.version,
        "type": dep.type,
        "rows": rows,
        "removed": dep.removed,
        "created_at": dep.created_at.isoformat() if dep.created_at else None,
    }

    if dep.type == DatasetDependencyType.STORAGE:
        info["listing_kind"] = (
            "current only"
            if dep.name.startswith(CURRENT_LISTING_PREFIX)
            else "all versions"
        )
    return info


def _dataset_info(catalog: "Catalog", name: str, version: str | None) -> dict[str, Any]:
    name, name_version = parse_dataset_with_version(name)
    namespace_name, project_name, name = catalog.get_full_dataset_name(name)
    dataset = catalog.get_dataset(
        name,
        namespace_name=namespace_name,
        project_name=project_name,
        versions=None,
        include_incomplete=False,
    )
    selected = dataset.get_version(version or name_version or dataset.latest_version)
    lineage: dict[str, dict[str, Any]] = {}

    def refs(deps: "list[DatasetDependency | None]") -> list[str | None]:
        result: list[str | None] = []
        for dep in deps:
            if dep is None:
                result.append(None)
                continue
            ref = create_dataset_full_name(
                dep.namespace, dep.project, dep.name, dep.version
            )
            if ref not in lineage:
                lineage[ref] = {"ref": ref, **_dependency_info(catalog, dep)}
                lineage[ref]["depends_on"] = refs(dep.dependencies)
            result.append(ref)
        return result

    depends_on = refs(catalog.get_dataset_dependencies_by_ids(dataset.id, selected.id))
    storage = [
        entry["name"]
        for entry in lineage.values()
        if entry["type"] == DatasetDependencyType.STORAGE
    ]
    dependents = catalog.metastore.get_direct_dataset_dependents(
        dataset, selected.version
    )
    return {
        "name": dataset.full_name,
        "version": selected.version,
        "description": dataset.description,
        "rows": selected.num_objects,
        "created_at": selected.created_at.isoformat(),
        "finished_at": selected.finished_at.isoformat()
        if selected.finished_at
        else None,
        "schema": selected.feature_schema,
        "versions": [
            {
                "version": v.version,
                "rows": v.num_objects,
                "created_at": v.created_at.isoformat(),
            }
            for v in sorted(dataset.versions)
        ],
        "depends_on": depends_on,
        "lineage": list(lineage.values()),
        "dependents": [_dependency_info(catalog, d) for d in dependents if d],
        "sources": sorted({*selected.sources.splitlines(), *storage} - {""}),
        "query_script": selected.query_script,
    }


def _dependency_label(entry: dict[str, Any]) -> str:
    kind = f" ({entry['listing_kind']})" if "listing_kind" in entry else ""
    return f"{entry['name']}{kind}@{entry['version']}"


def _version_label(entry: dict[str, Any]) -> str:
    rows = "" if entry["rows"] is None else f" ({entry['rows']} rows)"
    removed = " (removed)" if entry["removed"] else ""
    return f"{_dependency_label(entry)}{rows}{removed}"


def _refs_label(refs: list[str | None], labels: dict[str, str]) -> str:
    return ", ".join(labels[ref] if ref else "unavailable dependency" for ref in refs)


def describe_dataset(
    catalog: "Catalog", name: str, version: str | None = None, json_output: bool = False
) -> None:
    info = _dataset_info(catalog, name, version)
    if json_output:
        print(json.dumps(info, indent=2))
        return
    labels = {e["ref"]: _dependency_label(e) for e in info["lineage"]}
    print(f"{info['name']}@{info['version']}")
    print(f"Rows: {info['rows']}")
    print(f"Created: {info['created_at']}")
    if info["description"]:
        print(f"Description: {info['description']}")
    print("Versions: " + ", ".join(v["version"] for v in info["versions"]))
    print("Schema:")
    SignalSchema.deserialize(info["schema"]).print_tree(start_at=2)
    print("Sources: " + (", ".join(info["sources"]) or "none recorded"))
    print("Depends on: " + (_refs_label(info["depends_on"], labels) or "none recorded"))
    print("Lineage:" + ("" if info["lineage"] else " none recorded"))
    for entry in info["lineage"]:
        print(f"  {_version_label(entry)}")
        if entry["depends_on"]:
            print(f"    depends on: {_refs_label(entry['depends_on'], labels)}")
    print("Dependents:" + ("" if info["dependents"] else " none recorded"))
    for entry in info["dependents"]:
        print(f"  {_version_label(entry)}")
    print("Script:")
    print(info["query_script"] or "No script recorded.")
