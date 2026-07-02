import sys
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from tabulate import tabulate

from datachain import semver
from datachain.catalog import is_namespace_local
from datachain.cli.utils import determine_flavors
from datachain.config import Config
from datachain.error import (
    DataChainError,
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
)
from datachain.studio import list_datasets as list_datasets_studio

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
) -> None:
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = determine_flavors(studio, local, all, token)
    if name:
        latest_only = False

    local_datasets = set(list_datasets_local(catalog, name)) if all or local else set()
    studio_datasets = (
        set(list_datasets_studio(team=team, name=name)) if all or studio else set()
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
        )
        for n, (local_version, studio_version) in datasets
    ]

    print(tabulate(rows, headers="keys"))


def list_datasets_local(
    catalog: "Catalog", name: str | None = None
) -> Iterator[tuple[str, str]]:
    if name:
        yield from list_datasets_local_versions(catalog, name)
        return

    for d in catalog.ls_datasets():
        for v in d.versions:
            yield d.full_name, v.version


def list_datasets_local_versions(
    catalog: "Catalog", name: str
) -> Iterator[tuple[str, str]]:
    namespace_name, project_name, name = catalog.get_full_dataset_name(name)

    ds = catalog.get_dataset(
        name,
        namespace_name=namespace_name,
        project_name=project_name,
        versions=None,
        include_incomplete=False,
    )
    for v in ds.versions:
        yield name, v.version


def _datasets_tabulate_row(name, both, local_version, studio_version) -> dict[str, str]:
    row = {
        "Name": name,
    }
    if both:
        row["Studio"] = f"v{studio_version}" if studio_version else "\u2716"
        row["Local"] = f"v{local_version}" if local_version else "\u2716"
    else:
        latest_version = local_version or studio_version
        row["Latest Version"] = f"v{latest_version}" if latest_version else "\u2716"

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


def _format_stats_value(kind: str, info: dict) -> str:
    if kind == "numeric":
        mn, mx, avg = info.get("min"), info.get("max"), info.get("avg")
        avg_str = f"{avg:.4g}" if isinstance(avg, (int, float)) else "—"
        return f"min={mn}  max={mx}  avg={avg_str}"
    if kind == "boolean":
        return f"true={info.get('true_count', 0)}  false={info.get('false_count', 0)}"
    if kind == "temporal":
        return f"min={info.get('min')}  max={info.get('max')}"
    if kind == "categorical":
        distinct = info.get("distinct_count")
        prefix = "~" if info.get("distinct_approx") else ""
        top = info.get("top_k") or []
        top_str = ", ".join(f"{t['value']}({t['count']})" for t in top[:3])
        return f"distinct={prefix}{distinct}  top: {top_str}"
    return ""


def dataset_stats(
    catalog: "Catalog",
    name: str,
    version: str | None = None,
    force: bool = False,
    as_json: bool = False,
) -> None:
    from datachain.dataset import parse_dataset_with_version

    name, name_version = parse_dataset_with_version(name)
    version = version or name_version

    try:
        stats = catalog.get_dataset_stats(name, version, force=force)
    except (DatasetNotFoundError, DatasetVersionNotFoundError) as e:
        print(str(e) or "Dataset not found in local", file=sys.stderr)
        return

    if as_json:
        from datachain import json

        print(json.dumps(stats, indent=2))
        return

    sampled = stats.get("sampled")
    sample_note = f" (sampled {sampled['rows']} rows)" if sampled else ""
    print(f"rows: {stats['row_count']}{sample_note}")

    rows = [
        [
            col,
            info.get("type", ""),
            info.get("null_count", 0),
            _format_stats_value(info.get("kind", ""), info),
        ]
        for col, info in stats.get("columns", {}).items()
    ]
    if rows:
        print(
            tabulate(
                rows,
                headers=["column", "type", "nulls", "distribution"],
                disable_numparse=True,
            )
        )

    skipped = stats.get("skipped_columns") or {}
    if skipped:
        skipped_str = ", ".join(f"{c} ({reason})" for c, reason in skipped.items())
        print(f"\nskipped: {skipped_str}")
