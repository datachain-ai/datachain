"""Fetch data for all versions of a dataset in one call."""

import argparse
import json
import sys
from typing import TYPE_CHECKING

from changes import build_changes
from dataset import fetch_version_data
from schema import extract_preview, extract_schema, get_catalog
from utils import (
    collect_datasets,
    dc_import,
    dedupe_previous_scripts,
    dep_entry,
    drop_unchanged_scripts,
    parse_semver,
    write_json,
)

if TYPE_CHECKING:
    from datachain.skill.knowledge.types import (
        DatasetSnapshot,
        DatasetVersionEntry,
        DependencyEntry,
    )


def _fetch_all_versions(name: str) -> "DatasetSnapshot":  # noqa: C901, PLR0912, PLR0915
    dc = dc_import()

    dot_parts = name.split(".", 2)
    is_studio = len(dot_parts) == 3
    source = "studio" if is_studio else "local"

    warnings_list: list[str] = []

    if is_studio:
        all_entries = collect_datasets(dc, studio=True)
        version_entries = [e for e in all_entries if e["name"] == name]
        if not version_entries:
            print(json.dumps({"error": f"Dataset '{name}' not found"}), file=sys.stderr)
            sys.exit(1)
        versions_sorted = sorted(
            [e["version"] for e in version_entries if e["version"]],
            key=parse_semver,
        )
        versions_out: list[DatasetVersionEntry] = []
        ds_attrs: list[str] = []
        ds_description: str | None = None
        for idx, version in enumerate(versions_sorted):
            version_entry = next(
                (e for e in version_entries if e["version"] == version), None
            )
            data = fetch_version_data(f"{name}@{version}")
            if idx == 0:
                ds_attrs = list(data.get("attrs") or [])
                ds_description = data.get("description")
            is_latest = idx == len(versions_sorted) - 1
            ver_summary = None
            if is_latest:
                try:
                    studio_chain = dc.read_dataset(f"{name}@{version}")
                    from summary import dataset_summary_from_chain

                    ver_summary = dataset_summary_from_chain(studio_chain)
                except Exception as e:  # noqa: BLE001
                    warnings_list.append(f"summary: {e}")
            versions_out.append(
                {
                    "version": version,
                    "uuid": data.get("uuid"),
                    "records": version_entry.get("records") if version_entry else None,
                    "updated": (
                        (version_entry.get("finished") or version_entry.get("created"))
                        if version_entry
                        else None
                    ),
                    "schema": data.get("schema") if is_latest else {},
                    "preview": data.get("preview") if is_latest else None,
                    "summary": ver_summary,
                    "query_script": data.get("query_script"),
                    "changes": data.get("changes"),
                    "dependencies": data.get("dependencies", []),
                }
            )
        result: DatasetSnapshot = {
            "name": name,
            "source": source,
            "attrs": ds_attrs,
            "description": ds_description,
            "versions": versions_out,
        }
        if warnings_list:
            result["warnings"] = warnings_list
        return result

    bare_name = name

    catalog = None
    try:
        catalog = get_catalog()
    except Exception as e:  # noqa: BLE001
        warnings_list.append(f"catalog: {e}")

    versions_sorted_obj = []
    ds_attrs_main: list[str] = []
    ds_description_main: str | None = None
    if catalog is not None:
        try:
            dataset_record = catalog.get_dataset(
                bare_name, versions=None, include_incomplete=False
            )
            ds_attrs_main = list(dataset_record.attrs or [])
            ds_description_main = dataset_record.description
            versions_sorted_obj = sorted(
                dataset_record.versions, key=lambda v: v.version_value
            )
        except Exception as e:  # noqa: BLE001
            warnings_list.append(f"get_dataset: {e}")

    if not versions_sorted_obj:
        # Fallback: enumerate via dc.datasets() if catalog failed.
        all_entries = collect_datasets(dc, studio=False)
        version_entries = [e for e in all_entries if e["name"] == name]
        if not version_entries:
            print(json.dumps({"error": f"Dataset '{name}' not found"}), file=sys.stderr)
            sys.exit(1)
        versions_sorted_str = sorted(
            [e["version"] for e in version_entries if e["version"]],
            key=parse_semver,
        )
        versions_out = []
        ds_attrs_fb: list[str] = []
        ds_description_fb: str | None = None
        for idx, version in enumerate(versions_sorted_str):
            version_entry = next(
                (e for e in version_entries if e["version"] == version), None
            )
            data = fetch_version_data(f"{name}@{version}")
            if idx == 0:
                ds_attrs_fb = list(data.get("attrs") or [])
                ds_description_fb = data.get("description")
            is_latest = idx == len(versions_sorted_str) - 1
            fb_summary = None
            if is_latest:
                try:
                    fb_chain = dc.read_dataset(f"{name}@{version}")
                    from summary import dataset_summary_from_chain

                    fb_summary = dataset_summary_from_chain(fb_chain)
                except Exception as e:  # noqa: BLE001
                    warnings_list.append(f"summary: {e}")
            versions_out.append(
                {
                    "version": version,
                    "uuid": data.get("uuid"),
                    "records": version_entry.get("records") if version_entry else None,
                    "updated": (
                        (version_entry.get("finished") or version_entry.get("created"))
                        if version_entry
                        else None
                    ),
                    "schema": data.get("schema") if is_latest else {},
                    "preview": data.get("preview") if is_latest else None,
                    "summary": fb_summary,
                    "query_script": data.get("query_script"),
                    "changes": data.get("changes"),
                    "dependencies": data.get("dependencies", []),
                }
            )
        result = {
            "name": bare_name,
            "source": source,
            "attrs": ds_attrs_fb,
            "description": ds_description_fb,
            "versions": versions_out,
        }
        if warnings_list:
            result["warnings"] = warnings_list
        return result

    latest_ver_obj = versions_sorted_obj[-1]
    latest_version_str = latest_ver_obj.version
    chain = None
    schema: dict = {}
    preview = None
    try:
        chain = dc.read_dataset(f"{bare_name}@{latest_version_str}")
    except Exception as e:  # noqa: BLE001
        warnings_list.append(f"read_dataset: {e}")

    if chain is not None:
        try:
            schema = extract_schema(chain)
        except Exception as e:  # noqa: BLE001
            warnings_list.append(f"schema: {e}")
        preview = extract_preview(chain)
        if preview is None:
            warnings_list.append("preview: extraction failed")

    summary = None
    if chain is not None:
        try:
            from summary import dataset_summary_from_chain

            summary = dataset_summary_from_chain(chain)
        except Exception as e:  # noqa: BLE001
            warnings_list.append(f"summary: {e}")

    # Cache deps per version to avoid fetching prev-version deps twice.
    deps_by_version: dict[str, list[DependencyEntry]] = {}
    for ver_obj in versions_sorted_obj:
        v_str = ver_obj.version
        deps: list[DependencyEntry] = []
        try:
            raw_deps = (
                catalog.get_dataset_dependencies(  # type: ignore[union-attr]
                    name=bare_name, version=v_str, indirect=False
                )
                or []
            )
            deps = [dep_entry(d.name, d.version, d.type) for d in raw_deps if d]
        except Exception as e:  # noqa: BLE001
            warnings_list.append(f"deps {v_str}: {e}")
        deps_by_version[v_str] = deps

    versions_out = []
    for i, ver_obj in enumerate(versions_sorted_obj):
        v_str = ver_obj.version
        query_script = ver_obj.query_script or None
        ver_deps = deps_by_version.get(v_str, [])

        # Changes vs previous version — reuse cached deps, no extra queries.
        changes = None
        if i > 0:
            prev_ver_obj = versions_sorted_obj[i - 1]
            prev_v_str = prev_ver_obj.version
            prev_script = prev_ver_obj.query_script or None
            prev_deps = deps_by_version.get(prev_v_str, [])
            changes = build_changes(
                query_script,
                prev_v_str,
                prev_script,
                ver_deps,
                prev_deps,
            )

        is_latest = i == len(versions_sorted_obj) - 1
        versions_out.append(
            {
                "version": v_str,
                "uuid": getattr(ver_obj, "uuid", None),
                "records": ver_obj.num_objects,
                "updated": (ver_obj.finished_at or ver_obj.created_at).isoformat(),
                "schema": schema if is_latest else {},
                "preview": preview if is_latest else None,
                "summary": summary if is_latest else None,
                "query_script": query_script,
                "changes": changes,
                "dependencies": ver_deps,
            }
        )

    result = {
        "name": bare_name,
        "source": source,
        "attrs": ds_attrs_main,
        "description": ds_description_main,
        "versions": versions_out,
    }
    if warnings_list:
        result["warnings"] = warnings_list
    return result


def _merge_versions(existing_versions, new_versions, versions_to_fetch):
    """Merge existing and new version lists.

    Drop existing versions that are being refetched (in versions_to_fetch) or already
    present in new_versions — the new entries replace them. The `new_versions` guard
    prevents a version in both lists from appearing twice (the fetch returns every
    version, not just the planned ones).
    """
    fetch_set = set(versions_to_fetch)
    new_set = {v.get("version") for v in new_versions}
    merged = [
        v
        for v in existing_versions
        if v.get("version") not in fetch_set and v.get("version") not in new_set
    ]
    merged.extend(new_versions)
    merged.sort(key=lambda v: parse_semver(v.get("version", "0.0.0")))
    return merged


def cmd_dataset_all(
    name: str, plan_path: str | None = None, output_path: str | None = None
):
    """Fetch all versions of a dataset, optionally merge and save to file."""
    new_data = _fetch_all_versions(name)
    drop_unchanged_scripts(new_data["versions"])
    dedupe_previous_scripts(new_data["versions"])

    if not plan_path or not output_path:
        print(json.dumps(new_data, indent=2))
        return

    with open(plan_path) as f:
        plan = json.load(f)

    plan_entry = next((d for d in plan.get("datasets", []) if d["name"] == name), None)
    versions_to_fetch = plan_entry.get("versions_to_fetch", []) if plan_entry else []

    existing_versions = []
    try:
        with open(output_path) as f:
            existing_data = json.load(f)
        existing_versions = existing_data.get("versions", [])
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    merged_versions = _merge_versions(
        existing_versions, new_data.get("versions", []), versions_to_fetch
    )
    # Merge can reorder entries and change which version is latest — re-run both passes.
    drop_unchanged_scripts(merged_versions)
    dedupe_previous_scripts(merged_versions)

    result: DatasetSnapshot = {
        "name": name,
        "source": new_data.get("source", "local"),
        "attrs": new_data.get("attrs", []),
        "description": new_data.get("description"),
        "versions": merged_versions,
    }
    if new_data.get("warnings"):
        result["warnings"] = new_data["warnings"]

    write_json(output_path, result)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch data for all versions of a dataset."
    )
    parser.add_argument("name", help="Dataset name")
    parser.add_argument(
        "--plan", help="Path to .plan.json (enables merge + file output)"
    )
    parser.add_argument("--output", help="Output .json file path (requires --plan)")
    args = parser.parse_args()

    if bool(args.plan) != bool(args.output):
        parser.error("--plan and --output must be used together")

    cmd_dataset_all(args.name, plan_path=args.plan, output_path=args.output)


if __name__ == "__main__":
    main()
