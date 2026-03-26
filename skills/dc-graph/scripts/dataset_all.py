#!/usr/bin/env python3
"""Fetch data for all versions of a dataset in one call."""

import argparse
import json
import os
import sys

from changes import build_changes, dep_to_dict
from dataset import fetch_version_data
from schema import extract_preview, extract_schema, get_catalog
from utils import collect_datasets, dc_import, parse_semver


def _fetch_all_versions(name: str) -> dict:
    """Fetch data for all versions of a dataset. Returns result dict."""
    dc = dc_import()

    # Detect source
    dot_parts = name.split(".", 2)
    is_studio = len(dot_parts) == 3
    source = "studio" if is_studio else "local"

    warnings_list: list[str] = []

    if is_studio:
        # Studio path: use fetch_version_data per version.
        all_entries = collect_datasets(dc, studio=True)
        version_entries = [e for e in all_entries if e["name"] == name]
        if not version_entries:
            print(json.dumps({"error": f"Dataset '{name}' not found"}), file=sys.stderr)
            sys.exit(1)
        versions_sorted = sorted(
            [e["version"] for e in version_entries if e["version"]],
            key=parse_semver,
        )
        versions_out = []
        for version in versions_sorted:
            version_entry = next(
                (e for e in version_entries if e["version"] == version), None
            )
            data = fetch_version_data(f"{name}@{version}")
            versions_out.append(
                {
                    "version": version,
                    "num_objects": version_entry.get("num_objects")
                    if version_entry
                    else None,
                    "updated_at": version_entry.get("updated_at")
                    if version_entry
                    else None,
                    "schema": data.get("schema"),
                    "preview": data.get("preview"),
                    "query_script": data.get("query_script"),
                    "changes": data.get("changes"),
                    "dependencies": data.get("dependencies", []),
                }
            )
        result = {"name": name, "source": source, "versions": versions_out}
        if warnings_list:
            result["warnings"] = warnings_list
        return result

    # Local path: efficient single-catalog-query implementation.
    bare_name = name

    # 1. Get catalog once — independent of read_dataset.
    catalog = None
    try:
        catalog = get_catalog()
    except Exception as e:
        warnings_list.append(f"catalog: {e}")

    # 2. Get full dataset record once — has all versions with query_script, num_objects.
    versions_sorted_obj = []
    if catalog is not None:
        try:
            dataset_record = catalog.get_dataset(bare_name, include_incomplete=False)
            versions_sorted_obj = sorted(
                dataset_record.versions, key=lambda v: v.version_value
            )
        except Exception as e:
            warnings_list.append(f"get_dataset: {e}")

    if not versions_sorted_obj:
        # Fallback: enumerate via dc.datasets() if catalog failed.
        all_entries = collect_datasets(dc, studio=False)
        version_entries = [e for e in all_entries if e["name"] == name]
        if not version_entries:
            print(json.dumps({"error": f"Dataset '{name}' not found"}), file=sys.stderr)
            sys.exit(1)
        # Fall back to old per-version approach.
        versions_sorted_str = sorted(
            [e["version"] for e in version_entries if e["version"]],
            key=parse_semver,
        )
        versions_out = []
        for version in versions_sorted_str:
            version_entry = next(
                (e for e in version_entries if e["version"] == version), None
            )
            data = fetch_version_data(f"{name}@{version}")
            versions_out.append(
                {
                    "version": version,
                    "num_objects": version_entry.get("num_objects")
                    if version_entry
                    else None,
                    "updated_at": version_entry.get("updated_at")
                    if version_entry
                    else None,
                    "schema": data.get("schema"),
                    "preview": data.get("preview"),
                    "query_script": data.get("query_script"),
                    "changes": data.get("changes"),
                    "dependencies": data.get("dependencies", []),
                }
            )
        result = {"name": bare_name, "source": source, "versions": versions_out}
        if warnings_list:
            result["warnings"] = warnings_list
        return result

    # 3. Fetch schema + preview for latest version only.
    latest_ver_obj = versions_sorted_obj[-1]
    latest_version_str = latest_ver_obj.version
    chain = None
    schema: dict = {}
    preview = None
    try:
        chain = dc.read_dataset(f"{bare_name}@{latest_version_str}")
    except Exception as e:
        warnings_list.append(f"read_dataset: {e}")

    if chain is not None:
        try:
            schema = extract_schema(chain)
        except Exception as e:
            warnings_list.append(f"schema: {e}")
        preview = extract_preview(chain)
        if preview is None:
            warnings_list.append("preview: extraction failed")

    # 4. Fetch deps once per version — cache to avoid fetching prev-version deps twice.
    deps_by_version: dict[str, list[dict]] = {}
    for ver_obj in versions_sorted_obj:
        v_str = ver_obj.version
        deps: list[dict] = []
        try:
            raw_deps = (
                catalog.get_dataset_dependencies(
                    name=bare_name, version=v_str, indirect=True
                )
                or []
            )
            deps = [dep_to_dict(d) for d in raw_deps if d]
        except Exception as e:
            warnings_list.append(f"deps {v_str}: {e}")
        deps_by_version[v_str] = deps

    # 5. Build per-version output oldest-first.
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
                catalog=catalog,
            )

        is_latest = i == len(versions_sorted_obj) - 1
        versions_out.append(
            {
                "version": v_str,
                "num_objects": ver_obj.num_objects,
                "updated_at": ver_obj.finished_at.isoformat()
                if ver_obj.finished_at
                else None,
                "schema": schema if is_latest else {},
                "preview": preview if is_latest else None,
                "query_script": query_script,
                "changes": changes,
                "dependencies": ver_deps,
            }
        )

    result = {"name": bare_name, "source": source, "versions": versions_out}
    if warnings_list:
        result["warnings"] = warnings_list
    return result


def _merge_versions(existing_versions, new_versions, versions_to_fetch):
    """Merge existing and new version lists.

    Keep existing versions not in versions_to_fetch,
    replace/add versions from new data.
    """
    fetch_set = set(versions_to_fetch)
    merged = [v for v in existing_versions if v.get("version") not in fetch_set]
    merged.extend(new_versions)
    merged.sort(key=lambda v: parse_semver(v.get("version", "0.0.0")))
    return merged


def cmd_dataset_all(name: str, plan_path: str = None, output_path: str = None):
    """Fetch all versions of a dataset, optionally merge and save to file."""
    new_data = _fetch_all_versions(name)

    if not plan_path or not output_path:
        # No plan/output — just print JSON to stdout.
        print(json.dumps(new_data, indent=2))
        return

    # Read plan to get versions_to_fetch.
    with open(plan_path) as f:
        plan = json.load(f)

    plan_entry = next(
        (d for d in plan.get("datasets", []) if d["name"] == name), None
    )
    versions_to_fetch = plan_entry.get("versions_to_fetch", []) if plan_entry else []

    # Read existing JSON if it exists.
    existing_versions = []
    try:
        with open(output_path) as f:
            existing_data = json.load(f)
        existing_versions = existing_data.get("versions", [])
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Merge versions.
    merged_versions = _merge_versions(
        existing_versions, new_data.get("versions", []), versions_to_fetch
    )

    result = {
        "name": name,
        "source": new_data.get("source", "local"),
        "versions": merged_versions,
    }
    if new_data.get("warnings"):
        result["warnings"] = new_data["warnings"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch data for all versions of a dataset."
    )
    parser.add_argument("name", help="Dataset name")
    parser.add_argument("--plan", help="Path to .plan.json (enables merge + file output)")
    parser.add_argument("--output", help="Output .json file path (requires --plan)")
    args = parser.parse_args()

    if bool(args.plan) != bool(args.output):
        parser.error("--plan and --output must be used together")

    cmd_dataset_all(args.name, plan_path=args.plan, output_path=args.output)


if __name__ == "__main__":
    main()
