"""Fetch data for all versions of a dataset in one call."""

import argparse
import json
import sys
from typing import Any

from changes import build_changes, dep_to_dict
from dataset import fetch_version_data
from schema import extract_preview, extract_schema, get_catalog
from utils import collect_datasets, dc_import, parse_semver, write_json


def _build_version_entries(
    name: str,
    version_entries: list[dict],
    versions_sorted: list[str],
    dc: Any,
    warnings_list: list[str],
) -> tuple[list[dict], list[str], str | None]:
    """Build version entries from sorted version strings (studio/fallback)."""
    versions_out: list[dict] = []
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
                chain = dc.read_dataset(f"{name}@{version}")
                from summary import dataset_summary_from_chain

                ver_summary = dataset_summary_from_chain(chain)
            except Exception as e:  # noqa: BLE001
                warnings_list.append(f"summary: {e}")
        versions_out.append(
            {
                "version": version,
                "uuid": data.get("uuid"),
                "records": version_entry.get("records") if version_entry else None,
                "updated": version_entry.get("updated") if version_entry else None,
                "schema": data.get("schema"),
                "preview": data.get("preview"),
                "summary": ver_summary,
                "query_script": data.get("query_script"),
                "changes": data.get("changes"),
                "dependencies": data.get("dependencies", []),
            }
        )
    return versions_out, ds_attrs, ds_description


def _fetch_studio_versions(
    name: str,
    source: str,
    dc: Any,
) -> dict:
    """Fetch all versions for a studio dataset."""
    all_entries = collect_datasets(dc, studio=True)
    version_entries = [e for e in all_entries if e["name"] == name]
    if not version_entries:
        print(json.dumps({"error": f"Dataset '{name}' not found"}), file=sys.stderr)
        sys.exit(1)
    versions_sorted = sorted(
        [e["version"] for e in version_entries if e["version"]],
        key=parse_semver,
    )
    warnings_list: list[str] = []
    versions_out, ds_attrs, ds_description = _build_version_entries(
        name,
        version_entries,
        versions_sorted,
        dc,
        warnings_list,
    )
    result = {
        "name": name,
        "source": source,
        "attrs": ds_attrs,
        "description": ds_description,
        "versions": versions_out,
    }
    if warnings_list:
        result["warnings"] = warnings_list
    return result


def _fetch_local_catalog_versions(
    bare_name: str,
    source: str,
    catalog: Any,
    versions_sorted_obj: list[Any],
    ds_attrs_main: list[str],
    ds_description_main: str | None,
    dc: Any,
    warnings_list: list[str],
) -> dict:
    """Fetch version data using the optimized single-catalog-query path."""
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

    deps_by_version: dict[str, list[dict]] = {}
    for ver_obj in versions_sorted_obj:
        v_str = ver_obj.version
        deps: list[dict] = []
        try:
            raw_deps = (
                catalog.get_dataset_dependencies(  # type: ignore[union-attr]
                    name=bare_name, version=v_str, indirect=True
                )
                or []
            )
            deps = [dep_to_dict(d) for d in raw_deps if d]
        except Exception as e:  # noqa: BLE001
            warnings_list.append(f"deps {v_str}: {e}")
        deps_by_version[v_str] = deps

    versions_out: list[dict] = []
    for i, ver_obj in enumerate(versions_sorted_obj):
        v_str = ver_obj.version
        query_script = ver_obj.execution.query_script or None
        ver_deps = deps_by_version.get(v_str, [])

        changes = None
        if i > 0:
            prev_ver_obj = versions_sorted_obj[i - 1]
            prev_v_str = prev_ver_obj.version
            prev_script = prev_ver_obj.execution.query_script or None
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
                "uuid": getattr(ver_obj, "uuid", None),
                "records": ver_obj.stats.num_objects,
                "updated": (
                    ver_obj.timestamps.finished_at or ver_obj.timestamps.created_at
                ).isoformat(),
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


def _fetch_local_versions(
    name: str,
    source: str,
    dc: Any,
) -> dict:
    """Fetch all versions for a local dataset (catalog or fallback)."""
    warnings_list: list[str] = []
    bare_name = name

    catalog = None
    try:
        catalog = get_catalog()
    except Exception as e:  # noqa: BLE001
        warnings_list.append(f"catalog: {e}")

    versions_sorted_obj: list[Any] = []
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
        all_entries = collect_datasets(dc, studio=False)
        version_entries = [e for e in all_entries if e["name"] == name]
        if not version_entries:
            print(json.dumps({"error": f"Dataset '{name}' not found"}), file=sys.stderr)
            sys.exit(1)
        versions_sorted = sorted(
            [e["version"] for e in version_entries if e["version"]],
            key=parse_semver,
        )
        versions_out, ds_attrs_fb, ds_description_fb = _build_version_entries(
            name,
            version_entries,
            versions_sorted,
            dc,
            warnings_list,
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

    return _fetch_local_catalog_versions(
        bare_name,
        source,
        catalog,
        versions_sorted_obj,
        ds_attrs_main,
        ds_description_main,
        dc,
        warnings_list,
    )


def _fetch_all_versions(name: str) -> dict:
    """Fetch data for all versions of a dataset. Returns result dict."""
    dc = dc_import()

    dot_parts = name.split(".", 2)
    is_studio = len(dot_parts) == 3
    source = "studio" if is_studio else "local"

    if is_studio:
        return _fetch_studio_versions(name, source, dc)

    return _fetch_local_versions(name, source, dc)


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


def cmd_dataset_all(
    name: str, plan_path: str | None = None, output_path: str | None = None
):
    """Fetch all versions of a dataset, optionally merge and save to file."""
    new_data = _fetch_all_versions(name)

    if not plan_path or not output_path:
        # No plan/output — just print JSON to stdout.
        print(json.dumps(new_data, indent=2))
        return

    # Read plan to get versions_to_fetch.
    with open(plan_path) as f:
        plan = json.load(f)

    plan_entry = next((d for d in plan.get("datasets", []) if d["name"] == name), None)
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
