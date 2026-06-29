"""Fetch data for all versions of a dataset in one call."""

import argparse
import json
import sys
from typing import TYPE_CHECKING

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

from datachain.skill.knowledge.snapshot import build_dataset_snapshot

if TYPE_CHECKING:
    from datachain.skill.knowledge.types import (
        DatasetSnapshot,
        DatasetVersionEntry,
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

    def deps_provider(ver_obj):
        try:
            raw_deps = (
                catalog.get_dataset_dependencies(  # type: ignore[union-attr]
                    name=bare_name, version=ver_obj.version, indirect=False
                )
                or []
            )
        except Exception as e:  # noqa: BLE001
            warnings_list.append(f"deps {ver_obj.version}: {e}")
            return []
        return [dep_entry(d.name, d.version, d.type) for d in raw_deps if d]

    result = build_dataset_snapshot(
        name=bare_name,
        source=source,
        attrs=ds_attrs_main,
        description=ds_description_main,
        versions=versions_sorted_obj,
        deps_provider=deps_provider,
    )

    # overlay live-read schema/preview/summary onto the latest version
    if result["versions"]:
        latest_entry = result["versions"][-1]
        latest_entry["schema"] = schema
        latest_entry["preview"] = preview
        latest_entry["summary"] = summary

    if warnings_list:
        result["warnings"] = [*result.get("warnings", []), *warnings_list]
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
