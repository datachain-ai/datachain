import argparse
import json
from typing import TYPE_CHECKING

from dataset import fetch_dataset_snapshot
from utils import (
    dedupe_previous_scripts,
    drop_unchanged_scripts,
    parse_semver,
    write_json,
)

from datachain.error import DataChainError, NotFoundError

if TYPE_CHECKING:
    from datachain.skill.knowledge.types import DatasetSnapshot


def _fetch_all_versions(name: str) -> "DatasetSnapshot":
    return fetch_dataset_snapshot(name, summarize=True)


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

    try:
        cmd_dataset_all(args.name, plan_path=args.plan, output_path=args.output)
    except (DataChainError, NotFoundError) as exc:
        parser.exit(1, json.dumps({"error": str(exc)}) + "\n")


if __name__ == "__main__":
    main()
