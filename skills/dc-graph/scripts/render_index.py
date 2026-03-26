#!/usr/bin/env python3
"""Render index.md from a plan JSON file."""

import argparse
import json
import sys
from datetime import datetime, timezone

from utils import dataset_file_path


def render_index(plan: dict) -> str:
    """Render index.md markdown from a plan dict."""
    datasets = plan.get("datasets", [])
    local_count = sum(1 for d in datasets if d["source"] == "local")
    studio_count = sum(1 for d in datasets if d["source"] == "studio")

    # Frontmatter
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = ["---"]
    if "db_last_updated" in plan:
        lines.append(f"db_last_updated: {plan['db_last_updated']}")
    lines.append(f"generated_at: {now}")
    lines.append(f"local_dataset_count: {local_count}")
    if studio_count > 0:
        lines.append(f"studio_dataset_count: {studio_count}")
    lines.append("---")
    lines.append("")

    # Table
    lines.append("| Name | Source | Version | Objects | Updated |")
    lines.append("|------|--------|---------|---------|---------|")

    for ds in sorted(datasets, key=lambda d: d["name"]):
        name = ds["name"]
        source = ds["source"]
        version = ds.get("latest_version", "")
        num_objects = ds.get("num_objects") or ""
        updated_at = ds.get("updated_at") or ""
        if updated_at and "T" in updated_at:
            updated_at = updated_at.split("T")[0]

        file_path = ds.get("file_path", dataset_file_path(name, source))
        # Wikilink: [[path|display]]
        display = name
        link = f"[[{file_path}|{display}]]"

        lines.append(
            f"| {link} | {source} | {version} | {num_objects} | {updated_at} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Render index.md from plan JSON.")
    parser.add_argument("--plan", required=True, help="Path to .plan.json file")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    args = parser.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)

    result = render_index(plan)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
