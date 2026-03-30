"""Render index.md from a plan JSON file.

Supports both datasets and buckets sections in a single index.
"""

import argparse
import json
import os
from datetime import datetime, timezone

from utils import bucket_file_path, dataset_file_path, human_size, read_json_data


def render_index(plan: dict) -> str:
    """Render index.md markdown from a plan dict."""
    datasets = plan.get("datasets", [])
    buckets = plan.get("buckets", [])

    local_count = sum(1 for d in datasets if d["source"] == "local")
    studio_count = sum(1 for d in datasets if d["source"] == "studio")

    # Frontmatter
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    lines = ["---"]
    if "db_last_updated" in plan:
        lines.append(f"db_last_updated: {plan['db_last_updated']}")
    lines.append(f"generated_at: {now}")
    lines.append(f"local_dataset_count: {local_count}")
    if studio_count > 0:
        lines.append(f"studio_dataset_count: {studio_count}")
    if buckets:
        lines.append(f"bucket_count: {len(buckets)}")
    lines.append("---")
    lines.append("")

    # Datasets table
    if datasets:
        lines.append("## Datasets")
        lines.append("")
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
            link = f"[{name}]({file_path}.md)"

            lines.append(
                f"| {link} | {source} | {version} | {num_objects} | {updated_at} |"
            )

        lines.append("")

    # Buckets table
    if buckets:
        lines.append("## Buckets")
        lines.append("")
        lines.append("| Bucket | Scheme | Prefix | Files | Size | Scanned |")
        lines.append("|--------|--------|--------|------:|-----:|---------|")

        for b in sorted(buckets, key=lambda x: x["uri"]):
            uri = b["uri"]
            scheme = b["scheme"]
            prefix = b.get("prefix", "")
            file_path = b.get("file_path", bucket_file_path(uri))

            # Read JSON for rich stats
            abs_json_path = os.path.join("datachain/graph", file_path + ".json")
            data = read_json_data(abs_json_path)

            total_files = ""
            total_size = ""
            scanned_at = b.get("scanned_at") or ""

            if data:
                tf = data.get("total_files")
                total_files = f"{tf:,}" if tf else ""
                tb = data.get("total_size_bytes", 0)
                total_size = human_size(tb) if tb else ""
                scanned_at = data.get("scanned_at", scanned_at) or ""

            if scanned_at and "T" in scanned_at:
                scanned_at = scanned_at.split("T")[0]

            link = f"[{uri}]({file_path}.md)"

            row = (
                f"| {link} | {scheme} | {prefix}"
                f" | {total_files} | {total_size}"
                f" | {scanned_at} |"
            )
            lines.append(row)

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
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
