"""Render index.md from a plan JSON file.

Supports both datasets and buckets sections in a single index.
Reads enriched .md files for summaries and dependencies.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

from utils import bucket_file_path, dataset_file_path, human_size, read_json_data

BASE_DIR = "dc-knowledge"


def _read_md_frontmatter(md_path: str) -> dict:
    """Read YAML frontmatter from a markdown file. Returns dict or {}."""
    try:
        with open(md_path) as f:
            content = f.read()
    except Exception:  # noqa: BLE001
        return {}
    if not content.startswith("---"):
        return {}
    try:
        end = content.index("\n---", 3)
    except ValueError:
        return {}
    result = {}
    for line in content[4:end].splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def _read_md_info(md_path: str) -> dict:
    """Read metadata, description, and dependencies from an enriched dataset .md.

    Returns dict with keys: description, deps, last_version, records, updated.
    """
    info: dict = {
        "description": "",
        "deps": [],
        "last_version": "",
        "records": "",
        "num_versions": "",
        "updated": "",
    }

    try:
        with open(md_path) as f:
            content = f.read()
    except Exception:  # noqa: BLE001
        return info

    # Parse frontmatter
    fm = _read_md_frontmatter(md_path)
    info["last_version"] = fm.get("last_version", "")
    info["records"] = fm.get("records", "")
    known = fm.get("known_versions", "")
    if known.startswith("[") and known.endswith("]"):
        known = known[1:-1]
    versions_list = [v.strip() for v in known.split(",") if v.strip()]
    info["num_versions"] = str(len(versions_list)) if versions_list else ""
    updated = fm.get("updated", "")
    if updated and "T" in updated:
        updated = updated.split("T")[0]
    info["updated"] = updated

    # Strip frontmatter for body parsing
    if content.startswith("---"):
        try:
            end = content.index("\n---", 3)
            content = content[end + 4 :].strip()
        except ValueError:
            return info

    lines = content.split("\n")

    # Extract description: paragraph between # heading and first ## heading
    desc_lines: list[str] = []
    past_heading = False
    for line in lines:
        if not past_heading:
            if line.startswith("# "):
                past_heading = True
            continue
        if line.startswith("##"):
            break
        stripped = line.strip()
        if not stripped and desc_lines:
            break
        if stripped:
            desc_lines.append(stripped)
    info["description"] = " ".join(desc_lines)

    # Extract dependencies: preserve markdown links for clickability
    deps: list[str] = []
    in_deps = False
    for line in lines:
        if line.startswith("## Dependencies"):
            in_deps = True
            continue
        if in_deps:
            if line.startswith("##"):
                break
            # Preserve full markdown link syntax [name](path)
            link_matches = re.findall(r"\[[^\]]+\]\([^)]+\)", line)
            if link_matches:
                deps.extend(link_matches)
            elif line.strip().startswith("- "):
                name = line.strip().removeprefix("- ").strip()
                if name:
                    deps.append(name)
    info["deps"] = deps

    return info


def _render_dataset_table(
    datasets: list[dict], strip_namespace: bool = False
) -> list[str]:
    """Render a markdown table for a list of dataset entries."""
    lines = []
    lines.append("| Name | Updated | Dependencies | Summary |")
    lines.append("|------|---------|--------------|---------|")

    for ds in sorted(datasets, key=lambda d: d["name"]):
        name = ds["name"]
        source = ds["source"]
        file_path = ds.get("file_path", dataset_file_path(name, source))

        # Display name: strip namespace prefix if inside a namespace subsection
        display_name = name
        if strip_namespace:
            parts = name.split(".", 2)
            if len(parts) == 3:
                display_name = parts[2]

        link = f"[{display_name}]({file_path}.md)"

        # All metadata from enriched .md
        md_path = os.path.join(BASE_DIR, file_path + ".md")
        info = _read_md_info(md_path)
        updated = info["updated"]
        deps_str = ", ".join(info["deps"]) if info["deps"] else ""
        summary = info["description"]

        lines.append(f"| {link} | {updated} | {deps_str} | {summary} |")

    return lines


def render_index(plan: dict) -> str:
    """Render index.md markdown from a plan dict."""
    datasets = plan.get("datasets", [])
    buckets = plan.get("buckets", [])

    local_ds = [d for d in datasets if d["source"] == "local"]
    studio_ds = [d for d in datasets if d["source"] == "studio"]

    # Frontmatter
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    lines = ["---"]
    lines.append(f"generated: {now}")
    if "db_last_updated" in plan:
        lines.append(f"db_last_updated: {plan['db_last_updated']}")
    lines.append(f"datasets: {len(datasets)}")
    if buckets:
        lines.append(f"buckets: {len(buckets)}")
    lines.append("---")
    lines.append("")

    # Local datasets (default section — no "Local" header)
    if local_ds:
        lines.append("## Datasets")
        lines.append("")
        lines.extend(_render_dataset_table(local_ds))
        lines.append("")

    # Studio datasets grouped by namespace
    if studio_ds:
        # Group by namespace (namespace.project)
        by_ns: dict[str, list[dict]] = {}
        for ds in studio_ds:
            parts = ds["name"].split(".", 2)
            if len(parts) == 3:
                ns = f"{parts[0]}.{parts[1]}"
            else:
                ns = ""
            by_ns.setdefault(ns, []).append(ds)

        lines.append("## Studio")
        lines.append("")

        for ns in sorted(by_ns):
            if ns:
                lines.append(f"### {ns}")
            else:
                lines.append("### (default)")
            lines.append("")
            lines.extend(_render_dataset_table(by_ns[ns], strip_namespace=bool(ns)))
            lines.append("")

    # Buckets table
    if buckets:
        lines.append("## Buckets")
        lines.append("")
        lines.append("| Listing | Files | Size | Scanned |")
        lines.append("|---------|------:|-----:|---------|")

        for b in sorted(buckets, key=lambda x: x["uri"]):
            uri = b["uri"]
            file_path = b.get("file_path", bucket_file_path(uri))

            # Read JSON for rich stats
            abs_json_path = os.path.join(BASE_DIR, file_path + ".json")
            data = read_json_data(abs_json_path)

            total_files = ""
            total_size = ""
            scanned = b.get("scanned") or ""

            if data:
                tf = data.get("total_files")
                total_files = f"{tf:,}" if tf else ""
                tb = data.get("total_size_bytes", 0)
                total_size = human_size(tb) if tb else ""
                scanned = data.get("scanned", scanned) or ""

            if scanned and "T" in scanned:
                scanned = scanned.split("T")[0]

            link = f"[{uri}]({file_path}.md)"

            lines.append(f"| {link} | {total_files} | {total_size} | {scanned} |")

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
