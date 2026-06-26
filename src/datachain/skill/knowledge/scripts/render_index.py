"""Render index.md from a plan JSON file.

Supports both datasets and buckets sections in a single index.
Reads enriched .md files for summaries and dependencies.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

from utils import (
    bucket_file_path,
    dataset_file_path,
    escape_table_cell,
    extract_description,
    read_frontmatter,
    split_frontmatter,
    write_text,
)

BASE_DIR = "dc-knowledge"


def _parse_frontmatter_info(fm: dict) -> dict:
    """Extract normalized frontmatter fields for the info dict."""
    known = fm.get("known_versions", "")
    if known.startswith("[") and known.endswith("]"):
        known = known[1:-1]
    versions_list = [v.strip() for v in known.split(",") if v.strip()]
    updated = fm.get("updated", "")
    if updated and "T" in updated:
        updated = updated.split("T")[0]
    return {
        "last_version": fm.get("last_version", ""),
        "records": fm.get("records", ""),
        "num_versions": str(len(versions_list)) if versions_list else "",
        "updated": updated,
    }


def _extract_section_paragraph(lines: list[str], heading: str) -> str:
    """Paragraph under a specific `## heading`."""
    out: list[str] = []
    in_section = False
    for line in lines:
        if line.startswith(heading):
            in_section = True
            continue
        if not in_section:
            continue
        if line.startswith("##"):
            break
        stripped = line.strip()
        if not stripped and out:
            break
        if stripped:
            out.append(stripped)
    return " ".join(out)


def _extract_deps(lines: list[str]) -> list[str]:
    """Dependencies under `## Dependencies`, preserving markdown links."""
    deps: list[str] = []
    in_deps = False
    for line in lines:
        if line.startswith("## Dependencies"):
            in_deps = True
            continue
        if not in_deps:
            continue
        if line.startswith("##"):
            break
        link_matches = re.findall(r"\[[^\]]+\]\([^)]+\)", line)
        if link_matches:
            deps.extend(link_matches)
        elif line.strip().startswith("- "):
            name = line.strip().removeprefix("- ").strip()
            if name:
                deps.append(name)
    return deps


def _parse_list_field(raw: str) -> list[str]:
    """Parse a `[a, b, c]` list-style frontmatter value into a list of strings."""
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [v.strip() for v in raw.split(",") if v.strip()]


CAST_LAYERS = ("container", "asset", "sense", "task")


def _read_md_info(md_path: str) -> dict:
    """Read metadata, description, and dependencies from an enriched dataset .md.

    Returns dict with keys: description, deps, last_version, records, updated,
    cast_layer, cast_scope, cast_source, cast_parents.
    """
    info: dict = {
        "description": "",
        "session_context": "",
        "deps": [],
        "last_version": "",
        "records": "",
        "num_versions": "",
        "updated": "",
        "cast_layer": "",
        "cast_scope": "",
        "cast_source": "",
        "cast_parents": [],
    }

    try:
        with open(md_path) as f:
            content = f.read()
    except Exception:  # noqa: BLE001
        return info

    fm, body = split_frontmatter(content)
    info.update(_parse_frontmatter_info(fm))
    info["cast_layer"] = fm.get("cast_layer", "").strip().lower()
    info["cast_scope"] = fm.get("cast_scope", "").strip().lower()
    info["cast_source"] = fm.get("cast_source", "").strip()
    info["cast_parents"] = _parse_list_field(fm.get("cast_parents", ""))

    lines = body.split("\n")
    info["description"] = extract_description(lines)
    info["session_context"] = _extract_section_paragraph(lines, "## Session Context")
    info["deps"] = _extract_deps(lines)

    return info


def _collect_dataset_row(ds: dict, strip_namespace: bool = False) -> tuple[dict, dict]:
    """Read frontmatter + body info for a dataset entry.

    Returns (ds_with_link, info).
    """
    name = ds["name"]
    source = ds["source"]
    file_path = ds.get("file_path", dataset_file_path(name, source))

    display_name = name
    if strip_namespace:
        parts = name.split(".", 2)
        if len(parts) == 3:
            display_name = parts[2]

    md_path = os.path.join(BASE_DIR, file_path + ".md")
    info = _read_md_info(md_path)
    enriched = {
        "name": name,
        "display_name": display_name,
        "link": f"[{display_name}]({file_path}.md)",
        "file_path": file_path,
    }
    return enriched, info


def _render_dataset_table(
    datasets: list[dict], strip_namespace: bool = False
) -> list[str]:
    """Render a markdown table for a list of dataset entries (legacy/Studio shape)."""
    lines = []
    lines.append("| Name | Updated | Dependencies | Summary |")
    lines.append("|------|---------|--------------|---------|")

    for ds in sorted(datasets, key=lambda d: d["name"]):
        enriched, info = _collect_dataset_row(ds, strip_namespace=strip_namespace)
        updated = info["updated"]
        deps_str = ", ".join(info["deps"]) if info["deps"] else ""
        summary = info["description"]
        lines.append(
            f"| {escape_table_cell(enriched['link'])} | {updated} "
            f"| {escape_table_cell(deps_str)} | {escape_table_cell(summary)} |"
        )

    return lines


CAST_SECTION_NAMES = {
    "container": "Container",
    "asset": "Asset",
    "sense": "Sense",
    "task": "Task Dataset",
}

CAST_SECTION_BLURBS = {
    "container": "_File headers, listings, and sidecar metadata. One row per file._",
    "asset": (
        "_Raw extracted data (frames, clips, audio, parsed arrays) "
        "or training mixtures of multiple datasets._"
    ),
    "sense": (
        "_Model-derived signals: embeddings, classifications, "
        "transcriptions, LLM outputs._"
    ),
    "task": (
        "_Task-specific analytics and any dataset not tagged as "
        "Container, Asset, or Sense._"
    ),
}


def _render_cast_table(rows: list[tuple[dict, dict]], layer: str) -> list[str]:
    """Render a markdown table for one CAST layer.

    Columns are uniform across layers for readability: Name, Scope, Source,
    Parents, Updated, Records, Description.
    """
    lines = []
    lines.append(
        "| Name | Scope | Source | Parents | Updated | Records | Description |"
    )
    lines.append(
        "|------|-------|--------|---------|---------|--------:|-------------|"
    )
    for enriched, info in sorted(rows, key=lambda r: r[0]["name"]):
        parents = ", ".join(info["cast_parents"]) if info["cast_parents"] else ""
        lines.append(
            f"| {escape_table_cell(enriched['link'])} "
            f"| {escape_table_cell(info['cast_scope'])} "
            f"| {escape_table_cell(info['cast_source'])} "
            f"| {escape_table_cell(parents)} "
            f"| {info['updated']} "
            f"| {info['records']} "
            f"| {escape_table_cell(info['description'])} |"
        )
    return lines


def _render_cast_grouped(datasets: list[dict]) -> list[str]:
    """Render the local-datasets block as four CAST-grouped tables.

    Untagged datasets and any non-C/A/S `cast_layer` value fall under
    "Task Dataset" (the catch-all).
    """
    by_layer: dict[str, list[tuple[dict, dict]]] = {layer: [] for layer in CAST_LAYERS}
    for ds in datasets:
        enriched, info = _collect_dataset_row(ds)
        layer = info["cast_layer"]
        if layer not in CAST_LAYERS:
            layer = "task"
        by_layer[layer].append((enriched, info))

    lines: list[str] = []
    for layer in CAST_LAYERS:
        rows = by_layer[layer]
        if not rows and layer != "task":
            continue
        lines.append(f"### {CAST_SECTION_NAMES[layer]}")
        lines.append("")
        lines.append(CAST_SECTION_BLURBS[layer])
        lines.append("")
        if rows:
            lines.extend(_render_cast_table(rows, layer))
        else:
            lines.append("_No datasets yet._")
        lines.append("")
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

    # Local datasets, grouped by CAST layer
    # (Container, Asset, Sense, Task Dataset).
    # Untagged datasets fall into "Task Dataset" as the catch-all.
    if local_ds:
        lines.append("## Datasets")
        lines.append("")
        lines.extend(_render_cast_grouped(local_ds))

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

    # Buckets table — merge plan-derived entries with on-disk markdowns
    bucket_rows = _collect_bucket_rows(buckets)
    if bucket_rows:
        lines.append("## Buckets")
        lines.append("")
        lines.append("| Listing | Files | Size | Scanned |")
        lines.append("|---------|------:|-----:|---------|")
        for link, files_val, size_val, scanned in bucket_rows:
            lines.append(
                f"| {escape_table_cell(link)} | {files_val} | {size_val} | {scanned} |"
            )
        lines.append("")

    return "\n".join(lines)


def _collect_bucket_rows(buckets: list[dict]) -> list[tuple[str, str, str, str]]:
    """Return (link, files, size, scanned) rows for plan-derived bucket mds.

    Single source of truth is the markdown frontmatter — JSON is intermediate
    and may be cleaned up. Plan entries without an md on disk are skipped
    so the index never contains broken links.
    """
    rows: list[tuple[str, str, str, str]] = []

    for b in buckets:
        file_path = b.get("file_path", bucket_file_path(b["uri"]))
        md_path = os.path.join(BASE_DIR, file_path + ".md")
        if not os.path.exists(md_path):
            continue
        fm = read_frontmatter(md_path)
        uri = fm.get("uri", b["uri"])
        scanned = fm.get("scanned", "")
        if scanned and "T" in scanned:
            scanned = scanned.split("T")[0]
        rows.append(
            (
                f"[{uri}]({file_path}.md)",
                fm.get("files", ""),
                fm.get("size", ""),
                scanned,
            )
        )

    rows.sort(key=lambda r: r[0])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Render index.md from plan JSON.")
    parser.add_argument("--plan", required=True, help="Path to .plan.json file")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    args = parser.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)

    result = render_index(plan)

    if args.output:
        write_text(args.output, result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
