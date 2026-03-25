#!/usr/bin/env python3
"""Compute what needs updating and output a JSON plan."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from glob import glob

from utils import (
    collect_datasets,
    dataset_file_path,
    dc_import,
    parse_semver,
    read_file_versions,
    read_frontmatter,
    studio_available,
)


def cmd_plan(studio: bool = False):
    """Compute what needs updating and output a JSON plan."""
    dc = dc_import()

    # Step 1: get DB mtime
    matches = glob(".datachain/db*")
    if not matches:
        db_last_updated = "1970-01-01T00:00:00Z"
    else:
        mtime = max(os.path.getmtime(p) for p in matches)
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        db_last_updated = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Step 2: read existing index.md to check timestamp
    index_path = ".datachain/graph/index.md"
    index_fm = read_frontmatter(index_path)
    index_db_updated = index_fm.get("db_last_updated", "")

    # Step 3: early exit if timestamps match
    if db_last_updated and db_last_updated == index_db_updated:
        print(
            json.dumps(
                {
                    "up_to_date": True,
                    "db_last_updated": db_last_updated,
                    "datasets": [],
                }
            )
        )
        return

    # Step 4: collect datasets
    all_datasets = []
    for entry in collect_datasets(dc, studio=False):
        all_datasets.append(entry)
    if studio:
        seen_keys = {(e["name"], e["version"]) for e in all_datasets}
        for entry in collect_datasets(dc, studio=True):
            key = (entry["name"], entry["version"])
            if key not in seen_keys:
                seen_keys.add(key)
                all_datasets.append(entry)

    # Step 5: group by name
    by_name: dict[str, list[dict]] = {}
    for entry in all_datasets:
        by_name.setdefault(entry["name"], []).append(entry)

    # Step 6: for each dataset, compute plan info
    datasets_out = []
    for name in sorted(by_name):
        entries = by_name[name]
        source = entries[0]["source"]

        versions_sorted = sorted(
            [e["version"] for e in entries if e["version"]],
            key=parse_semver,
        )
        if not versions_sorted:
            continue
        latest_version = versions_sorted[-1]

        # Get latest entry metadata
        latest_entry = next(
            (e for e in entries if e["version"] == latest_version), entries[-1]
        )

        # Derive file path
        file_path = dataset_file_path(name, source)
        abs_file_path = os.path.join(".datachain/graph", file_path)

        # Read existing file
        file_exists = os.path.exists(abs_file_path)
        file_versions = read_file_versions(abs_file_path) if file_exists else []
        file_fm = read_frontmatter(abs_file_path) if file_exists else {}

        # versions_to_fetch: all versions not yet in file
        file_versions_set = set(file_versions)
        versions_to_fetch = [v for v in versions_sorted if v not in file_versions_set]

        # Always include latest_version to refresh frontmatter if stale
        if latest_version not in versions_to_fetch:
            file_latest = file_fm.get("latest_version", "")
            file_num_obj = file_fm.get("num_objects", "")
            latest_num_obj = str(latest_entry.get("num_objects") or "")
            if file_latest != latest_version or file_num_obj != latest_num_obj:
                versions_to_fetch.append(latest_version)

        # Determine status
        if not file_exists:
            status = "new"
        elif versions_to_fetch:
            status = "stale"
        else:
            status = "ok"

        # history_complete: false if oldest known version is not "1.0.0"
        history_complete = versions_sorted[0] == "1.0.0"

        datasets_out.append(
            {
                "name": name,
                "source": source,
                "file_path": file_path,
                "status": status,
                "latest_version": latest_version,
                "num_objects": latest_entry.get("num_objects"),
                "updated_at": latest_entry.get("updated_at"),
                "known_versions": versions_sorted,
                "file_versions": file_versions,
                "versions_to_fetch": versions_to_fetch,
                "history_complete": history_complete,
            }
        )

    up_to_date = bool(datasets_out) and all(
        d["status"] == "ok" for d in datasets_out
    )

    result: dict = {
        "up_to_date": up_to_date,
        "studio_available": studio_available(),
        "datasets": datasets_out,
    }
    if db_last_updated:
        result["db_last_updated"] = db_last_updated

    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(
        description="Compute what datasets need updating and output a JSON plan."
    )
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Include Studio datasets",
    )
    args = parser.parse_args()
    cmd_plan(studio=args.studio)


if __name__ == "__main__":
    main()
