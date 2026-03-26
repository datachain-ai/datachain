#!/usr/bin/env python3
"""Compute what needs updating and output a JSON plan."""

import argparse
import json
import os
from datetime import datetime, timezone
from glob import glob

from utils import (
    bucket_file_path,
    collect_datasets,
    dataset_file_path,
    dc_import,
    get_listing_finished_at,
    parse_semver,
    parse_uri,
    read_frontmatter,
    read_json_data,
    read_json_metadata,
    read_json_versions,
    studio_available,
)


def plan_datasets(
    dc, db_last_updated: str, studio: bool = False
) -> tuple[list[dict], bool]:
    """Plan dataset updates. Returns (datasets_out, up_to_date)."""
    # Read existing index.md to check timestamp
    index_path = ".datachain/graph/index.md"
    index_fm = read_frontmatter(index_path)
    index_db_updated = index_fm.get("db_last_updated", "")

    # Early exit if timestamps match
    if db_last_updated and db_last_updated == index_db_updated:
        return [], True

    # Collect datasets
    all_datasets = list(collect_datasets(dc, studio=False))
    if studio:
        seen_keys = {(e["name"], e["version"]) for e in all_datasets}
        for entry in collect_datasets(dc, studio=True):
            key = (entry["name"], entry["version"])
            if key not in seen_keys:
                seen_keys.add(key)
                all_datasets.append(entry)

    # Group by name
    by_name: dict[str, list[dict]] = {}
    for entry in all_datasets:
        by_name.setdefault(entry["name"], []).append(entry)

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

        latest_entry = next(
            (e for e in entries if e["version"] == latest_version), entries[-1]
        )

        file_path = dataset_file_path(name, source)
        abs_json_path = os.path.join(".datachain/graph", file_path + ".json")

        file_exists = os.path.exists(abs_json_path)
        file_versions = read_json_versions(abs_json_path) if file_exists else []
        file_fm = read_json_metadata(abs_json_path) if file_exists else {}

        file_versions_set = set(file_versions)
        versions_to_fetch = [v for v in versions_sorted if v not in file_versions_set]

        if latest_version not in versions_to_fetch:
            file_latest = file_fm.get("latest_version", "")
            file_num_obj = file_fm.get("num_objects", "")
            latest_num_obj = str(latest_entry.get("num_objects") or "")
            if file_latest != latest_version or file_num_obj != latest_num_obj:
                versions_to_fetch.append(latest_version)

        if not file_exists:
            status = "new"
        elif versions_to_fetch:
            status = "stale"
        else:
            status = "ok"

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

    up_to_date = bool(datasets_out) and all(d["status"] == "ok" for d in datasets_out)
    return datasets_out, up_to_date


def plan_buckets(uris: list[str]) -> list[dict]:
    """Plan bucket scans. Returns list of bucket entries with status."""
    buckets_out = []
    for uri in uris:
        if not uri.endswith("/"):
            uri = uri + "/"

        parts = parse_uri(uri)
        file_path = bucket_file_path(uri)
        abs_json_path = os.path.join(".datachain/graph", file_path + ".json")

        existing = read_json_data(abs_json_path)

        if existing is None:
            status = "new"
            scanned_at = None
        else:
            scanned_at = existing.get("scanned_at")
            listing_finished = get_listing_finished_at(uri)
            if (scanned_at and listing_finished and scanned_at >= listing_finished) or (
                scanned_at and not listing_finished
            ):
                status = "ok"
            else:
                status = "stale"

        buckets_out.append(
            {
                "uri": uri,
                "scheme": parts["scheme"],
                "bucket": parts["bucket"],
                "prefix": parts["prefix"],
                "file_path": file_path,
                "status": status,
                "scanned_at": scanned_at,
            }
        )
    return buckets_out


def cmd_plan(studio: bool = False, bucket_uris: list[str] | None = None):
    """Compute what needs updating and output a JSON plan."""
    dc = dc_import()

    # Get DB mtime
    matches = glob(".datachain/db*")
    if not matches:
        db_last_updated = "1970-01-01T00:00:00Z"
    else:
        mtime = max(os.path.getmtime(p) for p in matches)
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        db_last_updated = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Plan datasets (unless only buckets requested)
    if bucket_uris and not studio:
        # --buckets without --studio: skip dataset planning
        datasets_out = []
        datasets_up_to_date = True
    else:
        datasets_out, datasets_up_to_date = plan_datasets(dc, db_last_updated, studio)

    # Plan buckets
    buckets_out = plan_buckets(bucket_uris) if bucket_uris else []
    buckets_up_to_date = not buckets_out or all(
        b["status"] == "ok" for b in buckets_out
    )

    # Overall up_to_date: both must be true, and at least one must have entries
    has_work = bool(datasets_out) or bool(buckets_out)
    up_to_date = has_work and datasets_up_to_date and buckets_up_to_date

    result: dict = {
        "up_to_date": up_to_date,
        "studio_available": studio_available(),
    }
    if db_last_updated:
        result["db_last_updated"] = db_last_updated
    if datasets_out or not bucket_uris:
        result["datasets"] = datasets_out
    if buckets_out:
        result["buckets"] = buckets_out

    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Compute what datasets/buckets need updating and output a JSON plan."
    )
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Include Studio datasets",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        metavar="URI",
        help="One or more bucket URIs to plan (e.g., s3://bucket/ gs://bucket/prefix/)",
    )
    args = parser.parse_args()
    cmd_plan(studio=args.studio, bucket_uris=args.buckets)


if __name__ == "__main__":
    main()
