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
    parse_semver,
    parse_uri,
    read_frontmatter,
    read_json_data,
    read_json_metadata,
    read_json_versions,
    read_md_metadata,
    read_md_scanned,
    read_md_versions,
    studio_available,
    write_json,
)

# Calibration runs are session-scoped instrumentation, not knowledge — they
# don't get JSON/MD records, don't appear in the index, and don't show up in
# reuse recommendations. The skill prescribes `.persist()` for calibration
# runs; this filter is defense-in-depth for cases where one was accidentally
# `.save()`d under the `calib_` naming convention or tagged with
# `scope:calibration`. The legacy `pilot_` / `scope:pilot` aliases are also
# recognized.
_CALIB_NAME_PREFIXES = ("calib_", "pilot_")
_CALIB_ATTRS = frozenset({"scope:calibration", "scope:pilot"})


def _is_calibration(entry: dict) -> bool:
    name = entry.get("name") or ""
    if name.startswith(_CALIB_NAME_PREFIXES):
        return True
    if any(f".{p}" in name for p in _CALIB_NAME_PREFIXES):
        return True
    attrs = entry.get("attrs") or []
    return any(a in _CALIB_ATTRS for a in attrs)


def _process_dataset_entry(name: str, entries: list[dict]) -> dict | None:
    source = entries[0]["source"]
    versions_sorted = sorted(
        [e["version"] for e in entries if e["version"]],
        key=parse_semver,
    )
    if not versions_sorted:
        return None
    latest_version = versions_sorted[-1]

    latest_entry = next(
        (e for e in entries if e["version"] == latest_version), entries[-1]
    )

    file_path = dataset_file_path(name, source)
    abs_json_path = os.path.join("dc-knowledge", file_path + ".json")
    abs_md_path = os.path.join("dc-knowledge", file_path + ".md")

    if os.path.exists(abs_json_path):
        file_versions = read_json_versions(abs_json_path)
        file_fm = read_json_metadata(abs_json_path)
    elif os.path.exists(abs_md_path):
        file_versions = read_md_versions(abs_md_path)
        file_fm = read_md_metadata(abs_md_path)
    else:
        file_versions = []
        file_fm = {}

    file_exists = bool(file_versions)

    file_versions_set = set(file_versions)
    versions_to_fetch = [v for v in versions_sorted if v not in file_versions_set]

    if latest_version not in versions_to_fetch:
        file_latest = file_fm.get("last_version", "")
        file_records = file_fm.get("records", "")
        latest_records = str(latest_entry.get("records") or "")
        if file_latest != latest_version or file_records != latest_records:
            versions_to_fetch.append(latest_version)

    if not file_exists:
        status = "new"
    elif versions_to_fetch:
        status = "stale"
    else:
        status = "ok"

    history_complete = versions_sorted[0] == "1.0.0"

    return {
        "name": name,
        "source": source,
        "file_path": file_path,
        "status": status,
        "last_version": latest_version,
        "records": latest_entry.get("records"),
        "updated": latest_entry.get("updated"),
        "known_versions": versions_sorted,
        "file_versions": file_versions,
        "versions_to_fetch": versions_to_fetch,
        "history_complete": history_complete,
    }


def plan_datasets(
    dc, db_last_updated: str, studio: bool = False
) -> tuple[list[dict], bool]:
    """Plan dataset updates. Returns (datasets_out, up_to_date)."""
    index_path = "dc-knowledge/index.md"
    index_fm = read_frontmatter(index_path)
    index_db_updated = index_fm.get("db_last_updated", "")

    if db_last_updated and db_last_updated == index_db_updated:
        return [], True

    all_datasets = list(collect_datasets(dc, studio=False))
    if studio:
        seen_keys = {(e["name"], e["version"]) for e in all_datasets}
        for entry in collect_datasets(dc, studio=True):
            key = (entry["name"], entry["version"])
            if key not in seen_keys:
                seen_keys.add(key)
                all_datasets.append(entry)

    all_datasets = [e for e in all_datasets if not _is_calibration(e)]

    by_name: dict[str, list[dict]] = {}
    for entry in all_datasets:
        by_name.setdefault(entry["name"], []).append(entry)

    datasets_out = []
    for name in sorted(by_name):
        result = _process_dataset_entry(name, by_name[name])
        if result is not None:
            datasets_out.append(result)

    up_to_date = bool(datasets_out) and all(d["status"] == "ok" for d in datasets_out)
    return datasets_out, up_to_date


def plan_buckets() -> list[dict]:
    """Auto-discover all bucket listings from the catalog."""
    try:
        from datachain.query import Session

        session = Session.get()
        catalog = session.catalog
        listings = catalog.listings()
    except Exception:  # noqa: BLE001
        return []

    buckets_out = []
    for listing in listings:
        uri = listing.uri.rstrip("/") + "/"
        finished = listing.finished_at.isoformat() if listing.finished_at else None

        parts = parse_uri(uri)
        file_path = bucket_file_path(uri)
        abs_json_path = os.path.join("dc-knowledge", file_path + ".json")
        abs_md_path = os.path.join("dc-knowledge", file_path + ".md")

        existing = read_json_data(abs_json_path)
        if existing is None:
            md_scanned = read_md_scanned(abs_md_path)
            if md_scanned:
                existing = {"scanned": md_scanned}

        if existing is None:
            status = "new"
            scanned = None
        else:
            scanned = existing.get("scanned")
            if (
                scanned
                and finished
                and scanned.replace("T", " ")[:19] >= finished.replace("T", " ")[:19]
            ) or (scanned and not finished):
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
                "scanned": scanned,
                "listing_finished": finished,
                "listing_expired": listing.is_expired,
            }
        )
    return buckets_out


def cmd_plan(studio: bool = False, output: str | None = None):
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

    # Plan datasets
    datasets_out, datasets_up_to_date = plan_datasets(dc, db_last_updated, studio)

    # Auto-discover buckets from catalog listings
    buckets_out = plan_buckets()
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
    result["datasets"] = datasets_out
    result["buckets"] = buckets_out

    if output:
        write_json(output, result)
    else:
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute what datasets/buckets need updating and output a JSON plan."
        )
    )
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Include Studio datasets",
    )
    parser.add_argument("--output", help="Output JSON file path (default: stdout)")
    args = parser.parse_args()
    cmd_plan(studio=args.studio, output=args.output)


if __name__ == "__main__":
    main()
