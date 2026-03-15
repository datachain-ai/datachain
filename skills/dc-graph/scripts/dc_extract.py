#!/usr/bin/env python3
"""Standalone DataChain metadata extractor for the datachain-graph skill.

Usage:
    python3 dc_extract.py --db-mtime           # ISO-8601 UTC mtime of .datachain/db* files
    python3 dc_extract.py --list               # JSON list of all user datasets
    python3 dc_extract.py --dataset <name>     # JSON schema + preview for one dataset
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path


def cmd_db_mtime():
    matches = glob(".datachain/db*")
    if not matches:
        # No DB files found — return epoch so the graph is always stale
        print("1970-01-01T00:00:00Z")
        return
    mtime = max(os.path.getmtime(p) for p in matches)
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    print(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


def cmd_list():
    try:
        import datachain as dc
    except ImportError:
        print(
            json.dumps({"error": "datachain not installed"}),
            file=sys.stderr,
        )
        sys.exit(1)

    datasets = []
    for row in dc.datasets(column="dataset").to_iter():
        info = row[0]
        if getattr(info, "namespace", None) in ("system", "listing"):
            continue
        if getattr(info, "project", None) == "listing":
            continue
        if getattr(info, "is_temp", False):
            continue
        datasets.append(
            {
                "name": info.name,
                "version": str(info.version) if info.version is not None else None,
                "num_objects": getattr(info, "num_objects", None),
                "status": getattr(info, "status", None),
                "namespace": getattr(info, "namespace", None),
                "project": getattr(info, "project", None),
                "created_at": (
                    info.created_at.isoformat()
                    if getattr(info, "created_at", None) is not None
                    else None
                ),
                "updated_at": (
                    info.updated_at.isoformat()
                    if getattr(info, "updated_at", None) is not None
                    else None
                ),
            }
        )

    print(json.dumps({"datasets": datasets}))


def cmd_dataset(name: str):
    try:
        import datachain as dc
    except ImportError:
        print(
            json.dumps({"error": "datachain not installed"}),
            file=sys.stderr,
        )
        sys.exit(1)

    chain = dc.read_dataset(name)
    schema = {
        col: str(typ)
        for col, typ in chain.schema.items()
        if col != "sys" and not col.startswith("sys.")
    }
    cols = list(schema.keys())

    preview = []
    for row in chain.select(*cols).limit(10).to_iter():
        if isinstance(row, tuple):
            preview.append(dict(zip(cols, row)))
        else:
            preview.append({cols[0]: row} if cols else {})

    print(json.dumps({"name": name, "schema": schema, "preview": preview}))


def main():
    parser = argparse.ArgumentParser(
        description="DataChain metadata extractor for the datachain-graph skill."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--db-mtime",
        action="store_true",
        help="Print max mtime of .datachain/db* files as ISO-8601 UTC",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="Print JSON list of all user datasets",
    )
    group.add_argument(
        "--dataset",
        metavar="NAME",
        help="Print JSON schema and preview for the named dataset",
    )
    args = parser.parse_args()

    if args.db_mtime:
        cmd_db_mtime()
    elif args.list:
        cmd_list()
    elif args.dataset:
        cmd_dataset(args.dataset)


if __name__ == "__main__":
    main()
