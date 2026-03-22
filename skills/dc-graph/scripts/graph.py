#!/usr/bin/env python3
"""Standalone DataChain metadata extractor for the datachain-graph skill.

Usage:
    python3 graph.py --db-mtime               # ISO-8601 UTC mtime of .datachain/db* files
    python3 graph.py --list                   # JSON list of all user datasets
    python3 graph.py --dataset <name>         # JSON schema, preview, and dependencies
    python3 graph.py --dataset <name@version> # same, for a specific version
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from glob import glob


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


def cmd_dataset(name_version: str):
    try:
        import datachain as dc
    except ImportError:
        print(
            json.dumps({"error": "datachain not installed"}),
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse optional @version suffix
    if "@" in name_version:
        name, version = name_version.split("@", 1)
        chain = dc.read_dataset(name, version=version)
    else:
        name = name_version
        version = None
        chain = dc.read_dataset(name)

    schema = {
        col: str(typ)
        for col, typ in chain.schema.items()
        if col != "sys" and not col.startswith("sys.")
    }
    cols = list(schema.keys())

    def _serialize(val):
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        return str(val)

    preview = []
    for row in chain.select(*cols).limit(10).to_iter():
        if isinstance(row, tuple):
            preview.append({k: _serialize(v) for k, v in zip(cols, row)})
        else:
            preview.append({cols[0]: _serialize(row)} if cols else {})

    # Resolve version if not provided (needed for dependency lookup)
    if version is None:
        for row in dc.datasets(column="dataset").to_iter():
            info = row[0]
            if info.name == name:
                version = str(info.version)
                break

    # Fetch dependency tree
    dependencies = []
    try:
        catalog = chain.session.catalog
        if version:
            deps = catalog.get_dataset_dependencies(
                name=name, version=version, indirect=True
            )
            for dep in deps or []:
                if not dep:
                    continue
                dep_entry = {
                    "name": dep.name,
                    "version": str(dep.version) if dep.version is not None else None,
                    "type": str(dep.type) if dep.type is not None else None,
                    "dependencies": [
                        {
                            "name": child.name,
                            "version": (
                                str(child.version)
                                if child.version is not None
                                else None
                            ),
                            "type": str(child.type) if child.type is not None else None,
                        }
                        for child in (dep.dependencies or [])
                        if child
                    ],
                }
                dependencies.append(dep_entry)
    except Exception:
        pass  # dependencies are best-effort

    print(
        json.dumps(
            {
                "name": name,
                "schema": schema,
                "preview": preview,
                "dependencies": dependencies,
            }
        )
    )


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
