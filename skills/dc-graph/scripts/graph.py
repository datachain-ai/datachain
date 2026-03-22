#!/usr/bin/env python3
"""Standalone DataChain metadata extractor for the datachain-graph skill.

Usage:
    python3 graph.py --db-mtime               # ISO-8601 UTC mtime of .datachain/db* files
    python3 graph.py --list                   # JSON list of all user datasets
    python3 graph.py --dataset <name>         # JSON schema, preview, and dependencies
    python3 graph.py --dataset <name@version> # same, for a specific version
"""

import argparse
import inspect
import json
import os
import sys
import types
import typing
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


def _type_name(tp):
    if tp is type(None):
        return "None"
    if isinstance(tp, types.UnionType):  # Python 3.10+ X | Y
        return " | ".join(_type_name(a) for a in tp.__args__)
    origin = getattr(tp, "__origin__", None)
    if origin is typing.Union:
        return " | ".join(_type_name(a) for a in tp.__args__)
    if origin is list:
        args = getattr(tp, "__args__", ())
        return f"list[{_type_name(args[0])}]" if args else "list"
    if origin is dict:
        return "dict"
    return getattr(tp, "__name__", str(tp))


def _expand_signal(typ):
    """Return {"type": name, "fields": {name: type_str} | None}.
    Fields is None for File subclasses (well-known) and primitives."""
    from datachain.lib.data_model import DataModel
    from datachain.lib.file import File

    type_name = _type_name(typ)
    if not (inspect.isclass(typ) and issubclass(typ, DataModel)):
        return {"type": type_name, "fields": None}
    if issubclass(typ, File):
        return {"type": type_name, "fields": None}  # skip — covered by dc-core
    fields = {
        fname: _type_name(finfo.annotation) for fname, finfo in typ.model_fields.items()
    }
    return {"type": type_name, "fields": fields}


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
        col: _expand_signal(typ)
        for col, typ in chain.schema.items()
        if col != "sys" and not col.startswith("sys.")
    }

    def _serialize(val):
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        return str(val)

    df = chain.limit(10).to_pandas(flatten=True, include_hidden=False)
    preview_columns = list(df.columns)
    preview_rows = [[_serialize(v) for v in row] for row in df.itertuples(index=False)]
    preview = {"columns": preview_columns, "rows": preview_rows}

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
