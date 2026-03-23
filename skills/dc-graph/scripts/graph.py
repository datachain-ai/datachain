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


def _studio_available() -> bool:
    """Return True if a Studio token is configured (env var or config file)."""
    try:
        from datachain.remote.studio import is_token_set

        return is_token_set()
    except Exception:
        return False


def cmd_db_mtime():
    matches = glob(".datachain/db*")
    if not matches:
        if _studio_available():
            # No local DB but Studio is configured — signal Studio mode.
            # The skill will skip the timestamp comparison and always refresh.
            print("studio")
        else:
            # No DB, no Studio — return epoch so the graph is always stale
            print("1970-01-01T00:00:00Z")
        return
    mtime = max(os.path.getmtime(p) for p in matches)
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    print(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


def _collect_datasets(dc, studio: bool) -> list[dict]:
    """Return a list of dataset dicts from local or Studio source."""
    results = []
    try:
        for row in dc.datasets(column="dataset", studio=studio).to_iter():
            info = row[0]
            if getattr(info, "namespace", None) in ("system", "listing"):
                continue
            if getattr(info, "project", None) == "listing":
                continue
            if getattr(info, "is_temp", False):
                continue
            namespace = getattr(info, "namespace", None)
            project = getattr(info, "project", None)
            # Fully-qualify Studio dataset names so --dataset can route to the
            # Studio API when fetching metadata (query_script, changes, deps).
            if studio and namespace and project:
                full_name = f"{namespace}/{project}/{info.name}"
            else:
                full_name = info.name
            results.append(
                {
                    "name": full_name,
                    "version": str(info.version) if info.version is not None else None,
                    "num_objects": getattr(info, "num_objects", None),
                    "status": getattr(info, "status", None),
                    "namespace": namespace,
                    "project": project,
                    "source": "studio" if studio else "local",
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
    except Exception:
        pass  # best-effort per source
    return results


def cmd_list(studio: bool = False):
    try:
        import datachain as dc
    except ImportError:
        print(
            json.dumps({"error": "datachain not installed"}),
            file=sys.stderr,
        )
        sys.exit(1)

    datasets = []
    seen = set()  # (name, version) dedup across sources

    if studio:
        for entry in _collect_datasets(dc, studio=True):
            key = (entry["name"], entry["version"])
            if key not in seen:
                seen.add(key)
                datasets.append(entry)
    else:
        for entry in _collect_datasets(dc, studio=False):
            key = (entry["name"], entry["version"])
            if key not in seen:
                seen.add(key)
                datasets.append(entry)

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

    from datachain.dataset import parse_dataset_with_version

    name, version = parse_dataset_with_version(name_version)

    # Detect Studio routing: name is namespace/project/bare_name (slash-separated,
    # from --list) or namespace.project.bare_name (dot-separated, typed manually).
    slash_parts = name.split("/", 2)
    dot_parts = name.split(".", 2)
    if len(slash_parts) == 3:
        _namespace, _project, _bare_name = slash_parts
    elif len(dot_parts) == 3:
        _namespace, _project, _bare_name = dot_parts
    else:
        _namespace, _project, _bare_name = None, None, name

    chain = dc.read_dataset(name_version)

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
            if info.name == _bare_name:
                version = str(info.version)
                break

    # Fetch query_script and locate previous version for Changes section
    query_script = None
    _prev_version_info = None  # (prev_version_str, prev_script)
    dependencies = []
    try:
        catalog = chain.session.catalog
        if _namespace and _project:
            # Studio dataset — fetch full record (all versions + scripts) from Studio API
            dataset = catalog.get_remote_dataset(_namespace, _project, _bare_name)
        else:
            dataset = catalog.get_dataset(_bare_name, include_incomplete=False)
        resolved_ver = version or dataset.latest_version
        dv = dataset.get_version(resolved_ver)
        query_script = dv.query_script or None
        sorted_vers = sorted(dataset.versions, key=lambda v: v.version_value)
        idx = next(
            (i for i, v in enumerate(sorted_vers) if v.version == resolved_ver), None
        )
        if idx is not None and idx > 0:
            p = sorted_vers[idx - 1]
            _prev_version_info = (p.version, p.query_script or None)
    except Exception:
        pass  # query_script and previous-version lookup are best-effort

    try:
        catalog = chain.session.catalog
        if version:
            deps = catalog.get_dataset_dependencies(
                name=_bare_name, version=version, indirect=True
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

    # Compute Changes section vs previous version
    changes = None
    if _prev_version_info is not None:
        prev_version_str, prev_script = _prev_version_info
        script_changed = query_script != prev_script
        changes = {
            "previous_version": prev_version_str,
            "script_changed": script_changed,
            "previous_script": prev_script if script_changed else None,
            "deps_added": [],
            "deps_removed": [],
            "deps_updated": [],
        }
        try:
            catalog = chain.session.catalog
            prev_deps_raw = (
                catalog.get_dataset_dependencies(
                    name=_bare_name, version=prev_version_str, indirect=True
                )
                or []
            )
            curr_dep_map = {d["name"]: d["version"] for d in dependencies}
            prev_dep_map = {
                d.name: (str(d.version) if d.version is not None else None)
                for d in prev_deps_raw
                if d
            }
            curr_names = set(curr_dep_map)
            prev_names = set(prev_dep_map)
            changes["deps_added"] = [
                {"name": n, "version": curr_dep_map[n]}
                for n in sorted(curr_names - prev_names)
            ]
            changes["deps_removed"] = [
                {"name": n, "version": prev_dep_map[n]}
                for n in sorted(prev_names - curr_names)
            ]
            for n in sorted(curr_names & prev_names):
                cv, pv = curr_dep_map[n], prev_dep_map[n]
                if cv != pv:
                    entry = {
                        "name": n,
                        "version_from": pv,
                        "version_to": cv,
                        "script_changed": False,
                    }
                    try:
                        dep_ds = catalog.get_dataset(n, include_incomplete=False)
                        cs = dep_ds.get_version(cv).query_script or None
                        ps = dep_ds.get_version(pv).query_script or None
                        if cs != ps:
                            entry["script_changed"] = True
                            entry["previous_script"] = ps
                            entry["current_script"] = cs
                    except Exception:
                        pass
                    changes["deps_updated"].append(entry)
        except Exception:
            pass  # dep-level changes are best-effort

    print(
        json.dumps(
            {
                "name": _bare_name,
                "schema": schema,
                "preview": preview,
                "query_script": query_script,
                "changes": changes,
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
        help="Print JSON list of local datasets",
    )
    group.add_argument(
        "--list-studio",
        action="store_true",
        help="Print JSON list of Studio datasets (requires auth token)",
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
        cmd_list(studio=False)
    elif args.list_studio:
        cmd_list(studio=True)
    elif args.dataset:
        cmd_dataset(args.dataset)


if __name__ == "__main__":
    main()
