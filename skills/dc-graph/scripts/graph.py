#!/usr/bin/env python3
"""Standalone DataChain metadata extractor for the datachain-graph skill.

Usage:
    python3 graph.py --plan [--studio]         # JSON plan: what needs updating
    python3 graph.py --dataset-all <name>      # JSON data for all versions of a dataset
    python3 graph.py --dataset <name>          # JSON schema, preview, and dependencies
    python3 graph.py --dataset <name@version>  # same, for a specific version
    python3 graph.py --db-mtime               # ISO-8601 UTC mtime of .datachain/db* files
    python3 graph.py --list                   # JSON list of all user datasets
"""

import argparse
import inspect
import json
import os
import re
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


def _parse_semver(v):
    """Parse version string into a tuple for sorting."""
    try:
        return tuple(int(x) for x in str(v).split("."))
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _read_frontmatter(path):
    """Read YAML frontmatter from a markdown file. Returns dict or {}."""
    try:
        with open(path) as f:
            content = f.read()
        if not content.startswith("---"):
            return {}
        end = content.index("\n---", 3)
        fm_text = content[4:end]  # skip first "---\n"
        result = {}
        for line in fm_text.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                result[key.strip()] = val.strip().strip('"').strip("'")
        return result
    except Exception:
        return {}


def _read_file_versions(path):
    """Find all '### X.Y.Z' headings in a markdown file, in order."""
    try:
        with open(path) as f:
            content = f.read()
        return re.findall(r"^### (\d+\.\d+\.\d+)", content, re.MULTILINE)
    except Exception:
        return []


def _dataset_file_path(name, source):
    """Derive the relative file path (from .datachain/graph/) for a dataset."""
    dot_parts = name.split(".", 2)
    if source == "studio" and len(dot_parts) == 3:
        namespace, project, bare_name = dot_parts
        bare_name_slug = bare_name.lower().replace(".", "_")
        return f"datasets/{namespace}/{project}/{bare_name_slug}.md"
    else:
        name_slug = name.lower().replace(".", "_")
        return f"datasets/{name_slug}.md"


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
            # Fully-qualify Studio dataset names using dot-notation (namespace.project.name).
            # Dots are used in all human-visible content; / is used only for file paths.
            if studio and namespace and project:
                full_name = f"{namespace}.{project}.{info.name}"
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


def cmd_plan(studio: bool = False):
    """Compute what needs updating and output a JSON plan."""
    try:
        import datachain as dc
    except ImportError:
        print(json.dumps({"error": "datachain not installed"}), file=sys.stderr)
        sys.exit(1)

    # Step 1: get DB mtime
    matches = glob(".datachain/db*")
    db_last_updated = None
    studio_mode = False
    if not matches:
        if _studio_available():
            studio_mode = True
        else:
            db_last_updated = "1970-01-01T00:00:00Z"
    else:
        mtime = max(os.path.getmtime(p) for p in matches)
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        db_last_updated = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Step 2: read existing index.md to check timestamp
    index_path = ".datachain/graph/index.md"
    index_fm = _read_frontmatter(index_path)
    index_db_updated = index_fm.get("db_last_updated", "")

    # Step 3: early exit if timestamps match (local mode only)
    if not studio_mode and db_last_updated and db_last_updated == index_db_updated:
        print(json.dumps({"up_to_date": True, "db_last_updated": db_last_updated, "datasets": []}))
        return

    # Step 4: collect datasets
    all_datasets = []
    for entry in _collect_datasets(dc, studio=False):
        all_datasets.append(entry)
    if studio:
        seen_keys = {(e["name"], e["version"]) for e in all_datasets}
        for entry in _collect_datasets(dc, studio=True):
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
            key=_parse_semver,
        )
        if not versions_sorted:
            continue
        latest_version = versions_sorted[-1]

        # Get latest entry metadata
        latest_entry = next(
            (e for e in entries if e["version"] == latest_version), entries[-1]
        )

        # Derive file path
        file_path = _dataset_file_path(name, source)
        abs_file_path = os.path.join(".datachain/graph", file_path)

        # Read existing file
        file_exists = os.path.exists(abs_file_path)
        file_versions = _read_file_versions(abs_file_path) if file_exists else []
        file_fm = _read_frontmatter(abs_file_path) if file_exists else {}

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

        datasets_out.append({
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
        })

    up_to_date = bool(datasets_out) and all(d["status"] == "ok" for d in datasets_out)

    result: dict = {
        "up_to_date": up_to_date,
        "datasets": datasets_out,
    }
    if db_last_updated:
        result["db_last_updated"] = db_last_updated

    print(json.dumps(result))


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


def _fetch_version_data(name_version: str) -> dict:
    """Fetch schema, preview, query_script, changes, and dependencies for one dataset version.

    Returns a dict with keys: name, schema, preview, query_script, changes, dependencies.
    """
    import datachain as dc
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

    is_studio_dataset = bool(_namespace and _project)

    # --- Schema and Preview: requires live data access ---
    chain = None
    try:
        chain = dc.read_dataset(name_version)
    except Exception:
        pass

    schema = {}
    if chain is not None:
        try:
            schema = {
                col: _expand_signal(typ)
                for col, typ in chain.schema.items()
                if col != "sys" and not col.startswith("sys.")
            }
        except Exception:
            pass

    def _serialize(val):
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        return str(val)

    preview = None
    if chain is not None:
        try:
            df = chain.limit(10).to_pandas(flatten=True, include_hidden=False)
            preview_columns = list(df.columns)
            preview_rows = [
                [_serialize(v) for v in row] for row in df.itertuples(index=False)
            ]
            preview = {"columns": preview_columns, "rows": preview_rows}
        except Exception:
            pass  # data not locally accessible

    # --- Metadata: query_script, version history ---
    query_script = None
    _prev_version_info = None
    dependencies = []

    if is_studio_dataset:
        # Fetch full dataset record directly from Studio API — no local catalog needed.
        try:
            from datachain.dataset import DatasetRecord
            from datachain.remote.studio import StudioClient

            client = StudioClient()
            response = client.dataset_info(_namespace, _project, _bare_name)
            if response.ok and response.data:
                dataset = DatasetRecord.from_dict(response.data)
                resolved_ver = version or dataset.latest_version
                if version is None:
                    version = resolved_ver
                dv = dataset.get_version(resolved_ver)
                query_script = dv.query_script or None
                sorted_vers = sorted(dataset.versions, key=lambda v: v.version_value)
                idx = next(
                    (i for i, v in enumerate(sorted_vers) if v.version == resolved_ver),
                    None,
                )
                if idx is not None and idx > 0:
                    p = sorted_vers[idx - 1]
                    _prev_version_info = (p.version, p.query_script or None)
        except Exception:
            pass  # best-effort
    else:
        # Resolve version for local datasets
        if version is None:
            for row in dc.datasets(column="dataset").to_iter():
                info = row[0]
                if info.name == _bare_name:
                    version = str(info.version)
                    break

        # Fetch from local catalog
        try:
            catalog = chain.session.catalog if chain is not None else None
            if catalog is not None:
                dataset = catalog.get_dataset(_bare_name, include_incomplete=False)
                resolved_ver = version or dataset.latest_version
                dv = dataset.get_version(resolved_ver)
                query_script = dv.query_script or None
                sorted_vers = sorted(dataset.versions, key=lambda v: v.version_value)
                idx = next(
                    (i for i, v in enumerate(sorted_vers) if v.version == resolved_ver),
                    None,
                )
                if idx is not None and idx > 0:
                    p = sorted_vers[idx - 1]
                    _prev_version_info = (p.version, p.query_script or None)
        except Exception:
            pass

        # Dependencies (local metastore only)
        try:
            catalog = chain.session.catalog if chain is not None else None
            if catalog is not None and version:
                deps = catalog.get_dataset_dependencies(
                    name=_bare_name, version=version, indirect=True
                )
                for dep in deps or []:
                    if not dep:
                        continue
                    dep_entry = {
                        "name": dep.name,
                        "version": str(dep.version)
                        if dep.version is not None
                        else None,
                        "type": str(dep.type) if dep.type is not None else None,
                        "dependencies": [
                            {
                                "name": child.name,
                                "version": (
                                    str(child.version)
                                    if child.version is not None
                                    else None
                                ),
                                "type": str(child.type)
                                if child.type is not None
                                else None,
                            }
                            for child in (dep.dependencies or [])
                            if child
                        ],
                    }
                    dependencies.append(dep_entry)
        except Exception:
            pass

    # --- Changes section ---
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
        # Dep-level changes only available from local metastore
        if not is_studio_dataset:
            try:
                catalog = chain.session.catalog if chain is not None else None
                if catalog is not None:
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
                                dep_ds = catalog.get_dataset(
                                    n, include_incomplete=False
                                )
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
                pass

    # Return the fully-qualified dot-separated name for Studio datasets so the skill
    # uses it in headings and frontmatter; bare name for local datasets.
    output_name = (
        f"{_namespace}.{_project}.{_bare_name}" if is_studio_dataset else _bare_name
    )

    return {
        "name": output_name,
        "schema": schema,
        "preview": preview,
        "query_script": query_script,
        "changes": changes,
        "dependencies": dependencies,
    }


def cmd_dataset(name_version: str):
    try:
        import datachain as dc  # noqa: F401
    except ImportError:
        print(
            json.dumps({"error": "datachain not installed"}),
            file=sys.stderr,
        )
        sys.exit(1)

    print(json.dumps(_fetch_version_data(name_version)))


def cmd_dataset_all(name: str):
    """Fetch data for all versions of a dataset in one call."""
    try:
        import datachain as dc
    except ImportError:
        print(json.dumps({"error": "datachain not installed"}), file=sys.stderr)
        sys.exit(1)

    # Detect source
    dot_parts = name.split(".", 2)
    is_studio = len(dot_parts) == 3
    source = "studio" if is_studio else "local"

    # Collect all versions for this dataset
    all_entries = _collect_datasets(dc, studio=is_studio)
    version_entries = [e for e in all_entries if e["name"] == name]

    if not version_entries:
        print(json.dumps({"error": f"Dataset '{name}' not found"}), file=sys.stderr)
        sys.exit(1)

    versions_sorted = sorted(
        [e["version"] for e in version_entries if e["version"]],
        key=_parse_semver,
    )

    # Build per-version data (oldest first so changes chain correctly)
    versions_out = []
    for version in versions_sorted:
        version_entry = next(
            (e for e in version_entries if e["version"] == version), None
        )
        data = _fetch_version_data(f"{name}@{version}")
        versions_out.append({
            "version": version,
            "num_objects": version_entry.get("num_objects") if version_entry else None,
            "updated_at": version_entry.get("updated_at") if version_entry else None,
            "schema": data.get("schema"),
            "preview": data.get("preview"),
            "query_script": data.get("query_script"),
            "changes": data.get("changes"),
            "dependencies": data.get("dependencies", []),
        })

    print(json.dumps({
        "name": name,
        "source": source,
        "versions": versions_out,
    }))


def main():
    parser = argparse.ArgumentParser(
        description="DataChain metadata extractor for the datachain-graph skill."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--plan",
        action="store_true",
        help="Output JSON plan: what datasets exist and what needs updating",
    )
    group.add_argument(
        "--dataset-all",
        metavar="NAME",
        help="Output JSON with data for all versions of a dataset",
    )
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
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Include Studio datasets (used with --plan)",
    )
    args = parser.parse_args()

    if args.plan:
        cmd_plan(studio=args.studio)
    elif args.dataset_all:
        cmd_dataset_all(args.dataset_all)
    elif args.db_mtime:
        cmd_db_mtime()
    elif args.list:
        cmd_list(studio=False)
    elif args.list_studio:
        cmd_list(studio=True)
    elif args.dataset:
        cmd_dataset(args.dataset)


if __name__ == "__main__":
    main()
