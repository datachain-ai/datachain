"""Dependency serialization and version-diff logic."""


def dep_to_dict(dep) -> dict:
    """Convert a dependency object to a JSON-serializable dict."""
    return {
        "name": dep.name,
        "version": str(dep.version) if dep.version is not None else None,
        "type": str(dep.type) if dep.type is not None else None,
        "dependencies": [
            {
                "name": child.name,
                "version": str(child.version) if child.version is not None else None,
                "type": str(child.type) if child.type is not None else None,
            }
            for child in (dep.dependencies or [])
            if child
        ],
    }


def compute_dep_changes(
    curr_deps: list[dict], prev_deps: list[dict], catalog=None
) -> dict:
    """Compute added/removed/updated dependencies between two versions.

    curr_deps and prev_deps are lists of dicts with 'name' and 'version' keys
    (as returned by dep_to_dict). catalog is optional — used for script comparison.
    """
    curr_dep_map = {d["name"]: d["version"] for d in curr_deps}
    prev_dep_map = {d["name"]: d["version"] for d in prev_deps}
    curr_names = set(curr_dep_map)
    prev_names = set(prev_dep_map)

    deps_added = [
        {"name": n, "version": curr_dep_map[n]} for n in sorted(curr_names - prev_names)
    ]
    deps_removed = [
        {"name": n, "version": prev_dep_map[n]} for n in sorted(prev_names - curr_names)
    ]
    deps_updated = []
    for n in sorted(curr_names & prev_names):
        cv, pv = curr_dep_map[n], prev_dep_map[n]
        if cv != pv:
            entry: dict = {
                "name": n,
                "version_from": pv,
                "version_to": cv,
                "script_changed": False,
            }
            if catalog is not None:
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
            deps_updated.append(entry)

    return {
        "deps_added": deps_added,
        "deps_removed": deps_removed,
        "deps_updated": deps_updated,
    }


def build_changes(
    query_script,
    prev_version_str: str,
    prev_script,
    curr_deps: list[dict],
    prev_deps: list[dict],
    catalog=None,
) -> dict:
    """Build the full changes dict for a version, including script and dependency diffs."""
    script_changed = query_script != prev_script
    dep_changes = compute_dep_changes(curr_deps, prev_deps, catalog=catalog)
    return {
        "previous_version": prev_version_str,
        "script_changed": script_changed,
        "previous_script": prev_script if script_changed else None,
        **dep_changes,
    }
