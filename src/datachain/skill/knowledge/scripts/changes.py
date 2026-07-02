"""Dependency serialization and version-diff logic."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datachain.skill.knowledge.types import (
        ChangesEntry,
        DepChanges,
        DependencyEntry,
        DepRef,
        DepUpdate,
    )


def compute_dep_changes(
    curr_deps: "Sequence[DependencyEntry]",
    prev_deps: "Sequence[DependencyEntry]",
) -> "DepChanges":
    """Compute added/removed/updated dependencies between two versions.

    curr_deps and prev_deps are lists of dicts with 'name' and 'version' keys
    (as returned by dep_entry).
    """
    curr_dep_map = {name: d["version"] for d in curr_deps if (name := d["name"])}
    prev_dep_map = {name: d["version"] for d in prev_deps if (name := d["name"])}
    curr_names = set(curr_dep_map)
    prev_names = set(prev_dep_map)

    deps_added: list[DepRef] = [
        {"name": n, "version": curr_dep_map[n]} for n in sorted(curr_names - prev_names)
    ]
    deps_removed: list[DepRef] = [
        {"name": n, "version": prev_dep_map[n]} for n in sorted(prev_names - curr_names)
    ]
    deps_updated: list[DepUpdate] = [
        {"name": n, "version_from": prev_dep_map[n], "version_to": curr_dep_map[n]}
        for n in sorted(curr_names & prev_names)
        if curr_dep_map[n] != prev_dep_map[n]
    ]

    return {
        "deps_added": deps_added,
        "deps_removed": deps_removed,
        "deps_updated": deps_updated,
    }


def build_changes(
    query_script,
    prev_version_str: str,
    prev_script,
    curr_deps: "Sequence[DependencyEntry]",
    prev_deps: "Sequence[DependencyEntry]",
) -> "ChangesEntry":
    """Build the full changes dict for a version.

    Includes script and dependency diffs.
    """
    script_changed = query_script != prev_script
    dep_changes = compute_dep_changes(curr_deps, prev_deps)
    return {
        "previous_version": prev_version_str,
        "script_changed": script_changed,
        "previous_script": prev_script if script_changed else None,
        **dep_changes,
    }
