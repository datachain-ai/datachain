import inspect
import json
import types
import typing
from typing import TYPE_CHECKING, Any

from datachain.query.schema import DEFAULT_DELIMITER
from datachain.skill.knowledge.scripts.changes import build_changes
from datachain.skill.knowledge.scripts.utils import (
    dedupe_previous_scripts,
    dep_entry,  # noqa: F401
    drop_unchanged_scripts,
    is_sys_column,
    parse_semver,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from datachain.lib.data_model import DataType
    from datachain.skill.knowledge.types import (
        DatasetSnapshot,
        DatasetVersionEntry,
        DependencyEntry,
        PreviewData,
        SchemaEntry,
    )

    DepsProvider = Callable[[Any], "Sequence[DependencyEntry]"]


# Cap version history so the snapshot stays within the prompt budget.
MAX_VERSION_ENTRIES = 20

# Max chars per preview cell — stops one embedding/text value bloating the prompt.
MAX_PREVIEW_CELL_CHARS = 200


def build_dataset_snapshot(
    *,
    name: str,
    source: str,
    attrs: list[str],
    description: str | None,
    versions: "Sequence[Any]",
    deps_provider: "DepsProvider",
    max_version_entries: int = MAX_VERSION_ENTRIES,
) -> "DatasetSnapshot":
    """Assemble the ``DatasetSnapshot`` for one dataset."""
    versions_sorted = sorted(versions, key=lambda v: parse_semver(v.version))

    warnings: list[str] = []
    entries: list[DatasetVersionEntry] = []
    if not versions_sorted:
        warnings.append("dataset has no versions")
    else:
        if len(versions_sorted) > max_version_entries:
            warnings.append(
                f"version history truncated to {max_version_entries} most recent "
                f"of {len(versions_sorted)} versions"
            )
        latest = versions_sorted[-1]
        recent = versions_sorted[-max_version_entries:]
        # The version before the window anchors the oldest kept version's diff.
        window = versions_sorted[-max_version_entries - 1 :]
        deps = {v.version: list(deps_provider(v)) for v in window}

        truncated = len(versions_sorted) > max_version_entries
        prev = window[0] if truncated else None
        for version in recent:
            is_latest = version.version == latest.version
            if any(d["name"] is None for d in deps.get(version.version, [])):
                warnings.append(
                    f"version {version.version} depends on a deleted dataset "
                    "(lineage incomplete)"
                )
            entries.append(_version_entry(version, prev, deps, is_latest=is_latest))
            prev = version
        drop_unchanged_scripts(entries)
        dedupe_previous_scripts(entries)

    return {
        "name": name,
        "source": source,
        "attrs": list(attrs or []),
        "description": description or None,
        "versions": entries,
        "warnings": warnings,
    }


def version_schema(version: Any) -> "dict[str, SchemaEntry]":
    """Signal-level schema from the version's stored `feature_schema`."""
    feature_schema = getattr(version, "feature_schema", None)
    if feature_schema:
        from datachain.lib.signal_schema import SignalSchema

        try:
            signals = SignalSchema.deserialize(feature_schema)
        except Exception:  # noqa: BLE001, S110
            pass
        else:
            return {
                col: expand_signal(typ)
                for col, typ in signals.values.items()
                if not is_sys_column(col)
            }
    return {
        col: {"type": _flat_type(val), "fields": None}
        for col, val in (getattr(version, "schema", None) or {}).items()
        if not is_sys_column(col)
    }


def version_preview(version: Any) -> "PreviewData | None":
    """Coerce a version's stored preview into the {columns, rows} prompt shape."""
    try:
        raw = version.preview
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    if isinstance(raw, dict):
        rows = raw.get("rows")
        if not isinstance(rows, list):
            return raw  # type: ignore[return-value]
        capped = [[_cap_cell(c) for c in r] if isinstance(r, list) else r for r in rows]
        return {**raw, "rows": capped}  # type: ignore[return-value]
    columns: list[str] = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, dict):
            continue
        for col in row:
            if col in seen or is_sys_column(col):
                continue
            seen.add(col)
            columns.append(col)
    rows = [
        [_cap_cell(row.get(c)) for c in columns] for row in raw if isinstance(row, dict)
    ]
    if not rows:
        return None
    columns = [c.replace(DEFAULT_DELIMITER, ".") for c in columns]
    return {"columns": columns, "rows": rows}


def type_name(tp: Any) -> str:
    """Convert a Python type to a human-readable string."""
    if tp is type(None):
        return "None"
    if isinstance(tp, types.UnionType):  # Python 3.10+ X | Y
        return " | ".join(type_name(a) for a in tp.__args__)
    origin = getattr(tp, "__origin__", None)
    if origin is typing.Union:
        return " | ".join(type_name(a) for a in tp.__args__)
    if origin is list:
        args = getattr(tp, "__args__", ())
        return f"list[{type_name(args[0])}]" if args else "list"
    if origin is dict:
        return "dict"
    return getattr(tp, "__name__", str(tp))


def expand_signal(typ: "DataType") -> "SchemaEntry":
    """Expand a signal type into a schema entry with type and fields."""
    from datachain.lib.data_model import DataModel
    from datachain.lib.file import File

    tn = type_name(typ)
    if not (inspect.isclass(typ) and issubclass(typ, DataModel)):
        return {"type": tn, "fields": None}
    if issubclass(typ, File):
        return {"type": tn, "fields": None}  # skip — covered by dc-core
    fields = {
        fname: type_name(finfo.annotation) for fname, finfo in typ.model_fields.items()
    }
    return {"type": tn, "fields": fields}


def _version_entry(
    version: Any,
    prev: Any | None,
    deps_by_version: "dict[str, list[DependencyEntry]]",
    *,
    is_latest: bool,
) -> "DatasetVersionEntry":
    deps = deps_by_version.get(version.version, [])
    changes = None
    if prev is not None:
        prev_deps = deps_by_version.get(prev.version, [])
        changes = build_changes(
            version.query_script or None,
            prev.version,
            prev.query_script or None,
            deps,
            prev_deps,
        )
    return {
        "version": version.version,
        "uuid": str(version.uuid) if version.uuid else None,
        "records": version.num_objects,
        "updated": _updated(version),
        "schema": version_schema(version) if is_latest else {},
        "preview": version_preview(version) if is_latest else None,
        "summary": None,
        "query_script": version.query_script or None,
        "changes": changes,
        # name=None edges (deleted targets) drive the warning but aren't rendered.
        "dependencies": [d for d in deps if d["name"] is not None],
    }


def _updated(version: Any) -> str | None:
    ts = version.finished_at or version.created_at
    return ts.isoformat() if ts else None


def _flat_type(val: Any) -> str:
    return val if isinstance(val, str) else type_name(val)


def _cap_cell(value: Any) -> Any:
    if value is None or isinstance(value, int | float):
        return value
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= MAX_PREVIEW_CELL_CHARS:
        return value
    return text[:MAX_PREVIEW_CELL_CHARS] + "…"
