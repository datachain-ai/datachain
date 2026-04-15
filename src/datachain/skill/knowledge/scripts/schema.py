"""Type introspection, schema extraction, and preview generation."""

import inspect
import types
import typing

from utils import serialize, source_to_https


def get_catalog():
    """Get a DataChain catalog from the current session, independent of read_dataset."""
    from datachain import Session

    return Session.get().catalog


def type_name(tp):
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


def expand_signal(typ):
    """Return {"type": name, "fields": {name: type_str} | None}.
    Fields is None for File subclasses (well-known) and primitives."""
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


def parse_dataset_name(name: str) -> tuple:
    """Parse a dataset name into (namespace, project, bare_name).

    Returns (None, None, name) for local datasets,
    (namespace, project, bare_name) for Studio datasets
    (supports both slash and dot separators).
    """
    slash_parts = name.split("/", 2)
    dot_parts = name.split(".", 2)
    if len(slash_parts) == 3:
        return slash_parts[0], slash_parts[1], slash_parts[2]
    if len(dot_parts) == 3:
        return dot_parts[0], dot_parts[1], dot_parts[2]
    return None, None, name


def extract_schema(chain) -> dict:
    """Extract schema dict from a DataChain, filtering sys columns."""
    return {
        col: expand_signal(typ)
        for col, typ in chain.schema.items()
        if col != "sys" and not col.startswith("sys.")
    }


def _file_url_prefix(chain) -> str | None:
    """Return an HTTPS URL prefix for file paths, if the chain has a File column."""
    from datachain.lib.file import File

    for typ in chain.schema.values():
        if inspect.isclass(typ) and issubclass(typ, File):
            break
    else:
        return None
    try:
        df = chain.limit(1).to_pandas(flatten=True, include_hidden=True)
        for col in df.columns:
            if col.endswith(".source") or col == "source":
                val = df[col].iloc[0]
                if val:
                    return source_to_https(str(val))
    except Exception:  # noqa: BLE001, S110
        pass
    return None


def extract_preview(chain) -> dict | None:
    """Extract preview (columns + rows) from a DataChain. Returns None on failure."""
    try:
        df = chain.limit(10).to_pandas(flatten=True, include_hidden=False)
        result = {
            "columns": list(df.columns),
            "rows": [[serialize(v) for v in row] for row in df.itertuples(index=False)],
        }
        url_prefix = _file_url_prefix(chain)
        if url_prefix:
            result["file_url_prefix"] = url_prefix
        return result
    except Exception:  # noqa: BLE001
        return None
