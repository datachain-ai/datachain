"""Schema extraction and preview generation for a live DataChain."""

import inspect
from typing import TYPE_CHECKING

from utils import is_sys_column, serialize, source_to_https

from datachain.skill.knowledge.snapshot import expand_signal, type_name  # noqa: F401

if TYPE_CHECKING:
    from datachain import DataChain
    from datachain.skill.knowledge.types import PreviewData, SchemaEntry


def get_catalog():
    """Get a DataChain catalog from the current session, independent of read_dataset."""
    from datachain import Session

    return Session.get().catalog


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


def extract_schema(chain: "DataChain") -> "dict[str, SchemaEntry]":
    """Extract schema dict from a DataChain, filtering sys columns."""
    return {
        col: expand_signal(typ)
        for col, typ in chain.schema.items()
        if not is_sys_column(col)
    }


def _file_url_prefix(chain: "DataChain") -> str | None:
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


def extract_preview(chain: "DataChain") -> "PreviewData | None":
    """Extract preview (columns + rows) from a DataChain. Returns None on failure."""
    try:
        df = chain.limit(10).to_pandas(flatten=True, include_hidden=False)
        result: PreviewData = {
            "columns": list(df.columns),
            "rows": [[serialize(v) for v in row] for row in df.itertuples(index=False)],
        }
        url_prefix = _file_url_prefix(chain)
        if url_prefix:
            result["file_url_prefix"] = url_prefix
        return result
    except Exception:  # noqa: BLE001
        return None
