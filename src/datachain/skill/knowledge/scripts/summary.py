"""Compute per-field statistical summaries for a dataset.

Usage:
    python3 summary.py <name@version> [--output path.json]
"""

import argparse
import inspect
import json
import math
from collections import Counter
from datetime import datetime
from typing import get_origin

from schema import extract_schema
from utils import dc_import, human_size, write_json


def _hs(nbytes):
    """Human-readable size without space: 3.3KB, 1.1MB."""
    return human_size(nbytes).replace(" ", "")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_THRESHOLD = 50_000
SAMPLE_SIZE = 20_000

# Separator used in formatted lines
SEP = ", "


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------


def _is_file_subclass(typ):
    """Check if typ is a subclass of datachain File."""
    try:
        from datachain.lib.file import File

        return inspect.isclass(typ) and issubclass(typ, File)
    except ImportError:
        return False


def _classify_columns(schema_tree):
    """Classify leaf columns from get_flat_tree into categories.

    Returns dict of {dot_path: {"type": py_type, "category": str, "depth": int}}.
    """
    columns = {}
    file_parents = set()

    for path, typ, has_subtree, depth in schema_tree:
        dot_path = ".".join(path)
        if has_subtree:
            if _is_file_subclass(typ):
                file_parents.add(dot_path)
            continue

        parent = ".".join(path[:-1]) if len(path) > 1 else None
        field_name = path[-1]
        is_file_field = parent in file_parents if parent else False

        category = _classify_type(typ, field_name, is_file_field)
        columns[dot_path] = {
            "type": typ,
            "category": category,
            "depth": depth,
        }

    return columns


def _classify_type(typ, field_name, is_file_field):  # noqa: PLR0911
    """Return category string for a column type."""
    if field_name == "size" and is_file_field and typ in (int, float):
        return "file_size"
    if typ in (int, float):
        return "numeric"
    if typ is bool:
        return "bool"
    if typ is str:
        return "string"
    if typ is datetime:
        return "datetime"
    origin = get_origin(typ)
    if origin is list:
        return "list"
    return "unknown"


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


def _compute_aggregates(working, columns):
    """Single group_by for min/max/avg of numeric columns and sum of file sizes."""
    from datachain import C, func

    aggs = {"_count": func.count()}

    for col_path, info in columns.items():
        cat = info["category"]
        safe = col_path.replace(".", "__")
        if cat in ("numeric", "file_size"):
            aggs[f"{safe}__min"] = func.min(C(col_path))
            aggs[f"{safe}__max"] = func.max(C(col_path))
        if cat == "file_size":
            aggs[f"{safe}__sum"] = func.sum(C(col_path))
        if cat == "datetime":
            aggs[f"{safe}__min"] = func.min(C(col_path))
            aggs[f"{safe}__max"] = func.max(C(col_path))

    if len(aggs) <= 1:
        return {}

    try:
        rows = working.group_by(**aggs).to_list()
        if not rows:
            return {}
        row = rows[0]
        keys = list(aggs.keys())
        return {keys[i]: row[i] for i in range(len(keys))}
    except Exception:  # noqa: BLE001
        return {}


def _fetch_column_values(working, col_paths):
    """Fetch all values for given columns in a single to_list call.

    Returns dict of {col_path: list_of_values}.
    """
    if not col_paths:
        return {}
    try:
        rows = working.to_list(*col_paths)
    except Exception:  # noqa: BLE001
        return {}

    result = {col: [] for col in col_paths}
    for row in rows:
        for i, col in enumerate(col_paths):
            result[col].append(row[i])
    return result


def _percentiles(values, *quantiles):
    """Compute percentiles from a sorted list of non-None values."""
    n = len(values)
    if n == 0:
        return [None] * len(quantiles)
    result = []
    for q in quantiles:
        idx = min(int(n * q), n - 1)
        result.append(values[idx])
    return result


def _null_pct(values):
    """Compute null percentage from a list of values."""
    if not values:
        return 0.0
    n_null = sum(1 for v in values if v is None)
    return round(n_null / len(values) * 100, 1)


# ---------------------------------------------------------------------------
# Per-column summarizers
# ---------------------------------------------------------------------------


def _summarize_numeric(col_path, agg_data, values):
    """Summarize a numeric column: min-max, p50, p95."""
    safe = col_path.replace(".", "__")
    mn = agg_data.get(f"{safe}__min")
    mx = agg_data.get(f"{safe}__max")

    non_null = sorted(v for v in values if v is not None)
    null_rate = _null_pct(values)

    if not non_null:
        if mn is not None and mx is not None:
            line = f"{_sig(mn)} - {_sig(mx)}"
            return {
                "line": line,
                "stats": {"min": mn, "max": mx, "null_pct": null_rate},
            }
        return {"line": "", "stats": {"null_pct": null_rate}}

    p50, p95 = _percentiles(non_null, 0.5, 0.95)
    if mn is None:
        mn = non_null[0]
    if mx is None:
        mx = non_null[-1]

    parts = [
        f"{_sig(mn)} - {_sig(mx)}",
        f"p50={_sig(p50)}",
        f"p95={_sig(p95)}",
    ]
    if null_rate > 0:
        parts.append(f"{null_rate}% null")

    return {
        "line": SEP.join(parts),
        "stats": {
            "min": mn,
            "max": mx,
            "p50": p50,
            "p95": p95,
            "null_pct": null_rate,
        },
    }


def _summarize_file_size(col_path, agg_data, values):
    """Summarize a file size column: min-max, p50 in human units."""
    safe = col_path.replace(".", "__")
    mn = agg_data.get(f"{safe}__min")
    mx = agg_data.get(f"{safe}__max")
    total = agg_data.get(f"{safe}__sum")

    non_null = sorted(v for v in values if v is not None)
    null_rate = _null_pct(values)

    if not non_null and mn is None:
        return {"line": "", "stats": {"null_pct": null_rate}}

    if non_null:
        (p50,) = _percentiles(non_null, 0.5)
    else:
        p50 = None

    if mn is None and non_null:
        mn = non_null[0]
    if mx is None and non_null:
        mx = non_null[-1]

    parts = []
    if mn is not None and mx is not None:
        parts.append(f"{_hs(mn)} - {_hs(mx)}")
    if p50 is not None:
        parts.append(f"p50={_hs(p50)}")
    if null_rate > 0:
        parts.append(f"{null_rate}% null")

    stats = {"null_pct": null_rate}
    if mn is not None:
        stats["min"] = mn
    if mx is not None:
        stats["max"] = mx
    if p50 is not None:
        stats["p50"] = p50
    if total is not None:
        stats["sum"] = total

    return {"line": SEP.join(parts), "stats": stats}


def _summarize_string(col_path, values):
    """Summarize a string column: low-card top values or high-card uniqueness."""
    non_null = [v for v in values if v is not None]
    null_rate = _null_pct(values)
    total = len(non_null)

    if total == 0:
        return {
            "line": "100% null" if null_rate == 100 else "",
            "stats": {"null_pct": null_rate},
            "sub_category": "string_high_card",
        }

    unique_count = len(set(non_null))
    ratio = unique_count / total if total > 0 else 1.0

    if unique_count <= 20 or ratio < 0.05:
        return _summarize_string_low_card(
            non_null,
            total,
            unique_count,
            null_rate,
        )

    return _summarize_string_high_card(
        non_null,
        total,
        unique_count,
        ratio,
        null_rate,
    )


def _summarize_string_low_card(non_null, total, unique_count, null_rate):
    """Format low-cardinality string: top values with percentages."""
    counts = Counter(non_null).most_common(3)
    parts = []
    for val, cnt in counts:
        pct = round(cnt / total * 100, 0)
        display = str(val) if len(str(val)) <= 20 else str(val)[:17] + "..."
        parts.append(f"{display} {int(pct)}%")
    if null_rate > 0:
        parts.append(f"{null_rate}% null")
    return {
        "line": SEP.join(parts),
        "stats": {
            "top_values": [
                {"value": v, "pct": round(c / total * 100, 1)} for v, c in counts
            ],
            "unique_count": unique_count,
            "null_pct": null_rate,
        },
        "sub_category": "string_low_card",
    }


def _summarize_string_high_card(
    non_null,
    total,
    unique_count,
    ratio,
    null_rate,
):
    """Format high-cardinality string: uniqueness + avg length."""
    avg_len = sum(len(str(v)) for v in non_null) / total if total > 0 else 0
    uniq_pct = round(ratio * 100, 0)

    parts = []
    if uniq_pct >= 95:
        parts.append(f"{int(uniq_pct)}% unique")
    else:
        parts.append(f"{unique_count} unique")
    if avg_len > 0:
        parts.append(f"avg len {int(avg_len)}")
    if null_rate > 0:
        parts.append(f"{null_rate}% null")

    return {
        "line": SEP.join(parts),
        "stats": {
            "unique_count": unique_count,
            "unique_pct": round(ratio * 100, 1),
            "avg_len": round(avg_len, 1),
            "null_pct": null_rate,
        },
        "sub_category": "string_high_card",
    }


def _summarize_bool(values):
    """Summarize a bool column: true X%, false Y%."""
    non_null = [v for v in values if v is not None]
    null_rate = _null_pct(values)
    total = len(non_null)

    if total == 0:
        return {"line": "", "stats": {"null_pct": null_rate}}

    true_count = sum(1 for v in non_null if v)
    true_pct = round(true_count / total * 100, 0)
    false_pct = 100 - true_pct

    parts = [f"true {int(true_pct)}%", f"false {int(false_pct)}%"]
    if null_rate > 0:
        parts.append(f"{null_rate}% null")

    true_pct_exact = round(true_count / total * 100, 1)
    return {
        "line": SEP.join(parts),
        "stats": {"true_pct": true_pct_exact, "null_pct": null_rate},
    }


def _summarize_datetime(col_path, agg_data):
    """Summarize a datetime column: min to max."""
    safe = col_path.replace(".", "__")
    mn = agg_data.get(f"{safe}__min")
    mx = agg_data.get(f"{safe}__max")

    if mn is None and mx is None:
        return {"line": "", "stats": {}}

    def fmt(dt):
        if hasattr(dt, "strftime"):
            return dt.strftime("%Y-%m-%d")
        return str(dt)

    parts = []
    if mn is not None and mx is not None:
        parts.append(f"{fmt(mn)} - {fmt(mx)}")
    elif mn is not None:
        parts.append(f"from {fmt(mn)}")
    else:
        parts.append(f"until {fmt(mx)}")

    stats = {}
    if mn is not None:
        stats["min"] = str(mn)
    if mx is not None:
        stats["max"] = str(mx)

    return {"line": SEP.join(parts), "stats": stats}


def _summarize_list(values):
    """Summarize a list column: len min-max, p50."""
    non_null = [v for v in values if v is not None]
    null_rate = _null_pct(values)

    if not non_null:
        return {"line": "", "stats": {"null_pct": null_rate}}

    lengths = sorted(len(v) if isinstance(v, (list, tuple)) else 0 for v in non_null)
    mn = lengths[0]
    mx = lengths[-1]
    (p50,) = _percentiles(lengths, 0.5)

    parts = [f"len {mn} - {mx}", f"p50={p50}"]
    if null_rate > 0:
        parts.append(f"{null_rate}% null")

    return {
        "line": SEP.join(parts),
        "stats": {
            "len_min": mn,
            "len_max": mx,
            "len_p50": p50,
            "null_pct": null_rate,
        },
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _sig(val, n=3):  # noqa: PLR0911
    """Round to n significant digits for display."""
    if val is None:
        return "?"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == 0:
            return "0"
        magnitude = math.floor(math.log10(abs(val))) if val != 0 else 0
        if magnitude >= n:
            return str(round(val))
        decimals = max(0, n - 1 - magnitude)
        return f"{val:.{decimals}f}"
    return str(val)


def _format_count(count):
    """Format row count with K/M suffix."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


# ---------------------------------------------------------------------------
# Overview line
# ---------------------------------------------------------------------------


def _build_overview(total_rows, columns_info, col_results, schema):
    """Build the one-line dataset overview."""
    parts = [f"{_format_count(total_rows)} items", f"{len(schema)} cols"]

    total_size = 0
    for col_path, info in columns_info.items():
        if info["category"] == "file_size":
            s = col_results.get(col_path, {}).get("stats", {}).get("sum")
            if s:
                total_size += s
    if total_size > 0:
        parts.append(_hs(total_size))

    for col_path, res in col_results.items():
        field_name = col_path.split(".")[-1]
        is_format = field_name in ("format", "ext", "extension")
        is_low_card = res.get("category") == "string_low_card"
        if is_format and is_low_card:
            top = res.get("stats", {}).get("top_values", [])
            if top:
                parts.append(top[0]["value"])
            break

    for w_col, h_col in [
        ("width", "height"),
        ("info.width", "info.height"),
    ]:
        w_res = col_results.get(w_col)
        h_res = col_results.get(h_col)
        if w_res and h_res:
            w_p50 = w_res.get("stats", {}).get("p50")
            h_p50 = h_res.get("stats", {}).get("p50")
            if w_p50 is not None and h_p50 is not None:
                parts.append(f"~{int(w_p50)}x{int(h_p50)}")
            break

    return SEP.join(parts)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def dataset_summary_from_chain(chain) -> dict:
    """Compute per-field stats and overview from an open DataChain.

    Calls chain.persist() internally for efficient multi-query execution.
    """
    warnings_list: list[str] = []

    # Phase 0 -- Schema and classification
    schema = extract_schema(chain)
    flat_tree = list(
        chain.signals_schema.get_flat_tree(
            include_hidden=False,
            include_sys=False,
        )
    )
    columns_info = _classify_columns(flat_tree)

    try:
        total_rows = chain.count()
    except Exception as e:  # noqa: BLE001
        warnings_list.append(f"count: {e}")
        total_rows = 0

    if total_rows == 0:
        return {
            "overview": f"0 items {SEP} {len(schema)} cols",
            "total_rows": 0,
            "sampled": False,
            "columns": {},
            "warnings": warnings_list,
        }

    # Phase 0.5 -- Sample + Persist
    sampled = False
    sample_size = total_rows
    try:
        if total_rows > SAMPLE_THRESHOLD:
            working = chain.sample(SAMPLE_SIZE).persist()
            sampled = True
            sample_size = SAMPLE_SIZE
        else:
            working = chain.persist()
    except Exception as e:  # noqa: BLE001
        warnings_list.append(f"persist: {e}")
        overview = f"{_format_count(total_rows)} items"
        overview += f"{SEP}{len(schema)} cols"
        return {
            "overview": overview,
            "total_rows": total_rows,
            "sampled": False,
            "columns": {},
            "warnings": warnings_list,
        }

    # Phase 1 -- SQL aggregates (single group_by)
    agg_data = _compute_aggregates(working, columns_info)

    # Phase 2 -- Fetch all column values in one to_list call
    numeric_cols = [
        c for c, i in columns_info.items() if i["category"] in ("numeric", "file_size")
    ]
    string_cols = [c for c, i in columns_info.items() if i["category"] == "string"]
    bool_cols = [c for c, i in columns_info.items() if i["category"] == "bool"]
    list_cols = [c for c, i in columns_info.items() if i["category"] == "list"]

    all_fetch_cols = numeric_cols + string_cols + bool_cols + list_cols
    all_values = _fetch_column_values(working, all_fetch_cols) if all_fetch_cols else {}

    # Phase 3 -- Per-column summaries
    col_results = _build_column_results(
        columns_info,
        agg_data,
        all_values,
        warnings_list,
    )

    # Phase 4 -- Overview
    overview = _build_overview(
        total_rows,
        columns_info,
        col_results,
        schema,
    )

    result = {
        "overview": overview,
        "total_rows": total_rows,
        "sampled": sampled,
        "columns": col_results,
        "warnings": warnings_list,
    }
    if sampled:
        result["sample_size"] = sample_size

    return result


def _build_column_results(columns_info, agg_data, all_values, warnings_list):
    """Compute per-column summary dicts."""
    col_results = {}

    for col_path, info in columns_info.items():
        cat = info["category"]
        values = all_values.get(col_path, [])

        try:
            res = _summarize_column(cat, col_path, agg_data, values)
        except Exception as e:  # noqa: BLE001
            warnings_list.append(f"stats for {col_path}: {e}")
            res = {"line": "", "stats": {}}

        sub_cat = res.pop("sub_category", None)
        col_results[col_path] = {
            "category": sub_cat or cat,
            "line": res["line"],
            "stats": res.get("stats", {}),
        }

    return col_results


_SUMMARIZERS = {
    "numeric": _summarize_numeric,
    "file_size": _summarize_file_size,
    "string": lambda cp, _agg, vals: _summarize_string(cp, vals),
    "bool": lambda _cp, _agg, vals: _summarize_bool(vals),
    "datetime": lambda cp, agg, _vals: _summarize_datetime(cp, agg),
    "list": lambda _cp, _agg, vals: _summarize_list(vals),
}


def _summarize_column(cat, col_path, agg_data, values):
    """Dispatch to the right summarizer by category."""
    fn = _SUMMARIZERS.get(cat)
    if fn:
        return fn(col_path, agg_data, values)
    return {"line": "", "stats": {}}


def dataset_summary(name_version: str) -> dict:
    """Open dataset by name@version, compute summary.

    Args:
        name_version: Dataset name with optional @version,
            e.g. 'my_ds@1.0.3'.
    """
    dc = dc_import()
    chain = dc.read_dataset(name_version)
    return dataset_summary_from_chain(chain)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistical summary for a dataset."
    )
    parser.add_argument(
        "name",
        help="Dataset name with optional @version (e.g. my_dataset@1.0.3)",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path (default: stdout)",
    )
    args = parser.parse_args()

    result = dataset_summary(args.name)

    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
