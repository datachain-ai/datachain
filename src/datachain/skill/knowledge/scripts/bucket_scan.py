"""Scan a bucket URI: compute metadata aggregations and sample files.

Outputs a JSON data file with extensions, directories, size distribution,
time range, listing timestamps, and content samples.

Usage:
    python3 bucket_scan.py <uri> --output <path.json>
"""

import argparse
import json
import signal
import sys
from datetime import datetime, timezone
from typing import Any

from utils import dc_import, parse_uri, source_to_https, write_json

from datachain import bucket_status


class ScanTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise ScanTimeoutError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EXTENSIONS = 20
MAX_DIRECTORIES = 100
SAMPLES_PER_EXT = 5

# Extensions grouped by modality for sampling (without dot, matching file_ext output)
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "tiff", "tif", "bmp", "gif"}
AUDIO_EXTS = {"mp3", "wav", "flac", "ogg", "aac", "wma", "m4a"}
VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"}
STRUCTURED_EXTS = {"csv", "json", "jsonl", "parquet", "tsv", "ndjson"}
TEXT_EXTS = {"txt", "md", "log", "xml", "html", "yaml", "yml", "cfg", "ini"}


def get_listing_info(uri: str) -> dict:
    """Get listing metadata (timestamps, expiration) from DataChain catalog."""
    try:
        from datachain.query import Session

        session = Session.get()
        catalog = session.catalog

        for listing in catalog.listings():
            if listing.uri.rstrip("/") == uri.rstrip("/") or uri.rstrip("/").startswith(
                listing.uri.rstrip("/")
            ):
                return {
                    "listing_uuid": getattr(listing, "uuid", None),
                    "listing_created": (
                        listing.created_at.isoformat() if listing.created_at else None
                    ),
                    "listing_finished": (
                        listing.finished_at.isoformat() if listing.finished_at else None
                    ),
                    "listing_expires": (
                        listing.expires.isoformat() if listing.expires else None
                    ),
                    "listing_expired": listing.is_expired,
                }
    except Exception as e:  # noqa: BLE001
        print(f"[dc-knowledge warning] listing info: {e}", file=sys.stderr)

    return {
        "listing_uuid": None,
        "listing_created": None,
        "listing_finished": None,
        "listing_expires": None,
        "listing_expired": None,
    }


# ---------------------------------------------------------------------------
# Metadata aggregation (all use DataChain operations)
# ---------------------------------------------------------------------------


def compute_extensions(chain) -> list[dict[str, Any]]:
    """File extension breakdown via DataChain group_by."""
    from datachain import C, func

    ext_chain = chain.mutate(ext=func.path.file_ext(C("file.path"))).group_by(
        cnt=func.count(),
        total_bytes=func.sum(C("file.size")),
        partition_by="ext",
    )

    try:
        df = ext_chain.to_pandas()
    except Exception:  # noqa: BLE001
        return []

    total_files = int(df["cnt"].sum()) if len(df) > 0 else 0
    total_bytes_all = int(df["total_bytes"].sum()) if len(df) > 0 else 0

    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_ext = str(row.get("ext", "") or "")
        ext_display = f".{raw_ext}" if raw_ext else ""
        cnt = int(row["cnt"])
        tbytes = int(row["total_bytes"])
        results.append(
            {
                "ext": ext_display,
                "count": cnt,
                "total_bytes": tbytes,
                "pct_count": round(cnt / total_files * 100, 1) if total_files else 0,
                "pct_bytes": (
                    round(tbytes / total_bytes_all * 100, 1) if total_bytes_all else 0
                ),
            }
        )

    results.sort(key=lambda x: x["count"], reverse=True)

    if len(results) > MAX_EXTENSIONS:
        kept = results[:MAX_EXTENSIONS]
        rest = results[MAX_EXTENSIONS:]
        other_count = sum(r["count"] for r in rest)
        other_bytes = sum(r["total_bytes"] for r in rest)
        kept.append(
            {
                "ext": "(other)",
                "count": other_count,
                "total_bytes": other_bytes,
                "pct_count": (
                    round(other_count / total_files * 100, 1) if total_files else 0
                ),
                "pct_bytes": (
                    round(other_bytes / total_bytes_all * 100, 1)
                    if total_bytes_all
                    else 0
                ),
            }
        )
        results = kept

    return results


def compute_directories(chain) -> list[dict[str, Any]]:
    """Directory breakdown via DataChain group_by, capped at MAX_DIRECTORIES."""
    from datachain import C, func

    dir_chain = chain.mutate(dir=func.path.parent(C("file.path"))).group_by(
        cnt=func.count(),
        total_bytes=func.sum(C("file.size")),
        partition_by="dir",
    )

    try:
        df = dir_chain.to_pandas()
    except Exception:  # noqa: BLE001
        return []

    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        dir_path = str(row.get("dir", "") or "")
        if dir_path and not dir_path.endswith("/"):
            dir_path += "/"
        depth = dir_path.count("/") if dir_path else 0
        results.append(
            {
                "path": dir_path,
                "files": int(row["cnt"]),
                "bytes": int(row["total_bytes"]),
                "depth": depth,
            }
        )

    results.sort(key=lambda x: (x["depth"], -x["files"]))

    if len(results) <= MAX_DIRECTORIES:
        return results

    # Relevance-based capping
    kept = [r for r in results if r["depth"] <= 2]
    remaining = [r for r in results if r["depth"] > 2]

    deepest = sorted(remaining, key=lambda x: -x["depth"])[:5]
    deepest_paths = {r["path"] for r in deepest}
    kept.extend(deepest)

    slots_left = MAX_DIRECTORIES - len(kept)
    if slots_left > 0:
        rest = [r for r in remaining if r["path"] not in deepest_paths]
        rest.sort(key=lambda x: -x["files"])
        kept.extend(rest[:slots_left])

    kept.sort(key=lambda x: (x["depth"], -x["files"]))
    return kept


def compute_size_distribution(chain) -> dict:
    """Size distribution: min, max, percentiles, empty count."""
    from datachain import C, func

    try:
        stats_df = chain.group_by(
            min_size=func.min(C("file.size")),
            max_size=func.max(C("file.size")),
            total=func.count(),
        ).to_pandas()
        if len(stats_df) == 0:
            return {}
        min_bytes = int(stats_df["min_size"].iloc[0])
        max_bytes = int(stats_df["max_size"].iloc[0])
        total = int(stats_df["total"].iloc[0])
    except Exception:  # noqa: BLE001
        return {}

    # Percentiles via Python (DataChain lacks percentile funcs)
    try:
        if total <= 10000:
            sizes = sorted(v for (v,) in chain.to_iter("file.size"))
        else:
            sizes = sorted(
                v
                for (v,) in chain.mutate(rnd=func.rand())
                .order_by("rnd")
                .limit(10000)
                .to_iter("file.size")
            )
    except Exception:  # noqa: BLE001
        sizes = []

    result = {"min_bytes": min_bytes, "max_bytes": max_bytes}

    if sizes:
        n = len(sizes)
        result["median_bytes"] = int(sizes[n // 2])
        result["p10_bytes"] = int(sizes[n // 10]) if n >= 10 else int(sizes[0])
        result["p90_bytes"] = int(sizes[int(n * 0.9)]) if n >= 10 else int(sizes[-1])

    try:
        empty_df = (
            chain.filter(C("file.size") == 0).group_by(cnt=func.count()).to_pandas()
        )
        result["empty_count"] = int(empty_df["cnt"].iloc[0]) if len(empty_df) > 0 else 0
    except Exception:  # noqa: BLE001
        result["empty_count"] = 0

    return result


def compute_time_range(chain) -> dict:
    """Oldest and newest file timestamps."""
    from datachain import C, func

    try:
        df = chain.group_by(
            oldest=func.min(C("file.last_modified")),
            newest=func.max(C("file.last_modified")),
        ).to_pandas()
        if len(df) == 0:
            return {}
        oldest = df["oldest"].iloc[0]
        newest = df["newest"].iloc[0]
        return {
            "oldest": (
                oldest.isoformat()
                if hasattr(oldest, "isoformat")
                else str(oldest)
                if oldest
                else None
            ),
            "newest": (
                newest.isoformat()
                if hasattr(newest, "isoformat")
                else str(newest)
                if newest
                else None
            ),
        }
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_files(chain, extensions: list[dict]) -> dict:
    """Sample files per extension for content type detection."""
    from datachain import C, func

    samples = {}
    ext_list = [e["ext"] for e in extensions if e["ext"] and e["ext"] != "(other)"]

    for ext in ext_list:
        try:
            ext_bare = ext.lstrip(".")
            ext_chain = chain.filter(C("file.path").glob(f"*.{ext_bare}"))

            sample_rows = []
            for row in (
                ext_chain.mutate(rnd=func.rand())
                .order_by("rnd")
                .limit(SAMPLES_PER_EXT)
                .to_iter()
            ):
                sample_rows.append(row[0])

            if not sample_rows:
                continue

            type_detected = _detect_type(ext_bare)
            files_info = []
            for file_obj in sample_rows:
                info = {"path": file_obj.path, "size": file_obj.size}
                _enrich_sample(file_obj, ext_bare, type_detected, info)
                files_info.append(info)

            if files_info:
                samples[ext] = {"type_detected": type_detected, "files": files_info}
        except Exception as e:  # noqa: BLE001
            print(f"[dc-knowledge warning] sampling {ext}: {e}", file=sys.stderr)

    return samples


def _detect_type(ext: str) -> str:
    ext = ext.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in STRUCTURED_EXTS:
        return f"structured/{ext}"
    if ext in TEXT_EXTS:
        return "text"
    return "binary"


def _enrich_sample(file_obj, ext: str, type_detected: str, info: dict):
    try:
        if type_detected == "image":
            _enrich_image(file_obj, info)
        elif type_detected == "audio":
            _enrich_audio(file_obj, info)
        elif type_detected == "video":
            _enrich_video(file_obj, info)
        elif type_detected.startswith("structured/"):
            _enrich_structured(file_obj, ext, info)
        elif type_detected == "text":
            _enrich_text(file_obj, info)
    except Exception as e:  # noqa: BLE001
        info["sample_error"] = str(e)


def _enrich_image(file_obj, info):
    try:
        import io

        from PIL import Image

        data = file_obj.read()
        img = Image.open(io.BytesIO(data))
        info["width"] = img.width
        info["height"] = img.height
        info["format"] = img.format
    except Exception:  # noqa: BLE001, S110
        pass


def _enrich_audio(file_obj, info):
    try:
        from datachain import AudioFile

        if isinstance(file_obj, AudioFile):
            ai = file_obj.get_info()
            info["duration"] = ai.duration
            info["sample_rate"] = ai.sample_rate
            info["channels"] = ai.channels
    except Exception:  # noqa: BLE001, S110
        pass


def _enrich_video(file_obj, info):
    try:
        from datachain import VideoFile

        if isinstance(file_obj, VideoFile):
            vi = file_obj.get_info()
            info["duration"] = vi.duration
            info["fps"] = vi.fps
            info["codec"] = vi.codec
    except Exception:  # noqa: BLE001, S110
        pass


def _enrich_structured(file_obj, ext, info):
    try:
        if ext == "parquet":
            import io

            import pyarrow.parquet as pq

            pf = pq.ParquetFile(io.BytesIO(file_obj.read()))
            info["columns"] = [f.name for f in pf.schema_arrow]
            info["row_count"] = pf.metadata.num_rows
        elif ext in ("csv", "tsv"):
            text = file_obj.read_text()
            lines = text.split("\n")
            if lines:
                sep = "\t" if ext == "tsv" else ","
                info["columns"] = [c.strip().strip('"') for c in lines[0].split(sep)]
                info["row_count_approx"] = len([ln for ln in lines if ln.strip()]) - 1
        elif ext in ("json", "jsonl", "ndjson"):
            info["snippet"] = file_obj.read_text()[:500]
    except Exception:  # noqa: BLE001, S110
        pass


def _enrich_text(file_obj, info):
    try:
        info["snippet"] = file_obj.read_text()[:500]
    except Exception:  # noqa: BLE001, S110
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def compute_bucket_metadata(
    chain,
    uri: str,
    is_anon: bool,
    sampled: bool = False,
    dataset_name: str | None = None,
) -> dict:
    """Aggregate metadata + samples from a File-shape DataChain into a bucket JSON dict.

    Reusable across full bucket scans and sampled overviews. `sampled=True`
    means the chain represents a subset, not a full enumeration; downstream
    enrichment surfaces this prominently in the bucket markdown. When
    `dataset_name` is provided, it is recorded so readers can recover the
    actual File rows via `dc.read_dataset(dataset_name)`.
    """
    from datachain import C, func

    parts = parse_uri(uri)
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    listing_info = get_listing_info(uri) if not sampled else {}

    try:
        totals = chain.group_by(
            total_files=func.count(),
            total_bytes=func.sum(C("file.size")),
        ).to_pandas()
        total_files = int(totals["total_files"].iloc[0]) if len(totals) > 0 else 0
        total_size_bytes = int(totals["total_bytes"].iloc[0]) if len(totals) > 0 else 0
    except Exception as e:  # noqa: BLE001
        print(f"[dc-knowledge error] totals: {e}", file=sys.stderr)
        total_files = 0
        total_size_bytes = 0

    extensions = compute_extensions(chain)
    directories = compute_directories(chain)
    size_distribution = compute_size_distribution(chain)
    time_range = compute_time_range(chain)
    samples = sample_files(chain, extensions)
    url_prefix = source_to_https(uri)

    result = {
        "uri": uri,
        "scheme": parts["scheme"],
        "bucket": parts["bucket"],
        "prefix": parts["prefix"],
        "anon": is_anon,
        "sampled": sampled,
        "scanned": now,
        **listing_info,
        "total_files": total_files,
        "total_size_bytes": total_size_bytes,
        "max_depth": max((d["depth"] for d in directories), default=0),
        "extensions": extensions,
        "directories": directories,
        "size_distribution": size_distribution,
        "time_range": time_range,
        "samples": samples,
    }
    if url_prefix and is_anon:
        result["file_url_prefix"] = url_prefix
    if dataset_name:
        result["dataset_name"] = dataset_name
    return result


def scan_bucket(uri: str, output: str | None = None, timeout: int = 0):
    """Full bucket scan: list via read_storage and emit a bucket JSON."""

    if timeout > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)

    parts = parse_uri(uri)
    status = bucket_status(f"{parts['scheme']}://{parts['bucket']}/")
    if not status.exists or status.access == "denied":
        print(json.dumps({"error": status.error, "uri": uri}), file=sys.stderr)
        sys.exit(1)
    is_anon = status.access == "anonymous"

    try:
        dc = dc_import()
        # Never pass update=True. Listing is already cached from a prior
        # read_storage() call. Pass anon=True for public buckets so listing
        # doesn't hang on credential lookup.
        chain = dc.read_storage(uri, anon=True) if is_anon else dc.read_storage(uri)
        result = compute_bucket_metadata(chain, uri, is_anon)

        if timeout > 0:
            signal.alarm(0)

        if output:
            write_json(output, result)
        else:
            print(json.dumps(result, indent=2, default=str))
    except ScanTimeoutError:
        print(
            json.dumps({"error": "timeout", "uri": uri, "timeout": timeout}),
            file=sys.stderr,
        )
        sys.exit(124)


def main():
    parser = argparse.ArgumentParser(
        description="Scan a bucket URI and output metadata + samples as JSON."
    )
    parser.add_argument("uri", help="Storage URI (e.g., s3://bucket/prefix/)")
    parser.add_argument("--output", help="Output JSON file path (default: stdout)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout in seconds (0 = no timeout). Exit code 124 on timeout.",
    )
    args = parser.parse_args()
    scan_bucket(args.uri, args.output, args.timeout)


if __name__ == "__main__":
    main()
