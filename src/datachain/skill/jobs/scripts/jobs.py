"""Standalone DataChain Studio job fetcher for the datachain-jobs skill.

Usage:
    python3 jobs.py --plan                          # JSON: staleness check for index.md
    python3 jobs.py --fetch [--days N] [--limit N]  # JSON: fetch jobs from Studio
    python3 jobs.py --fetch --enrich                # also fetch per-job details
    python3 jobs.py --clusters                      # JSON: list available clusters
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

STALE_AFTER_HOURS = 12
DEFAULT_DAYS = 30
DEFAULT_LIMIT = 500
ENRICH_LIMIT = 200
INDEX_PATH = "datachain/graph/jobs/index.md"

TERMINAL_STATUSES = {"complete", "failed", "canceled", "task"}


def _studio_available() -> bool:
    """Return True if a Studio token is configured (env var or config file)."""
    try:
        from datachain.remote.studio import is_token_set

        return is_token_set()
    except Exception:  # noqa: BLE001
        return False


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
    except Exception:  # noqa: BLE001
        return {}


def _duration_str(seconds: int) -> str:
    """Format duration as plain seconds string."""
    return f"{seconds}s"


def _strip_ordinal(value: str) -> str:
    """Strip trailing ordinal suffix like ' (3rd)' or ' (4th)' from a string."""
    import re

    return re.sub(r"\s*\(\w+\)\s*$", "", value).strip()


def _parse_dt(s) -> datetime | None:
    """Parse ISO datetime string to UTC-aware datetime."""
    if not s:
        return None
    try:
        # Handle both "Z" suffix and "+00:00" offset
        s = str(s).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        return None


def _normalize_status(status) -> str:
    """Normalize status value to lowercase string."""
    if status is None:
        return "unknown"
    return str(status).lower()


def cmd_plan():
    """Check staleness of the jobs index file."""
    now = datetime.now(tz=timezone.utc)
    studio_ok = _studio_available()

    result = {
        "up_to_date": False,
        "index_path": INDEX_PATH,
        "index_generated_at": None,
        "index_age_hours": None,
        "stale_after_hours": STALE_AFTER_HOURS,
        "studio_available": studio_ok,
    }

    if not studio_ok:
        result["error"] = (
            "Studio token not set. Run `datachain auth login`"
            " or set DATACHAIN_STUDIO_TOKEN."
        )
        print(json.dumps(result))
        return

    fm = _read_frontmatter(INDEX_PATH)
    generated_at_str = fm.get("generated_at")
    generated_at = _parse_dt(generated_at_str)

    if generated_at:
        age_hours = (now - generated_at).total_seconds() / 3600
        result["index_generated_at"] = generated_at_str
        result["index_age_hours"] = round(age_hours, 2)
        result["up_to_date"] = age_hours < STALE_AFTER_HOURS

    print(json.dumps(result))


def cmd_clusters():
    """List available Studio clusters."""
    from datachain.remote.studio import StudioClient

    client = StudioClient()
    response = client.get_clusters()
    if not response.ok:
        print(
            json.dumps({"error": response.message or "Failed to fetch clusters"}),
            file=sys.stderr,
        )
        sys.exit(1)

    clusters = []
    for c in response.data or []:
        clusters.append(
            {
                "id": c.get("id"),
                "name": c.get("name"),
                "cloud_provider": c.get("cloud_provider"),
                "max_workers": c.get("max_workers"),
                "is_active": c.get("is_active", True),
                "is_default": c.get("default", False),
            }
        )

    print(json.dumps({"clusters": clusters}))


def _enrich_job(client, job: dict) -> dict:
    """Fetch per-job details and merge into the job dict."""
    job_id = job.get("id")
    if not job_id:
        return job
    try:
        response = client.get_jobs(job_id=job_id)
        if response.ok and response.data and len(response.data) > 0:
            detail = response.data[0]
            # Merge fields that may be richer in the per-job response
            for field in (
                "workers",
                "finished_at",
                "python_version",
                "cluster",
                "compute_cluster_name",
                "cluster_name",
            ):
                if detail.get(field) is not None:
                    job[field] = detail[field]
    except Exception:  # noqa: BLE001, S110
        pass
    return job


def cmd_fetch(days: int, limit: int, enrich: bool):  # noqa: C901
    """Fetch jobs from Studio and output JSON."""
    from datachain.remote.studio import StudioClient

    client = StudioClient()
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=days)

    # Fetch clusters for name reference
    clusters_by_id = {}
    clusters_list = []
    try:
        cr = client.get_clusters()
        if cr.ok:
            for c in cr.data or []:
                entry = {
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "cloud_provider": c.get("cloud_provider"),
                    "max_workers": c.get("max_workers"),
                    "is_active": c.get("is_active", True),
                    "is_default": c.get("default", False),
                }
                clusters_list.append(entry)
                if c.get("id"):
                    clusters_by_id[c["id"]] = c.get("name", c["id"])
                if c.get("name"):
                    clusters_by_id[c["name"]] = c["name"]
    except Exception:  # noqa: BLE001, S110
        pass

    # Fetch jobs
    response = client.get_jobs(limit=limit)
    if not response.ok:
        print(
            json.dumps({"error": response.message or "Failed to fetch jobs"}),
            file=sys.stderr,
        )
        sys.exit(1)

    raw_jobs = response.data or []
    fetched_count = len(raw_jobs)
    truncated = fetched_count >= limit

    # Filter by date window (client-side — API has no date filter)
    filtered = []
    for j in raw_jobs:
        created_dt = _parse_dt(j.get("created_at"))
        if created_dt and created_dt >= cutoff:
            filtered.append(j)
        elif created_dt is None:
            filtered.append(j)  # include if we can't parse date

    # Optionally enrich terminal-state jobs
    to_enrich = []
    if enrich:
        to_enrich = [
            j
            for j in filtered
            if _normalize_status(j.get("status")) in TERMINAL_STATUSES
        ]
        n = min(len(to_enrich), ENRICH_LIMIT)
        if len(to_enrich) > ENRICH_LIMIT:
            print(
                f"Warning: {len(to_enrich)} terminal jobs found,"
                f" enriching first {ENRICH_LIMIT} only.",
                file=sys.stderr,
            )
            to_enrich = to_enrich[:ENRICH_LIMIT]
        elif n > 100:
            print(
                f"Enriching {n} jobs with per-job API calls...",
                file=sys.stderr,
            )
        to_enrich_ids = {j.get("id") for j in to_enrich}
        enriched_map = {}
        for j in to_enrich:
            enriched_j = _enrich_job(client, dict(j))
            enriched_map[j.get("id")] = enriched_j
        filtered = [
            enriched_map.get(j.get("id"), j) if j.get("id") in to_enrich_ids else j
            for j in filtered
        ]

    # Normalize each job
    jobs_out = []
    for j in filtered:
        created_dt = _parse_dt(j.get("created_at"))
        finished_dt = _parse_dt(j.get("finished_at"))

        duration_seconds = None
        if created_dt and finished_dt:
            dur = int((finished_dt - created_dt).total_seconds())
            if dur >= 0:
                duration_seconds = dur

        # Resolve cluster name: try multiple field names the API might use
        cluster_name = (
            j.get("cluster_name")
            or j.get("compute_cluster_name")
            or clusters_by_id.get(j.get("cluster"))
            or j.get("cluster")
        )

        # Strip ordinal suffixes (e.g. " (3rd)") from ID
        raw_id = j.get("id") or ""
        job_id = _strip_ordinal(raw_id) if raw_id else None

        # Format created_at as "YYYY-MM-DD HH:MM" for display
        created_at_display = (
            created_dt.strftime("%Y-%m-%d %H:%M") if created_dt else j.get("created_at")
        )

        jobs_out.append(
            {
                "id": job_id,
                "name": j.get("name"),
                "status": _normalize_status(j.get("status")),
                "created_at": j.get("created_at"),
                "created_at_display": created_at_display,
                "created_by": j.get("created_by"),
                "finished_at": j.get("finished_at"),
                "duration_seconds": duration_seconds,
                "duration_str": _duration_str(duration_seconds)
                if duration_seconds is not None
                else None,
                "workers": j.get("workers") or 1,
                "cluster_name": cluster_name,
                "python_version": j.get("python_version"),
            }
        )

    # Sort newest-first
    jobs_out.sort(key=lambda j: j.get("created_at") or "", reverse=True)

    # Compute status counts
    failed_count = sum(1 for j in jobs_out if j["status"] == "failed")
    complete_count = sum(1 for j in jobs_out if j["status"] == "complete")
    running_count = sum(1 for j in jobs_out if j["status"] == "running")

    print(
        json.dumps(
            {
                "generated_at": now.isoformat().replace("+00:00", "Z"),
                "days_covered": days,
                "fetched_count": fetched_count,
                "filtered_count": len(jobs_out),
                "truncated": truncated,
                "enriched": enrich,
                "failed_count": failed_count,
                "complete_count": complete_count,
                "running_count": running_count,
                "other_count": len(jobs_out)
                - failed_count
                - complete_count
                - running_count,
                "clusters": clusters_list,
                "jobs": jobs_out,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description="DataChain Studio job fetcher")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true", help="Check index.md staleness")
    group.add_argument("--fetch", action="store_true", help="Fetch jobs from Studio")
    group.add_argument(
        "--clusters", action="store_true", help="List available clusters"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Days to look back (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max jobs to fetch (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Fetch per-job details for workers/duration/cluster",
    )

    args = parser.parse_args()

    if args.plan:
        cmd_plan()
    elif args.clusters:
        cmd_clusters()
    elif args.fetch:
        cmd_fetch(days=args.days, limit=args.limit, enrich=args.enrich)


if __name__ == "__main__":
    main()
