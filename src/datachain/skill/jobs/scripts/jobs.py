"""Standalone DataChain Studio job fetcher for the datachain-jobs skill.

Usage:
    python3 jobs.py --plan                          # JSON: staleness check for index.md
    python3 jobs.py --fetch [--days N] [--limit N]  # JSON: fetch jobs from Studio
    python3 jobs.py --fetch --enrich                # also fetch per-job details
    python3 jobs.py --clusters                      # JSON: list available clusters

Output shapes
-------------
`--clusters` prints `{"clusters": [...]}`; `--fetch` prints the same list under
`"clusters"` alongside `"jobs"` and the counts the index frontmatter records.

Clusters come back exactly as Studio returns them - `ClusterData` in
`datachain.remote.studio` is the model, and these are the fields that matter here:
    uuid                the cluster's identifier, and what a job's cluster_uuid
                        points at. Prefer it over the legacy numeric `id`
    name, status, cloud_provider, is_active, default
    cloud_region        where it runs, e.g. us-west-2
    instance_type       machine type or family, e.g. m5.xlarge
    compute_class       node class, e.g. Performance or gpu - NOT spot vs on-demand
    disk_size           temporary storage a worker requests, e.g. 100Gi. NOT the
                        capacity of the volumes it is given, and no basis for a
                        storage cost
    job_quota           configured limit on the cluster's workers
    max_workers         the live value of that limit, as the cluster reports it

Each job carries:
    id, name, status, created, created_display, created_by, finished
    duration_seconds    wall clock, submit to finish; null while running
    duration_str        the same as "9000s"
    workers             machines the job asked for, 1 when unreported
    cluster_name        the cluster it ran on
    cluster_uuid        joins to a cluster's uuid; --enrich only
    python_version
    stages              {stage name: seconds}; --enrich only, {} otherwise.
                        Every returned job has them, not just the enriched ones
    queue_seconds       time in the `waiting` stage; null when unknown
    run_seconds         time in the `running_query` stage; null when unknown

A null is always "Studio did not report it", never zero.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

STALE_AFTER_HOURS = 12
DEFAULT_DAYS = 30
DEFAULT_LIMIT = 500
ENRICH_LIMIT = 200
INDEX_PATH = "dc-knowledge/jobs/index.md"

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
        s = str(s).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        return None


def _normalize_status(status) -> str:
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
        "index_generated": None,
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
    generated_str = fm.get("generated")
    generated_dt = _parse_dt(generated_str)

    if generated_dt:
        age_hours = (now - generated_dt).total_seconds() / 3600
        result["index_generated"] = generated_str
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

    # Exactly as Studio returns it - see ClusterData in datachain.remote.studio.
    print(json.dumps({"clusters": list(response.data or [])}))


def _stage_seconds(steps) -> dict:
    """Seconds spent in each job stage, keyed by stage name.

    Stages come from Studio only when asked for, and a job carries only the ones it
    reached - `waiting`, `requesting_workers`, `preparation`, `virtualenv`,
    `downloading_files`, `dw_wake_up`, `running_query`. A stage with no end (still
    running, or a job that stopped mid-stage) is left out rather than counted as 0.
    """
    stages = {}
    for step in steps or []:
        started = _parse_dt(step.get("started_at"))
        finished = _parse_dt(step.get("finished_at"))
        name = step.get("name")
        if not name or not started or not finished:
            continue
        seconds = int((finished - started).total_seconds())
        if seconds >= 0:
            stages[name] = seconds
    return stages


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
                "compute_cluster_uuid",
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

    # Fetch clusters for name reference, keyed by uuid and by name.
    clusters_by_key: dict[Any, Any] = {}
    clusters_list = []
    try:
        cr = client.get_clusters()
        if cr.ok:
            for c in cr.data or []:
                clusters_list.append(dict(c))
                if c.get("uuid"):
                    clusters_by_key[c["uuid"]] = c.get("name") or c["uuid"]
                if c.get("name"):
                    clusters_by_key[c["name"]] = c["name"]
    except Exception:  # noqa: BLE001, S110
        pass

    # include_steps fills every returned job's stages in this one call. Asking per
    # job instead would cap them at ENRICH_LIMIT and leave later jobs untimed.
    response = client.get_jobs(limit=limit, include_steps=enrich)
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

    # Annotated because the rows hold a mix - strings, ints, the stages dict - and
    # the sort below needs the inferred value type to stay comparable.
    jobs_out: list[dict[str, Any]] = []
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
            or clusters_by_key.get(j.get("compute_cluster_uuid"))
            or clusters_by_key.get(j.get("cluster"))
            or j.get("cluster")
        )

        raw_id = j.get("id") or ""
        job_id = _strip_ordinal(raw_id) if raw_id else None

        created_display = (
            created_dt.strftime("%Y-%m-%d %H:%M") if created_dt else j.get("created_at")
        )

        # Empty unless --enrich asked the list call for the job's stages.
        stages = _stage_seconds(j.get("steps"))

        jobs_out.append(
            {
                "id": job_id,
                "name": j.get("name"),
                "status": _normalize_status(j.get("status")),
                "created": j.get("created_at"),
                "created_display": created_display,
                "created_by": j.get("created_by"),
                "finished": j.get("finished_at"),
                "duration_seconds": duration_seconds,
                "duration_str": _duration_str(duration_seconds)
                if duration_seconds is not None
                else None,
                "workers": j.get("workers") or 1,
                "cluster_name": cluster_name,
                "cluster_uuid": j.get("compute_cluster_uuid"),
                "python_version": j.get("python_version"),
                "stages": stages,
                "queue_seconds": stages.get("waiting"),
                "run_seconds": stages.get("running_query"),
            }
        )

    jobs_out.sort(key=lambda j: j.get("created") or "", reverse=True)

    failed_count = sum(1 for j in jobs_out if j["status"] == "failed")
    complete_count = sum(1 for j in jobs_out if j["status"] == "complete")
    running_count = sum(1 for j in jobs_out if j["status"] == "running")

    print(
        json.dumps(
            {
                "generated": now.isoformat().replace("+00:00", "Z"),
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
