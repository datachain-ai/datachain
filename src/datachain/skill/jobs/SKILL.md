---
name: datachain-jobs
description: Use when asked about Studio job analytics — compute hours, user spend, failure rates, cost estimation, cluster usage. Generates and maintains dc-knowledge/jobs/index.md.
triggers:
  - "how many hours"
  - "compute time"
  - "who ran jobs"
  - "failed jobs"
  - "job cost"
  - "cluster usage"
  - "studio jobs"
  - "job analytics"
  - "job history"
  - "how much did we spend"
---

You are now loaded with the datachain-jobs skill. Maintain a jobs analytics file at `dc-knowledge/jobs/index.md`. Follow the 3-step flow below exactly.

## What Studio reports

Two things are available, and the script fetches both. Know what is in them before telling a user something cannot be answered.

**Clusters** (`--clusters`, and the `clusters` array of `--fetch`) — one entry per compute cluster, exactly as Studio returns it: `uuid`, `name`, `status`, `cloud_provider`, `is_active`, `default`, `max_workers`, `active_workers`, `busy_workers`, and what a worker is: `cloud_region`, `instance_type`, `compute_class`, `disk_size`, `job_quota`. That identifies the machine for a rate lookup; it is not a price. Identify a cluster by `uuid` — an `id` is also returned, but it is legacy and on its way out.

**Jobs** (the `jobs` array of `--fetch`) — `id`, `name`, `status`, `created`, `created_by`, `finished`, `duration_seconds`/`duration_str`, `workers`, `cluster_name`, `python_version`. Every job also carries `cluster_uuid`, which joins to a cluster's `uuid`. With `--enrich` each one carries `stages` too, a `{stage name: seconds}` map behind `queue_seconds` and `run_seconds`. The stages are `waiting`, `requesting_workers`, `preparation`, `virtualenv`, `downloading_files`, `dw_wake_up`, `running_query`. A job carries the ones Studio recorded a timing for.

A null anywhere means Studio did not report it, never zero. The script's module docstring (`{skill_dir}/scripts/jobs.py`) is the authoritative field list.

---

## Step 1 — Check Staleness

```
python3 {skill_dir}/scripts/jobs.py --plan
```

- If `"studio_available": false` → report the `error` message and stop.
- If `"up_to_date": true` → skip to Step 3.
- If `"up_to_date": false` → continue to Step 2.

---

## Step 2 — Fetch & Write

```
python3 {skill_dir}/scripts/jobs.py --fetch [--days N] [--limit N] [--enrich]
```

- Use `--days N` from the user's request if stated (e.g. "last 7 days" → `--days 7`). Default: `--days 30`.
- Add `--enrich` when the question needs stage timings (Queue and Run). It costs one request either way — duration, workers and cluster come back without it.
- If the script fails → report the error and stop.

Write `dc-knowledge/jobs/index.md` using EXACTLY this format:

```markdown
---
generated: <generated from script output>
days_covered: <days_covered>
total_jobs: <filtered_count>
failed_count: <failed_count>
complete_count: <complete_count>
running_count: <running_count>
other_count: <other_count>
enriched: <true|false>
duration_note: "Wall-clock duration (submit→finish). Null while a job is still running."
stage_note: "Queue/Run come from job stages. Null when enriched=false, or when that timing is unavailable."
truncated: <true|false>
---

## Clusters

| UUID | Name | Cloud | Region | Instance Type | Compute Class | Disk Request | Job Quota | Max Workers | Default |
|------|------|-------|--------|---------------|---------------|--------------|-----------|-------------|---------|
| <uuid> | <name> | <cloud_provider> | <cloud_region> | <instance_type> | <compute_class> | <disk_size> | <job_quota> | <max_workers> | <yes if default else no> |

## Jobs

| Date | ID | Name | Status | User | Workers | Duration | Queue | Run | Cluster | Cluster UUID | Python |
|------|----|------|--------|------|---------|----------|-------|-----|---------|--------------|--------|
| <created_display> | <id> | <name> | <status> | <created_by> | <workers> | <duration_str> | <queue_seconds as Ns> | <run_seconds as Ns> | <cluster_name> | <cluster_uuid> | <python_version> |
```

**Section rules:**
- Every cell: write `—` when, and only when, the value is null. `0` and `false` are values — write them.
- Omit `## Clusters` if the `clusters` array is empty.
- A null instance type or region means Studio has no record of it, not that the cluster lacks one.
- Carry both UUIDs through. They are what a job joins to its cluster on: names can be reused, and a retired cluster keeps its jobs while dropping out of the cluster list.
- Job Quota is the configured worker limit, which the cluster reports live as Max Workers. It is not a number of jobs each worker runs.
- Disk Request (`disk_size`) is requested temporary storage per worker; allocated capacity may differ.
- Duration cell: `duration_str` value, e.g. `"9000s"`.
- Queue and Run: `queue_seconds` and `run_seconds` written as `Ns` (e.g. `4s`). They come from the job's stages, so they are `—` unless `enriched: true`. Duration covers everything from submit to finish, so Queue + Run is normally less than it — the difference is setup (`preparation`, `virtualenv`, `downloading_files`, `dw_wake_up`).
- Workers: always a number (`workers` field, defaults to 1).
- Date column: `created_display` (`YYYY-MM-DD HH:MM` UTC).
- Rows: newest-first (already sorted by script).
- If `truncated: true`, add after the table: `_(Results truncated at <limit> jobs. Use --limit N for more.)_`

---

## Step 3 — Answer

Read `dc-knowledge/jobs/index.md` and answer the user's question.

### Duration arithmetic
Duration cells contain plain seconds strings like `"9000s"`. Parse the integer before `s`, sum, then convert:
- Example: filter rows for user "alice" in the last 7 days, sum all Duration values → total seconds → divide by 3600 for hours.
- Missing durations do not mean the index needs re-fetching — duration does not depend on `--enrich`. A `—` means the job has no recorded finish, usually because it is still running. Say so, and state the coverage when aggregating.

### Stage breakdown — where the time went
The Queue and Run columns split a job's wall clock: waiting before it started, running the query. The remainder is setup.
- "How long do jobs wait?" → sum or average the Queue column. It shows where the time went, not why. Check cluster capacity or diagnostics before explaining a wait, and never recommend more workers on queue time alone.
- "Why is this job slow?" → compare Queue, Run, and `Duration − Queue − Run` (setup: dependency installs, file downloads, warehouse wake-up).
- Missing timings: check `enriched` in the frontmatter, not the cells. If it is `false`, re-run Step 2 with `--enrich` yourself as part of answering. If it is `true` and the cells are still `—`, the timings were never recorded for those jobs — say "Stage timings are unavailable for these jobs", and state the coverage (how many of how many) whenever you aggregate.
- For a stage the table does not hold - download time, dependency install - Queue and Run are not enough, and `enriched: true` does not help: it records that stages were fetched, not that all of them were saved. Re-run Step 2 with `--enrich` and answer from the `stages` map in its output.

### Failure rate
- Overall: `failed_count / total_jobs * 100` from frontmatter.
- Per user or per day: count rows matching Status = `failed` in the table.

### Price estimation
When the user asks for cost:
1. Match each job to its cluster on the UUIDs, never on the name. A retired cluster keeps its jobs but drops out of the cluster list, and Studio lets a later cluster take its name — matching on name can price a job against a machine it never ran on. A job whose Cluster UUID is in no Clusters row ran on a cluster that is gone: say its details are unavailable rather than guessing one. Use the Cluster name for display either way.
2. Establish the hourly rate from that row — instance type and region are what a cloud price list is keyed on. State the rate you used and where it came from. If the row's instance type is `—`, or you have no price for it, ask: "What is the hourly rate in $/hr for <cluster> (<instance type> in <region>)?"
3. Nothing in the index says whether a cluster runs on spot or on-demand capacity — Compute Class is a node class (e.g. `Performance`, `gpu`), not a purchase model. Spot can cost a fraction of on-demand, so when the answer turns on it, ask rather than assume: "Is <cluster> running spot or on-demand capacity?"
4. If Workers column is all `—` → ask: "How many workers per job?" or compute single-worker cost and note it.
5. Compute per job: `duration_seconds / 3600 × rate × workers`. Group by user/day/cluster as requested.
6. Storage cost requires allocated capacity, storage type, and rate; Disk Request is insufficient. Ask for the three, or say the cluster data does not carry them.
7. Present as a table: User | Compute-hours | Est. cost (@$X/hr × N workers), with a line naming the rate and instance type per cluster.

### Per-cluster / per-user analytics
Filter the Jobs table by the Cluster or User column. Aggregate `(Ns)` Duration values for totals.
