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

**Clusters** (`--clusters`, and the `clusters` array of `--fetch`) — one entry per compute cluster, exactly as Studio returns it: `uuid`, `name`, `status`, `cloud_provider`, `is_active`, `default`, `max_workers`, `active_workers`, `busy_workers`, and the cost-relevant `cloud_region`, `instance_type`, `compute_class`, `disk_size`, `job_quota`. Identify a cluster by `uuid` — an `id` is also returned, but it is legacy and on its way out.

**Jobs** (the `jobs` array of `--fetch`) — `id`, `name`, `status`, `created`, `created_by`, `finished`, `duration_seconds`/`duration_str`, `workers`, `cluster_name`, `python_version`. With `--enrich` each terminal job also carries `cluster_uuid` (joins to a cluster's `uuid`) and `stages`, a `{stage name: seconds}` map behind `queue_seconds` and `run_seconds`. The stages a job can have are `waiting`, `requesting_workers`, `preparation`, `virtualenv`, `downloading_files`, `dw_wake_up`, `running_query` — which ones it actually has depends on when it ran and how far it got.

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
- Add `--enrich` only when the question requires duration, workers, cluster, or stage timings AND `enriched: false` in an existing index — tell the user it makes one API call per terminal job.
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
duration_note: "Wall-clock duration (submit→finish). Null when enriched=false or job still running."
stage_note: "Queue/Run come from job stages. Null when enriched=false, or when the job never reached that stage."
truncated: <true|false>
---

## Clusters

| Name | Cloud | Region | Instance Type | Compute Class | Disk Request | Job Quota | Max Workers | Default |
|------|-------|--------|---------------|---------------|--------------|-----------|-------------|---------|
| <name> | <cloud_provider> | <cloud_region or —> | <instance_type or —> | <compute_class or —> | <disk_size or —> | <job_quota or —> | <max_workers> | <yes if default else no> |

## Jobs

| Date | ID | Name | Status | User | Workers | Duration | Queue | Run | Cluster | Python |
|------|----|------|--------|------|---------|----------|-------|-----|---------|--------|
| <created_display> | <id> | <name> | <status> | <created_by> | <workers> | <duration_str or —> | <queue_seconds as Ns, or —> | <run_seconds as Ns, or —> | <cluster_name or —> | <python_version or —> |
```

**Section rules:**
- Omit `## Clusters` if the `clusters` array is empty.
- Cluster cells: use `—` when a field is null. A null instance type or region means Studio has no record of it, not that the cluster lacks one.
- Job Quota is the configured limit on the cluster's workers, which the cluster reports live as Max Workers. It is not a number of jobs each worker runs.
- Disk Request (`disk_size`) is temporary storage a worker asks for, not the capacity of the volumes it gets. It cannot price storage.
- Duration cell: `duration_str` value (e.g. `"9000s"`) when known, `—` when null.
- Queue and Run: `queue_seconds` and `run_seconds` written as `Ns` (e.g. `4s`), `—` when null. They come from the job's stages, so they are `—` unless `enriched: true`. Duration covers everything from submit to finish, so Queue + Run is normally less than it — the difference is setup (`preparation`, `virtualenv`, `downloading_files`, `dw_wake_up`).
- Workers: always a number (`workers` field, defaults to 1).
- Cluster, Python: use `—` when null.
- Date column: `created_display` (`YYYY-MM-DD HH:MM` UTC).
- Rows: newest-first (already sorted by script).
- If `truncated: true`, add after the table: `_(Results truncated at <limit> jobs. Use --limit N for more.)_`

---

## Step 3 — Answer

Read `dc-knowledge/jobs/index.md` and answer the user's question.

### Duration arithmetic
Duration cells contain plain seconds strings like `"9000s"`. Parse the integer before `s`, sum, then convert:
- Example: filter rows for user "alice" in the last 7 days, sum all Duration values → total seconds → divide by 3600 for hours.
- If all Duration cells are `—` (enriched: false) → say: "Duration data requires enrichment. Re-fetch with: `python3 {skill_dir}/scripts/jobs.py --fetch --enrich`" and offer to do so.

### Stage breakdown — where the time went
The Queue and Run columns split a job's wall clock: `waiting` before it started, `running_query` doing the work. The remainder is setup.
- "How long do jobs wait?" → sum or average the Queue column. A queue that rivals Run means the cluster is at its worker cap, not that jobs are slow.
- "Why is this job slow?" → compare Queue, Run, and `Duration − Queue − Run` (setup: dependency installs, file downloads, warehouse wake-up).
- All `—` → the index is not enriched. Say so and offer: `python3 {skill_dir}/scripts/jobs.py --fetch --enrich`.
- For a finer split than Queue/Run, read the `stages` map from the script output directly rather than the index.

### Failure rate
- Overall: `failed_count / total_jobs * 100` from frontmatter.
- Per user or per day: count rows matching Status = `failed` in the table.

### Price estimation
When the user asks for cost:
1. Establish the hourly rate per cluster from its Clusters row — instance type and region are what a cloud price list is keyed on. State the rate you used and where it came from. If the row's instance type is `—`, or you have no price for it, ask: "What is the hourly rate in $/hr for <cluster> (<instance type> in <region>)?"
2. Nothing in the index says whether a cluster runs on spot or on-demand capacity — Compute Class is a node class (e.g. `Performance`, `gpu`), not a purchase model. Spot can cost a fraction of on-demand, so when the answer turns on it, ask rather than assume: "Is <cluster> running spot or on-demand capacity?"
3. If Workers column is all `—` → ask: "How many workers per job?" or compute single-worker cost and note it.
4. Compute per job: `duration_seconds / 3600 × rate × workers`. Group by user/day/cluster as requested.
5. Do not derive storage cost from Disk Request. It is how much temporary storage a worker asks for, not how much it is given — a `1Gi` request can sit on volumes of `20Gi` or `500Gi`, and the request does not move when they change. If the user asks for storage cost, say the cluster data does not carry the billable capacity, and ask for the volume size and type and the applicable rate.
6. Present as a table: User | Compute-hours | Est. cost (@$X/hr × N workers), with a line naming the rate and instance type per cluster.

### Per-cluster / per-user analytics
Filter the Jobs table by the Cluster or User column. Aggregate `(Ns)` Duration values for totals.
