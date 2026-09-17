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
- Add `--enrich` only when the question requires duration, workers, or cluster data AND `enriched: false` in an existing index — tell the user it makes one API call per terminal job.
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
truncated: <true|false>
---

## Clusters

| Name | Cloud | Region | Instance Type | Compute Class | Disk | Jobs per Worker | Max Workers | Default |
|------|-------|--------|---------------|---------------|------|-----------------|-------------|---------|
| <name> | <cloud_provider> | <cloud_region or —> | <instance_type or —> | <compute_class or —> | <disk_size or —> | <job_quota or —> | <max_workers> | <yes if is_default else no> |

## Jobs

| Date | ID | Name | Status | User | Workers | Duration | Cluster | Python |
|------|----|------|--------|------|---------|----------|---------|--------|
| <created_display> | <id> | <name> | <status> | <created_by> | <workers> | <duration_str or —> | <cluster_name or —> | <python_version or —> |
```

**Section rules:**
- Omit `## Clusters` if the `clusters` array is empty.
- Cluster cells: use `—` when a field is null. A null instance type or region means Studio has no record of it, not that the cluster lacks one.
- Duration cell: `duration_str` value (e.g. `"9000s"`) when known, `—` when null.
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

### Failure rate
- Overall: `failed_count / total_jobs * 100` from frontmatter.
- Per user or per day: count rows matching Status = `failed` in the table.

### Price estimation
When the user asks for cost:
1. Establish the hourly rate per cluster from its Clusters row — instance type, region, and compute class are what a cloud price list is keyed on, and compute class says whether the rate is spot or on-demand. State the rate you used and where it came from. If the row's instance type is `—`, or you have no price for it, ask: "What is the hourly rate in $/hr for <cluster> (<instance type> in <region>)?"
2. If Workers column is all `—` → ask: "How many workers per job?" or compute single-worker cost and note it.
3. Compute per job: `duration_seconds / 3600 × rate × workers`. Group by user/day/cluster as requested.
4. Where a cluster's Jobs per Worker is above 1, jobs share a worker: say so, and treat the per-job figure as an upper bound rather than silently dividing by it.
5. Disk is the worker's disk request, not a separate billed volume — fold it in only if the user asks for storage cost, and say the rate is per-GB-month.
6. Present as a table: User | Compute-hours | Est. cost (@$X/hr × N workers), with a line naming the rate and instance type per cluster.

### Per-cluster / per-user analytics
Filter the Jobs table by the Cluster or User column. Aggregate `(Ns)` Duration values for totals.
