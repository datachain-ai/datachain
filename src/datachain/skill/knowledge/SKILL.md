---
name: datachain-knowledge
description: Use whenever datasets, cloud storage buckets, or data pipelines are mentioned — creating, saving, querying, listing, exploring, deleting, or processing data in S3, GCS, Azure Blob, or local storage. Also use when running any script that may create datasets as a side effect. Maintains a knowledge base at dc-knowledge/ (JSON + markdown). ALWAYS use this skill when the user creates a dataset, saves pipeline output, runs a data script, or references any storage bucket.
triggers:
  # Discovery
  - "what datasets exist"
  - "show me the schema"
  - "list datasets"
  - "datachain knowledge"
  - "update the knowledge base"
  - "refresh dataset docs"
  - "what's in this bucket"
  - "explore bucket"
  - "scan bucket"
  - "bucket overview"
  - "what files are in s3://"
  - "what files are in gs://"
  # Creation & mutation
  - "create dataset"
  - "save dataset"
  - "delete dataset"
  - "new dataset"
  - "build dataset"
  - "make dataset"
  - "generate dataset"
  # Pipeline output
  - "save the results"
  - "save to dataset"
  # Storage references
  - "s3://"
  - "gs://"
  - "az://"
  - "read_storage"
  - "from bucket"
  - "from s3"
  - "from gcs"
  # Data processing
  - "process files"
  - "extract metadata"
  - "filter dataset"
  - "query dataset"
  # Script execution (may create datasets as side effects)
  - "run script"
  - "run pipeline"
  - "python scan"
  - "run scan"
---

Maintain a knowledge base at `dc-knowledge/`. `.md` files are the persistent
output. `.json` files are intermediate (generated in Step 3, consumed in
Step 4, then deleted).

`CAST.md` (sibling to this file) is the canonical methodology — the four
layers, naming + tagging, layer-ladder planning, calibration, dialogue
template, reuse rules, methodology transmission. Mode B reads it in full
as a precondition. When something methodology-related needs to change,
change `CAST.md`, not this file.

## Critical Rules

`CAST.md` §6 owns the CAST-doctrine rules (follow CAST, never bypass
DataChain, C/A/S substrate mandatory, one script per stage, one
`.save()` per script). The rules below are operational additions unique
to this skill.

1. **Path is `dc-knowledge/`** — NOT `.datachain/`. The `.datachain/` directory is the internal database; the knowledge base lives at `dc-knowledge/`.
2. **Never pass `update=True`** to `dc.read_storage()` in Task or exploration code unless the user explicitly asks to refresh the listing. L1/L2/L3 build scripts are the exception (`CAST.md` §5).
3. **Prefer DataChain operations** over plain Python for all metadata analysis.
4. **Bounded output** — JSON and markdown files stay small regardless of data size.
5. **Stop on auth/connection errors** — `bucket_scan.py` runs a fast access check. If it exits with an error JSON on stderr, **stop immediately** and show the error to the user. Do not retry with different regions, profiles, or endpoints — ask for the missing credentials.
6. **Follow the enrichment prompt template literally** in Step 4. Downstream tooling (`render_index.py`, `cast_layer` resolution) parses the exact frontmatter the prompt prescribes.

## Common gotchas in UDF scripts

- **`parallel=N` vs `workers=N`.** `parallel=N` is local multiprocessing (works anywhere). `workers=N` is Studio-only and MUST be guarded: `chain = chain.settings(parallel=N); if dc.is_studio(): chain = chain.settings(workers=N)`.
- **No `from __future__ import annotations` in UDF modules.** It stringifies type hints and DataChain's signal-schema resolution rejects the string-vs-class mismatch.
- **Type the UDF return precisely.** `Iterator[object]` / `Iterator[Any]` / bare `dict` fail schema resolution. Return a specific `Iterator[T]`, a Pydantic `BaseModel`, or a primitive.
- **Generators aren't subscriptable.** Iterators returned by file APIs do not support `[:N]`. Use `enumerate` + `break`, or `list(...)` only when the result is genuinely small.
- **Use `datachain.__version__` to get the package version** (e.g. `dc.__version__`).

---

## Workflow Mode Detection

**Mode A — Discovery/Exploration** (e.g., "what datasets exist", "show schema", "explore bucket"):
→ If the user references a specific bucket URI, run **Step 1** (Bucket Enlistment) for its root first.
→ Then run Steps 2–7.

**Mode B — Dataset Creation/Pipeline** (e.g., "create dataset X from ...", "process files and save"):

> **Precondition (do this FIRST — before ANY tool call):**
>
>     $ cat dc-knowledge/index.md
>     $ cat {skill_dir}/CAST.md
>
> If `index.md` exists and the task can be solved by reading an existing
> dataset, do not write a pipeline — read it directly with
> `dc.read_dataset("name")` and filter/merge/extend from there. This avoids
> recomputing expensive operations.
>
> `CAST.md` drives every layer / scope / shape decision. Re-read on each
> new task so the layer-ladder walk and dialogue template are in working
> context when you plan.
>
> **Never parse files under `dc-knowledge/datasets/*.json` or
> `dc-knowledge/buckets/**/*.json` directly** — those are pre-render
> intermediates that get deleted. The information you need is in `index.md`.
>
> If `dc-knowledge/index.md` does not exist, proceed with Steps 1–7 to build it.

→ **If the pipeline reads from a bucket**, run **Step 1** (Bucket Enlistment) for the bucket root first.
→ **Run the access check** (if not already done in Step 1): `datachain bucket status <uri>`. If `not found` / `denied`, stop and ask for credentials.
→ Read `{skill_dir}/../core/SKILL.md` for DataChain SDK rules.
→ Follow `CAST.md` §4 (planning) and §4.10 (dialogue) before writing pipeline code.
→ **While the pipeline is running**, enrich any Step 1 bucket JSON that does not yet have a `.md` (parallel work).
→ After the pipeline completes, run Steps 2–7 to update the knowledge base.
→ Report both: pipeline result AND knowledge base update status.

**Mode C — Script Execution** (e.g., user runs an existing `.py` file that touches data):
→ If the script references bucket URIs, run **Step 1** for each bucket root first.
→ Scripts can create datasets as side effects.
→ **While the script is running**, enrich Step 1 bucket JSON in parallel.
→ After ANY data-related script finishes, run Steps 2–7 to detect and record new/changed datasets.

**Mode D — Knowledge Base Maintenance** (e.g., "update the knowledge base", "refresh dataset docs"):
→ Run Steps 2–7. Existing session context in `.md` files is preserved automatically during re-enrichment.

---

## Step 1 — Bucket Enlistment

When any storage URI is encountered, enlist the whole bucket first.

1. **Extract bucket root.** From any URI, derive `{scheme}://{bucket}/`.
2. **Check if already enlisted.** Look for `dc-knowledge/buckets/{scheme}/{bucket_slug}.md` or `.json`. If either exists, skip.
3. **Access check.** Run `datachain bucket status {root_uri}`. If denied / not found, stop and ask.
4. **Scan with timeout.** Default 60s; user can override:
   ```bash
   python3 {skill_dir}/scripts/bucket_scan.py {root_uri} \
     --output dc-knowledge/buckets/{scheme}/{bucket_slug}.json --timeout 60
   ```
5. **Handle timeout** (exit code 124). Run the hierarchical fallback:
   ```bash
   python3 {skill_dir}/scripts/bucket_overview.py {root_uri} \
     --bucket-json dc-knowledge/buckets/{scheme}/{bucket_slug}.json
   ```
6. **Report.** "Enlisted bucket {bucket} — {N} files, total size {size}, primarily {top 2-3 extensions}." Do **not** enrich here; Step 4 batches it.

Step 1 runs **once per bucket root** per session.

---

## Step 2 — Sync

```bash
python3 {skill_dir}/scripts/plan.py [--studio] --output dc-knowledge/.plan.json
```

Buckets are auto-discovered from catalog listings. Do **not** add `--studio` unless requested. If `"up_to_date": true`, print "Knowledge base is up to date." and stop. Entries with `status` of `"new"` or `"stale"` need processing in Step 3.

---

## Step 3 — Save Data

For each dataset where `status != "ok"`:
```bash
python3 {skill_dir}/scripts/dataset_all.py <name> \
  --plan dc-knowledge/.plan.json --output dc-knowledge/<file_path>.json
```

For each bucket where `status != "ok"` (and not enlisted in Step 1):
```bash
python3 {skill_dir}/scripts/bucket_scan.py <uri> --output dc-knowledge/<file_path>.json
```

Run independent calls concurrently.

---

## Step 4 — Enrich

Generate `.md` from `.json` for each entry processed in Step 3 (and any Step 1 bucket JSON that lacks a `.md`).

- Datasets: read `{skill_dir}/prompts/enrich.md`, then write `dc-knowledge/<file_path>.md` per the template.
- Buckets: read `{skill_dir}/prompts/enrich_bucket.md`, then write `dc-knowledge/<file_path>.md`.

The prompt template is authoritative — downstream tooling parses the exact frontmatter it prescribes. Skip this step only if the user requests raw output only.

---

## Step 5 — Build Index

```bash
python3 {skill_dir}/scripts/render_index.py --plan dc-knowledge/.plan.json --output dc-knowledge/index.md
```

---

## Step 6 — Cleanup

```bash
python3 {skill_dir}/scripts/cleanup_json.py --plan dc-knowledge/.plan.json
```

Keeps `.plan.json` for Step 7. Skip if the user asks to retain JSON for debugging.

---

## Step 7 — Report

```
Knowledge base updated: <N> datasets (<M> updated, <K> unchanged), <B> buckets (<X> scanned, <Y> unchanged).
```

If any buckets have `listing_expired: true`, add:
```
Warning: Listing for <bucket> is expired (last scanned: <date>). Run dc.read_storage("<uri>", update=True) to refresh.
```
