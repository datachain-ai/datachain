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
  - "process images"
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

You are now loaded with the datachain-knowledge skill. Maintain a knowledge base at `dc-knowledge/`. `.md` files are the persistent output — they contain frontmatter metadata, schema, code, and version history. `.json` files are intermediate (generated in Step 3, consumed in Step 4, then deleted). Follow the workflow below.

## Critical Rules

1. **Path is `dc-knowledge/`** — NOT `.datachain/`. The `.datachain/` directory is the internal database; the knowledge base lives at `dc-knowledge/`.
2. **Never pass `update=True`** to `dc.read_storage()` unless the user explicitly asks to refresh the listing.
3. **Prefer DataChain operations** over plain Python for all metadata analysis.
4. **Bounded output** — JSON and markdown files stay small regardless of data size.
5. **Stop on auth/connection errors** — `bucket_scan.py` runs a fast access check before scanning (uses cloud SDKs, no DC listing). If it exits with an error JSON on stderr, **stop immediately** and show the error to the user. Do not retry with different regions, credential profiles, or endpoint variations — ask the user for the missing credentials or configuration.

---

## Workflow Mode Detection

When loaded, determine the user's intent:

**Mode A — Discovery/Exploration** (e.g., "what datasets exist", "show schema", "explore bucket"):
→ If the user references a specific bucket URI, run **Step 1** (Bucket Enlistment) for its root first.
→ Then run Steps 2–5 as normal.

**Mode B — Dataset Creation/Pipeline** (e.g., "create dataset X from ...", "process images and save"):
→ **If the pipeline reads from a bucket** (`read_storage`), run **Step 1** (Bucket Enlistment) for the bucket root first. The bucket overview in the knowledge base may inform the pipeline design.
→ **Before building anything**, read `dc-knowledge/index.md` and check whether an existing dataset already covers the data the user needs. If one does, start from `dc.read_dataset("name")` — filter, merge, or extend it instead of re-reading raw storage. This avoids recomputing expensive operations (LLM calls, model inference) and reuses proven code patterns.
→ **Run the access check** (if not already done in Step 1):
  ```bash
  datachain bucket status <uri>
  ```
  Prints `Status: exists|not found` and `Access: anonymous|authenticated|denied`. Exit code 0 = exists, 1 = not found. If status is `not found` or access is `denied` → stop and ask the user for credentials. If access is `anonymous` → pass `anon=True` to `read_storage()`.
→ Read `{skill_dir}/../core/SKILL.md` for DataChain SDK rules and patterns.
→ **Superlative defaults.** When the user asks "best/worst/highest/lowest" without naming a metric, pick the most natural metric the data supports, name it in a one-line comment, and emit `n` per group. Drop groups with `n < max(5, sqrt(median_n))` before ranking. Ask one clarifying question only if multiple metrics would materially change the ranking.
→ **Slice sanity check.** Before committing to a prefix or glob, verify the slice still contains every entity dimension the question groups, compares, or ranks over. If too large, narrow on an orthogonal dimension — never on one the question depends on.
→ **Multi-axis classification batching.** When a per-row LLM/VLM call classifies on the axis the user asked about, extend the prompt to return plausible related axes in the same call. Variant questions then hit cache on the saved dataset.
→ Build and execute the pipeline the user requested, following core skill rules.
→ **While the pipeline is running**, enrich any Step 1 bucket JSON that does not yet have a `.md` — read `{skill_dir}/prompts/enrich_bucket.md` and generate the markdown in parallel with the running script.
→ After the pipeline completes, **always** run Steps 2–5 to update the knowledge base.
→ **During Step 4 (Enrich)**, when writing the `.md` for a dataset created in this session: add a `## Session Context` section with 1-3 sentences summarizing why the dataset was created — the user's goal, the analytical question, or the discussion that led to it. Only add this if the session provides meaningful context. If the dataset is a routine output with no notable motivation, omit it.
→ Report both: pipeline result AND knowledge base update status.

**Mode C — Script Execution** (e.g., user runs an existing script, or agent runs a .py file that touches data):
→ If the script references bucket URIs, run **Step 1** (Bucket Enlistment) for each bucket root first.
→ Scripts can create datasets as side effects (e.g., `scan.py` calls `.save()` internally).
→ **While the script is running**, enrich any Step 1 bucket JSON that does not yet have a `.md` — read `{skill_dir}/prompts/enrich_bucket.md` and generate the markdown in parallel with the running script.
→ After ANY data-related script finishes, run Steps 2–5 to detect and record new/changed datasets.
→ This applies even if the script was not written by the agent — always check the DB afterward.
→ Do not add `## Session Context` for script-executed datasets unless the there is a specific context in a session about why the dataset was created or why script is being run.

**Mode D — Knowledge Base Maintenance** (e.g., "update the knowledge base", "refresh dataset docs"):
→ Run Steps 2–5 as normal.
→ Do not add new `## Session Context` sections during maintenance refreshes. Existing session context in `.md` files is preserved automatically during re-enrichment.

---

## Step 1 — Bucket Enlistment

When any storage URI is encountered, **enlist the whole bucket first** before doing any other work. This gives the knowledge base a complete bucket overview rather than fragmented prefix-level entries.

### Procedure

1. **Extract bucket root.** From any URI (e.g., `s3://my-bucket/data/images/`), derive the root: `{scheme}://{bucket}/` (e.g., `s3://my-bucket/`).

2. **Check if already enlisted.** Look for `dc-knowledge/buckets/{scheme}/{bucket_slug}.md` or `.json` (where `bucket_slug` is the bucket name lowercased with non-alphanumeric characters replaced by `_`). If either file exists, skip enlistment — the bucket is already known.

3. **Communicate.** Tell the user: "Enlisting bucket {bucket}..."

4. **Access check.** Run:
   ```bash
   datachain bucket status {root_uri}
   ```
   If access is `denied` or bucket is `not found` → stop and ask the user. Note the access level for the scan step.

5. **Scan with timeout.** Run:
   ```bash
   python3 {skill_dir}/scripts/bucket_scan.py {root_uri} \
     --output dc-knowledge/buckets/{scheme}/{bucket_slug}.json \
     --timeout 60
   ```
   - Default timeout: **60 seconds**.
   - If the user has indicated the bucket is large: use **180** or the timeout the user specifies.

6. **Handle timeout.** If the command exits with code 124 (timeout):
   - Run the hierarchical fallback: `python3 {skill_dir}/scripts/bucket_overview.py {root_uri} --bucket-json dc-knowledge/buckets/{scheme}/{bucket_slug}.json` (add `--anon` for public buckets).
   - It saves a sampled DataChain dataset of File rows and writes a bucket-shape JSON the enrich step turns into a bucket markdown marked `sampled: true`. Continue with Steps 2–5 as normal.

7. **Report.** On success, read the JSON output and report a quick summary:
   > Enlisted bucket {bucket} — {N} files, total size {size}, primarily {top 2-3 extensions}.

   Read `total_files`, `total_size_bytes`, and the top entries from `extensions[]` in the JSON. Do **not** enrich (generate markdown) here — that happens later in Step 4, batched with all other enrichments.

### Notes
- Step 1 runs **once per bucket root**. If multiple URIs reference the same bucket (e.g., `s3://bucket/data/` and `s3://bucket/meta/`), only one enlistment is needed.
- Step 1 does **not** replace Steps 2–5. Prefix-level entries may still be created in Steps 2–3 if the catalog has prefix-level listings.
- Step 1 leaves a `.json` file — enrichment to `.md` is deferred. **Parallelism opportunity:** In Modes B and C, enrich the Step 1 JSON while a pipeline or script is running (see mode instructions below).

---

## Step 2 — Sync

Plan what needs updating. The plan auto-discovers both datasets and buckets from the catalog.

```bash
python3 {skill_dir}/scripts/plan.py [--studio] --output dc-knowledge/.plan.json
```

- Buckets are auto-discovered from catalog listings (every bucket that `dc.read_storage()` has ever listed). No flag needed.
- Do **NOT** add `--studio` unless the user explicitly requests it.
- If `"up_to_date": true` → print "Knowledge base is up to date." and stop.
- If the output contains `"warnings"` → report them after the update summary.

Review `.plan.json`. Entries with `status` of `"new"` or `"stale"` need processing in Step 3. Entries with `"ok"` are skipped.

---

## Step 3 — Save Data

### Datasets

For each dataset where `status != "ok"` in `plan.datasets[]`:

```bash
python3 {skill_dir}/scripts/dataset_all.py <name> \
  --plan dc-knowledge/.plan.json \
  --output dc-knowledge/<file_path>.json
```

- `<name>` and `<file_path>` come from the plan's `datasets[]` entries.
- Do **not** modify the output — it is deterministic and complete.

### Buckets

For each bucket where `status != "ok"` in `plan.buckets[]`:

**Skip buckets enlisted in Step 1.** If a bucket root was already scanned in Step 1 during this session, treat it as `"ok"` regardless of what the plan says — the JSON is already up to date.

```bash
python3 {skill_dir}/scripts/bucket_scan.py <uri> \
  --output dc-knowledge/<file_path>.json
```

- `<uri>` and `<file_path>` come from the plan's `buckets[]` entries.
- The script aggregates metadata (extensions, directories, sizes, timestamps) and samples files.

Run independent `dataset_all.py` and `bucket_scan.py` calls concurrently when multiple items need processing.

---

## Step 4 — Enrich

For each dataset or bucket processed in Step 3, generate a human-readable markdown summary from the JSON data.

### Datasets

1. Read the enrichment prompt at `{skill_dir}/prompts/enrich.md`.
2. For each dataset, read `dc-knowledge/<file_path>.json`.
3. Following the prompt, write `dc-knowledge/<file_path>.md`.

### Buckets

1. Read the enrichment prompt at `{skill_dir}/prompts/enrich_bucket.md`.
2. For each bucket processed in Step 3 **and** any bucket JSON from Step 1 that does not yet have a corresponding `.md`, read `dc-knowledge/<file_path>.json`.
3. Following the prompt, write `dc-knowledge/<file_path>.md`.

Skip this step only if the user requests raw output only.

---

## Step 4.5 — Build Index

Update the index (must run after enrichment so it can read dataset `.md` summaries and `.json` dependencies):
```bash
python3 {skill_dir}/scripts/render_index.py --plan dc-knowledge/.plan.json --output dc-knowledge/index.md
```

---

## Step 4.6 — Cleanup

Delete intermediate `.json` files for all datasets and buckets processed in Step 3. Keep `.plan.json` (needed for Step 5 report).

```bash
python3 {skill_dir}/scripts/cleanup_json.py --plan dc-knowledge/.plan.json
```

Skip this step if the user explicitly requests to keep JSON files (e.g., for debugging).

---

## Step 5 — Report

```
Knowledge base updated: <N> datasets (<M> updated, <K> unchanged), <B> buckets (<X> scanned, <Y> unchanged).
```

If any buckets have `listing_expired: true`, add:
```
Warning: Listing for <bucket> is expired (last scanned: <date>). Run dc.read_storage("<uri>", update=True) to refresh.
```

Include any warnings collected from Steps 2-3.
