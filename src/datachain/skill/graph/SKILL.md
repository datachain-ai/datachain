---
name: datachain-graph
description: Use whenever datasets, cloud storage buckets, or data pipelines are mentioned — creating, saving, querying, listing, exploring, deleting, or processing data in S3, GCS, Azure Blob, or local storage. Also use when running any script that may create datasets as a side effect. Maintains a knowledge base at datachain/graph/ (JSON + markdown). ALWAYS use this skill when the user creates a dataset, saves pipeline output, runs a data script, or references any storage bucket.
triggers:
  # Discovery
  - "what datasets exist"
  - "show me the schema"
  - "list datasets"
  - "datachain graph"
  - "update the graph"
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

You are now loaded with the datachain-graph skill. Maintain a knowledge base at `datachain/graph/`. Both datasets and buckets have a `.json` file (structured data, source of truth) and a `.md` file (AI-generated human-readable summary). Follow the 4-step flow below.

## Critical Rules

1. **Path is `datachain/graph/`** — NOT `.datachain/graph/`. The `.datachain/` directory is the internal database; the knowledge base lives at `datachain/graph/` (no leading dot).
2. **Never pass `update=True`** to `dc.read_storage()` unless the user explicitly asks to refresh the listing.
3. **Prefer DataChain operations** over plain Python for all metadata analysis.
4. **Bounded output** — JSON and markdown files stay small regardless of data size.

---

## Workflow Mode Detection

When loaded, determine the user's intent:

**Mode A — Discovery/Exploration** (e.g., "what datasets exist", "show schema", "explore bucket"):
→ Run Steps 1–4 as normal.

**Mode B — Dataset Creation/Pipeline** (e.g., "create dataset X from ...", "process images and save"):
→ Read `{skill_dir}/../core/SKILL.md` for DataChain SDK rules and patterns.
→ Build and execute the pipeline the user requested, following core skill rules.
→ After the pipeline completes, **always** run Steps 1–4 to update the knowledge base.
→ Report both: pipeline result AND graph update status.

**Mode C — Script Execution** (e.g., user runs an existing script, or agent runs a .py file that touches data):
→ Scripts can create datasets as side effects (e.g., `scan.py` calls `.save()` internally).
→ After ANY data-related script finishes, run Steps 1–4 to detect and record new/changed datasets.
→ This applies even if the script was not written by the agent — always check the DB afterward.

**Mode D — Graph Maintenance** (e.g., "update the graph", "refresh dataset docs"):
→ Run Steps 1–4 as normal.

---

## Step 1 — Sync

Plan what needs updating. The plan auto-discovers both datasets and buckets from the catalog.

```bash
python3 {skill_dir}/scripts/plan.py [--studio] --output datachain/graph/.plan.json
```

- Buckets are auto-discovered from catalog listings (every bucket that `dc.read_storage()` has ever listed). No flag needed.
- Do **NOT** add `--studio` unless the user explicitly requests it.
- If `"up_to_date": true` → print "Graph is up to date." and stop.
- If the output contains `"warnings"` → report them after the update summary.

Review `.plan.json`. Entries with `status` of `"new"` or `"stale"` need processing in Step 2. Entries with `"ok"` are skipped.

---

## Step 2 — Save Data

### Datasets

For each dataset where `status != "ok"` in `plan.datasets[]`:

```bash
python3 {skill_dir}/scripts/dataset_all.py <name> \
  --plan datachain/graph/.plan.json \
  --output datachain/graph/<file_path>.json
```

- `<name>` and `<file_path>` come from the plan's `datasets[]` entries.
- Do **not** modify the output — it is deterministic and complete.

### Buckets

For each bucket where `status != "ok"` in `plan.buckets[]`:

```bash
python3 {skill_dir}/scripts/bucket_scan.py <uri> \
  --output datachain/graph/<file_path>.json
```

- `<uri>` and `<file_path>` come from the plan's `buckets[]` entries.
- The script aggregates metadata (extensions, directories, sizes, timestamps) and samples files.

Run independent `dataset_all.py` and `bucket_scan.py` calls concurrently when multiple items need processing.

Then update the index (must run after data scripts so it can read their JSON output):
```bash
python3 {skill_dir}/scripts/render_index.py --plan datachain/graph/.plan.json --output datachain/graph/index.md
```

---

## Step 3 — Enrich

For each dataset or bucket processed in Step 2, generate a human-readable markdown summary from the JSON data.

### Datasets

1. Read the enrichment prompt at `{skill_dir}/prompts/enrich.md`.
2. For each dataset, read `datachain/graph/<file_path>.json`.
3. Following the prompt, write `datachain/graph/<file_path>.md`.

### Buckets

1. Read the enrichment prompt at `{skill_dir}/prompts/enrich_bucket.md`.
2. For each bucket, read `datachain/graph/<file_path>.json`.
3. Following the prompt, write `datachain/graph/<file_path>.md`.

Skip this step only if the user requests raw output only.

---

## Step 4 — Report

```
Graph updated: <N> datasets (<M> updated, <K> unchanged), <B> buckets (<X> scanned, <Y> unchanged).
```

If any buckets have `listing_expired: true`, add:
```
Warning: Listing for <bucket> is expired (last scanned: <date>). Run dc.read_storage("<uri>", update=True) to refresh.
```

Include any warnings collected from Steps 1-2.
