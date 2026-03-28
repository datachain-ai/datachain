---
name: datachain-graph
description: Use when you need to discover, understand, or navigate DataChain datasets or cloud storage buckets. Generates and maintains a knowledge base at datachain/graph/ (JSON data + AI-enriched markdown) so context is always available without running queries or browsing cloud consoles.
triggers:
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
---

You are now loaded with the datachain-graph skill. Maintain a knowledge base at `datachain/graph/`. Both datasets and buckets have a `.json` file (structured data, source of truth) and a `.md` file (AI-generated human-readable summary). Follow the 4-step flow below.

## Critical Rules

1. **Path is `datachain/graph/`** — NOT `.datachain/graph/`. The `.datachain/` directory is the internal database; the knowledge base lives at `datachain/graph/` (no leading dot).
2. **Never pass `update=True`** to `dc.read_storage()` unless the user explicitly asks to refresh the listing.
3. **Prefer DataChain operations** over plain Python for all metadata analysis.
4. **Bounded output** — JSON and markdown files stay small regardless of data size.

---

## Step 1 — Sync

Plan what needs updating. The plan covers datasets, buckets, or both.

**Datasets only** (default):
```bash
python3 {skill_dir}/scripts/plan.py [--studio] > datachain/graph/.plan.json
```

**Buckets only** (user provides URIs):
```bash
python3 {skill_dir}/scripts/plan.py --buckets <uri> [<uri> ...] > datachain/graph/.plan.json
```

**Both** (datasets + buckets):
```bash
python3 {skill_dir}/scripts/plan.py --studio --buckets <uri> [<uri> ...] > datachain/graph/.plan.json
```

- Do **NOT** add `--studio` unless the user explicitly requests it.
- Do **NOT** add `--buckets` unless the user provides bucket URIs.
- If `"up_to_date": true` → print "Graph is up to date." and stop.
- If the output contains `"warnings"` → report them after the update summary.

Then update the index:
```bash
python3 {skill_dir}/scripts/render_index.py --plan datachain/graph/.plan.json --output datachain/graph/index.md
```

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
- Progress is printed to stderr.

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
