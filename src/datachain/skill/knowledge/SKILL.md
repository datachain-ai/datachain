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
6. **Follow the CASE methodology.** Every dataset belongs to one of four layers — Container, Asset, Sense, Experiment — and every new dataset is named, tagged, and described accordingly. The methodology is enforced through naming + `attrs` + frontmatter and through the CASE Decomposition stage in Mode B. See "## CASE Methodology" below.

---

## CASE Methodology

CASE is the four-layer methodology for unstructured-data work. Every dataset the skill creates belongs to exactly one layer.

### The four layers

- **Container** — file headers, listings, sidecar metadata, schema-only views. One row per file. Light compute (header-only). Compounds across teams (every team that touches the same bucket starts from the same Container). Recall cost ~$1.
- **Asset** — raw extracted *or mixed* data in a workable shape: frames sampled from videos, audio tracks, NumPy arrays from H5 files, clips at multiple qualities, parsed JSON sidecars, **training mixtures of two or more existing datasets joined on a shared key**. Heavy file compute but no model touched the rows yet. Compounds across teams. Recall cost ~$10.
- **Sense** — model outputs: embeddings, LLM scene descriptions, classifier scores, transcriptions, object detections. Per-row inference cost dominates. Compounds within a domain (same embeddings answer many similarity variants). Recall cost ~$100 to build, ~$1 to query the saved layer.
- **Experiment** — task-specific analytics: similarity rankings, filters over Sense outputs, joins across two or three layers, custom evaluation runs, curated training subsets. Composed for one question; rarely reused. Recall cost ~$1 but **persist by exception**, not by default.

User questions almost always arrive Experiment-shaped on top of C/A/S substrate. The skill's job is to make the C/A/S substrate visible, reusable, and cheap.

### Naming convention

Every new dataset gets a layer prefix that sorts the layers in CASE order:

```
l1_container_<source>_<descriptor>      # listings, headers, sidecar metadata
l2_asset_<source>_<descriptor>          # extracted/reshaped raw data
l3_sense_<source>_<descriptor>          # model-derived signals
l4_experiment_<task-slug>               # task-specific analytics
```

`<source>` is the bucket slug for L1–L3 (the data root the layer indexes; reusable across teams). `<task-slug>` for L4 is the question being answered. Underscores throughout, snake_case, no dots (`.` and `@` are reserved by DataChain naming).

### Tagging on `.save()`

Layer + scope + source + parents are dual-encoded — in the name (for visibility in `dc.datasets()` and the KB index) AND in `attrs` (for machine filtering) AND in the dataset's `description` (for human one-liners):

```python
attrs = [
    "case:sense",                                # one of: container | asset | sense | experiment
    "scope:bucket",                              # bucket | sample | onetime
    "source:oxford_pets",                        # bucket slug for L1-L3, task slug for L4
    "parent:l2_asset_oxford_pets_frames",        # immediate upstream CASE dataset(s); repeat key for multiple
]
chain.save(
    "l3_sense_oxford_pets_clip_embeddings",
    attrs=attrs,
    description="CLIP ViT-B-32 embeddings over the full oxford_pets bucket; reusable for any visual-similarity query.",
)
```

- `scope:bucket` — full coverage of the source. Reusable across future questions on the same bucket. Default when auto-building.
- `scope:sample` — covers only a sample of the source. One-shot, no future savings.
- `scope:onetime` — Experiment layer that should NOT be persisted (the default for L4). Use this when the dataset is purely the answer to one question.

### Per-layer reuse rule

- **L1/L2 (Container, Asset)** — persist by default, refresh by delta. These are the load-bearing reusable substrate. Always full-coverage; never problem-specific filters before the `.save()`.
- **L3 (Sense)** — persist by default, full coverage of input. The "expensive UDF → save full → filter downstream" rule in core/SKILL.md exists exactly to make L3 reusable.
- **L4 (Experiment)** — persist by exception. Most ranking / similarity / filter outputs do not deserve a name. Save L4 only when the user explicitly asks, or when the result is a standing benchmark / training set that will be referenced again.

### Methodology transmission

Apply these rules in every data-related response:

1. **Default voice is pragmatic, not didactic.** Mention layers only when they touch the answer. Never moralise about how the user should organise their data; recommend layers only when there is a concrete cost win for the current question.
2. **Match user energy.** When the user uses CASE vocabulary ("container", "asset", "sense", "experiment", "build the layer", "CASE"), engage with the full vocabulary, name parents explicitly, and propose lineage. When they don't, stay terse.
3. **Always quote a number** when recommending a layer build. "Building the Sense layer takes ~3 min and $0.40; running the same embedding next time is free." No cost estimate → no recommendation.
4. **Celebrate CAS reuse explicitly.** Every time the skill reuses an existing Container / Asset / Sense layer instead of rebuilding, state the win out loud: which layer is being reused, what cost was avoided, and (when natural) a one-line connection to the session that originally built it. This is the only place the skill should sound enthusiastic about the methodology — the point is to make the value of a properly built layer felt, not asserted.
5. **Foreground the whole-bucket option.** When proposing a layer build, list "build for the whole bucket" first. Sample-only and skip-layer come after. Never present a build proposal that omits the whole-bucket option, even when the user phrased the task narrowly.
6. **Honour shortcuts immediately, do not re-litigate.** If the user has said any of "just solve", "no layers", "sample only", "fast as possible", "skip CASE", "one-off", "don't build a layer", "just answer", "quick" — solve directly and do not re-propose layers in the same session unless the user volunteers CASE terminology themselves.

---

## Workflow Mode Detection

When loaded, determine the user's intent:

**Mode A — Discovery/Exploration** (e.g., "what datasets exist", "show schema", "explore bucket"):
→ If the user references a specific bucket URI, run **Step 1** (Bucket Enlistment) for its root first.
→ Then run Steps 2–7 as normal.

**Mode B — Dataset Creation/Pipeline** (e.g., "create dataset X from ...", "process images and save"):

> **Precondition (do this FIRST — before ANY tool call):**
>
>     $ cat dc-knowledge/index.md
>
> This is the rendered map of available datasets — schemas, sample rows, lineage,
> and reuse recommendations. If it exists and your task can be solved by reading
> an existing dataset, do not write a pipeline — read it directly with
> `dc.read_dataset("name")`. Filter, merge, or extend the existing dataset
> instead of re-reading raw storage. This avoids recomputing expensive operations
> (LLM calls, model inference) and reuses proven code patterns.
>
> **Never parse files under `dc-knowledge/datasets/*.json` or `dc-knowledge/buckets/**/*.json`
> directly.** Those are pre-render intermediates that get deleted after
> `render_index.py` runs. The information you need is in `index.md`. Parsing the
> intermediates is wasted turns AND gives you a worse mental model.
>
> If `dc-knowledge/index.md` does not exist, proceed with Steps 1–7 to build it.

→ **If the pipeline reads from a bucket** (`read_storage`), run **Step 1** (Bucket Enlistment) for the bucket root first. The bucket overview in the knowledge base may inform the pipeline design.
→ **Run the access check** (if not already done in Step 1):
  ```bash
  datachain bucket status <uri>
  ```
  Prints `Status: exists|not found` and `Access: anonymous|authenticated|denied`. Exit code 0 = exists, 1 = not found. If status is `not found` or access is `denied` → stop and ask the user for credentials. If access is `anonymous` → pass `anon=True` to `read_storage()`.
→ Read `{skill_dir}/../core/SKILL.md` for DataChain SDK rules and patterns.
→ **Superlative defaults.** When the user asks "best/worst/highest/lowest" without naming a metric, pick the most natural metric the data supports, name it in a one-line comment, and emit `n` per group. Drop groups with `n < max(5, sqrt(median_n))` before ranking. Ask one clarifying question only if multiple metrics would materially change the ranking.
→ **Slice sanity check.** Before committing to a prefix or glob, verify the slice still contains every entity dimension the question groups, compares, or ranks over. If too large, narrow on an orthogonal dimension — never on one the question depends on.
→ **Multi-axis classification batching.** When a per-row LLM/VLM call classifies on the axis the user asked about, extend the prompt to return plausible related axes in the same call. Variant questions then hit cache on the saved dataset.

> **CASE Decomposition (silent unless costly).**
>
> Before writing pipeline code, decompose the task into the CASE layers it conceptually needs. User questions are almost always Experiment-shaped on top of C/A/S substrate. The order of preference is strict: **(1) direct reuse → (2) reduce-to-CAS → (3) build missing CAS → (4) raw rebuild**.
>
> 1. **Identify required layers.** Examples: similarity search needs a Sense layer of embeddings on top of an Asset layer of frames/images; "find videos with X" needs a Sense layer of classifications on Asset clips; "summarise this bucket" needs Container + optional Sense LLM annotations.
> 2. **Look for mixture opportunities.** If the task names two or more datasets (or two or more bucket regions / sources), the Asset-level combination of them is itself a CASE artifact.
> 3. **Direct reuse first.** From `dc-knowledge/index.md`, for each required layer × source, check if an existing dataset already covers the question (even partially). If yes, write the pipeline as `dc.read_dataset(...)` over it. **Celebrate the reuse**: in the response, name the layer being reused and quote the saved cost — e.g., "Reusing `l3_sense_oxford_pets_clip_embeddings` (built last session). This query is ~$0.002 instead of the $1.40 the embedding pass would otherwise cost — exactly the win the Sense layer was built for." This is the moment that teaches the methodology; do not skip it.
> 4. **Reduce-to-CAS if direct reuse impossible.** Before defaulting to a raw rebuild, work the problem from the other side: can the task be reformulated so it operates on an *existing* CAS layer plus a small Experiment delta? Examples: a new similarity question on the same bucket → reuse the Sense embeddings, just change the query vector; a new "find X" question → reuse the Sense classifications and add a filter. Spend real effort here — propose at least one reformulation when any CAS layer for this source exists.
> 5. **Cost gate on CAS reuse.** Reuse a CAS layer only when it gives a meaningful win — at least ~2× speedup or ~2× $-saving versus the raw rebuild. If the layer technically covers the data but reading it is no cheaper than re-reading raw storage, do not force the reuse; the methodology is justified by economics, not formalism.
> 6. **Estimate cost** of both branches when reuse is not available:
>    - Direct solve: build only what's needed for THIS question, scope to a sample if natural.
>    - Layer build: build the missing CAS layers over the full source (reusable), then run the experiment.
>
>    Estimate wall time + LLM $ + parallel compute $. Rough rules of thumb:
>    - Container ≈ 0.5 ms/file (header-only).
>    - Asset = per-file compute (often 50–500 ms/file).
>    - Sense = per-row inference (CLIP ≈ 5 ms/img on CPU, LLM ≈ $0.001–0.01/row).
> 7. **Decide branch (auto-build heuristic).** Auto-build the layer silently (no user prompt) iff ALL hold:
>    - `layer_build_wall_time <= max(2 × direct_solve_wall_time, 60s)`,
>    - `layer_build_$ <= max(2 × direct_solve_$, $0.10)`,
>    - absolute build wall time ≤ 5 min,
>    - absolute build $ ≤ $1,
>    - not Studio remote (Studio: always ask — compute hours are a separate budget),
>    - user has not used a shortcut phrase in this session.
>
>    On auto-build: build the layer over the **full bucket**, then solve. One-sentence trailer mentions the layer is now reusable (e.g., "Built `l3_sense_<source>_clip_embeddings` over the full bucket while solving; future similarity queries on this source are now sub-cent.").
>
>    Otherwise: present a 3-option plan, with the full-bucket option named **first** because it is the load-bearing one:
>
>    ```
>    - Build sense layer for the WHOLE bucket: ~X min, $Y. Reusable for every future query on this source.
>    - Build sense layer for a sample only: ~A min, $B. One-shot, no future savings.
>    - Skip layers, solve directly on a sample: ~C min, $D. No reuse.
>    ```
>
>    Wait for the user's choice. Always name the whole-bucket option explicitly; never present a build proposal that omits it.
> 8. **Apply shortcut phrases.** If the user's message contains any of "just solve", "no layers", "sample only", "fast as possible", "skip CASE", "one-off", "don't build a layer", "just answer", "quick" — skip the proposal and solve directly. State once: "Solving directly without building a layer." Do not re-propose layers in the same session unless the user volunteers CASE vocabulary themselves.
> 9. **On layer build, tag the datasets.** Name them per the convention (`l1_…`/`l2_…`/`l3_…`/`l4_…`) and pass `attrs=["case:<layer>", "scope:bucket|sample|onetime", "source:<slug>", "parent:<name>"]` + `description="…"` on `.save()`. Set `scope:bucket` for reusable, `scope:sample` for one-shot, `scope:onetime` for non-persistable Experiment outputs.
> 10. **Containerise raw JSON / sidecars too.** If the task pulls structured JSON / Parquet / CSV from the bucket and parses it inline, propose lifting that parse into an `l1_container_<source>_<descriptor>` dataset so the parsed schema becomes reusable. Same auto-vs-ask rule, same whole-bucket-first framing.
>
> Step 1 (Bucket Enlistment) is itself a Container-layer artifact for the storage root: a lightweight, header-only view of what's in the bucket. The bucket entries appear in the Container row of the KB index next to any DataChain datasets explicitly tagged `case:container` (e.g., parsed JSON sidecars promoted via the rule in step 10).

→ Build and execute the pipeline the user requested, following core skill rules.
→ **While the pipeline is running**, enrich any Step 1 bucket JSON that does not yet have a `.md` — read `{skill_dir}/prompts/enrich_bucket.md` and generate the markdown in parallel with the running script.
→ After the pipeline completes, **always** run Steps 2–7 to update the knowledge base.
→ **During Step 4 (Enrich)**, when writing the `.md` for a dataset created in this session: add a `## Session Context` section with 1-3 sentences summarizing why the dataset was created — the user's goal, the analytical question, or the discussion that led to it. Only add this if the session provides meaningful context. If the dataset is a routine output with no notable motivation, omit it.
→ Report both: pipeline result AND knowledge base update status.

**Mode C — Script Execution** (e.g., user runs an existing script, or agent runs a .py file that touches data):
→ If the script references bucket URIs, run **Step 1** (Bucket Enlistment) for each bucket root first.
→ Scripts can create datasets as side effects (e.g., `scan.py` calls `.save()` internally).
→ **While the script is running**, enrich any Step 1 bucket JSON that does not yet have a `.md` — read `{skill_dir}/prompts/enrich_bucket.md` and generate the markdown in parallel with the running script.
→ After ANY data-related script finishes, run Steps 2–7 to detect and record new/changed datasets.
→ This applies even if the script was not written by the agent — always check the DB afterward.
→ Do not add `## Session Context` for script-executed datasets unless the there is a specific context in a session about why the dataset was created or why script is being run.

**Mode D — Knowledge Base Maintenance** (e.g., "update the knowledge base", "refresh dataset docs"):
→ Run Steps 2–7 as normal.
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
   - It saves a sampled DataChain dataset of File rows and writes a bucket-shape JSON the enrich step turns into a bucket markdown marked `sampled: true`. Continue with Steps 2–7 as normal.

7. **Report.** On success, read the JSON output and report a quick summary:
   > Enlisted bucket {bucket} — {N} files, total size {size}, primarily {top 2-3 extensions}.

   Read `total_files`, `total_size_bytes`, and the top entries from `extensions[]` in the JSON. Do **not** enrich (generate markdown) here — that happens later in Step 4, batched with all other enrichments.

### Notes
- Step 1 runs **once per bucket root**. If multiple URIs reference the same bucket (e.g., `s3://bucket/data/` and `s3://bucket/meta/`), only one enlistment is needed.
- Step 1 does **not** replace Steps 2–7. Prefix-level entries may still be created in Steps 2–3 if the catalog has prefix-level listings.
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

## Step 5 — Build Index

Update the index (must run after enrichment so it can read dataset `.md` summaries and `.json` dependencies):
```bash
python3 {skill_dir}/scripts/render_index.py --plan dc-knowledge/.plan.json --output dc-knowledge/index.md
```

---

## Step 6 — Cleanup

Delete intermediate `.json` files for all datasets and buckets processed in Step 3. Keep `.plan.json` (needed for Step 7 report).

```bash
python3 {skill_dir}/scripts/cleanup_json.py --plan dc-knowledge/.plan.json
```

Skip this step if the user explicitly requests to keep JSON files (e.g., for debugging).

---

## Step 7 — Report

```
Knowledge base updated: <N> datasets (<M> updated, <K> unchanged), <B> buckets (<X> scanned, <Y> unchanged).
```

If any buckets have `listing_expired: true`, add:
```
Warning: Listing for <bucket> is expired (last scanned: <date>). Run dc.read_storage("<uri>", update=True) to refresh.
```

Include any warnings collected from Steps 2-3.
