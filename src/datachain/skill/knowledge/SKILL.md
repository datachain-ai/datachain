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
2. **Never pass `update=True`** to `dc.read_storage()` in Task or exploration code unless the user explicitly asks to refresh the listing. L1/L2/L3 build scripts are the exception — they always pass `update=True, delta=True` per step 12 (running a build script IS the refresh).
3. **Prefer DataChain operations** over plain Python for all metadata analysis.
4. **Bounded output** — JSON and markdown files stay small regardless of data size.
5. **Stop on auth/connection errors** — `bucket_scan.py` runs a fast access check before scanning (uses cloud SDKs, no DC listing). If it exits with an error JSON on stderr, **stop immediately** and show the error to the user. Do not retry with different regions, credential profiles, or endpoint variations — ask the user for the missing credentials or configuration.
6. **Follow the CAST methodology.** Every dataset belongs to one of four layers — Container, Asset, Sense, Task — and every new dataset is named, tagged, and described accordingly. The methodology is enforced through naming + `attrs` + frontmatter and through the CAST Decomposition stage in Mode B. See "## CAST Methodology" below.
7. **NEVER bypass DataChain for results.** UDF outputs (embeddings, LLM, classification, file extraction) MUST land via `.save("name", attrs=[...], description=...)`. Writing to local `.json` / `.csv` / `.parquet` files via `open()`, `json.dump`, `pandas.to_csv`, etc., bypasses lineage and the KB — the most common regression.
8. **C/A/S substrate is mandatory.** Any UDF-bearing task builds and saves **at least one** Container / Asset / Sense layer. "At least one" means exactly that — Sense alone satisfies the rule. Building a Container or Asset dataset *by reflex*, when neither adds information, is the regression (see CAST Decomposition step 1 for the Container justification test). "Persist by exception" applies ONLY to the final Task ranking, never to substrate that does meaningfully reduce future cost. "One-off question" / "small dataset" / "in-memory is enough" are not legal opt-outs.
9. **One script per stage.** Multi-stage pipelines (2+ named datasets) decompose into separate scripts named after the dataset each produces. Never one monolith with multiple `.save()` calls. See `core/SKILL.md` "Code-level decomposition" for the canonical pattern.
10. **One script, one `.save()`. Never manually batch or shard.** A long L2/L3 build is ONE script with ONE terminal `.save()` over the whole source. DataChain checkpoints UDF progress internally; rerunning the same script after a kill resumes from where it stopped. **Do NOT** split the source into batches and save them as `name_b00`/`name_b01`/…, do NOT union shards together, do NOT loop `.save()` calls. Manual sharding is the regression this rule used to recommend; DataChain's checkpoint recovery made that obsolete.

---

## CAST Methodology

CAST is the four-layer pattern the skill suggests for organising unstructured-data work — Container, Asset, Sense, Task. It is a recommendation backed by recall-economics math, not a mandate; users may decline at any time via a shortcut phrase (see Methodology transmission rule 5). When the skill does create datasets, each one belongs to exactly one CAST layer.

### The four layers

- **Container** — A typed, queryable index of what each file IS without decoding its full content: paths, sizes, format headers (e.g., MP4 codec/duration, HDF5 attributes, DICOM tags, Parquet schema), sidecar metadata (JSON/XML), and external-DB joins. *Build it when the task needs to answer "what files are here, what format/metadata do they carry, which match this predicate" without paying decode cost.* One row per source file. Light compute (header bytes only). Recall cost ~$1.
- **Asset** — Raw extracted or mixed data in a workable shape: video frames, audio tracks, decoded arrays, or a training mixture of multiple source datasets joined on a shared key. *Build it when downstream work needs decoded or combined content rather than file pointers — annotation, training, multi-source analytics.* Heavy file compute, no model has touched the rows yet. Recall cost ~$10.
- **Sense** — What a model said about the data: embeddings, LLM annotations, classifier scores, transcriptions, detections. *Build it when the task needs semantic or learned signals — similarity search, classification, captioning, retrieval, RAG.* Per-row model inference dominates the cost (LLM ~$0.001–0.01/row, CLIP ~ms/image). Pay once at build time, query at warehouse speed forever. Recall cost ~$100 build, ~$1 query.
- **Task** — A task-specific composition on top of C/A/S: a similarity ranking, filter over Sense outputs, join across layers, eval set, curated training subset. *Build it (and usually do NOT save it) when answering one specific question.* **Persist by exception** — only when the result becomes a standing benchmark or shared training set. Recall cost ~$1.

Most user questions arrive Task-shaped on top of C/A/S substrate. The skill's job is to make the C/A/S substrate visible, reusable, and cheap. See Critical Rules 7–9 above for the enforcement bottom line.

### Naming convention

Every new dataset gets a layer prefix that sorts the layers in CAST order:

```
l1_container_<source>_<descriptor>      # listings, headers, sidecar metadata
l2_asset_<source>_<descriptor>          # extracted/reshaped raw data
l3_sense_<source>_<descriptor>          # model-derived signals
<descriptor>                            # Task outputs (and anything not C/A/S) — no prefix
```

`<source>` is the bucket slug for L1–L3 (the data root the layer indexes; reusable across teams). Task-layer datasets carry no prefix; the name describes the question (`products_similar_to_query`, `recsys_eval_runs`, `cik_text_stats`), and layer membership is recorded only via `attrs=["cast:task", …]` and the resulting `cast_layer: task` frontmatter. Underscores throughout, snake_case, no dots (`.` and `@` are reserved by DataChain naming).

**Source slug — strip uninformative prefixes.** Strip these from the bucket name (case-insensitive) before using it as the source slug: `datachain-`, `dc-`, `iterative-`, `gs-`, `s3-`, `aws-`, `gcp-`, `azure-`, `public-`, `private-`. Then replace `-` with `_`. Example: `datachain-starss23` → `starss23`.

**Preset/config → `attrs`, never in the name.** Names describe content (frames, embeddings, transcriptions), not parameters. Right: `l2_asset_starss23_frames` + `attrs=["cast:asset", "source:starss23", "preset:1fps_640", "format:jpeg"]`. The agent does not append preset suffixes to names by default.

**Cap names at 40 characters.** If a meaningful name would exceed, shorten the descriptor (`frames` not `video_frames` when source implies video).

**Task names are JUST the question** — no source slug, no layer prefix (`cik_text_stats`, `products_similar_to_query`). Add `_<source_slug>` only when the same question runs on two different sources and disambiguation is genuinely needed.

### Tagging on `.save()`

Layer + scope + source + parents are dual-encoded — in the name (for visibility in `dc.datasets()` and the KB index) AND in `attrs` (for machine filtering) AND in the dataset's `description` (for human one-liners):

```python
attrs = [
    "cast:sense",                                # one of: container | asset | sense | task
    "scope:bucket",                              # bucket | sample | onetime
    "source:product_catalog",                    # bucket slug for L1-L3, task slug for Task
    # DataChain tracks lineage automatically — no need for parent: attrs.
    # Direct dependencies live in DatasetDependency; query via
    # Catalog.get_dataset_dependencies(name, version).
    "model:clip_vit_b32@open_clip-2.24",         # build-signature: model id @ version (for refresh-signature check, step 12)
    "preset:cpu_fp32",                           # build-signature: preset/config tag
]
chain.save(
    "l3_sense_product_catalog_clip_embeddings",
    attrs=attrs,
    description="CLIP ViT-B-32 embeddings over the full product-catalog bucket; reusable for any visual-similarity query.",
)
```

- `scope:bucket` — full coverage of the **bucket root** (the storage root such as `s3://my-bucket/`). **Default for any auto-build, even when the user's prompt named a subdirectory.** Reusable across every team and every future query on the same bucket.
- `scope:directory` — coverage of a specific subdirectory under the bucket. Set ONLY when the user explicitly opted into a narrower scope via the scope-and-preset dialogue. The `source` slug includes the directory path so different subdirectories of the same bucket do not collide (e.g., `source:my_bucket__images_subset`). Never auto-build at directory scope without asking.
- `scope:sample` — covers only a sample of the source. One-shot, no future savings.
- `scope:onetime` — Task layer that should NOT be persisted (the default for Task). Use this when the dataset is purely the answer to one question.

**`.save()` checklist for L1/L2/L3.** Before calling `.save()`, verify every item:

- `attrs` includes `cast:<layer>`, `scope:<scope>`, `source:<slug>` — slug stripped per the prefix-strip rule above (e.g. `starss23`, not `datachain_starss23`).
- Do NOT add `parent:<dataset_name>` to attrs. Lineage is tracked natively by DataChain — when you do `dc.read_dataset("foo").map(...).save("bar")`, the catalog records `bar`'s dependency on `foo` automatically (queryable via `Catalog.get_dataset_dependencies`). Duplicating lineage in attrs creates two sources of truth that drift.
- `attrs` includes `model:<id>@<version>` and `preset:<name>` whenever a model/encoding was used. These are the build signature read by step 12 on refresh.
- `description="…"` set to a one-line human summary.
- The corresponding `read_storage` includes `update=True, delta=True` (defaults for the rest — see step 12).
- The Pydantic output type carries the operation's full output, not the current question's projection (see "Saved schema captures what was produced" below). Derived columns — counts, booleans, top-k labels, aggregates of the actual output — live in Task, not on the substrate row.

### Per-layer reuse rule

- **L1/L2 (Container, Asset)** — persist by default, refresh by delta. These are the load-bearing reusable substrate. Always full-coverage; never problem-specific filters before the `.save()`.
- **L3 (Sense)** — persist by default, full coverage of input. The "expensive UDF → save full → filter downstream" rule in core/SKILL.md exists exactly to make L3 reusable.
- **L4 (Task)** — persist by exception. Most ranking / similarity / filter outputs do not deserve a name. Save only when the user explicitly asks, or when the result is a standing benchmark / training set referenced again. (Critical Rule 8 already pins down that this exception applies ONLY to the final Task output, never to the C/A/S substrate underneath.)

**Saved schema captures what was produced, not what's currently needed.** Every L1/L2/L3 row holds everything the upstream operation produced — its substantive output plus per-call telemetry. Don't trim to the columns the current question reads. Three categories of upstream operation, with what to capture from each:

- **Decode / parse / read** (file headers, schemas, sidecars, manifests, archive indexes): every field the parser surfaces, not just the one the current question filters on.
- **Materialize** (decoded payloads, derived files, segments): the payload as a typed reference (`dc.File` or similar), plus auxiliary metadata about how it was produced (preset, encoding params, source/timestamp link).
- **Inference / external call** (any model call, any paid API): the operation's complete structured output, plus per-call telemetry — operation id + version, latency, finish/status code, and units consumed (e.g. tokens for LLMs, requests for rate-limited APIs).

The Task layer projects to the asked question; per-question cost is computed there. Operation cost dominates schema cost — never trade dollars or compute for bytes. For free-form LLM/VLM prompts, see Multi-axis classification batching in CAST Decomposition (same principle applied at the prompt-construction step).

**No projection columns alongside the full output.** A projection column is one *derived* from the operation's output by filtering, counting, aggregating, comparing, thresholding, or top-k-picking — not a field the operation itself returned. Saving a projection alongside the full output bakes the current question into the substrate. Litmus: *"Is this column something the operation directly returned, or did I derive it from the actual output?"* Derived → it belongs in an Task `.save()` or a downstream `.read_dataset().filter().group_by()` query, not on the L3 row. The substrate row holds full output + per-call telemetry — that's it.

### Methodology transmission

Apply these rules in every data-related response:

1. **Suggest CAST, do not push it.** When telling the user about layers, the tone is "here is the cheap way to do this and why" — never "you must follow the methodology". Mention layers only when they touch the answer. Recommend a layer build only when there is a concrete cost win for the current question (always quote a number, see rule 3). Never moralise about how the user should organise their data. If the user declines a layer suggestion, solve directly — that's their call, and re-proposing the same layer later in the session is pushing.
2. **Match user energy.** When the user uses CAST vocabulary ("container", "asset", "sense", "task", "build the layer", "CAST"), engage with the full vocabulary, name parents explicitly, and propose lineage. When they don't, stay terse.
3. **Always quote a number** when recommending a layer build. "Building the Sense layer takes ~3 min and $0.40; running the same embedding next time is free." No cost estimate → no recommendation.
4. **Celebrate CAS reuse explicitly.** Every time the skill reuses an existing Container / Asset / Sense layer instead of rebuilding, state the win out loud: which layer is being reused, what cost was avoided, and (when natural) a one-line connection to the session that originally built it. This is the only place the skill should sound enthusiastic about the methodology — the point is to make the value of a properly built layer felt, not asserted.
5. **Honour shortcuts immediately, do not re-litigate.** If the user has said any of "just solve", "no layers", "sample only", "fast as possible", "skip CAST", "one-off", "don't build a layer", "just answer", "quick" — solve directly and do not re-propose layers in the same session unless the user volunteers CAST terminology themselves.

---

## Common gotchas in UDF scripts

These are the failure modes that have actually consumed sessions. Apply them whenever writing a `.map`/`.gen`/`.agg` script.

- **`parallel=N` vs `workers=N`.** `parallel=N` is local multiprocessing (works anywhere). `workers=N` is distributed UDF processing — Studio-only. Guard with `dc.is_studio()`, never with env vars: `chain = chain.settings(parallel=N); if dc.is_studio(): chain = chain.settings(workers=N)`. If you see `DATACHAIN_DISTRIBUTED import path is required` locally, a script is using `workers=` without the guard — fix the script.
- **No `from __future__ import annotations` in UDF modules.** It stringifies type hints; DataChain's signal-schema resolution then rejects the string-vs-class mismatch (`SignalResolvingError: types mismatch`). Use plain runtime annotations.
- **Type the UDF return precisely.** `Iterator[object]`/`Iterator[Any]`/bare `dict` fail schema resolution. Return `Iterator[dc.VideoFrame]`/`Iterator[dc.VideoFragment]`, a Pydantic `BaseModel`, or a primitive.
- **Generators aren't subscriptable.** `file.get_frames(step=…)` returns a generator; `frames[:2]` raises `TypeError`. Use `enumerate` + `break`, or `list(...)` only when the result is genuinely small.
- **`datachain.__version__` does not exist.** The module doesn't export `__version__` (it's missing from `__init__.py`). Use `from importlib.metadata import version; version("datachain")` if you need the installed version. Same pattern for any package whose `__version__` attribute isn't set; `importlib.metadata` always works for installed packages.

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

> **CAST Decomposition (silent unless costly).**
>
> Before writing pipeline code, decompose the task into the CAST layers it conceptually needs. User questions are almost always Task-shaped on top of C/A/S substrate. The order of preference is strict: **(1) direct reuse → (2) reduce-to-CAS → (3) build missing CAS → (4) raw rebuild**.
>
> 1. **Identify required layers.** The output of this step is a non-empty list of CAS layers the task depends on, NOT a decision to skip layers. If the task involves a UDF (embedding model, LLM call, classifier, file decode), at least one CAS layer is required and must be saved. Examples: similarity search → Sense layer of embeddings on top of an Asset layer of frames/images; "find videos with X" → Sense layer of classifications (and possibly thin Asset of materialized frames); "summarise this bucket" → Container of header metadata + optional Sense LLM annotations; "extract frames from videos" → Asset layer of frames on top of bucket scan. There is no "no layer needed" branch for UDF tasks; the only decisions are *which* layers and *what scope*.
>
>    **Container layer = partial-read or metadata-only work.** Justified when per-row work reads only a bounded prefix of each file (headers, schema, EXIF, tags, footer) or a sidecar, not the full file body. **Any partial read without a full-file read is a sign of L1.** Header/schema reads are native in PIL Image (dims/mode/EXIF), pyarrow Parquet (schema + footer stats), h5py (file structure + dataset attrs), `dc.VideoFile.get_info()` (codec/fps/duration), pydicom (tags), `file.read(N)` with bounded N. Sidecar JSON/XML/CSV/YAML next to primary files, and cross-bucket joins on parsed identifiers, are also Container. **Full file decode → L2 Asset, not L1** (decoded pixels, audio samples, parsed CSV/Parquet rows, video frames are payload, not metadata). **Filter-only Container datasets are forbidden** — `.filter(glob)` reads zero bytes; inline into `read_storage` glob or an L2/L3 `.filter()`. The bucket scan in `dc-knowledge/buckets/` already covers "what files are here". Critical Rule 8's "at least one" is satisfied by Sense alone.
> 2. **Look for mixture opportunities.** If the task names two or more datasets (or two or more bucket regions / sources), the Asset-level combination of them is itself a CAST artifact.
> 3. **Direct reuse first.** From `dc-knowledge/index.md`, for each required layer × source, check if an existing dataset already covers the question (even partially). If yes, write the pipeline as `dc.read_dataset(...)` over it. **Celebrate the reuse**: in the response, name the layer being reused and quote the saved cost — e.g., "Reusing `l3_sense_product_catalog_clip_embeddings` (built last session). This query is ~$0.002 instead of the $1.40 the embedding pass would otherwise cost — exactly the win the Sense layer was built for." This is the moment that teaches the methodology; do not skip it.
> 4. **Reduce-to-CAS if direct reuse impossible.** Before defaulting to a raw rebuild, work the problem from the other side: can the task be reformulated so it operates on an *existing* CAS layer plus a small Task delta? Examples: a new similarity question on the same bucket → reuse the Sense embeddings, just change the query vector; a new "find X" question → reuse the Sense classifications and add a filter. Spend real effort here — propose at least one reformulation when any CAS layer for this source exists.
> 5. **Cost gate on CAS reuse.** Reuse a CAS layer only when it gives a meaningful win — at least ~2× speedup or ~2× $-saving versus the raw rebuild. If the layer technically covers the data but reading it is no cheaper than re-reading raw storage, do not force the reuse; the methodology is justified by economics, not formalism.
> 5.5. **Derive task minimum BEFORE the calibration run.** For decode-heavy sources (video/audio/large H5/NIfTI/multi-page PDF), name the minimum fidelity the task requires across three axes:
>     - **Temporal sampling:** shortest event the answer depends on. People on screen → 1 fps; brief impacts/flashes → ≥5 fps; one scene/video → 1 frame.
>     - **Spatial resolution:** smallest visual detail. Object detection of people/cars → 640px; OCR / small text → full resolution.
>     - **Encoding:** audio full-rate (transcription) vs resampled (speech-vs-silence); PDF OCR vs text-layer.
>
>     This minimum is the FLOOR for any thin-Asset preset in step 7 and what the calibration measures against. Never calibrate at coarser sampling than the task requires. If genuinely ambiguous, ask one targeted question — do not guess.
>
> 6. **Estimate cost** by measurement, never guess. The number going into the 7c gate MUST be calibration-derived. Two branches when reuse is unavailable: direct solve (this question only) vs layer build (full source, reusable).
>
>    **6a. Lead with sizes.** Before the calibration run, quote bucket footprint from the scan (total GB, files, avg size, top extensions) and a size-derived I/O baseline: `total_GB / bandwidth` where bandwidth ≈ 50–150 MB/s GCS→Mac, 80–200 S3→local, 500+ same-region cloud. The calibration refines; the size baseline anchors.
>
>    **6b. Calibration procedure** (mandatory for any `.map`/`.gen` over >50 source items, or any UDF that decodes/downloads/calls a model). Pick `N = min(50, max(5, ⌈total/100⌉))`. Run two calibration runs back-to-back sharing one decode-once `.map`, each ending in `.persist()` (never `.save()`):
>
>    1. **no-Asset calibration:** emit only the aggregate/Sense output.
>    2. **thin-Asset calibration:** same `.map`, ALSO encode the materialized payload at task-minimum fidelity and return `sum(len(bytes))` as a column. **Do NOT upload during calibration** — encoding suffices to measure compute + size; upload happens only at full-run time.
>
>    60–120 s timeout per run. If neither finishes, auto-build is off → go to 7d dialogue. Record wall seconds, GB read, rows out, plus encoded-bytes sum for thin-Asset.
>
>    Extrapolate: `wall_full = (wall_sample/N) × total × 1.5`; `asset_full_bytes = (sample_bytes/N) × total × 1.5`. Cost in $ from the recall-economics tier (lines 72–75) unless calibration disagrees by >3×.
>
>    **Sanity floor:** estimate disagreeing with the recall-economics tier by >100× → re-calibrate, do not paper over.
>
>    **Fallback rules of thumb** (use only when calibration is impossible — empty/unreachable source). Representative cost ranges per operation type:
>    - Container (header-only parse): ~0.5 ms/file.
>    - Asset (full-file decode, materialize): 50–500 ms/file; for heavy-source codecs (video/audio/large blobs) budget per source file, not per unit.
>    - Sense — paid API (LLM/VLM, hosted embeddings): ~$0.001–0.01/row plus latency; cost dominates, batch when prompt allows.
>    - Sense — local model on CPU: ms-to-seconds/item depending on size (small classifiers/embeddings: 5–50 ms; mid-size detection/segmentation: 50–500 ms; ASR ≈ 0.1–0.5× realtime). Add source-read/decode I/O for non-cached inputs.
>    - Sense — local model on GPU: ~10–100× faster than CPU baselines above (when GPU is available; check `dc.is_studio()` for distributed Studio context).
>
>    **L2 shape — three valid options for heavy-decode sources** (video frames, audio segments, PDF pages, archive entries, multi-channel sensor data, etc.). Pick deliberately based on selection criteria; none is forbidden.
>
>    1. **Pointer-row L2.** Emit `Iterator[dc.VideoFrame]` (or `Iterator[dc.VideoFragment]` for audio) from `.gen()` walking `file.get_frames(step=N)`. Saved row = `{video: VideoFile, frame: int, timestamp: float}` — pure metadata, bytes per row. Downstream `frame.get_np()` calls `video.open()` which streams from storage with DataChain's local caching (see `lib/video.py:398`). **No data duplication, no re-download per frame.** Per-frame cost is local CPU + cached disk I/O. Simplest shape; the right default for in-DataChain consumption (subsequent queries via `read_dataset` + `.map`/`.filter`/`.gen`).
>    2. **Materialized thumbnail L2.** Emit rows with `frame: dc.File` pointing at written JPEGs / segment files in storage. Choose when downstream needs files-on-disk for **non-DataChain interop** — annotation UIs, training pipelines that consume files directly, browsing in the storage browser. Format/destination/schema details below apply to this shape.
>    3. **Fused decode-once L3 (no separate L2).** `.gen(VideoFile → Iterator[Detection])` reads `read_storage` directly, decodes the source once sequentially, emits per-detection (or per-frame) rows. Choose when there's no Asset reuse case — the Sense layer is the only consumer; no intermediate substrate needed.
>
>    A prior version of this rule flagged shape (1) as "forbidden / re-decode trap". That was wrong (over-calibrated against a misdiagnosed earlier trace). All three shapes are valid; pick by selection criteria.
>
>    **Shape (2) — materialized thumbnail — details follow.** Skip this block for shapes (1) and (3).
>
>    - **Storage shape, by format:**
>      - **Standard containers** (JPEG/PNG/WEBP/GIF/BMP/WAV/MP3/FLAC/OGG/MP4/MOV/WEBM) → file in storage + `dc.File` pointer on the row **by default**. Bytes column only on explicit user request ("store inline" / "blob column").
>      - **Custom binary** (numpy, embedding tensors, intermediate features) → bytes column on row.
>      - **Text/JSON** (transcripts, summaries) → string column on row.
>    - **Destination** (asked once per session, then reused): source on GCS/S3/Azure → derivative on cloud in same scheme, default proposal mirrors the dataset name `gs://<user-bucket>/<dataset_name>/` — agent ASKS to confirm bucket + prefix before writing. Source on local FS → parallel local prefix like `./<dataset_name>/`, still confirm. User-specified → respect verbatim. **NEVER default to `.datachain/thin-assets/`** — invisible to the team — unless the user asks for local-only.
>    - **Row schema** uses typed file objects, never bare paths or bytes columns by default:
>      ```python
>      class VideoFrameAsset(BaseModel):
>          source: dc.VideoFile      # preserves path/size/etag and the .get_info()/.get_frames() API
>          timestamp: float
>          sampled_frame_index: int
>          frame: dc.File            # pointer to materialized payload in storage
>      ```
>      Downstream UDFs declare `params=["frame"]` with `frame: dc.File`, not `params=["frame.jpeg"]` over a bytes column. Never `source_path: str`.
>    - **Default presets** (must meet step 5.5 minimum; adjust up when minimum demands it): **video** — 1 fps × 640px long side × JPEG q80; **audio** — 1-sec windows × 16 kHz mono PCM.
>
>    **Calibration policy.** `.persist()` not `.save()`; no `attrs` / `description` on calibration runs; no enrichment. End state: no `calib_*` (or legacy `pilot_*`) row in `dc.datasets()` and no `calib_*` / `pilot_*` in `dc-knowledge/`. `plan.py` filters `calib_*` / `pilot_*` names and `scope:calibration` / `scope:pilot` attrs as backstop.
> 7. **Decide branch** — strict order, bucket-root scope throughout.
>
>    **7a. Bucket root from URI.** From `s3://dc-readme/oxford-pets-micro/images/`, root is `s3://dc-readme/`. A subdir the user phrased the task around is not the root.
>
>    **7b. Compute cost AT BUCKET-ROOT SCOPE.** Carrying a subdir cost into 7c biases every downstream decision toward narrowing.
>
>    **7c. Auto-build heuristic** (ALL must hold, bucket-root scope): `layer_build_wall_time ≤ max(2 × direct_solve_wall_time, 60s)`; `layer_build_$ ≤ max(2 × direct_solve_$, $0.10)`; absolute wall ≤ 5 min; absolute $ ≤ $1; not Studio remote; no shortcut phrase used.
>
>    **7d. Decide. Two independent decisions:**
>    1. **Silent vs ask** (wall time): `wall ≤ 5 min` AND 7c passes → silent at bucket-root. `5 min < wall ≤ 8 h` → open dialogue, always. `wall > 8 h` → require explicit user override; surface architectural alternatives (smaller model, GPU, Studio, restricted subdir, coarser sampling).
>    2. **Shape** (Asset × 2): `R = thin_asset_cost / no_asset_cost`. In the silent branch R picks the shape (thin when R ≤ 2, no-Asset when R > 2). In the dialogue branch **recommendation is ALWAYS thin Asset** (CAST doctrine — substrate compounds); R is shown as information. When R > 2 in dialogue, surface a cost-premium framing line; recommendation does not flip.
>
>    Never silently auto-build at directory or sample scope. Format times as `~Xh Ym` / `~Xm` (concrete numbers, not band names).
>
>    **Scope-and-preset dialogue template** (substitute the measured calibration numbers verbatim; ≥3 options, more when warranted; the recommended option is ALWAYS thin Asset):
>    ```
>    Source: {bucket_size} / {n_total} {ext_summary} / avg file {avg_size}.
>    Task minimum (5.5): {sample_rate}, {resolution}, {format}.
>    Calibration ({n_calib}): no-Asset ~{wall_calib_no}s, thin-Asset ~{wall_calib_thin}s (+{thin_storage_calib}); R = {R:.1f}.
>    {if R > 2:} thin Asset costs ~{R:.1f}× more — substrate-vs-immediate-cost trade-off. Recommendation follows CAST (substrate compounds); pick no-Asset only for genuinely one-shot answers.
>
>    Full-run options (~{baseline_io} unavoidable I/O):
>    - **WHOLE bucket, thin Asset {preset}** [recommended — CAST substrate]: ~{wall_full_thin}, ${cost_full_thin}, Asset ~{asset_full_gb}. Reusable across future queries at fidelity ≥ task minimum. Destination: {default_destination_proposal} — confirm or override.
>    - **WHOLE bucket, no Asset**: ~{wall_full_no}, ${cost_full_no}. Re-decodes on the next question.
>    - **{Largest subdir} only**, thin Asset ({subdir_n}): ~{wall_sub}, ${cost_sub}. Won't cover sibling subdirs. ← include largest 1–2 subdirs from the scan
>    - **Tighter preset** (1/5 sec or 320px), whole bucket: ~{wall_tight}, ${cost_tight}, Asset ~{asset_tight_gb}. Must still meet step 5.5 minimum.
>    - **Sample only** ({sample_n}), no Asset: ~{wall_sample}, ${cost_sample}. One-shot.
>    - {if prior dataset on this source:} KB shows {prior_dataset} — thin Asset is the right call.
>    ```
>    Full-Asset (every frame, original resolution) is deliberately not in the default options; surface only on explicit user request. If you catch yourself rationalizing a silent narrow ("the directory is small", "the prompt was task-specific"), STOP and re-run 7a–7d.
>
>    **Render the template verbatim — do not paraphrase the recommendation.** Three specific failure modes that defeat CAST doctrine:
>    - **Do NOT reorder** so no-Asset comes first. Thin Asset is option 1, always.
>    - **Do NOT replace `[recommended — CAST substrate]` with conditional phrasing** like "recommended if you expect future queries" / "useful if you'll re-ask". The recommendation is unconditional in the dialogue.
>    - **Do NOT add competing labels to no-Asset** like "best for this one answer" or "best for the current question". That's the agent applying its own immediate-cost framing — exactly the regression Round 3 fixed. The user picks; the agent does not nudge with "best for X" tags.
>
>    The agent's instinct to optimize for "the user's current question" is what CAST doctrine deliberately overrides. Render the options, render the [recommended] tag, let the user decide.
> 8. **Apply shortcut phrases.** If the user's message contains any of "just solve", "no layers", "sample only", "fast as possible", "skip CAST", "one-off", "don't build a layer", "just answer", "quick" — skip the proposal and solve directly. State once: "Solving directly without building a layer." Do not re-propose layers in the same session unless the user volunteers CAST vocabulary themselves.
> 9. **On layer build, tag the datasets.** Name them per the convention (`l1_…`/`l2_…`/`l3_…` for C/A/S; no prefix for Task outputs) and pass `attrs=["cast:<layer>", "scope:bucket|directory|sample|onetime", "source:<slug>"]` + `description="…"` on `.save()`. Lineage (parent dataset references) is tracked automatically by DataChain — do not duplicate it in attrs. `scope:bucket` covers the bucket root (the default for auto-build) and the source slug is the bucket name. `scope:directory` is set ONLY when the user explicitly opted into a subdirectory in the scope-and-preset dialogue; the source slug includes the directory path (e.g., `source:my_bucket__images_subset`) so it does not collide with bucket-root layers. `scope:sample` is one-shot. `scope:onetime` is for non-persistable Task outputs.
> 10. **Containerise raw JSON / sidecars too.** If the task pulls structured JSON / Parquet / CSV from the bucket and parses it inline, propose lifting that parse into an `l1_container_<source>_<descriptor>` dataset so the parsed schema becomes reusable. Same auto-vs-ask rule, same whole-bucket-first framing.
> 11. **Mid-flight monitoring + early abort.** For any job estimated >5 min, watch the first 60–90 s of stdout for a throughput line (DataChain emits `Processed: N rows [elapsed, rate]`) and compute observed per-row rate.
>     - `observed_rate ≥ 0.66 × calib_rate` → carry on.
>     - `observed_rate < 0.5 × calib_rate` → **kill**, report the gap and the revised estimate, re-open the dialogue from 7d at the new band.
>     - No throughput line within 2 min → kill, investigate (startup cost, model download, auth retry), report.
> 12. **L1/L2/L3 builds always take the delta path; L4 (Task) and queries do NOT.** For L1/L2/L3 builds whose source is `dc.read_storage()`, pass `update=True, delta=True`. The defaults are right: `delta_on` defaults to `("file.path", "file.etag", "file.version")` (catches re-uploads with same path but different content); `delta_compare=None` uses all non-`delta_on` fields. Override only when you have a specific reason (e.g., local FS sources where `etag` is empty — then pass `delta_compare="file.mtime"`). Same code on first build (SDK processes everything, since no prior version) and on subsequent runs (SDK processes only new/changed source items). When the source is `dc.read_dataset()` (chaining from a parent CAS layer), the parent's delta semantics propagate — no extra args needed. See `core/SKILL.md` Section 8 "Delta updates".
>
>     **Do NOT apply `update=True` / `delta=True` to L4 Task code or regular queries / exploration.** Those read the cached listing (per Critical Rule 2) unless the user explicitly asks to refresh. The delta-path rule is scoped to the C/A/S substrate; it does not generalize to downstream consumption.
>
>     Build signature lives in `attrs`: `model:<id>@<version>`, `preset:<name>`, optional `udf_hash:<short>` (skip when not trivially computable). When the current build signature differs from the prior version's `attrs`, do a **full rebuild** — old rows are stale even though source unchanged. Schema changes are a different dataset (different name), not a refresh of this one. Source-file deletions leave orphaned L2 storage files — keep them for audit and lineage; never auto-delete.
>
> Step 1 (Bucket Enlistment) is itself a Container-layer artifact for the storage root: a lightweight, header-only view of what's in the bucket. The bucket entries appear in the Container row of the KB index next to any DataChain datasets explicitly tagged `cast:container` (e.g., parsed JSON sidecars promoted via the rule in step 10).

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

**MANDATORY:** read the prompt file in full and follow its template literally — frontmatter shape, section order, schema/stats/preview/version blocks. Do NOT hand-roll markdown via a Python script that emits a different structure; downstream tooling (`render_index.py`, `cast_layer` resolution) parses the exact frontmatter the prompt prescribes, so shortcuts produce lower-quality enrichment that may break the index.

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
