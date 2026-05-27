# CAST — The Methodology

CAST is the doctrine that drives every layer / scope / shape decision the
datachain-knowledge skill makes. This file is the canonical source. The
operational pipeline in `SKILL.md` (Modes A–D, Steps 1–7) is the *how*;
CAST is the *what* and the *why*.

`SKILL.md`'s Mode B precondition reads this file in full before any
pipeline-planning work. Do not skim it; the rules below override the
agent's default instinct to "answer the immediate question with minimum
work" — which is exactly the regression CAST exists to prevent.

---

## 1. Mission

### Job of this skill

The skill is hired to grow the user's reusable substrate for unstructured
data. Answering the current question is a **byproduct** of that job, not
the job itself. Every Task-shaped question is also an opportunity to leave
behind one or more reusable layers — Container, Asset, or Sense — that
make the next question cheaper.

If the skill answers the question fast and leaves nothing reusable behind,
that is a regression even when the immediate answer was correct. The user
explicitly delegated *substrate building* to this skill; under-communicating
to "stay out of the way" defeats the delegation. The only legal opt-out
is a shortcut phrase from §7 (`just solve`, `no layers`, `quick`, etc.).
Wall time alone is **not** a reason to go silent.

### Substrate rows are general-purpose, not question-specific

Every C/A/S row is built once and queried many times. So every row holds
the **full** output of the operation that produced it, plus all per-call
telemetry, plus all "free" auxiliary fields the operation surfaced —
NOT a projection shaped to the current question. The current question
projects downstream at Task time, where projecting is free.

By operation type:

- **Parse / decode / read** — every field the parser surfaces (all
  schema columns, all header fields, all sidecar keys), not just the
  one the current filter uses.
- **Materialize** — typed file reference (`dc.File` and subclasses),
  the preset that produced it, source link, encoding parameters.
- **Inference / model call** — the model's complete structured output
  (full detection list, full embedding vector, full LLM response),
  plus per-call telemetry: `model_id`, `model_version`, `inference_ms`,
  `prompt_tokens` / `completion_tokens` for LLMs, `finish_reason`,
  `request_id` for paid APIs.

If the agent finds itself writing `if conf > 0.5` or `if label == X`
before `.save()`, that filter is the current question leaking into the
substrate — push it downstream into a Task query.

**Worked counter-example:**

```python
# WRONG — `counts` is derived from `boxes.name`. It bakes the current
# question ("how many of these target classes?") into the L3 row.
chain.map(boxes=detect).map(counts=count_targets).save("l3_…")

# RIGHT — keep only what the operation returned.
chain.map(boxes=detect).save("l3_…")
# Counts (and any filter / threshold / agg) live downstream in
# read_dataset(…).map(counts=…).save("…") as a Task, or a one-shot query.
```

The only legal exceptions are size-prohibitive blobs (e.g. raw model
activations on a 1B-row dataset). Those get gated by an explicit user
decision, not by the agent guessing what's "needed".

---

## 2. The Four Layers

- **Container** — A typed, queryable index of what each file IS without
  decoding its full content: paths, sizes, format headers, sidecar
  metadata (JSON/XML), external-DB joins. *Build it when the task needs
  "what files are here, what format/metadata do they carry, which match
  this predicate" without paying decode cost.* One row per source file.
  Light compute (bounded-prefix reads only). **Recall cost ~$1.**

- **Asset** — Raw extracted or mixed data in a workable shape: decoded
  units (sub-file segments, parsed arrays, derived chunks), or a
  training mixture of multiple source datasets joined on a shared key.
  *Build it when downstream work needs decoded or combined content
  rather than file pointers — annotation, training, multi-source
  analytics.* Heavy per-file compute; no model has touched the rows
  yet. **Recall cost ~$10.**

- **Sense** — What a model said about the data: embeddings, LLM
  annotations, classifier scores, transcriptions, detections. *Build it
  when the task needs semantic or learned signals — similarity search,
  classification, retrieval, RAG.* Per-row model inference dominates
  cost. Pay once at build time, query at warehouse speed forever.
  **Recall cost ~$100 build, ~$1 query.**

- **Task** — A task-specific composition on top of C/A/S: a similarity
  ranking, a filter over Sense outputs, a join across layers, an eval
  set, a curated training subset. *Build it (and usually do NOT save it)
  when answering one specific question.* **Persist by exception** — only
  when the result becomes a standing benchmark or shared training set.
  **Recall cost ~$1.**

Most user questions arrive Task-shaped on top of C/A/S substrate. The
skill's job is to make the C/A/S substrate visible, reusable, and cheap.

---

## 3. Naming + Tagging

### Names

```
l1_<source>_<descriptor>      # listings, headers, sidecar metadata
l2_<source>_<descriptor>      # extracted/reshaped raw data
l3_<source>_<descriptor>      # model-derived signals
<descriptor>                  # Task outputs — no prefix
```

- **`l1_` / `l2_` / `l3_` prefix is enough — do NOT add the layer-type
  infix.** `l1_<source>_info` ✓; `l1_container_<source>_info` ✗. The
  layer is encoded by the L-number prefix and by `cast:<layer>` in
  `attrs`. Repeating it in the middle adds chars for zero new
  information.
- `<source>` is the bucket slug for L1–L3.
- **Strip uninformative prefixes** from the bucket name (case-insensitive)
  before using it as the source slug: `datachain-`, `dc-`, `iterative-`,
  `gs-`, `s3-`, `aws-`, `gcp-`, `azure-`, `public-`, `private-`. Replace
  `-` with `_`.
- **Preserve source-path word order in the slug.** Write subdirs
  left-to-right matching the bucket path; never reverse for a
  "friendlier" slug — that desyncs the KB from the bucket browser.
- **Preset/config → `attrs`, never in the name.** Names describe content,
  not parameters.
- **One content noun per name.** Pick one — don't stack redundant
  content markers.
- **Cap names at 30 characters, shorter is better.** Longer names get
  truncated in UI lists and table headers.
- **Task names are JUST the question** — no source slug, no layer prefix
  (`cik_text_stats`, `products_similar_to_query`). Add `_<source_slug>`
  only when the same question runs on two different sources and
  disambiguation is genuinely needed.

### Tagging on `.save()`

Layer + scope + source are dual-encoded — in the name (visible in
`dc.datasets()` and the KB index) AND in `attrs` (machine filtering) AND
in `description` (one-line human summary):

```python
attrs = [
    "cast:sense",                              # container | asset | sense | task
    "scope:bucket",                            # bucket | directory | sample | onetime
    "source:product_catalog",                  # bucket slug for L1-L3, task slug for Task
    "model:clip_vit_b32@open_clip-2.24",       # build signature (model id @ version)
    "preset:cpu_fp32",                         # build signature (preset/config tag)
]
chain.save(
    "l3_product_catalog_clip",
    attrs=attrs,
    description="CLIP ViT-B-32 embeddings over the full product-catalog bucket; reusable for any visual-similarity query.",
)
```

Lineage is tracked automatically when `dc.read_dataset("foo").map(...).save("bar")`.
Do NOT add `parent:<dataset_name>` to attrs — duplicating lineage creates two
sources of truth that drift.

### Scopes

- `scope:bucket` — full coverage of the bucket root. Default for auto-builds.
- `scope:directory` — coverage of a subdirectory. Set ONLY when the user
  explicitly opted into a narrower scope via the §4.10 dialogue. Include
  the directory path in the source slug to avoid collisions.
- `scope:sample` — covers only a sample. One-shot, no future savings.
- `scope:onetime` — Task layer that should NOT be persisted (default for Task).

### `.save()` checklist for L1/L2/L3

Verify every item before calling `.save()`:

- `attrs` includes `cast:<layer>`, `scope:<scope>`, `source:<slug>`.
- `attrs` includes `model:<id>@<version>` and `preset:<name>` whenever a
  model/encoding was used. These are the build signature read on refresh (§5).
- `description="…"` set to a one-line human summary.
- The `read_storage` includes `update=True, delta=True` for L1/L2/L3 builds (§5).
- **The Pydantic output type carries the operation's full output**, not
  the current question's projection (§1.2).

### `.gen()` pre-flight checklist

When you call `.gen(name=fn)`:

- `fn` has an **explicit** `-> Iterator[T]` annotation. Not bare return,
  not `Iterator`, not `Iterator[Any]`. Missing annotation collapses the
  saved schema to `str` over every row.
- `T` is either a primitive or a Pydantic `BaseModel` declared at
  **module level** — not nested inside the function, not imported lazily.
- No `from __future__ import annotations` anywhere in the module. It
  stringifies hints and DataChain's signal-schema resolution rejects the
  string-vs-class mismatch.

### Row shape — prefer nested over flat

When a source decomposes into sub-units (segments, frames, chunks,
events, time-windows, records), the default row shape is **nested**:
one row per source, with sub-units as nested Pydantic lists.

```python
class SegmentInfo(BaseModel):
    start: float
    end: float
    mask: list[float]

class FrameAnalysis(BaseModel):
    timestamp: float
    frame_index: int
    detections: list[Detection]
    segments: list[SegmentInfo]

class VideoRow(BaseModel):
    source: dc.VideoFile
    info: dc.Video                  # codec, fps, duration, dims (folded L1, §4.7b)
    frames: list[FrameAnalysis]
```

The flat alternative — one row per terminal unit with parent dimensions
as scalar references — wins only in two cases: (a) a downstream tool
requires one-row-per-unit (annotation UI, training pipeline reading
filesystem rows), or (b) per-unit aggregations dominate over hierarchy
queries.

The full menu of row shapes (nested, flat-pointer, flat-materialized,
fused-L3) and the dialogue rules for picking among them live in §4.8
and §4.10.0. Nested is always the recommended default in the §4.10
dialogue; flat shapes appear as the alternative for cases (a) and (b).

### Typed File references on substrate rows — never bare paths

A substrate row that points at a file MUST carry the typed reference,
not a string path. Use `dc.File` or its modality-specific subclass.
A bare `str` loses:

- **Lineage / consistency.** DataChain can no longer trace the row back
  to the source listing; etag-change detection breaks.
- **UI visualisation.** Studio renders `dc.File` subclasses with
  previews. A bare string is just text.
- **Downstream methods.** Modality methods like `.get_info()`,
  `.read()`, `.open()` require the typed object on the row.
- **Access config.** The `anon=True` / credentials from the original
  `read_storage` are carried by the typed File, not by the string.

```python
class Row(BaseModel):
    source: dc.VideoFile      # ✓ typed reference — lineage + methods + UI
    timestamp: float
    detections: list[Detection]
```

The only legal exception is when the user explicitly asks for a
string-only payload. In that case, keep the typed File AND add the
string as a secondary derived column — don't replace.

---

## 4. Planning a Task — the Layer-Ladder Walk

The agent's instinct is to identify the **minimum** layer the user's
question requires and build only that. This section overrides that
instinct: the agent walks the ladder **top-down** (Task → Sense → Asset →
Container) AND **bottom-up** (does this need to be combined / mixed?),
then surfaces every candidate layer that would compound for future
questions on the same source.

Strict order of preference: **(1) direct reuse → (2) reduce-to-CAS →
(3) build missing CAS → (4) raw rebuild**.

### 4.1 Identify required layers

The output of this step is a non-empty list of CAS layers the task
depends on, NOT a decision to skip layers. If the task involves any UDF
(decode, embedding model, LLM call, classifier, file parse), **at least
one CAS layer is required and must be saved.** There is no "no layer
needed" branch for UDF tasks.

**Container layer = partial-read or metadata-only work.** Justified when
per-row work reads only a bounded prefix of each file (headers, schema,
sidecar) — not the full file body. **Full file decode → L2 Asset, not
L1.** **Filter-only Container datasets are forbidden** — `.filter(glob)`
reads zero bytes; inline into `read_storage` glob or an L2/L3 `.filter()`.
The bucket scan in `dc-knowledge/buckets/` already covers "what files
are here". §6's "at least one CAS layer" is satisfied by Sense alone.

### 4.2 Look for mixture opportunities

If the task names two or more datasets (or two or more bucket regions /
sources), the Asset-level combination of them is itself a CAST artifact.
Surface it.

### 4.3 Direct reuse first

From `dc-knowledge/index.md`, for each required layer × source, check if
an existing dataset already covers the question (even partially). If yes,
write the pipeline as `dc.read_dataset(...)` over it.

**Celebrate the reuse.** In the response, name the layer being reused and
quote the saved cost — e.g., "Reusing `l3_product_catalog_clip`. This
query is ~$0.002 instead of the ~$1.40 the embedding pass would
otherwise cost — exactly the win the Sense layer was built for." This is
the moment that teaches the methodology to the user; do not skip it.

### 4.4 Reduce-to-CAS if direct reuse impossible

Before defaulting to a raw rebuild, can the task be reformulated to
operate on an *existing* CAS layer plus a small Task delta? Examples: a
new similarity question on the same source → reuse Sense embeddings,
just change the query vector; a new "find X" question → reuse Sense
classifications and add a filter. Spend real effort here — propose at
least one reformulation when any CAS layer for this source exists.

### 4.5 Cost gate on CAS reuse

Reuse a CAS layer only when it gives a meaningful win — at least ~2×
speedup or ~2× $-saving versus the raw rebuild. If the layer technically
covers the data but reading it is no cheaper than re-reading raw storage,
do not force the reuse; the methodology is justified by economics, not
formalism.

### 4.6 Derive task minimum BEFORE the calibration run

For decode-heavy sources, name the minimum fidelity the task requires
along whichever axes apply to the format:

- **Source granularity.** Smallest unit of decomposition the answer
  depends on (per-segment, per-page, per-event, per-grid-cell, per-record).
- **Fidelity.** Smallest detail the answer depends on (resolution,
  sample rate, bit depth, precision).
- **Encoding.** Lossy vs lossless; full-rate vs resampled;
  decoded vs raw container.

This minimum is the **floor** for any thin-Asset preset and what the
calibration measures against. Never calibrate at coarser fidelity than
the task requires. If genuinely ambiguous, ask one targeted question —
do not guess.

**Silent defaults are forbidden.** Any parameter that materially affects
the result or the cost — model id and version, threshold, granularity,
fidelity, preset — is surfaced to the user, never chosen silently. Two
levels:

- **ASK** (via AskUserQuestion, default pre-selected, one alternative
  shown) for parameters that change *what gets counted* — model + version,
  threshold. Two questions max; combine into one multi-select if
  appropriate.
- **NOTIFY** (render verbatim in the §4.10 dialogue's `Task minimum`
  line; no explicit confirmation needed, but the value must be visible)
  for parameters derived deterministically from the task minimum.

A different model + threshold on the same source gives different
numbers. The user cannot tell from the agent's output which produced
which unless the dialogue made it explicit. Build signature lives in
`attrs` (§3) AND in the rendered dialogue.

### 4.7 Walk the ladder top-down

After identifying the minimum layer the question requires (usually Sense
or Task), walk DOWN the ladder and for each layer underneath, answer
three questions in writing:

1. **"Would this layer, if it existed, cut my current build by ≥2× OR be
   plausibly reused by a future question on this source?"** If yes →
   it's a build candidate.
2. **"What shape would it have?"**
   - *Container*: which header / metadata fields.
   - *Asset*: which preset + which destination.
   - *Sense*: which model + which output schema. Save the full output
     (§1.2), not a projection.
3. **"Is the operation prompt/call generalisable to cover related
   axes?"** For deterministic ops (decode, header parse) — list every
   field the parser exposes. For LLM/VLM calls — extend the prompt
   while you're paying for one call. Same API cost, multiplies reuse.

Surface ALL candidates in §4.10's dialogue. The user picks which to
build. The agent does **not** pre-filter to "just what the current
question needs" — that's exactly the substrate-erosion regression CAST
exists to prevent.

### 4.7b L1 wiring rule — three modes

L1 captures task-agnostic source characteristics (codec, dimensions,
schema, format). It is the most general substrate; future tasks
(unknown today) will query headers regardless. So L1 data is always
worth capturing — but *whether* it lives as a standalone dataset or
folded onto the L2 row depends on the substrate's maturity on the source.

- **Mode 1 — First L2 on this source, no L1 yet → fold into L2.** Add
  the header read as a column on the L2 row. The L2 UDF usually
  already needs the header to compute its sampling step, so capture is
  essentially free. Do NOT build a separate L1 at this stage — the
  folded column IS the L1 data, colocated. Future tasks can still query
  via `dc.read_dataset("l2_…").select("source", "info")`.
- **Mode 2 — L1 already exists on this source → L2 reads from L1, no
  re-call.** Never re-call the header op inside the L2 UDF — "L1 paid
  for this once" is the whole point of the layer.
- **Mode 3 — 3+ L2 layers on this source carry folded headers → extract
  L1.** Header data is now duplicated; refresh coordination gets
  expensive. Break L1 out as a canonical dataset, then migrate L2 reads
  to Mode 2 on next refresh.

Standalone L1 appears as a separate build candidate in §4.10 ONLY when:
the task is genuinely header-only (no decode needed), OR ≥3 L2s on this
source make duplication bite, OR the user explicitly asks.

### 4.8 Estimate cost by measurement, never guess

The number going into the §4.9 gate MUST be calibration-derived.

**4.8a — Lead with sizes.** Before the calibration run, quote bucket
footprint from the scan (total GB, files, avg size, top extensions) and a
size-derived I/O baseline: `total_GB / bandwidth` where bandwidth ≈
50–150 MB/s GCS→Mac, 80–200 S3→local, 500+ same-region cloud.

**4.8b — Calibration procedure** (mandatory for any `.map`/`.gen` over
>50 source items, or any UDF that decodes / downloads / calls a model).

Pick `N = min(50, max(5, ⌈total/100⌉))`. Run **two** calibration runs
back-to-back sharing one decode-once `.map`, each ending in `.persist()`
(never `.save()`):

1. **no-Asset calibration:** emit only the aggregate / Sense output.
2. **thin-Asset calibration:** same `.map`, ALSO encode the materialized
   payload at task-minimum fidelity and return `sum(len(bytes))` as a
   column. **Do NOT upload during calibration** — encoding suffices to
   measure compute + size; upload happens only at full-run time.

60–120 s timeout per run. Record wall seconds, GB read, rows out, plus
encoded-bytes sum for thin-Asset.

**Both branches are mandatory.** Single-branch calibration leaves
R = thin/no undefined — which forces a fused-L3 default in §4.9 and
prevents the layer choice. If only one branch was run, re-run the
missing branch before proceeding to §4.9. If the full calibration is
genuinely infeasible (empty source, network unreachable), auto-build is
OFF and the agent goes to the §4.10 dialogue with the fallback rules of
thumb below.

**Never quote unmeasured calibration numbers — say "unmeasured"
instead.** If a calibration branch timed out, errored, or never ran,
the dialogue MUST report the unmeasured branch literally as
`not measured` and R as `unknown`. Do NOT extrapolate the measured
branch's wall onto the unmeasured one. Do NOT paraphrase ("about the
same as no-Asset"). Do NOT guess from recall-economics tiers and
present the guess as measurement.

When R is unknown, the §4.10 dialogue still surfaces the layer-ladder
options — but the dialogue cannot recommend thin-Asset vs no-Asset
shape on cost grounds. Surface this honestly: *"R is unknown — pick
thin-Asset to preserve substrate reuse, or no-Asset to keep this
build minimal."* The user picks under uncertainty; the agent does not
fabricate the missing measurement. If calibration keeps timing out,
options: (a) shrink N by 5× and re-run; (b) widen the timeout (with
explicit user opt-in); (c) accept R-unknown and surface honestly.

Extrapolate: `wall_full = (wall_sample / N) × total × 1.5`;
`asset_full_bytes = (sample_bytes / N) × total × 1.5`. Cost in $ from the
recall-economics tier unless calibration disagrees by >3×.

**Calibration measurements are reusable across scope changes** within
the same media / decode profile. When the user narrows or broadens
scope after the dialogue was rendered, do NOT re-calibrate "for
cleanliness" — just re-extrapolate the existing per-item numbers to
the new item count. See §4.10.5.

**Sanity floor.** Estimate disagreeing with the recall-economics tier by
>100× → re-calibrate, do not paper over.

**Fallback rules of thumb** (use only when calibration is impossible):
- Container (header-only parse): ~0.5 ms/file.
- Asset (full-file decode, materialize): 50–500 ms/file; for heavy-decode
  sources budget per source file, not per emitted unit.
- Sense — paid API (LLM/VLM, hosted embeddings): ~$0.001–0.01/row plus
  latency.
- Sense — local model on CPU: 5–50 ms (small classifiers), 50–500 ms
  (mid-size detection / segmentation), 0.1–0.5× realtime for streaming
  models.
- Sense — local model on GPU: ~10–100× faster than CPU baselines above.

**L2 row-shape options — pick deliberately.** This is the data-model
dimension surfaced in §4.10.0. Four shapes, in default-recommended order:

| Shape | Row schema | When to choose |
|---|---|---|
| **Nested** (recommended default) | `SourceRow(source: dc.File, info: ..., units: list[Unit])` where `Unit` may itself nest deeper (`Unit.sub_units: list[SubUnit]`). One row per source. | Default for any source that decomposes into sub-units. Composes with downstream queries (`.select("source", "units.sub_units")` projects the right slice). Source fetched once per row. |
| **Flat-pointer** | `UnitRow(source: dc.File, offset, index)` — one row per terminal unit; source is a reference, no payload bytes on the row. | Downstream consumes one-row-per-unit AND downstream is in-DataChain only (no external interop). **Consume via `.gen()` over the source File**, never per-row `.map()` that re-opens the file per emitted unit (cache key is per source file, not per unit). |
| **Flat-materialized** | `UnitRow(source: dc.File, unit_file: dc.File)` — one row per terminal unit; `unit_file` points at a written derivative at task-minimum preset. | Downstream tool needs files-on-disk for non-DataChain interop (annotation UIs, training pipelines that read filesystem rows). Destination: cloud-source → cloud-derivative same scheme; confirm bucket + prefix with the user, never default to `.datachain/`. |
| **Fused-L3** (skip L2) | `SenseRow(source: dc.File, ...)` — `.gen(File → Iterator[SenseUnit])` decodes once, emits per-unit Sense rows, no separate L2. | User explicitly opted out of L2 in the dialogue, OR L2 has no plausible reuse (one-pass embedding only). Cheapest single pass; re-decodes from raw storage on the next question. |

**Why nested is the default.** Nested rows let downstream queries project
to any granularity (`.select("source", "units")` for per-source aggregates;
`.gen(u=lambda row: row.units)` to flatten to per-unit). Flat rows can
only fan out, never re-aggregate without a costly `group_by`. Nested
also fetches the source once per row by construction — no per-unit
cache-miss trap.

**Flat is right when:** (a) a downstream tool requires one-row-per-unit
(annotation UI, training pipeline reading filesystem rows), OR (b)
per-unit aggregations dominate (`.group_by(unit_class, partition_by=...)`
across millions of units, where a nested schema would force a huge
`.gen()` fan-out before the aggregation).

**Calibration policy.** `.persist()` not `.save()`; no `attrs` /
`description` on calibration runs; no enrichment. End state: no `calib_*`
row in `dc.datasets()` and no `calib_*` in `dc-knowledge/`.

### 4.9 Decide branch — strict order, bucket-root scope throughout

**4.9a Bucket root from URI.** From `s3://bucket/sub/dir/`, root is
`s3://bucket/`. A subdir the user phrased the task around is NOT the root.

**4.9b Compute cost AT BUCKET-ROOT SCOPE.** Carrying a subdir cost into
the gate biases every downstream decision toward narrowing.

**4.9c Auto-build heuristic** (ALL must hold, bucket-root scope):
`layer_build_wall_time ≤ max(2 × direct_solve_wall_time, 60s)`;
`layer_build_$ ≤ max(2 × direct_solve_$, $0.10)`; absolute wall ≤ 5 min;
absolute $ ≤ $1; not Studio remote; no shortcut phrase used.

### 4.10 Render the layer-ladder dialogue — almost always

**Dialogue is mandatory whenever ANY new CAS layer would be built.**
Wall time does NOT gate the dialogue — the dialogue exists to make
substrate visible, not to gate on cost. Silent path is allowed **only**
when:

- the task is fully solved by reading an existing CAS layer (no new
  build), OR
- the user used a shortcut phrase from §7.

For every other case, render the template below — even for sub-minute
builds.

#### 4.10.0 Four dimensions — ask separately, never mix

A layer-ladder dialogue settles **four orthogonal dimensions** with the
user. Each is asked as its own question; never bundle options from two
dimensions into one menu. Order is fixed — scope is last because its
cost depends on every answer above.

| # | Dimension | Question | Default | Source |
|---|---|---|---|---|
| 1 | **Method** | Which library or approach realises the Sense (or heavy Asset) operation — e.g. a segmentation library vs an LLM vs custom code; OCR vs text-layer; an embedding family. | ASK whenever ≥2 plausible methods exist with different precision/recall/cost profiles. NOTIFY when only one is reasonable. | This section |
| 2 | **Parameters** | Model id + version, threshold, fidelity — anything within the chosen method that materially changes results. | Task-minimum-derived (NOTIFY) for fidelity; ASK for model+version and threshold. | §4.6 ASK / NOTIFY table |
| 3 | **Data model** | Row schema, per layer that decomposes into sub-units: **nested** (one row per source, sub-units as nested Pydantic lists) vs **flat-pointer** vs **flat-materialized** vs **fused-L3** (skip L2). | **Nested.** See §4.8 L2-row-shape table for the full menu and when flat wins. | §4.8 |
| 4 | **Scope** | Bucket / directory / sample. | Bucket (whole root). Per §4.9. | §4.9 |

Method comes first because parameters live *inside* a method — `threshold`
and `model_id` only make sense once the library is picked. Scope is last
because its cost depends on every answer above.

**Separation rule.** Render every AskUserQuestion call with **exactly
one dimension per question**. AskUserQuestion accepts up to 4 questions
per call, presented to the user as a sequential wizard ("answer Q1,
then Q2, then Q3, then Q4, submit"). Pack the dimensions that genuinely
need an answer into that one call, in order: method → parameters →
data-model → scope. Do NOT collapse two dimensions into one option list
(e.g., `[a) WHOLE bucket nested, b) directory flat-pointer]` is a bug
— that mixes scope AND data-model).

**Skip rule.** Drop a dimension from the wizard when it has no real
choice:

- Method: if only one library/approach is plausible, NOTIFY in the
  rendered prose, skip the question.
- Parameters: if §4.6 lists no ASK rows for this task, skip.
- Data model: nested is the default; ask only when there's a concrete
  reason to consider flat (downstream tool requires it, or per-unit
  aggregations dominate).
- Scope: always ask.

The minimum wizard is one question (scope). The maximum is four.

#### 4.10.1 Rendered dialogue template

Substitute measured calibration numbers verbatim. Render every layer
that passed §4.7's "build candidate" test.

```
Task: {one-line user goal}.
Source: {bucket_size} / {n_total} {ext_summary} / avg file {avg_size}.
Task minimum (§4.6): {granularity}, {fidelity}, {encoding}.
Calibration (N={n_calib}): no-Asset ~{wall_calib_no}s, thin-Asset ~{wall_calib_thin}s
                           (+{thin_storage_calib}); R = {R:.1f}.
{if R > 2:} thin Asset costs ~{R:.1f}× more — substrate-vs-immediate
trade-off. Recommendation follows CAST (substrate compounds); pick no-Asset
only for genuinely one-shot answers.

Build candidates (§4.7 walk):
- L1 Container — {what it'd hold}; row: {field list}
- L2 Asset     — {what it'd hold + preset + destination}; row: {field list}
- L3 Sense     — {what it'd hold, model + full output schema}; row: {field list}
                 (no derived counts/booleans — those are Task queries, §1.2)

──────────────────────────────────────────────────────────────────────
Dimensions to settle (each in its own question):

Q1. METHOD — which library / approach
  • {method_a} — {one-line precision/recall/cost hint}     [recommended]
  • {method_b} — {hint}
  • {method_c} — {hint}
  (Skip the question if only one method is plausible; NOTIFY instead.)

Q2. PARAMETERS — only ASK rows from §4.6, for the method picked in Q1
  • model:    {default} | {alternative-1 (+precision/-wall)} | {alternative-2}
  • threshold:{default} | tighter ({alt}) | looser ({alt})
  (NOTIFY rows shown above in `Task minimum`; user can override but no question.)

Q3. DATA MODEL — row shape (per layer that decomposes)
  • nested:           one row per source, sub-units as nested lists     [recommended]
                      composes with downstream queries; source fetched once per row
  • flat-pointer:     one row per unit, source as reference
                      pick when downstream is in-DC AND consumes per-unit
  • flat-materialized: one row per unit, unit file written to storage
                      pick when external tool needs files-on-disk
  • fused-L3 (no L2): emit Sense rows directly from source decode
                      pick when no L2 reuse is plausible

Q4. SCOPE — coverage of the source
  • WHOLE bucket: ~{wall_full_bucket}, ${cost_full_bucket}     [recommended]
  • directory only ({subdir}): ~{wall_dir}, ${cost_dir}
  • sample ({sample_n}): ~{wall_sample}, ${cost_sample}
  • skip the build

{if LLM/VLM Sense, add to Q1 (it's a method-axis sub-choice):}
  Prompt extension — same API cost, multiplies reuse:
  • default prompt covers {axis_1}
  • extended covers {axis_1}, {axis_2}, {axis_3}
  • custom

──────────────────────────────────────────────────────────────────────

Recommendation: {explicit combination across all four dimensions, e.g.
"method: {library} / params: {model_id}@{threshold} / data model: nested / scope: bucket"}
Reasoning: {one line — what compounds across future questions}.
```

**Render the template verbatim — do not paraphrase the recommendation.**
Failure modes that defeat CAST doctrine:

- **Do NOT collapse dimensions into one menu.** "Pick one of: A) WHOLE
  bucket nested, B) directory flat-pointer, C) sample fused" mixes
  scope AND data-model AND L2-shape into one question. That's the
  exact bug the four-dimension split fixes.
- **Do NOT reorder so flat / fused-L3 / no-Asset come first.** The
  recommended option is the recommended option; render it first.
- **Do NOT replace `[recommended]` with conditional phrasing** like
  "recommended if you expect future queries". The recommendation is
  unconditional.
- **Do NOT add competing labels to flat / fused-L3 / no-Asset** like
  "best for this one answer". The user picks; the agent does not nudge.

**Always quote a number.** Every option carries an estimate. If
calibration is feasible but wasn't run, run it first and then render the
dialogue.

**AskUserQuestion options MUST mirror the rendered prose dialogue.**
Every named option (a / b / c / d) from the prose dialogue must appear
as an `AskUserQuestion` option with the same label, and the `description`
field must include the wall-time estimate, the dollar estimate, and the
one-line trade-off. The prose render is the source of truth;
AskUserQuestion is just the UI affordance.

**One AskUserQuestion call carries all dimensions that need answers.**
Pack Q1..Q4 (whichever apply) into a single AskUserQuestion call as
separate `questions[]` entries. The runtime renders them as a wizard
— the user fills each, then submits all together. Do NOT make four
sequential AskUserQuestion tool calls; that fragments the user's
attention and forces them to wait between rounds.

### 4.10b Small-scope dialogue variant

When the chosen scope has **<50 source items** OR the full L3 wall time
is **<5 min**, the L1+L2+L3 ladder is often over-engineering: the
substrate covers only the narrow subdirectory, not its siblings.
Render the small-scope template:

```
Scope: {dir}, {n} items, full L3 wall ~{wall_l3}.
Substrate at this scope covers ONLY {dir} — siblings ({sibling list})
would need a separate build.

Two ways forward:

  a) [recommended for one-shot] fused-L3 only
     ~{wall_l3_fused}, leaves no reusable substrate.
  b) full ladder L1 + L2 + L3 at directory scope
     ~{wall_full_dir}, covers ONLY {dir}.
     Worth it only if you expect >2 future questions on this exact subdirectory.
  c) full ladder L1 + L2 + L3 at bucket scope (broader)
     ~{wall_full_bucket}, substrate compounds across siblings.
     The CAST default; pick this when bucket-wide substrate is the real goal.
```

Lead with (a) — the user narrowed scope; respect the intent. Surface
(c) so the bucket-wide alternative is visible. Never silently default to
(b) just because directory scope was named.

### 4.10.5 Dimension-change re-dialogue protocol

Any post-dialogue user reply that changes one of the four dimensions
(§4.10.0) re-opens the dialogue. The agent re-renders at the new state
before proceeding to script generation.

**Triggers — any of these:**

1. **Free-text reply outside the offered AskUserQuestion options.** The
   user said something the wizard didn't surface — treat as a
   dimension change.
2. **Parameter-change keywords**: a different model name, a new
   threshold value, "higher resolution", "more precise".
3. **Method-change keywords**: "use {other library}", "try LLM
   instead", "switch to {framework}".
4. **Data-model-change keywords**: "flat", "one row per {unit}",
   "nested", "annotate this", "I need files on disk", "skip L2".
5. **Scope-narrowing keywords**: "only", "just", "limit to", "single",
   "this dir only", "subset", "narrow to", "skip the …".
6. **Scope-broadening keywords**: "whole bucket", "all of …", "every …",
   "across all", "include the train/eval split".
7. **Explicit dir / prefix / glob** named by the user that wasn't in
   the original dialogue.

When any trigger fires, reuse the existing calibration measurements and
re-extrapolate (parameters and data-model changes can affect the
extrapolation — see §4.8 sanity rules). Render the dialogue again at
the new state (§4.10 or §4.10b). Never proceed to script generation off
the prior recommendation.

**Do NOT re-calibrate by default.** Re-running calibration on the same
media/decode profile is wasted wall time. The only cases that warrant a
fresh calibration: media-type change within the same task, or
decode-profile change where the prior per-item wall is unlikely to
extrapolate. Even then, ASK rather than auto-recalibrate. Default to
extrapolation when in doubt.

**The recommendation can flip even when the measurements are reused.**
Re-deriving the recommendation is the whole point of re-rendering the
dialogue.

**Orphan-substrate warning.** When the user picks the full ladder at
`scope:directory`, surface in the post-build report: *"L1+L2 substrate
covers only {dir}. A future task on a sibling will require a separate
build, or you can promote to `scope:bucket` later by re-running with the
bucket-root URI."* Never auto-rebuild at bucket scope without asking.

### 4.11 Containerise raw JSON / sidecars too

If the task pulls structured JSON / Parquet / CSV from the bucket and
parses it inline, propose lifting that parse into an
`l1_<source>_<descriptor>` dataset so the parsed schema becomes
reusable. Same dialogue rule, same whole-bucket-first framing.

### 4.12 Mid-flight monitoring + early abort

For any job estimated >5 min, watch the first 60–90 s of stdout for a
throughput line (DataChain emits `Processed: N rows [elapsed, rate]`) and
compute observed per-row rate.

- `observed_rate ≥ 0.66 × calib_rate` → carry on.
- `observed_rate < 0.5 × calib_rate` → **kill**, report the gap and the
  revised estimate, re-open the dialogue at the new band.
- No throughput line within 2 min → kill, investigate (startup cost,
  model download, auth retry), report.

---

## 5. Reuse Rules

- **L1/L2 (Container, Asset).** Persist by default, refresh by delta.
  Load-bearing reusable substrate. Always full-coverage; never
  problem-specific filters before `.save()`.
- **L3 (Sense).** Persist by default, full coverage of input. The
  "expensive UDF → save full → filter downstream" rule exists exactly to
  make L3 reusable.
- **L4 (Task).** Persist by exception. Most ranking / similarity / filter
  outputs do not deserve a name. Save only when the user explicitly asks,
  or when the result is a standing benchmark / training set referenced
  again.

### Delta path

**L1/L2/L3 builds always take the delta path; L4 and queries do NOT.**
For L1/L2/L3 builds whose source is `dc.read_storage()`, pass
`update=True, delta=True`. Defaults are right: `delta_on` defaults to
`("file.path", "file.etag", "file.version")`; `delta_compare=None` uses
all non-`delta_on` fields. Same code on first build and subsequent runs.

When the source is `dc.read_dataset()` (chaining from a parent CAS
layer), the parent's delta semantics propagate — no extra args needed.

**Do NOT apply `update=True` / `delta=True` to L4 Task code or regular
queries.** Those read the cached listing unless the user explicitly asks
to refresh.

### Build signature

Build signature lives in `attrs`: `model:<id>@<version>`, `preset:<name>`,
optional `udf_hash:<short>`. When the current build signature differs from
the prior version's `attrs`, do a **full rebuild** — old rows are stale
even though source unchanged. Schema changes are a different dataset
(different name), not a refresh of this one. Source-file deletions leave
orphaned L2 storage files — keep them for audit and lineage; never
auto-delete.

---

## 6. Critical Rules (CAST-adjacent)

These five rules encode CAST doctrine directly. `SKILL.md` cross-references
this section rather than restating the rules.

- **Follow CAST.** Every dataset belongs to one of four layers — Container,
  Asset, Sense, Task — and every new dataset is named, tagged, and
  described accordingly.
- **NEVER bypass DataChain for results.** UDF outputs MUST land via
  `.save("name", attrs=[...], description=...)`. Writing to local `.json`
  / `.csv` / `.parquet` via `open()`, `json.dump`, `pandas.to_csv`, etc.,
  bypasses lineage and the KB.
- **C/A/S substrate is mandatory.** Any UDF-bearing task builds and saves
  **at least one** Container / Asset / Sense layer. "Persist by exception"
  applies ONLY to the final Task ranking, never to substrate that
  meaningfully reduces future cost.
- **One script per stage.** Multi-stage pipelines (2+ named datasets)
  decompose into separate scripts named after the dataset each produces.
  Never one monolith with multiple `.save()` calls.
- **One script, one `.save()`. Never manually batch or shard.** DataChain
  checkpoints UDF progress internally; rerunning the same script after a
  kill resumes from where it stopped.

---

## 7. Methodology Transmission

Apply these in every data-related response.

1. **Surface the layer ladder by default.** When the task requires any
   UDF (decode, model call, parse), render the full CAST ladder
   (Container? Asset? Sense?) and the scope ladder (bucket / directory /
   sample) with numbers. Do not stay silent on fast jobs — the dialogue
   exists for substrate visibility, not cost gating.
2. **Match user energy.** When the user uses CAST vocabulary
   ("container", "asset", "sense", "task", "build the layer", "CAST"),
   engage with the full vocabulary, name parents explicitly, and propose
   lineage. When they don't, stay concrete but still render the dialogue.
3. **Always quote a number** when recommending a layer build. "Building
   the Sense layer takes ~3 min and $0.40; running the same embedding
   next time is free." No cost estimate → run the calibration first, then
   render the dialogue.
4. **Celebrate CAS reuse explicitly.** Every time the skill reuses an
   existing C/A/S layer instead of rebuilding, state the win out loud:
   which layer is being reused, what cost was avoided, and (when natural)
   a one-line connection to the session that originally built it. This is
   the only place the skill should sound enthusiastic about the methodology
   — the point is to make the value of a properly built layer felt, not
   asserted.
5. **Honour shortcuts immediately, do not re-litigate.** If the user has
   said any of "just solve", "no layers", "sample only", "fast as
   possible", "skip CAST", "one-off", "don't build a layer", "just
   answer", "quick" — solve directly and do not re-propose layers in the
   same session unless the user volunteers CAST vocabulary themselves.
   State once: "Solving directly without building a layer." Shortcuts are
   the user-side opt-out; until they're spoken, default is
   over-communicate.
6. **Narrowing scope mid-dialogue is a soft shortcut.** When the user
   voluntarily narrows scope after seeing a bucket-scope dialogue ("just
   this directory", "only the sony split", "drop eval"), that's a signal
   they want a smaller investment than the full ladder. Re-render via
   §4.10.5 — and if the new scope is small (§4.10b applies), **lead with
   fused-L3**, not the full ladder. The user can still pick the ladder,
   but the recommendation order flips. Do NOT treat the narrow as "same
   plan, smaller domain" — that defeats the point of letting the user
   redirect.
