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

### Row granularity — one big table, pick the right grain

Every saved dataset is **one big Pydantic table** — that part is fixed.
The methodology choice is **granularity**: what entity is "one row"?

**Default: row granularity matches the Task entity.** Identify what
the user's question filters or groups on; make that the row. An ideal
dataset answers the Task with one `.filter()` / `.group_by()` — no
`.map()` to walk nested lists.

**Finer sub-units stay nested as Pydantic lists** (preserves §1.2
full-output rule). One row per Task entity; everything below sits in
nested fields that downstream `.gen` / `.map` can fan out when needed,
but Task-level analytics never touch.

```python
# Task: "find videos with high pedestrian density"
# Row = per-video. Frames + detections live as nested lists.
class VideoRow(BaseModel):
    source: dc.VideoFile
    info: dc.Video                          # folded L1 (§4.7b)
    frames: list[FrameAnalysis]             # nested
    pedestrian_density: float               # derived column for the Task filter
```

**Anti-pattern: the Task's filter target is buried in nested lists.**
If the user filters on something inside `list[list[T]]`, the
granularity is wrong — pick one level deeper. Lists at the root level
force `.map()` for every analytic query and defeat Data Memory's
warehouse-speed promise.

Granularity is a first-class user question in the §4.10 dialogue
(§4.10.0 Q3), phrased in domain terms (per-video / per-frame /
per-segment) with the Task-matched grain pre-selected.

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

### 4.6 Derive task minimum BEFORE estimating cost

For decode-heavy sources, name the minimum fidelity the task requires
along whichever axes apply to the format:

- **Source granularity.** Smallest unit of decomposition the answer
  depends on (per-segment, per-page, per-event, per-grid-cell, per-record).
- **Fidelity.** Smallest detail the answer depends on (resolution,
  sample rate, bit depth, precision).
- **Encoding.** Lossy vs lossless; full-rate vs resampled;
  decoded vs raw container.

This minimum is the **floor** for any Asset preset and the input to the
§4.8 estimate (or calibration, when one runs). Never estimate at coarser
fidelity than the task requires. If genuinely ambiguous, ask one
targeted question — do not guess.

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

### 4.8 Estimate cost — prior first, calibrate only when needed

**Default: estimate from priors.** Quote bucket footprint
(`total_GB`, `total_files`, `avg_size`, top extensions) and a wall
estimate built from the table below:

```
wall ≈ files × per-row × 1.5 / parallel
```

Most tasks need nothing more — standard models on familiar source
shapes are predictable to within 2-3×.

| Op class | Per-row | Notes |
|---|---|---|
| Container header parse | ~1 ms | bounded-prefix reads |
| Asset decode (file body) | size / 10-50 MB/s | I/O dominates |
| Sense — small CPU classifier | 5-50 ms | text, light CV |
| Sense — mid CPU model | 50-500 ms | detection, segmentation |
| Sense — streaming CPU model | 0.1-0.5× realtime | ASR, audio |
| Sense — local GPU | 10-100× faster than CPU | depends on model size |
| Sense — paid API (LLM/VLM) | $0.001-0.01/row + 0.5-2s | rate-limited |

Quote in §4.10 as `Estimated wall: ~X (prior)`.

**Calibrate only when the prior is unreliable:**

1. **Untested implementation** — smoke-test 2-3 items to catch schema
   mismatches, missing deps, model-load failures.
2. **Unknown model or library** — no row in the table covers it.
3. **Heavy-decode files** (avg >100 MB) where decode dominates and the
   table is too loose. For files >300 MB, sample headers, don't decode.
4. **Prior is within 3× of the §4.9c auto-build threshold** — a soft
   prior on a multi-hour run is risky.

Skip otherwise: small source (<50 items or <100 MB), model in the
table, or shortcut phrase (§7).

**Fast calibration.** Budget **60s wall**. `N = 3-5` items,
`.persist()` (never `.save()`), no `attrs`. Extrapolate
`wall_full = (wall_sample / N) × total_files × 1.5`. Kill at 60s;
fall back to the prior with a `calibration timed out` flag.

**Don't conflate estimate vs measurement.** A prior-derived number is
`Estimated wall: ~X (prior)`. A measured number is `Calibration (N=5): ~X`.
Never paraphrase a missing measurement — say `not measured` literally.
If the estimate disagrees with the recall-economics tier by >100×,
re-derive before quoting (wrong table row? wrong file count?).

**Asset payload — where decoded bytes live.** Once granularity is
picked (§4.10.0 Q3), one sub-question remains for L2: where does the
decoded payload sit on the row?

- **On the row** (default for compact payloads — embeddings, transcripts,
  small structured fields). Compose with downstream queries; no extra
  round-trip.
- **Source reference only** (`source: dc.File` on the row; decode lazily
  at consume time). Pick for large payloads (>10 MB/row) when downstream
  is in-DataChain only. Consume via `.gen()` over the source — never
  per-row `.map()` (cache key is per source file, re-opens per row).
- **Materialised derivative file** (`unit_file: dc.File` pointing at a
  written derivative at task-minimum preset). Pick when an external tool
  needs files-on-disk (annotation UI, training pipeline reading
  filesystem rows). Destination: cloud-source → cloud-derivative same
  scheme; confirm bucket + prefix with the user, never default to
  `.datachain/`.

**Skip L2 entirely (fused-L3)** when L2 has no plausible reuse beyond
the current Sense pass (one-pass embedding, no future per-unit
questions). This is a §4.7 ladder-walk decision — drop L2 from the
build candidates, emit Sense rows directly from source decode.

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
| 3 | **Granularity** | One row per what domain entity? Always one big table; the question is which entity is "one row". Sub-units below the chosen grain stay nested as Pydantic lists. | **Per-Task-entity** — the grain at which the Task answers in one `.filter()` / `.group_by()` (no `.map()` over nested lists). | §3 "Row granularity" |
| 4 | **Scope** | Bucket / directory / sample. | Bucket (whole root). Per §4.9. | §4.9 |

Method comes first because parameters live *inside* a method — `threshold`
and `model_id` only make sense once the library is picked. Scope is last
because its cost depends on every answer above.

**Separation rule.** Render every AskUserQuestion call with **exactly
one dimension per question**. AskUserQuestion accepts up to 4 questions
per call, presented to the user as a sequential wizard ("answer Q1,
then Q2, then Q3, then Q4, submit"). Pack the dimensions that genuinely
need an answer into that one call, in order: method → parameters →
granularity → scope. Do NOT collapse two dimensions into one option
list (e.g., `[a) WHOLE bucket per-video, b) directory per-frame]` is a
bug — that mixes scope AND granularity).

**Skip rule.** Drop a dimension from the wizard when it has no real
choice:

- Method: if only one library/approach is plausible, NOTIFY in the
  rendered prose, skip the question.
- Parameters: if §4.6 lists no ASK rows for this task, skip.
- Granularity: **always ask**, in domain terms. The user often anticipates
  future questions at a finer grain than the current Task strictly needs.
- Scope: always ask.

The minimum wizard is one question (scope). The maximum is four.

#### 4.10.1 Rendered dialogue template

Substitute numbers verbatim (estimate or calibration — label which).
Render every layer that passed §4.7's "build candidate" test.

```
Task: {one-line user goal}.
Source: {bucket_size} / {n_total} {ext_summary} / avg file {avg_size}.
Task minimum (§4.6): {granularity}, {fidelity}, {encoding}.
{if estimated:} Estimated wall: ~{wall_estimate} (prior: {avg_size} × {throughput_class}).
{if calibrated:} Calibration (N={n_calib}): ~{wall_calib} per-row × {n_total} files = ~{wall_full}.
{if calibration timed out:} Calibration timed out at 60s; falling back to estimate ~{wall_estimate} (prior).

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

Q3. GRANULARITY — one row per {task_entity}? (phrased in domain terms)
  • per-{task_entity}: Task answers in one .filter()/.group_by()     [recommended]
  • per-{finer_entity}: finer than Task needs; reusable for future per-{finer} questions
  • per-{coarser_entity}: coarser; sub-units nested as list, Task needs .map() (avoid)

  Pick {task_entity} from the user's question — what does it filter / group on?
  E.g. "videos with people" → per-video; "frames containing cars" → per-frame;
  "segments above threshold" → per-segment. Sub-units below the chosen grain
  stay nested as Pydantic lists (§1.2 full-output rule still holds).

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
  bucket per-video, B) directory per-frame, C) sample fused" mixes
  scope AND granularity AND ladder choice into one question. That's the
  exact bug the four-dimension split fixes.
- **Do NOT reorder so the Task-matched grain isn't first.** The
  recommended option is the recommended option; render it first.
- **Do NOT replace `[recommended]` with conditional phrasing** like
  "recommended if you expect future queries". The recommendation is
  unconditional.
- **Do NOT add competing labels to flat / fused-L3** like "best for
  this one answer". The user picks; the agent does not nudge.

**Always quote a number.** Every option carries a wall + cost. Derive
the number from §4.8 priors by default; calibrate only when §4.8b's
triggers fire. Label the source (`estimate` or `calibration`) so the
user knows which.

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

When any trigger fires, reuse the existing per-row numbers (estimate
or calibration — either is fine) and re-extrapolate to the new state.
Parameter and data-model changes can shift the per-row baseline — see
§4.8 sanity floor. Render the dialogue again at the new state (§4.10 or
§4.10b). Never proceed to script generation off the prior recommendation.

**Do NOT re-measure by default.** Re-running calibration on the same
model + file-size profile is wasted wall time. Fresh measurement is
worth it only when the model or decode profile actually changed (new
library, very different file size class). Default to re-extrapolation
when in doubt.

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
   next time is free." Derive the number from §4.8 priors by default;
   calibrate only when the prior is unreliable.
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
