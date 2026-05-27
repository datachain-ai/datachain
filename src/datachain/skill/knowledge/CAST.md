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
to "stay out of the way" defeats the delegation.

The only legal opt-out is a shortcut phrase from §7 below (`just solve`,
`no layers`, `quick`, etc.). Wall time alone is **not** a reason to go
silent.

### Substrate rows are general-purpose, not question-specific

Every C/A/S row is built once and queried many times. So every row holds
the **full** output of the operation that produced it, plus all per-call
telemetry, plus all "free" auxiliary fields the operation surfaced —
NOT a projection shaped to the current question. The current question
projects downstream at Task time, where projecting is free.

Concretely, by operation type:

- **Parse / decode / read** — every field the parser surfaces (all EXIF
  tags, all Parquet schema columns, all video codec metadata, all sidecar
  JSON keys), not just the field the current filter uses.
- **Materialize** — typed file reference (`dc.File` / `dc.ImageFile` /
  `dc.VideoFile`), the preset that produced it, source link, encoding
  params.
- **Inference / model call** — the model's complete structured output
  (all detections / all classes / all confidences; full LLM/VLM response;
  full embedding vector), plus per-call telemetry: `model_id`,
  `model_version`, `inference_ms` (or API latency), `prompt_tokens` /
  `completion_tokens` for LLMs, `finish_reason`, `request_id` for paid
  APIs.

If the agent finds itself writing `if conf > 0.5` or `if label == "person"`
before `.save()`, that filter is the current question leaking into the
substrate — push it downstream into a Task query.

**Worked counter-example (the test/ regression — do NOT replicate):**

```python
# WRONG — `counts` is derived from `boxes.name`. It bakes the current
# "people / cars / trucks / bags" question into the L3 row.
chain.map(boxes=detect_objects).map(counts=count_targets).save("l3_sense_…")

# RIGHT — keep only what YOLO returned.
chain.map(boxes=detect_objects).save("l3_sense_…")
# Counts (and any filter / threshold / agg) live in a downstream
# read_dataset(…).map(counts=…).save("…") Task, or a one-shot query.
```

The only legal exceptions are size-prohibitive blobs (e.g. raw model
activations on a 1B-row dataset). Those get gated by an explicit user
decision, not by the agent guessing what's "needed".

---

## 2. The Four Layers

- **Container** — A typed, queryable index of what each file IS without
  decoding its full content: paths, sizes, format headers (MP4
  codec/duration, HDF5 attributes, DICOM tags, Parquet schema), sidecar
  metadata (JSON/XML), external-DB joins. *Build it when the task needs
  "what files are here, what format/metadata do they carry, which match
  this predicate" without paying decode cost.* One row per source file.
  Light compute (header bytes only). **Recall cost ~$1.**

- **Asset** — Raw extracted or mixed data in a workable shape: video
  frames, audio segments, decoded arrays, or a training mixture of
  multiple source datasets joined on a shared key. *Build it when
  downstream work needs decoded or combined content rather than file
  pointers — annotation, training, multi-source analytics.* Heavy file
  compute; no model has touched the rows yet. **Recall cost ~$10.**

- **Sense** — What a model said about the data: embeddings, LLM
  annotations, classifier scores, transcriptions, detections. *Build it
  when the task needs semantic or learned signals — similarity search,
  classification, captioning, retrieval, RAG.* Per-row model inference
  dominates cost. Pay once at build time, query at warehouse speed
  forever. **Recall cost ~$100 build, ~$1 query.**

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
l1_container_<source>_<descriptor>      # listings, headers, sidecar metadata
l2_asset_<source>_<descriptor>          # extracted/reshaped raw data
l3_sense_<source>_<descriptor>          # model-derived signals
<descriptor>                            # Task outputs — no prefix
```

- `<source>` is the bucket slug for L1–L3.
- **Strip uninformative prefixes** from the bucket name (case-insensitive)
  before using it as the source slug: `datachain-`, `dc-`, `iterative-`,
  `gs-`, `s3-`, `aws-`, `gcp-`, `azure-`, `public-`, `private-`. Replace
  `-` with `_`. Example: `datachain-starss23` → `starss23`.
- **Preset/config → `attrs`, never in the name.** Names describe content,
  not parameters.
- **One content noun per name.** `yolo` and `detections` are the same
  content axis — pick one. `l3_sense_starss23_yolo` ✓.
  `l3_sense_starss23_dts_yolo` ✗ — the `dts` infix is redundant content
  marker the agent invented.
- **Cap names at 40 characters.** If a meaningful name would exceed,
  shorten the descriptor (`frames` not `video_frames` when source implies
  video).
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
    "l3_sense_product_catalog_clip_embeddings",
    attrs=attrs,
    description="CLIP ViT-B-32 embeddings over the full product-catalog bucket; reusable for any visual-similarity query.",
)
```

Lineage (parent dataset references) is tracked automatically by DataChain
when you do `dc.read_dataset("foo").map(...).save("bar")`. Do NOT add
`parent:<dataset_name>` to attrs — duplicating lineage creates two sources
of truth that drift.

### Scopes

- `scope:bucket` — full coverage of the **bucket root** (e.g.
  `s3://my-bucket/`). The default when an auto-build is justified.
- `scope:directory` — coverage of a specific subdirectory under the
  bucket. Set ONLY when the user explicitly opted into a narrower scope
  via the dialogue (§4.7). Source slug includes the directory path so
  different subdirs do not collide (e.g.,
  `source:my_bucket__images_subset`).
- `scope:sample` — covers only a sample of the source. One-shot, no
  future savings.
- `scope:onetime` — Task layer that should NOT be persisted (default for
  Task).

### `.save()` checklist for L1/L2/L3

Before calling `.save()`, verify every item:

- `attrs` includes `cast:<layer>`, `scope:<scope>`, `source:<slug>`
  (stripped per the prefix-strip rule above).
- `attrs` includes `model:<id>@<version>` and `preset:<name>` whenever a
  model/encoding was used. These are the build signature read on refresh
  (§5).
- `description="…"` set to a one-line human summary.
- The `read_storage` includes `update=True, delta=True` for L1/L2/L3
  builds (see §5).
- **The Pydantic output type carries the operation's full output**, not
  the current question's projection (§1.2). Derived columns — counts,
  booleans, top-k labels, aggregates — live in a Task, not on the
  substrate row.

### `.gen()` pre-flight checklist

(Added because the test/ run lost a build to a missing return
annotation — schema collapsed to `str` over 1,737 rows.)

When you call `.gen(name=fn)`:

- `fn` has an **explicit** `-> Iterator[T]` annotation. Not bare return,
  not `Iterator`, not `Iterator[Any]`.
- `T` is either a primitive or a Pydantic `BaseModel` declared at
  **module level** — not nested inside the function, not imported lazily.
- No `from __future__ import annotations` anywhere in the module. It
  stringifies hints and DataChain's signal-schema resolution rejects the
  string-vs-class mismatch.

A subtle wrong:

```python
def extract_frames(file: dc.VideoFile):           # ← no return annotation
    for video_frame in file.get_frames(step=N):
        yield Starss23FrameAsset(...)             # ← BaseModel exists, but
                                                  #   the saved schema saw `object`
                                                  #   and collapsed to str(file.path)
```

The right form:

```python
def extract_frames(file: dc.VideoFile) -> Iterator[Starss23FrameAsset]:
    ...
```

### Heavy-init resources go through `.setup()`, never module-level lazy globals

Models, tokenizers, DB clients, any object with a non-trivial init cost
load via `chain.setup(resource=load_fn)` — NOT a `_resource = None`
module-level lazy global. The lazy-global pattern works for
`parallel=1` but has subtle pitfalls under `parallel=N` (each worker
process initializes its own copy on first call rather than at chain
start, the resource is invisible in the chain definition, and global
state across multiple chains in the same process can leak). The
canonical pattern and its rationale live in `core/SKILL.md`
"Pre-Generation Checklist"; the knowledge skill enforces them
unconditionally when generating any L2/L3 build script.

### Typed File references on substrate rows — never bare paths

A substrate row that points at a file MUST carry the typed reference,
not a string path. For video: `dc.VideoFile`. For images:
`dc.ImageFile`. For audio: `dc.AudioFile`. For generic: `dc.File`. The
typed reference preserves path, etag, size, version, access config,
AND the methods the file's modality needs (`.get_info()`,
`.get_frames()`, `.read()`, `.open()`).

```python
class FrameDetections(BaseModel):
    source: dc.VideoFile      # ✓ typed reference — full lineage + methods + UI render
    timestamp: float
    detections: list[Detection]
```

```python
class FrameDetections(BaseModel):
    video_path: str           # ✗ bare path — lineage lost, no methods,
                              #   no UI render, no consistency check
    timestamp: float
    detections: list[Detection]
```

A row with `video_path: str` instead of `source: dc.VideoFile` loses:

- **Lineage / consistency.** DataChain can no longer trace the row back
  to the source listing; etag-change detection breaks.
- **UI visualisation.** Studio and the local UI render `dc.File`
  subclasses with previews / thumbnails / sample players. A bare
  string is just text.
- **Downstream methods.** `frame.get_np()` and similar require the
  typed object on the row; `path.get_np()` does not exist.
- **Access config.** The `anon=True` / credentials from the original
  `read_storage` are carried by the `VideoFile`, not by the string.

The **only** legal exception is when the user explicitly asks for a
string-only payload ("I just need the path", "no DataChain pointer
here"). In that case, keep the typed `dc.VideoFile` on the row AND add
the string as a secondary derived column — don't replace.

(Common cause of this regression: `.gen()` dropping a parent column,
agent's response is to write a string column to "carry the path
forward". The correct fix is to put `source: dc.VideoFile` in the
yielded Pydantic model and use the source row's `file` directly, not
synthesize a string.)

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
depends on, NOT a decision to skip layers.

If the task involves any UDF (decode, embedding model, LLM call, classifier,
file decode), **at least one CAS layer is required and must be saved.**
There is no "no layer needed" branch for UDF tasks.

Examples:
- similarity search → Sense layer of embeddings, possibly on top of Asset
  layer of frames/images
- "find videos with X" → Sense layer of classifications (and possibly
  thin Asset of materialized frames)
- "summarise this bucket" → Container of header metadata + optional Sense
  LLM annotations
- "extract frames from videos" → Asset layer of frames on top of bucket scan

**Container layer = partial-read or metadata-only work.** Justified when
per-row work reads only a bounded prefix of each file (headers, schema,
EXIF, tags, footer) or a sidecar — not the full file body. **Full file
decode → L2 Asset, not L1**. Header reads are native in PIL Image, pyarrow
Parquet, h5py, `dc.VideoFile.get_info()`, pydicom, `file.read(N)` with
bounded N. **Filter-only Container datasets are forbidden** —
`.filter(glob)` reads zero bytes; inline into `read_storage` glob or an
L2/L3 `.filter()`. The bucket scan in `dc-knowledge/buckets/` already
covers "what files are here". §6 Critical Rule "at least one" is
satisfied by Sense alone.

### 4.2 Look for mixture opportunities

If the task names two or more datasets (or two or more bucket regions /
sources), the Asset-level combination of them is itself a CAST artifact.
Surface it.

### 4.3 Direct reuse first

From `dc-knowledge/index.md`, for each required layer × source, check if
an existing dataset already covers the question (even partially). If yes,
write the pipeline as `dc.read_dataset(...)` over it.

**Celebrate the reuse.** In the response, name the layer being reused and
quote the saved cost — e.g., "Reusing `l3_sense_product_catalog_clip_embeddings`
(built last session). This query is ~$0.002 instead of the $1.40 the
embedding pass would otherwise cost — exactly the win the Sense layer was
built for." This is the moment that teaches the methodology to the user;
do not skip it.

### 4.4 Reduce-to-CAS if direct reuse impossible

Before defaulting to a raw rebuild, work the problem from the other
side: can the task be reformulated to operate on an *existing* CAS layer
plus a small Task delta? Examples: a new similarity question on the same
bucket → reuse Sense embeddings, just change the query vector; a new
"find X" question → reuse Sense classifications and add a filter. Spend
real effort here — propose at least one reformulation when any CAS layer
for this source exists.

### 4.5 Cost gate on CAS reuse

Reuse a CAS layer only when it gives a meaningful win — at least ~2×
speedup or ~2× $-saving versus the raw rebuild. If the layer technically
covers the data but reading it is no cheaper than re-reading raw storage,
do not force the reuse; the methodology is justified by economics, not
formalism.

### 4.6 Derive task minimum BEFORE the calibration run

For decode-heavy sources (video / audio / large H5 / NIfTI / multi-page
PDF), name the minimum fidelity the task requires across three axes:

- **Temporal sampling.** Shortest event the answer depends on. People on
  screen → 1 fps; brief impacts/flashes → ≥5 fps; one scene/video → 1
  frame.
- **Spatial resolution.** Smallest visual detail. Object detection of
  people/cars → 640px; OCR / small text → full resolution.
- **Encoding.** Audio full-rate (transcription) vs resampled (speech-vs-silence);
  PDF OCR vs text-layer.

This minimum is the **floor** for any thin-Asset preset and what the
calibration measures against. Never calibrate at coarser sampling than
the task requires. If genuinely ambiguous, ask one targeted question —
do not guess.

**Silent defaults are forbidden.** The skill picks five things by
default that materially affect the result and the cost — each one
must be surfaced to the user, not chosen silently:

| Parameter           | Default rule              | User must see / pick                                |
|---------------------|---------------------------|-----------------------------------------------------|
| Model id + version  | from task type            | **ASK** — model and version change results materially (yolo11n vs yolov8n vs RT-DETR; clip-vit-b32 vs siglip; gpt-4o-mini vs gpt-4o). Quote one alternative with a "+precision / +recall / +wall" hint. |
| Confidence threshold| 0.25 for detection        | **ASK** — directly determines what gets counted. Different defaults across models. |
| Sample rate (fps / window) | from task-minimum  | NOTIFY in the §4.10 dialogue's `Task minimum` line. User can override but does not need to confirm. |
| Image size (imgsz)  | model-native (often 640)  | NOTIFY. User overrides only if the task minimum (§4.6 spatial) demands higher. |
| Preset / quality    | task-minimum-derived      | NOTIFY in `Task minimum` line. |

**ASK** = surface via an AskUserQuestion (with the default
pre-selected and one alternative shown) BEFORE running calibration.
Two questions max; combine into one multi-select if appropriate.
**NOTIFY** = render verbatim in the `Task minimum` line of the §4.10
dialogue; do not require explicit confirmation, but the value must
be visible.

Why this matters: a yolov8n@0.25 result and a yolo11n@0.50 result on
the same bucket give different person/car/truck counts. The user
asking for "objects in videos" cannot tell from the agent's output
which model/threshold produced the numbers unless the dialogue made
it explicit. The model and threshold MUST be in the substrate `attrs`
(per §3 `.save()` checklist) AND in the conversation the user can
audit later.

### 4.7 Walk the ladder top-down (NEW — the test_c2 fix)

After identifying the minimum layer the question requires (usually Sense
or Task), walk DOWN the ladder and for each layer underneath, answer
three questions in writing:

1. **"Would this layer, if it existed, cut my current build by ≥2× OR be
   plausibly reused by a future question on this source?"** If yes →
   it's a build candidate.
2. **"What shape would it have?"**
   - *Container*: which header fields. (e.g., `width`, `height`, `codec`,
     `fps`, `duration_s`, `audio_channels`, `audio_sample_rate`.)
   - *Asset*: which preset + which destination. (e.g., 0.5 fps × 640px
     JPEGs at `gs://<user-bucket>/<dataset_name>/`.)
   - *Sense*: which model + which output schema. Save the full output
     (§1.2), not a projection.
3. **"Is the operation prompt/call generalisable to cover related
   axes?"** For deterministic ops (decode, header parse) — list every
   field the parser exposes. For LLM/VLM calls — extend the prompt
   while you're paying for one call. Same API cost, multiplies reuse.

Concrete walk for "find videos with people":

- Minimum layer = Sense (YOLO detections).
- Walk down to Asset: thin-Asset of materialized 0.5 fps frames in S3 —
  YES, it's a candidate. Decouples decode from inference, reusable for any
  other per-frame model (CLIP, BLIP, OCR), reusable for human annotation
  UIs. Preset: 0.5 fps × 640px × JPEG q80. Destination: `gs://<user-bucket>/l2_asset_<source>_frames/`.
- Walk down to Container: `video_info` via `get_info()` — YES, it's a
  candidate. Codec/fps/duration/dimensions/audio is reusable across every
  video task on this bucket.

Concrete walk for "is this product description promotional?" (LLM case):

- Minimum layer = Sense (LLM classifier on text).
- Generalise the prompt (question 3): same call can also extract `tone`,
  `claims_made`, `target_audience`, `language`, `length_bucket`. Same
  API cost, multiplies reuse. Row schema includes `prompt_tokens`,
  `completion_tokens`, `model_id`, `model_version`, `finish_reason`.

Surface ALL candidates in §4.10's dialogue. The user picks which to build.
The agent does **not** pre-filter to "just what the current question
needs" — that's exactly the substrate-erosion regression CAST exists to
prevent.

**Anti-pattern (Round 4 regression, test_c2).** Choosing fused-L3
(shape 3 below) silently because "the Sense layer is the only consumer
for this question" — without surfacing what Asset / Container would
unlock for the next question. Fused-L3 is a valid shape when the user
has explicitly chosen it, not the agent's silent default.

### 4.7b L1 wiring rule — three modes

L1 captures task-agnostic source characteristics (codec, dimensions,
fps, duration, format, schema). It is the most general substrate;
future tasks (unknown today) will query headers regardless of what the
current task wants. So L1 data is always worth capturing — but
*whether* it lives as a standalone dataset or folded onto the L2 row
depends on the substrate's maturity on the source.

**Mode 1 — First L2 on this source, no L1 yet → fold into L2.**

Add the header read as a column on the L2 row. The L2 UDF usually
already needs the header to compute its sampling step, so capture is
essentially free.

```python
class VideoFrameAsset(BaseModel):
    source: dc.VideoFile
    info: dc.Video                # folded L1 — codec, fps, duration, dims
    timestamp: float
    sampled_frame_index: int
    frame: dc.File                # only for materialized-thumbnail L2
```

Do NOT build a separate L1 dataset at this stage. The folded column IS
the L1 data, just colocated. Future tasks can still query headers via
`dc.read_dataset("l2_…").select("source", "info")`. At one L2, a
separate L1 is a dataset with no second consumer — pure bookkeeping
cost.

**Mode 2 — L1 already exists on this source → L2 reads from L1, no
re-call.**

```python
def to_frames(file: VideoFile, info: Video) -> Iterator[VideoFrame]:
    step = max(1, int(round(info.fps / TARGET_FPS)))
    yield from file.get_frames(step=step)

chain = dc.read_dataset("l1_container_<source>_headers").gen(frame=to_frames)
```

Never re-call `file.get_info()` inside the UDF — "L1 paid for this
once" is the whole point of the layer. Re-calling it is the test_c2
regression: L1 becomes decorative substrate.

**Mode 3 — 3+ L2 layers on this source carry folded headers → extract
L1.**

Header data is now duplicated across L2 versions; refresh
coordination gets expensive. Break L1 out into a canonical dataset:

```python
# project from an existing L2:
(dc.read_dataset("l2_…")
   .select("source", "info")
   .save("l1_container_<source>_headers", attrs=[...]))
# OR rebuild from raw storage, then migrate L2 reads to read from L1
# on next refresh.
```

After extraction, future L2 builds switch to Mode 2.

**Dialogue impact.** §4.10 and §4.10b should reflect this: the
recommended Asset shape is "L2 with folded `info` column" when no L1
exists yet on the source. Standalone L1 appears as a separate build
candidate ONLY when:

- The task is genuinely header-only (the user asked "what codecs are
  used" or "list all 4K videos" and no decode is needed) — build L1
  alone, no L2.
- ≥3 L2s already exist on this source and the duplication is biting.
- The user explicitly asks for it.

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

**Both branches are mandatory** (test_c2 regression fix). Single-branch
calibration leaves R = thin/no undefined — which forces a fused-L3
default in §4.9 and prevents the layer choice. If only one branch was
run, re-run the missing branch before proceeding to §4.9. If the full
calibration is genuinely infeasible (empty source, network unreachable),
auto-build is OFF and the agent goes to the §4.10 dialogue with the
fallback rules of thumb below.

Extrapolate: `wall_full = (wall_sample / N) × total × 1.5`;
`asset_full_bytes = (sample_bytes / N) × total × 1.5`. Cost in $ from the
recall-economics tier unless calibration disagrees by >3×.

**Calibration measurements are reusable across scope changes** within
the same media / decode profile. When the user narrows or broadens
scope after the dialogue was rendered, do NOT re-calibrate "for
cleanliness" — just re-extrapolate the existing per-item numbers to
the new item count. See §4.10.5 for the narrow set of cases that
warrant a fresh run, and for the user-facing question to ask when in
doubt.

**Sanity floor.** Estimate disagreeing with the recall-economics tier by
>100× → re-calibrate, do not paper over.

**Fallback rules of thumb** (use only when calibration is impossible):
- Container (header-only parse): ~0.5 ms/file.
- Asset (full-file decode, materialize): 50–500 ms/file; for video / audio
  / large blobs budget per source file, not per unit.
- Sense — paid API (LLM/VLM, hosted embeddings): ~$0.001–0.01/row plus
  latency.
- Sense — local model on CPU: 5–50 ms (small classifiers), 50–500 ms
  (mid-size detection / segmentation), 0.1–0.5× realtime (ASR).
- Sense — local model on GPU: ~10–100× faster than CPU baselines above.

**L2 shape — three valid options for heavy-decode sources** (video frames,
audio segments, PDF pages, archive entries, multi-channel sensor data,
etc.). Pick deliberately based on selection criteria; none is forbidden.

1. **Pointer-row L2.** Emit `Iterator[dc.VideoFrame]` (or
   `Iterator[dc.VideoFragment]` for audio) from `.gen()` walking
   `file.get_frames(step=N)`. Saved row = `{video: VideoFile, frame: int,
   timestamp: float}` — pure metadata, bytes per row. Downstream
   `frame.get_np()` calls `video.open()` which streams from storage with
   DataChain's local caching. **No data duplication, no re-download per
   frame.** Simplest shape; the right default for in-DataChain consumption.
2. **Materialized thumbnail L2.** Emit rows with `frame: dc.File` pointing
   at written JPEGs / segment files in storage. Choose when downstream
   needs files-on-disk for **non-DataChain interop** — annotation UIs,
   training pipelines that consume files directly, browsing in the storage
   browser. **Row schema uses typed file objects:**

   ```python
   class VideoFrameAsset(BaseModel):
       source: dc.VideoFile      # preserves path/size/etag + APIs
       timestamp: float
       sampled_frame_index: int
       frame: dc.File            # pointer to materialized payload in storage
   ```

   - **Storage shape, by format:** standard containers (JPEG/PNG/WAV/MP4/…)
     → file in storage + `dc.File` pointer; custom binary (numpy,
     embedding tensors) → bytes column on row; text/JSON (transcripts,
     summaries) → string column on row.
   - **Destination** (ask once per session, then reuse): source on
     cloud → derivative on cloud same scheme, default proposal
     `gs://<user-bucket>/<dataset_name>/` — agent ASKS to confirm bucket
     + prefix. Source on local FS → parallel local prefix like
     `./<dataset_name>/`, still confirm. **NEVER default to
     `.datachain/thin-assets/`** unless the user asks for local-only.
   - **Default presets** (must meet §4.6 minimum): video — 1 fps × 640px
     long side × JPEG q80; audio — 1-sec windows × 16 kHz mono PCM.
3. **Fused decode-once L3 (no separate L2).** `.gen(VideoFile →
   Iterator[Detection])` reads `read_storage` directly, decodes the source
   once sequentially, emits per-detection (or per-frame) rows. Choose
   **only when the user has explicitly opted out of L2** via the
   dialogue, OR when L2 doesn't apply (e.g. one-pass embedding pipeline
   with no plausible reuse). Never the agent's silent default.

**Calibration policy.** `.persist()` not `.save()`; no `attrs` /
`description` on calibration runs; no enrichment. End state: no `calib_*`
row in `dc.datasets()` and no `calib_*` in `dc-knowledge/`.

### 4.9 Decide branch — strict order, bucket-root scope throughout

**4.9a Bucket root from URI.** From `s3://dc-readme/oxford-pets-micro/images/`,
root is `s3://dc-readme/`. A subdir the user phrased the task around is
NOT the root.

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

**Template** (substitute measured calibration numbers verbatim; render
every layer that passed §4.7's "build candidate" test):

```
Task: {one-line user goal}.
Source: {bucket_size} / {n_total} {ext_summary} / avg file {avg_size}.
Task minimum (§4.6): {sample_rate}, {resolution}, {format}.
Calibration (N={n_calib}): no-Asset ~{wall_calib_no}s, thin-Asset ~{wall_calib_thin}s
                           (+{thin_storage_calib}); R = {R:.1f}.
{if R > 2:} thin Asset costs ~{R:.1f}× more — substrate-vs-immediate
trade-off. Recommendation follows CAST (substrate compounds); pick no-Asset
only for genuinely one-shot answers.

What to leave behind on this source — pick any combination:

──────────────────────────────────────────────────────────────────────
L1 Container — {what it'd hold, e.g. video headers via get_info()}
  Row schema: {field list}
  a) WHOLE bucket: ~{wall_l1_bucket}, ${cost_l1_bucket}     [recommended]
  b) directory only ({subdir}): ~{wall_l1_dir}, ${cost_l1_dir}
  c) skip

L2 Asset — {what it'd hold, e.g. 0.5 fps × 640px JPEGs at gs://…/l2_…/}
  Row schema: {field list, with full operation output, no projections}
  a) WHOLE bucket, thin {preset}: ~{wall_l2_bucket}, ${cost_l2_bucket}, ~{asset_gb}GB     [recommended — CAST substrate]
     Destination proposal: gs://<user-bucket>/<l2_asset_name>/ — confirm or override.
  b) directory only, thin {preset}: ~{wall_l2_dir}, ${cost_l2_dir}
  c) tighter preset (1/5 sec or 320px), whole bucket: ~{wall_l2_tight}, ${cost_l2_tight}
  d) skip (fused-L3 below re-decodes on the next question)

L3 Sense — {what it'd hold, e.g. YOLO11n full COCO output per frame}
  Row schema: {full operation output} + telemetry (inference_ms, model_id, model_version)
  (No `person_count`, `has_person`, etc. — those are Task queries — §1.2)
  a) WHOLE bucket on top of L2: ~{wall_l3_bucket}, ${cost_l3_bucket}     [recommended]
  b) directory only on top of L2: ~{wall_l3_dir}, ${cost_l3_dir}
  c) fused-L3 directly from files (skips L2): ~{wall_l3_fused}, ${cost_l3_fused}
  d) sample only ({sample_n}): ~{wall_sample}, ${cost_sample}

{if LLM/VLM Sense:}
L3 Sense — optional prompt extension (same API cost, multiplies reuse)
  Default prompt covers: {axis_1}.
  Optional axes (free at same call): {axis_2}, {axis_3}, {axis_4}.
  Pick: default | extended | custom
──────────────────────────────────────────────────────────────────────

Recommendation: {explicit combination, e.g. "L1 (a) + L2 (a) + L3 (a)"}
Reasoning: {one line — what compounds across future questions}.
```

**Render the template verbatim — do not paraphrase the recommendation.**
Failure modes that defeat CAST doctrine:

- **Do NOT reorder** so no-Asset comes first. Thin Asset is the recommended
  option, always.
- **Do NOT replace `[recommended]` with conditional phrasing** like
  "recommended if you expect future queries" / "useful if you'll re-ask".
  The recommendation is unconditional.
- **Do NOT add competing labels to no-Asset / fused-L3** like "best for
  this one answer". The user picks; the agent does not nudge with
  "best for X" tags.

**Always quote a number.** Every option carries an estimate. If
calibration is feasible but wasn't run, run it first and then render the
dialogue.

**AskUserQuestion options MUST mirror the rendered prose dialogue.**
Every named option (a / b / c / d) from the prose dialogue must appear
as an `AskUserQuestion` option with the same label and the cost
numbers in the option's description. Do NOT collapse the rich prose
options to short labels like "Recommended: L1+L2+L3" — the user needs
the wall, the dollars, and the trade-offs visible at the moment they
pick. If the prose lists four trade-offs, the AskUserQuestion has
four options with those four descriptions. The prose render is the
source of truth; AskUserQuestion is just the UI affordance.

Concretely: every option's `description` field includes the wall-time
estimate, the dollar estimate, and the one-line trade-off from the
prose. The `label` field uses the same short tag (e.g.
"WHOLE bucket thin Asset", "Directory only", "Fused-L3", "Sample") —
do not abbreviate beyond what the prose used.

### 4.10b Small-scope dialogue variant

When the chosen scope has **<50 source items** OR the full L3 wall time
is **<5 min**, the L1+L2+L3 ladder is often over-engineering: the
substrate covers only the narrow subdirectory, not its siblings, so the
per-question reuse payoff is small while the bookkeeping cost (three
saved datasets, three KB entries, three rebuilds on schema change) is
unchanged. Render the small-scope template instead of §4.10:

```
Scope: {dir}, {n} items, full L3 wall ~{wall_l3}.
Substrate at this scope covers ONLY {dir} — siblings ({sibling list})
would need a separate build.

Two ways forward:

  a) [recommended for one-shot] fused-L3 only
     ~{wall_l3_fused}, leaves no reusable substrate; re-decodes if
     you ask another question on this directory later.

  b) full ladder L1 + L2 + L3 at directory scope
     ~{wall_full_dir}, $L1+L2+L3 substrate covers ONLY {dir}.
     Worth it only if you expect >2 future questions on this exact
     subdirectory.

  c) full ladder L1 + L2 + L3 at bucket scope (broader)
     ~{wall_full_bucket}, substrate compounds across siblings
     ({n_sibling_items} extra items, ~{extra_wall} extra wall).
     The CAST default; pick this when bucket-wide substrate is the
     real goal.
```

Lead with (a) — the user narrowed scope; respect the intent. Surface
(c) so the bucket-wide alternative is visible (this is the "compounds"
nudge); never silently default to (b) just because directory scope was
named.

If neither §4.10 nor §4.10b clearly applies (mid-size scope, ambiguous
wall), default to §4.10 (full template).

### 4.10.5 Scope-change re-dialogue protocol

**Triggers — any of these is a scope-change event.** The agent must
detect this mechanically BEFORE writing any code or issuing the next
AskUserQuestion:

1. **Free-text reply outside the offered AskUserQuestion options.** If
   the user did not pick any of the labelled options (a/b/c/d) and
   instead typed a free-form constraint, that's an override — treat it
   as a scope change, even if the words sound like a "small
   refinement". (The test_c2 trace: user replied "only one dir" when
   none of the four bucket-wide options matched. Agent must NOT just
   ask "which dir?" and proceed — must re-render at the new scope
   first.)
2. **Scope-narrowing keywords** in any user message: "only", "just",
   "limit to", "single", "one dir", "this dir only", "subset",
   "narrow to", "skip the …".
3. **Scope-broadening keywords**: "whole bucket", "all of …", "every
   …", "across all", "include the train/eval split".
4. **Explicit dir / prefix / glob** named by the user that wasn't in
   the original dialogue (e.g. "do `video_dev/dev-test-sony/`
   instead").

When any trigger fires, reuse the existing calibration measurements
and re-extrapolate to the new item count. Render the dialogue again
with the updated cost numbers — picking §4.10 or §4.10b based on the
*new* scope's size. Never proceed to script generation off the prior
recommendation.

**Do NOT re-calibrate by default.** Re-running 60–240 s of calibration
on the same media/decode profile is wasted wall time. The only cases
that warrant a fresh calibration are:

- **Media-type change** within the same task (video task → audio task
  on the same bucket).
- **Decode-profile change** (e.g., 360° 1920×960 MP4 → 4K MP4, or
  small JPEG thumbnails → original-resolution TIFF) where the prior
  per-item wall is unlikely to extrapolate.

Even then, do NOT auto-recalibrate — ASK the user one targeted
question:

```
Scope changed from {old} to {new}. Existing calibration was on
{sample-shape}; the new scope's items are {different-shape}. Re-run
calibration (~{calib_wall}) or extrapolate from the existing one?
Extrapolation is fine for most narrowings within the same media type.
```

Default to extrapolation when in doubt.

**The recommendation can flip even when the measurements are reused.**
A bucket-scope dialogue may have recommended the full L1+L2+L3 ladder;
the same measurements at directory scope may now point to fused-L3
(via §4.10b). Re-deriving the recommendation is the whole point of
re-rendering the dialogue — never silently carry the prior
recommendation forward at the new scope.

**Orphan-substrate warning.** When the user picks the full ladder at
`scope:directory` (small-scope variant option (b)), surface the trade
explicitly in the post-build report: *"L1+L2 substrate is
directory-scoped — it covers only {dir}. A future task on a sibling
subdirectory ({sibling list}) will require a separate build, or you
can promote to `scope:bucket` later by re-running the build scripts
with the bucket-root URI."* The agent should NOT auto-rebuild at
bucket scope without asking.

### 4.11 Containerise raw JSON / sidecars too

If the task pulls structured JSON / Parquet / CSV from the bucket and
parses it inline, propose lifting that parse into an
`l1_container_<source>_<descriptor>` dataset so the parsed schema becomes
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
`("file.path", "file.etag", "file.version")` (catches re-uploads with
same path but different content); `delta_compare=None` uses all
non-`delta_on` fields. Same code on first build and subsequent runs.

When the source is `dc.read_dataset()` (chaining from a parent CAS layer),
the parent's delta semantics propagate — no extra args needed.

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

## 6. Critical Rules (CAST-adjacent — full list in SKILL.md)

These five rules are restated here because they encode CAST doctrine
directly. SKILL.md holds the full list; the bullets below cannot drift.

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
