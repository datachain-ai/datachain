---
name: datachain-core
description: Use ONLY for abstract DataChain SDK questions — API usage, method signatures, or code patterns — when no specific dataset or bucket is referenced. If the request mentions creating, saving, listing, exploring datasets or buckets, use datachain-graph instead.
---

You are now loaded with expert-level DataChain SDK context. Apply every rule below when generating DataChain Python code. Do not deviate.

### Graph context (read-only)

If `datachain/graph/index.md` exists, read it at conversation start for dataset and bucket awareness. When the user mentions a specific dataset or bucket by name, read the matching `.md` file under `datachain/graph/datasets/` or `datachain/graph/buckets/`. Never create or modify files under `datachain/graph/` — that directory is owned by the datachain-graph skill.

---

## Section 1 — Critical Rules (Must-Never-Break)

```
1. TRAILING SLASH: Always add / to bucket/prefix paths.
   ✓ dc.read_storage("s3://bucket/images/")
   ✗ dc.read_storage("s3://bucket/images")  ← permission error on anon access

2. TYPE HINTS NOT params/output: Never pass params= or output= to map/gen/agg.
   DataChain auto-infers from function signature and return type annotation.
   ✓ def fn(file: ImageFile) -> list[float]: ...
     chain.map(emb=fn)
   ✗ chain.map(fn, params=["file"], output={"emb": list[float]})

3. LAMBDA ONLY FOR STR: Lambdas have no type annotation → DataChain defaults to str.
   Use lambdas only when return type is str. For everything else, write a named function.
   ✓ chain.map(name=lambda file: file.path.split("/")[-1])   # str is fine
   ✗ chain.map(size=lambda file: file.size)                  # int → treated as str → broken

4. COLUMN NAMING: keyword in map/gen/agg = new column name.
   chain.map(embedding=fn)  → column is named "embedding"
   chain.gen(frame=fn)      → column is named "frame"

5. INPUT PARAM: The file column is always named "file" regardless of modality.
   def process(file: ImageFile) -> ...  ← always "file", not "image"

6. ALWAYS PARALLEL: Every chain with map/gen/agg must call .settings(parallel=True).
   Without it, execution is single-threaded. Only reduce parallelism for heavy
   per-worker memory loads (large ML models).
   ✓ chain.settings(parallel=True).map(emb=fn)
   ✗ chain.map(emb=fn)  ← single-threaded, wastes cores

7. COLUMN-COLUMN ARITHMETIC: Use chain.column() instead of C() when combining two columns.
   C() does not carry type info → engine can't infer the result type.
   chain.column("name") returns a typed column derived from the chain's schema.
   ✓ chain.mutate(total=chain.column("price") * chain.column("qty"))
   ✓ chain.mutate(discounted=C("price") * 0.9)          # scalar literal → type inferred
   ✗ chain.mutate(total=C("price") * C("qty"))           # no type → error

8. READ NOT FROM: Use dc.read_* module functions, not deprecated DataChain.from_* methods.
   ✓ dc.read_csv("s3://data.csv")
   ✗ DataChain.from_csv("s3://data.csv")  ← deprecated

9. ONE SIGNAL PER MAP/GEN/AGG: Each call accepts exactly one signal (keyword).
   For multiple columns, chain calls or return a DataModel.
   ✓ chain.map(a=fn1).map(b=fn2)          # chained — two columns
   ✓ chain.map(info=fn)                    # DataModel with named fields
   ✗ chain.map(a=fn1, b=fn2)              # ERROR: multiple signals

10. NO TUPLE RETURNS: Always prefer DataModel classes to tuple in map/gen/agg functions
    until user directly asks for tuple.
    ✓ def fn(file: File) -> MyModel: ...   # named fields via DataModel
    ✓ def fn(file: File) -> int: ...       # single scalar
    ✗ def fn(file: File) -> tuple[int, int]: ...  # → col_0, col_1
```

---

## Section 2 — Golden Rule

```
Prefer metadata operations over Python operations.
Metadata ops run in the engine (SQLite local / ClickHouse Studio) at warehouse speed.
Python ops (map/gen/agg) spin up Python workers and are expensive.

Use Python ops ONLY when you need:
  - File content (file.read(), file.open())
  - ML model inference
  - LLM calls
  - External API calls

Everything else → use filter/mutate/group_by/merge with func.*
```

---

## Section 3 — Import Cheat Sheet

```python
import datachain as dc
from datachain import DataChain, C, DataModel, File
from datachain import TextFile, ImageFile, VideoFile, AudioFile
from datachain import BBox, OBBox, Pose, Pose3D, Segment
from datachain import func
```

Stateful UDF base classes (two equivalent styles):
```python
# Style 1 (preferred shorthand)
class MyMapper(dc.Mapper): ...
class MyGen(dc.Generator): ...
class MyAgg(dc.Aggregator): ...

# Style 2 (explicit import)
from datachain.udf import Mapper, Generator, Aggregator
```

---

## Section 4 — Core API Reference

**Entry points:**
```python
dc.read_storage("s3://bucket/prefix/", type="image")   # File / ImageFile etc.
dc.read_csv("s3://bucket/data.csv")
dc.read_json("s3://bucket/ann.json", jmespath="images")
dc.read_parquet("s3://bucket/data/*.parquet")
dc.read_hf("dataset-name", split="train")
dc.read_pandas(df)
dc.read_values(scores=[1.2, 3.4])
dc.read_records([{"a": 1}, ...])
dc.read_database("SELECT * FROM t", "sqlite:///local.db")
dc.read_dataset("name")                    # latest version
dc.read_dataset("name", version="2.0.0")  # specific version
```

**Metadata operations (run in engine, fast):**
```python
chain.filter(C("file.size") > 1000)
chain.filter((C("det.label") == "cat") & (C("det.conf") > 0.9))
chain.filter(C("file.path").glob("*.jpg"))
# String methods (filter)
chain.filter(C("name").contains("alice"))
chain.filter(C("name").startswith("al"))
chain.filter(C("name").endswith("ob"))
chain.filter(C("name").like("%ob"))
chain.filter(C("name").ilike("AL%"))
chain.filter(C("name").regexp_match("^al"))
# NULL checks
chain.filter(C("name").isnot(None))
chain.filter(C("name").is_(None))
# Range / membership
chain.filter(C("price").between(10, 25))
chain.filter(C("name").in_(["alice", "bob"]))
# Logical combinators: & (and), | (or), ~ (not) -- always parenthesize
chain.filter((C("x") > 1) & (C("y") < 10))
chain.filter(~(C("x") > 1))
chain.mutate(ext=func.path.file_ext(C("file.path")))
chain.mutate(dist=func.cosine_distance(C("emb"), reference))
# Column-column arithmetic (use chain.column(), not C())
chain.mutate(total=chain.column("price") * chain.column("qty"))
chain.mutate(discounted=C("price") * 0.9)    # scalar → C() is fine
# Floor division, modulo, negation
chain.mutate(bucket=C("price") // 10, remainder=C("qty") % 3, neg=-C("score"))
# Type conversion
chain.mutate(price_int=chain.column("price").cast(sa.Integer))  # import sqlalchemy as sa
chain.group_by(cnt=func.count(), total=func.sum(C("file.size")), partition_by="category")
chain.order_by("dist")
chain.order_by("score", descending=True)
chain.distinct("response.text")
chain.limit(100)
chain.select("file", "score", "label")
chain.select_except("internal_id")
chain.merge(other, on="id", right_on="meta.id")
chain.union(other)
chain.subtract(other)
chain.diff(other, on="id", compare=["score"])
chain.file_diff(other)
```

**Python operations (run in Python workers, expensive):**
```python
chain.map(col_name=fn)        # 1 input → 1 output record
chain.gen(col_name=fn)        # 1 input → N output records
chain.agg(col_name=fn, partition_by="key")  # group → aggregate
chain.batch_map(fn, batch_size=32)
```

**Setup and execution settings:**
```python
chain.setup(model=lambda: load_model())   # initialize once per worker
chain.settings(parallel=True, cache=True, prefetch=10, workers=50)
```

**Terminal operations (trigger execution):**
```python
chain.save("dataset_name")                     # versioned named dataset
chain.save("ns.proj.name", update_version="minor")
chain.persist()                                # anonymous cache
chain.show(limit=10)
chain.collect("col1", "col2")                  # → list of tuples
chain.to_pandas()
chain.to_parquet("output.parquet")
chain.to_csv("output.csv")
chain.to_pytorch(transform=..., tokenizer=...)
chain.to_storage("s3://output/", signal="file", placement="filepath")
chain.count()
chain.sum("column")
chain.avg("column")
```

**Delta + incremental:**
```python
dc.read_storage("s3://bucket/", update=True, delta=True,
                delta_on="file.path", delta_compare="file.mtime")
```

---

## Section 5 — Type System

**DataModel -- custom structured types:**
```python
from datachain import DataModel

class Detection(DataModel):
    label: str
    confidence: float
    bbox: BBox

# External Pydantic models must be registered:
dc.DataModel.register(MistralResponse)
```

**File types (all inherit from File → DataModel):**

| Type | `type=` param | `.read()` returns | Extra methods |
|---|---|---|---|
| `File` | (default) | `bytes` | `.read_text()`, `.open()`, `.ensure_cached()` |
| `TextFile` | `"text"` | `str` | `.read_text()` |
| `ImageFile` | `"image"` | `PIL.Image` | `.get_info()` → `Image(width,height,format)` |
| `VideoFile` | `"video"` | -- | `.get_frames(step=N)` → `VideoFrame[]`, `.get_fragments(duration)` → `VideoFragment[]`, `.get_info()` → `Video(fps,duration,codec,...)` |
| `AudioFile` | `"audio"` | -- | `.get_fragments(duration)` → `AudioFragment[]`, `.get_info()` → `Audio(sample_rate,channels,duration,...)` |

Sub-file units (DataModel):
- `VideoFrame` -- `.get_np()` → ndarray, `.save(path)`
- `VideoFragment` -- `.save(path)`
- `AudioFragment` -- `.get_np()` → `(ndarray, sample_rate)`, `.save(path)`

**Annotation types:**
```python
BBox(title="car", coords=[x1,y1,x2,y2])            # PASCAL VOC
BBox.from_coco([x,y,w,h], title="car")
BBox.from_yolo([cx,cy,w,h], img_size=(640,480))
BBox.from_albumentations([x1n,y1n,x2n,y2n], img_size)
bbox.to_coco() / .to_yolo(img_size) / .to_albumentations(img_size) / .to_voc()
bbox.point_inside(x, y)  # → bool
bbox.pose_inside(pose)   # → bool

OBBox(...)                # oriented bbox -- four corner points

Pose(x=[...], y=[...])                       # 2D keypoints
Pose3D(x=[...], y=[...], visible=[...])      # 3D with visibility

Segment(title="road", x=[...], y=[...])      # instance segmentation polygon
```

**Column references:**
```python
C("file.size")               # top-level column
C("det.bbox.x1")             # nested field access
C("file.path").glob("*.jpg") # path glob
chain.column("price")        # typed column for arithmetic between columns
```

---

## Section 6 — func Module

All run natively in the metadata engine (no Python, no deserialization):

```python
# Distance (for vector search)
func.cosine_distance(C("emb"), reference_list)
func.euclidean_distance(C("emb"), reference_list)
func.l2_distance(C("emb"), reference_list)

# Aggregate (use in group_by)
func.count()
func.sum(C("file.size"))
func.avg(C("score"))
func.min(C("val"))
func.max(C("val"))
func.collect(C("label"))   # list aggregation
func.first(C("path"))

# Path
func.path.file_ext(C("file.path"))     # → "jpg"
func.path.file_stem(C("file.path"))    # → "image01"
func.path.name(C("file.path"))         # → "image01.jpg"
func.path.parent(C("file.path"))       # → "folder/subfolder"

# Conditional
func.case((C("score") > 0.9, "high"), (C("score") > 0.5, "medium"), else_="low")
func.ifelse(func.isnone(C("result")), "pending", "done")

# String
func.string.length(C("text"))
func.string.split(C("path"), "/")

# Window (both partition_by and order_by are required)
w = func.window(partition_by="category", order_by="created_at")
chain.mutate(row_num=func.row_number().over(w),
             rank=func.rank().over(w),
             first=func.first(C("path")).over(w))

# Ranking (in group_by)
func.rank()
func.dense_rank()
func.row_number()

# Hashing / sampling (ClickHouse only -- not available on local SQLite)
func.sip_hash_64(C("file.path"))
func.int_hash_64(C("file.path"))
```

---

## Section 7 — Common Pipeline Templates

**Basic: read → filter → map → save**
```python
import datachain as dc
from datachain import C, ImageFile

def compute_embedding(file: ImageFile) -> list[float]:
    img = file.read().convert("RGB")
    return model.encode(img).tolist()

(
    dc.read_storage("s3://bucket/images/", type="image")
    .filter(C("file.size") > 1000)
    .settings(parallel=True, cache=True)
    .map(emb=compute_embedding)
    .save("image_embeddings")
)
```

**Stateful class (model loaded once per worker):**
```python
import datachain as dc
import open_clip

class ImageEncoder(dc.Mapper):
    def setup(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", "laion2b_s34b_b79k"
        )

    def process(self, file: dc.ImageFile) -> list[float]:
        img = self.preprocess(file.read()).unsqueeze(0)
        return self.model.encode_image(img)[0].tolist()

(
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=True, cache=True)
    .map(emb=ImageEncoder("ViT-B-32", "laion2b_s34b_b79k"))
    .save("image_embeddings")
)
```

**Inline setup() for model/client initialization:**
```python
from datachain import File

def caption(file: File, pipeline) -> str:
    return pipeline(file.read().convert("RGB"))[0]["generated_text"]

(
    dc.read_storage("gs://bucket/images/", type="image")
    .settings(cache=True, parallel=True)
    .setup(pipeline=lambda: load_pipeline("image-to-text", model="blip-large"))
    .map(caption=caption)
    .save("captions")
)
```

**Multi-stage pipeline:**
```python
# Stage 1
dc.read_storage("s3://docs/*.pdf").settings(parallel=True).gen(chunk=split_pdf).save("chunks")

# Stage 2
(dc.read_dataset("chunks")
   .setup(model=lambda: load_embedding_model())
   .settings(parallel=True)
   .map(emb=embed_chunk)
   .save("chunk_embeddings"))

# Stage 3
(dc.read_dataset("chunk_embeddings")
   .setup(client=lambda: create_llm_client())
   .settings(parallel=True)
   .map(category=classify)
   .save("classified_chunks"))
```

**Generator: 1 input → many outputs (video frames, audio segments, PDF pages):**
```python
from datachain import VideoFile, VideoFragment
from typing import Iterator

def split_clips(file: VideoFile) -> Iterator[VideoFragment]:
    yield from file.get_fragments(duration=10.0)

(
    dc.read_storage("s3://videos/", type="video")
    .settings(parallel=True)
    .gen(frag=split_clips)
    .save("video_clips")
)
```

**Merge sidecar metadata:**
```python
images = dc.read_storage("gs://bucket/images/", type="image", anon=True)
meta = dc.read_json("gs://bucket/annotations.json", jmespath="images")
annotated = images.merge(meta, on="file.path", right_on="images.file_name")
```

**Vector similarity search:**
```python
import datachain as dc
from datachain import C, func

(
    dc.read_dataset("image_embeddings")
    .mutate(dist=func.cosine_distance(C("emb"), query_embedding))
    .order_by("dist")
    .limit(10)
    .show()
)
```

**LLM extraction with structured Pydantic output:**
```python
from datachain import DataModel, File

class Analysis(DataModel):
    sentiment: str
    confidence: float
    topics: list[str]

def analyze(file: File, client) -> Analysis:
    resp = client.messages.create(model="claude-sonnet-4-6", ...)
    return Analysis.model_validate_json(resp.content[0].text)

(dc.read_storage("s3://docs/")
   .setup(client=lambda: anthropic.Anthropic())
   .settings(parallel=True)
   .map(result=analyze)
   .save("analyzed"))
```

**Metadata analytics (no Python needed):**
```python
(
    dc.read_storage("gs://bucket/")
    .filter(C("file.size") > 0)
    .group_by(
        count=func.count(),
        total=func.sum(C("file.size")),
        partition_by=func.path.file_ext(C("file.path")),
    )
    .order_by("total", descending=True)
    .show()
)
```

**Delta updates (incremental, process only new/changed files):**
```python
(
    dc.read_storage("s3://bucket/data/", update=True, delta=True,
                    delta_on="file.path", delta_compare="file.mtime")
    .settings(parallel=True)
    .map(result=process_file)
    .save("processed_data")
)
```

**File paths to File objects (manifest CSV → file processing):**
```python
import datachain as dc
from datachain import File

def to_file(path: str) -> File:
    return File.at(path)

def process_file(file: File) -> str:
    return summarize(file.read_text())

(
    dc.read_csv("s3://data/manifest.csv")
    .map(file=to_file)
    .settings(parallel=True, prefetch=3, cache=True)
    .map(result=process_file)
    .save("summaries")
)
```

**In-memory data (from Python lists / dicts):**
```python
# read_values: keyword args become typed columns
dc.read_values(score=[0.9, 0.7, 0.3], label=["cat", "dog", "fish"]).show()

# read_records: list of dicts → rows; types inferred from first record
dc.read_records([
    {"path": "img/a.jpg", "label": "cat", "conf": 0.95},
    {"path": "img/b.jpg", "label": "dog", "conf": 0.82},
]).show()

# Typical use: join in-memory labels with storage files
labels = dc.read_records([{"name": "a.jpg", "cls": "cat"}, ...])
images = dc.read_storage("s3://bucket/images/")
combined = images.merge(labels, on="file.name", right_on="labels.name")
```

---

## Section 8 — Anti-Patterns

```
✗ Omitting trailing slash → permission error on anonymous/restricted storage
✗ Using params= or output= with map/gen/agg → breaks auto-inference
✗ Lambda for non-str return types → DataChain defaults to str → wrong schema
✗ Pulling all data to Python for filtering:
    chain.to_list() then iterating in Python  ← never do this for metadata ops
    Use chain.filter(C("x") > 0) instead
✗ External Pydantic model not registered:
    dc.DataModel.register(ExternalModel)  ← required for non-DataModel subclasses
✗ Missing .settings(parallel=True) for Python operations → single-threaded execution
✗ Using C() for column-column arithmetic:
    C("price") * C("qty")  ← no type info → engine error
    Use chain.column("price") * chain.column("qty") instead
✗ Materializing to Pandas/list for aggregation:
    chain.to_pandas() then df.groupby(...)  ← never do this
    Use chain.group_by(...) natively instead
✗ Reading files in a Python loop outside the chain:
    rows = chain.to_list(); for path in rows: open(path)  ← no parallelism, no cache
    Use File.at(path) inside map() instead
✗ Using deprecated DataChain.from_*() methods:
    DataChain.from_csv(...)  ← deprecated
    Use dc.read_csv(...) instead
✗ Using .concat() in mutate() → engine can't infer type; use it only in filter()
✗ Using C("col").asc() / .desc() in order_by():
    Use order_by("col", descending=True) instead
✗ Using Python @property / .name / .parent in metadata ops:
    C("file.parent")  ← not a stored column
    Use func.path.parent(C("file.path")) instead
✗ Forgetting .setup() when loading models -- loading in process() means one model
    load per record, not per worker
✗ Using DATACHAIN_IGNORE_CHECKPOINTS carelessly -- clears progress for long jobs
✗ Using stateful python operation based on dc.Mapper/Generator/Aggregator when inline .setup() is sufficient:
    Stateful classes are only needed when setup requires MULTIPLE self.* fields
    or complex initialization that cannot fit in a single lambda per signal.
    For a single model or client, always use inline .setup():
    ✓ .setup(model=lambda: load_model())                   # one resource → inline
    ✓ .setup(client=lambda: anthropic.Anthropic())         # one client → inline
    ✗ class MyMapper(dc.Mapper):
          def setup(self): self.model = load_model()       # overkill for one resource
    Use dc.Mapper when you need:
      self.model + self.tokenizer + self.config (multiple fields)
      custom __init__ args passed at pipeline construction time
✗ Multiple signals in one map/gen/agg call:
    chain.map(a=fn1, b=fn2)  ← UdfSignatureError
    Instead, use: chain.map(a=fn1).map(b=fn2)
✗ Tuple return type in map/gen/agg:
    def fn(...) -> tuple[int, int]: ...  ← creates col_0, col_1
    Always prefer using DataModel for named fields instead
```
