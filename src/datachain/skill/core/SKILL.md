---
name: datachain-core
description: Use ONLY for abstract DataChain SDK questions — API usage, method signatures, or code patterns — when no specific dataset or bucket is referenced. If the request mentions creating, saving, listing, exploring datasets or buckets, use datachain-graph instead.
---

You are now loaded with expert-level DataChain SDK context. Apply every rule below when generating DataChain Python code. Do not deviate.

### Code style

Write self-explanatory code. Use clear variable names, function names, and type hints instead of comments. Add a comment only when the code alone cannot convey *why* something is done (non-obvious workarounds, surprising constraints). Never add comments that restate what the code does.

### Graph context (read-only)

If `datachain/graph/index.md` exists, read it at conversation start for dataset and bucket awareness. When the user mentions a specific dataset or bucket by name, read the matching `.md` file under `datachain/graph/datasets/` or `datachain/graph/buckets/`. Never create or modify files under `datachain/graph/` — that directory is owned by the datachain-graph skill.

---

## Section 1 — Critical Rules (Must-Never-Break)

```
1. TRAILING SLASH: Always add / to bucket/prefix paths.
   ✓ dc.read_storage("s3://bucket/images/")
   ✗ dc.read_storage("s3://bucket/images")  ← permission error on anon access

1a. ANON FOR GCS: Always add anon=True for gs:// buckets. Without it, the GCS
    client tries credential discovery first, adding ~1 min delay. Remove anon=True
    only if the user says they need authenticated access to a private bucket.
    ✓ dc.read_storage("gs://bucket/data/", anon=True)
    ✗ dc.read_storage("gs://bucket/data/")  ← 1 min credential timeout

2. TYPE HINTS NOT params/output: Never pass params= or output= to map/gen/agg.
   DataChain auto-infers from function signature and return type annotation.
   ✓ def fn(file: dc.ImageFile) -> list[float]: ...
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
   def process(file: dc.ImageFile) -> ...  ← always "file", not "image"

6. PARALLEL WHEN NEEDED: Only use .settings(parallel=True) when files are large
   or per-row processing is expensive (ML inference, LLM calls, heavy file I/O).
   For simple/lightweight operations, omit parallel — sequential is the default.
   ✓ chain.settings(parallel=True).map(emb=model_fn)  # ML inference → parallel
   ✓ chain.map(label=classify)                         # lightweight → sequential OK

7. PREFETCH FOR FILE-READING UDFs: When generating UDF that reads file content,
   estimate the average file size from context (file type, domain knowledge —
   e.g. small XMLs ~2KB, JPEGs ~200KB, PDFs ~2MB) and compute:
     prefetch = clamp(4MB / estimated_avg_size, 2, 128)
   Only add .settings(prefetch=N) to the generated code if N > 4 (default is 2).
   Skip for UDFs that don't read file content (metadata-only operations).
   Skip if the user explicitly sets prefetch.
   ✓ # small XML files ~2KB → prefetch = 4MB/2KB = 2048 → clamped to 128
     chain.settings(prefetch=128).map(bbox=parse_xml)
   ✓ # JPEGs ~200KB → prefetch = 4MB/200KB = 20
     chain.settings(prefetch=20).map(emb=encode_image)
   ✓ # large PDFs ~5MB → prefetch = 4MB/5MB ≈ 1 → clamped to 2, skip (≤ 4)
     chain.map(result=process_pdf)

8. CACHE ONLY WHEN NEEDED: Do not add cache=True by default. Only use it when:
   - The user explicitly asks for caching, OR
   - The same files are read multiple times in the pipeline (e.g., multi-stage
     pipelines where stage 2 re-reads files processed in stage 1).
   Caching downloads files to local disk — unnecessary for single-pass pipelines.
   ✓ chain.settings(cache=True).map(emb=fn)            # multi-pass → cache
   ✗ dc.read_storage("s3://b/").settings(cache=True).map(fn).save("out")  # single pass

9. COLUMN-COLUMN ARITHMETIC: Use chain.column() instead of C() when combining two columns.
   C() does not carry type info → engine can't infer the result type.
   chain.column("name") returns a typed column derived from the chain's schema.
   ✓ chain.mutate(total=chain.column("price") * chain.column("qty"))
   ✓ chain.mutate(discounted=C("price") * 0.9)          # scalar literal → type inferred
   ✗ chain.mutate(total=C("price") * C("qty"))           # no type → error

10. READ NOT FROM: Use dc.read_* module functions, not deprecated DataChain.from_* methods.
   ✓ dc.read_csv("s3://data.csv")
   ✗ DataChain.from_csv("s3://data.csv")  ← deprecated

11. GLOB IN PATH: When filtering by file extension or name pattern, put the glob
   directly in the read_storage() path instead of a separate .filter() call.
   This preserves per-file lineage (not just directory-level).
   ✓ dc.read_storage("s3://bucket/images/**/*.{jpg,jpeg,png}", type="image")
   ✓ dc.read_storage("s3://data/**/*.csv")
   ✗ dc.read_storage("s3://bucket/images/", type="image")
     .filter(dc.C("file.path").glob("*.jpg") | dc.C("file.path").glob("*.png"))

12. SINGLE FILE vs MULTI FILE: Use the right API for the job.
    - For one known file: use dc.File.at() (or dc.TextFile.at(), dc.ImageFile.at()).
    - For one known CSV/JSON/Parquet: use dc.read_csv(), dc.read_json(), dc.read_parquet().
    - For a small fixed set of known files: use one read_storage() with a glob pattern.
    - For listing/processing many files in a directory: use dc.read_storage().
    read_storage() is designed for directory listing — don't use it for a single known file.

    File.at() signature: dc.File.at(uri) — accepts ONLY the URI string.
    ✓ file = dc.File.at("s3://bucket/config.json")
      data = json.loads(file.read_bytes())
    ✓ dc.read_csv("s3://bucket/labels.csv")
    ✓ dc.read_storage("s3://bucket/**/{trainval,test}.txt")  # fixed set via glob
    ✓ dc.read_storage("s3://bucket/images/", type="image")   # many files
    ✗ dc.read_storage("s3://bucket/config.json")              # single file
    ✗ dc.read_storage("s3://bucket/trainval.txt")             # single known file
      dc.read_storage("s3://bucket/test.txt")                 # ← two listings for two files

    For small files (<300 MB) that fit in memory, it is fine to read the whole
    file and process in Python if the code is simpler and cleaner.
    ✓ img = dc.ImageFile.at("s3://b/photo.jpg").read()       # PIL Image in memory
    ✓ text = dc.TextFile.at("s3://b/readme.txt").read()      # string in memory

13. ONE SIGNAL PER MAP/GEN/AGG: Each call accepts exactly one signal (keyword).
   For multiple columns, chain calls or return a Pydantic BaseModel.
   ✓ chain.map(a=fn1).map(b=fn2)          # chained — two columns
   ✓ chain.map(info=fn)                    # BaseModel with named fields
   ✗ chain.map(a=fn1, b=fn2)              # ERROR: multiple signals

14. NO TUPLE RETURNS: Always prefer Pydantic BaseModel classes to tuple in map/gen/agg
    functions until user directly asks for tuple.
    ✓ def fn(file: dc.File) -> MyModel: ...   # named fields via BaseModel
    ✓ def fn(file: dc.File) -> int: ...       # single scalar
    ✗ def fn(file: dc.File) -> tuple[int, int]: ...  # → col_0, col_1

15. MERGE NOT DICTS: When combining multiple data sources, read each as its own
    chain, parse inside map()/gen(), then merge(). Never build Python dicts outside
    the chain and close over them in map()/gen() — even via dc.TextFile.at().read().
    ✓ annotations = dc.read_storage("./**/list.txt", type="text").gen(ann=parse_list)
      images = dc.read_storage("./images/", type="image")
      images.merge(annotations, on=dc.func.path.file_stem(dc.C("file.path")),
                   right_on="ann.name")
    ✗ text = dc.TextFile.at("list.txt").read()
      lookup = {name: label for name, label in parse(text)}  ← dict outside chain
      dc.read_storage("./images/").map(ann=lambda f: lookup[f.path])  ← closure

16. SHARED LISTING PREFIX: When multiple read_storage() calls target the same
    bucket or directory tree, use the common parent prefix and glob from there.
    DataChain caches listings by prefix — shared prefix = one listing + cache hits.
    ✓ dc.read_storage("s3://bucket/data/**/*.txt", type="text")
      dc.read_storage("s3://bucket/data/**/*.xml")
      dc.read_storage("s3://bucket/data/**/*.jpg", type="image")
    ✗ dc.read_storage("s3://bucket/data/annotations/**/*.txt", type="text")
      dc.read_storage("s3://bucket/data/annotations/xmls/**/*.xml")
      dc.read_storage("s3://bucket/data/images/**/*.jpg", type="image")

17. LAZY CHAINS — NO DOUBLE EXECUTION: Chains are lazy — each terminal operation
    (save, show, to_list, …) re-executes the entire pipeline. Never call two
    terminal operations on the same unmaterialized chain.
    - After save(): use the returned chain (save() returns the saved dataset).
    - When a chain is reused multiple times: call .persist() to materialize it,
      then use the persisted chain for all subsequent operations.
    ✓ saved = chain.save("my_data")
      saved.show(5)
    ✓ materialized = chain.persist()       # materialize once
      materialized.show(5)                 # reuse without re-executing
      materialized.to_csv("out.csv")
    ✗ chain.save("my_data")
      chain.show(5)               ← runs the whole pipeline a second time
    ✗ chain.show(5)
      chain.to_csv("out.csv")    ← also double execution

18. INLINE FUNC EXPRESSIONS: Pass func/C expressions directly to on=,
    right_on=, partition_by=, order_by=. Do not mutate() a throwaway column.
    ✓ chain.merge(other, on=dc.func.path.file_stem(dc.C("file.path")), ...)
    ✓ chain.group_by(..., partition_by=dc.func.path.parent(dc.C("file.path")))
    ✗ chain.mutate(stem=dc.func.path.file_stem(dc.C("file.path")))
            .merge(other, on="stem", ...)  ← unnecessary column

19. SELECT_EXCEPT AFTER MERGE: After merge(), use select_except() to drop
    duplicated/unwanted columns. Never write a long select() list (>4 columns).
    ✓ chain.merge(other, ...).select_except("right_file", "ann.name")
    ✗ chain.merge(other, ...).select("file", "col1", "col2", ..., "col18")
```

---

## Section 2 — Golden Rule

```
1. ALWAYS use DataChain to access files in storage (local, S3, GCS, Azure).
   NEVER use os.walk, os.listdir, glob.glob, pathlib, or any manual file-system
   traversal to discover or read data files.
   - Single known file: dc.File.at(), dc.TextFile.at(), dc.ImageFile.at()
   - Single CSV/JSON/Parquet: dc.read_csv(), dc.read_json(), dc.read_parquet()
   - Many files in a directory: dc.read_storage() (vectorised, preserves lineage)
   ✓ dc.read_storage("/data/images/", type="image").map(emb=encode)
   ✓ dc.File.at("s3://bucket/config.json").read_bytes()
   ✗ for f in os.listdir("/data/images/"): ...   ← breaks lineage, slow
   ✗ paths = glob.glob("*.csv"); dc.read_values(paths=paths)  ← no lineage

2. Prefer engine-side operations over Python operations.
   Engine-side ops — filter(), mutate(), group_by(), order_by(), select(),
   merge(), union(), distinct(), limit() — run in SQL (SQLite local /
   ClickHouse Studio) at warehouse speed using dc.C() and dc.func.*.
   Python ops (map/gen/agg) spin up Python workers and are expensive.

   Use Python ops ONLY when you need:
     - File content (file.read(), file.open())
     - ML model inference
     - LLM calls
     - External API calls

   Everything else → use filter/mutate/group_by/merge with dc.C() and dc.func.*

3. EXTRACTING RESULTS: Use to_values() for a single column (returns flat list),
   to_list() for multiple columns (returns list of tuples).
   Never use to_iter() — it loses parallelism and lineage compared to map().
   For processing, use map()/gen() instead of extracting and looping.
   ✓ ids = chain.to_values("id")              # → [5, 8, 15]
   ✓ files = chain.to_values("file")          # → [File(...), File(...), ...]
   ✓ rows = chain.to_list("file", "label")    # → [(File, "cat"), (File, "dog")]
   ✗ for f, id in chain.to_list("file", "id"): # to_list returns tuples
   ✓ chain.map(result=process_fn).save("output")
   ✗ for row in chain.to_iter("file"): ...    # to_iter is an anti-pattern
   ✗ for row in chain.to_list("file"):        # to_list returns tuples
       row.read_text()                         # ← AttributeError: tuple has no read_text
```

---

## Section 3 — Import Cheat Sheet

```python
import datachain as dc
from pydantic import BaseModel
```

Always use `dc.` prefix for DataChain types: `dc.File`, `dc.ImageFile`, `dc.C`, `dc.func`,
 `dc.Mapper`, etc. Do NOT use `from datachain import File, C, func, ...`. Exception
 is `from datachain import model` for computer vision and audio data models.

---

## Section 4 — Core API Reference

**Entry points:**

`read_storage()` creates a **listing** — a cached snapshot of the bucket or directory.
The listing prefix is the path before the first glob (`**`). Subsequent `read_storage()`
calls with the same prefix reuse the cached listing. Use a shared parent prefix when
reading multiple file types from the same tree (see rule 16).

```python
dc.read_storage("s3://bucket/prefix/", type="image")   # File / ImageFile etc.
dc.read_storage("s3://bucket/imgs/**/*.{jpg,png}", type="image")  # glob in path
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
In all examples below, `C` = `dc.C` and `func` = `dc.func`.
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
chain.merge(other, on="id", right_on="meta.id")  # duped cols get right_ prefix
chain.merge(...).select_except("right_file")     # drop duped columns after merge
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
```

**Setup and execution settings:**
```python
chain.setup(model=lambda: load_model())   # initialize once per worker
chain.settings(parallel=True, cache=True, prefetch=10, workers=50)  # cache only if needed
```

**Terminal operations (trigger execution):**
```python
chain.save("dataset_name")                     # versioned named dataset
chain.save("ns.proj.name", update_version="minor")
chain.persist()                                # anonymous cache
chain.show(limit=10)
chain.to_values("col")                         # → flat list: [val1, val2, ...]
chain.to_list("col1", "col2")                  # → list of tuples: [(v1,v2), ...]
# NOTE: to_values for 1 column, to_list for 2+. Do NOT use to_iter (anti-pattern).
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

**Structured types — use Pydantic BaseModel:**
```python
from datachain import model

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: model.BBox
```

**File types (all inherit from dc.File):**

| Type | `type=` param | `.read()` returns | Extra methods |
|---|---|---|---|
| `dc.File` | (default) | `bytes` | `.read_text()`, `.open()`, `.ensure_cached()` |
| `dc.TextFile` | `"text"` | `str` | `.read_text()` |
| `dc.ImageFile` | `"image"` | `PIL.Image` | `.get_info()` → `Image(width,height,format)` |
| `dc.VideoFile` | `"video"` | -- | `.get_frames(step=N)` → `VideoFrame[]`, `.get_fragments(duration)` → `VideoFragment[]`, `.get_info()` → `Video(fps,duration,codec,...)` |
| `dc.AudioFile` | `"audio"` | -- | `.get_fragments(duration)` → `AudioFragment[]`, `.get_info()` → `Audio(sample_rate,channels,duration,...)` |

Sub-file units:
- `VideoFrame` -- `.get_np()` → ndarray, `.save(path)`
- `VideoFragment` -- `.save(path)`
- `AudioFragment` -- `.get_np()` → `(ndarray, sample_rate)`, `.save(path)`

**Annotation types (prefer these over custom BaseModels for bbox/pose/segment):**
```python
from datachain import model # import is mandatory, dc.model.BBox is not enough

model.BBox(title="car", coords=[x1,y1,x2,y2])            # PASCAL VOC
model.BBox.from_coco([x,y,w,h], title="car")
model.BBox.from_yolo([cx,cy,w,h], img_size=(640,480))
model.BBox.from_albumentations([x1n,y1n,x2n,y2n], img_size)
bbox.to_coco() / .to_yolo(img_size) / .to_albumentations(img_size) / .to_voc()
bbox.point_inside(x, y)  # → bool
bbox.pose_inside(pose)   # → bool

model.OBBox(...)                # oriented bbox -- four corner points

model.Pose(x=[...], y=[...])                       # 2D keypoints
model.Pose3D(x=[...], y=[...], visible=[...])      # 3D with visibility

model.Segment(title="road", x=[...], y=[...])      # instance segmentation polygon
```

**Column references:**
```python
dc.C("file.size")               # top-level column
dc.C("det.bbox.x1")             # nested field access
dc.C("file.path").glob("*.jpg") # path glob
chain.column("price")           # typed column for arithmetic between columns
```

---

## Section 6 — func Module

All run natively in the metadata engine (no Python, no deserialization).
In examples below, `C` = `dc.C`, `func` = `dc.func`.

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

def compute_embedding(file: dc.ImageFile) -> list[float]:
    img = file.read().convert("RGB")
    return model.encode(img).tolist()

(
    dc.read_storage("s3://bucket/images/", type="image")
    .filter(dc.C("file.size") > 1000)
    .settings(parallel=True)
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
    .settings(parallel=True)
    .map(emb=ImageEncoder("ViT-B-32", "laion2b_s34b_b79k"))
    .save("image_embeddings")
)
```

**Inline setup() for model/client initialization:**
```python
def caption(file: dc.File, pipeline) -> str:
    return pipeline(file.read().convert("RGB"))[0]["generated_text"]

(
    dc.read_storage("gs://bucket/images/", type="image")
    .settings(parallel=True)
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
from typing import Iterator

def split_clips(file: dc.VideoFile) -> Iterator[dc.VideoFragment]:
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
images = dc.read_storage("gs://bucket/images/", type="image")
meta = dc.read_json("gs://bucket/annotations.json", jmespath="images")
annotated = images.merge(meta, on="file.path", right_on="images.file_name")
```

**Multi-source dataset (images + annotations + XMLs → merge):**
```python
# Shared prefix (rule 16), inline func (rule 18), select_except (rule 19)
annotations = dc.read_storage("gs://b/**/*.txt", type="text").gen(ann=parse_list)
xmls = dc.read_storage("gs://b/**/*.xml").settings(prefetch=128).map(xml=parse_xml)
images = dc.read_storage("gs://b/**/*.jpg", type="image")

# func in on= (rule 18), select_except after merge (rule 19)
(
    images
    .merge(annotations,
           on=dc.func.path.file_stem(dc.C("file.path")), right_on="ann.name")
    .merge(xmls,
           on=dc.func.path.file_stem(dc.C("file.path")),
           right_on=dc.func.path.file_stem(dc.C("file.path")))
    .select_except("right_file", "ann.name")
    .save("annotated_images")
)
```

**Vector similarity search:**
```python
import datachain as dc

(
    dc.read_dataset("image_embeddings")
    .mutate(dist=dc.func.cosine_distance(dc.C("emb"), query_embedding))
    .order_by("dist")
    .limit(10)
    .show()
)
```

**LLM extraction with structured Pydantic output:**
```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

def analyze(file: dc.File, client) -> Analysis:
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
    .filter(dc.C("file.size") > 0)
    .group_by(
        count=dc.func.count(),
        total=dc.func.sum(dc.C("file.size")),
        partition_by=dc.func.path.file_ext(dc.C("file.path")),
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
    .map(result=process_file)
    .save("processed_data")
)
```

**File paths to File objects (manifest CSV → file processing):**
```python
import datachain as dc

def to_file(path: str) -> dc.File:
    return dc.File.at(path)

def process_file(file: dc.File) -> str:
    return summarize(file.read_text())

(
    dc.read_csv("s3://data/manifest.csv")
    .map(file=to_file)
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
✗ Using os.walk / os.listdir / glob.glob / pathlib to discover or read data files:
    NEVER walk the filesystem or open files manually — it breaks lineage and
    skips optimisations. Always use dc.read_storage() for local dirs, S3, GCS,
    or Azure paths, and parse file content inside map()/gen().
    ✗ for f in Path("/data").rglob("*.jpg"): ...
    ✗ files = glob.glob("s3://bucket/*.csv")
    ✓ dc.read_storage("/data/", type="image")
    ✓ dc.read_storage("s3://bucket/", type="csv")
✗ Reading data files outside the chain to build Python dicts/lists:
    NEVER parse files outside the chain and reconstruct with read_values(),
    read_records(), or Python dicts used as lookup tables in map()/gen().
    This loses lineage to the source files and bypasses DataChain's engine.
    ✗ with open("annotations.txt") as f:           ← manual file I/O
          rows = [parse(line) for line in f]
      dc.read_values(label=[r["label"] for r in rows])  ← no lineage
    ✗ for xml in Path("xmls/").glob("*.xml"):      ← manual walk + parse
          data.append(parse_xml(xml))
      dc.read_records(data)                         ← no lineage
    Same applies when using DataChain's single-file API to build lookup dicts:
    ✗ text = dc.TextFile.at("list.txt").read()     ← outside the chain
      lookup = {line.split()[0]: line.split()[1] for line in text.splitlines()}
      dc.read_storage("./images/").map(ann=lambda f: lookup[stem(f)])  ← closure
    ✓ Read ALL files (images, annotations, XMLs) via dc.read_storage(),
      then parse inside map()/gen() and merge():
      annotations = dc.read_storage("./annotations/**/list.txt", type="text")
        .gen(info=parse_list_file)
      xmls = dc.read_storage("./annotations/xmls/").map(bbox=parse_xml)
      images = dc.read_storage("./images/", type="image")
      dataset = images.merge(annotations, ...).merge(xmls, ...)
✗ Importing symbols from datachain:
    from datachain import File, C, func  ← clutters namespace
    Use dc.File, dc.C, dc.func after `import datachain as dc`
✗ Omitting trailing slash → permission error on anonymous/restricted storage
✗ Using params= or output= with map/gen/agg → breaks auto-inference
✗ Lambda for non-str return types → DataChain defaults to str → wrong schema
✗ Pulling all data to Python for filtering:
    chain.to_list() then iterating in Python  ← never do this for metadata ops
    Use chain.filter(C("x") > 0) instead
✗ Using DataModel instead of BaseModel:
    Use pydantic.BaseModel for custom types — DataChain accepts it natively
✗ Using parallel=True for lightweight operations → unnecessary overhead
   Only use parallel for expensive per-row work (ML, LLM, heavy file I/O)
✗ Using cache=True by default in single-pass pipelines → wastes disk, no benefit
   Only cache when files are read multiple times or the user explicitly requests it
✗ Long select() list after merge — use select_except() instead (rule 19)
✗ select() right after map/gen to "pick" the new column:
    chain.map(ann=parse_xml).select("ann")  ← redundant, map already created "ann"
    The keyword in map/gen IS the column name (rule 4) — no select needed
✗ Using C() for column-column arithmetic:
    C("price") * C("qty")  ← no type info → engine error
    Use chain.column("price") * chain.column("qty") instead
✗ Materializing to Pandas/list for aggregation:
    chain.to_pandas() then df.groupby(...)  ← never do this
    Use chain.group_by(...) natively instead
✗ Reading files in a Python loop outside the chain:
    rows = chain.to_list(); for path in rows: open(path)  ← no parallelism, no cache
    Use dc.File.at(path) inside map() instead
✗ Using to_iter() — always an anti-pattern. Use to_values() for one column,
    to_list() for multiple columns, map()/gen() for processing.
    ✗ for row in chain.to_iter("file"): row.read_text()  ← tuple, not File
    ✗ for row in chain.to_list("file"): row.read_text()  ← also tuple
    ✓ for f in chain.to_values("file"): f.read_text()    ← flat list of File objects
✗ DEPRECATED APIs — never use these:
    DataChain.from_storage()  → dc.read_storage()
    DataChain.from_dataset()  → dc.read_dataset()
    DataChain.from_csv()      → dc.read_csv()
    DataChain.from_json()     → dc.read_json()
    DataChain.from_parquet()  → dc.read_parquet()
    DataChain.from_values()   → dc.read_values()
    DataChain.from_pandas()   → dc.read_pandas()
    DataChain.from_hf()       → dc.read_hf()
    DataChain.from_records()  → dc.read_records()
    DataChain.datasets()      → dc.datasets()
    DataChain.listings()      → dc.listings()
    chain.collect()           → chain.to_list() / chain.to_iter() / chain.to_values()
    File.get_uri()            → file.get_fs_path()
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
    Always prefer using BaseModel for named fields instead
✗ Using filter() with C("file.path").glob() for file extension filtering right after read_storage():
    dc.read_storage("s3://bucket/images/", type="image")
      .filter(dc.C("file.path").glob("*.jpg") | dc.C("file.path").glob("*.png"))
    ← lineage points to directory, not individual files; verbose
    Put glob patterns directly in the read_storage() path:
    ✓ dc.read_storage("s3://bucket/images/**/*.{jpg,png}", type="image")
✗ Using read_storage() for a single known file:
    dc.read_storage("s3://bucket/annotations.json")  ← overkill, listing machinery
    Use the single-file API instead:
    ✓ dc.File.at("s3://bucket/annotations.json").read_bytes()
    ✓ dc.read_json("s3://bucket/annotations.json")   # if you want a DataChain
    ✓ dc.read_csv("s3://bucket/labels.csv")           # single CSV → DataChain
✗ Custom BaseModel for bounding boxes, poses, or segments:
    Use built-in model.BBox / model.Pose / model.Segment instead of custom classes.
    ✗ class HeadBBox(BaseModel): xmin: int; ymin: int; xmax: int; ymax: int
    ✓ model.BBox(title="head", coords=[xmin, ymin, xmax, ymax])  # PASCAL VOC
```
