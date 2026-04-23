---
title: Best Practices
---

# Best Practices

Rules for writing correct, idiomatic DataChain code. Follow these to avoid the most common pitfalls.

## Import Convention

```python
# CORRECT
import datachain as dc

# CORRECT (for annotation types and custom models)
from datachain import model
from pydantic import BaseModel

# WRONG -- never import individual symbols
# from datachain import File, C, func
```

## Always Type Your UDFs

Every function passed to `map()`, `gen()`, or `agg()` must have a return type annotation. Missing annotations default to `str` and crash at runtime.

```python
# GOOD
def classify(file: dc.File) -> str:
    return "positive"

# GOOD -- Pydantic model for multiple outputs
from pydantic import BaseModel

class Result(BaseModel):
    label: str
    confidence: float

def classify(file: dc.File) -> Result:
    return Result(label="positive", confidence=0.95)

# BAD -- no return type
def classify(file):
    return 0.95  # crashes: default is str
```

## Prefer Type Hints Over params/output

Rely on function annotations for auto-inference. Use `params=` only for nested column binding. Use `output=` only for non-str lambdas.

```python
# GOOD -- auto-inferred
def caption(file: dc.ImageFile) -> str:
    return describe(file.read())

# GOOD -- params for nested columns
chain.map(ext=lambda path: path.rsplit(".", 1)[-1], params=["file.path"])

# UNNECESSARY
chain.map(caption=caption, params=["file"], output=str)
```

## save() Before filter() on Expensive Operations

Filtering discards work that was never persisted. Save expensive UDF output first, then filter.

```python
# GOOD
chain.map(result=expensive_llm_call).save("all_results")
dc.read_dataset("all_results").filter(dc.C("result.score") > 0.9).save("good_results")

# BAD -- discarded results are lost forever
chain.map(result=expensive_llm_call).filter(dc.C("result.score") > 0.9).save("good_results")
```

## Materialize Before Reuse

When the same chain is consumed by multiple terminal operations, `save()` or `persist()` it first.

```python
# GOOD
base = chain.map(emb=compute_embedding).persist()
base.filter(dc.C("file.size") > 10_000).save("large")
base.filter(dc.C("file.size") <= 10_000).save("small")

# BAD -- embeddings computed twice
chain.map(emb=compute_embedding).filter(dc.C("file.size") > 10_000).save("large")
chain.map(emb=compute_embedding).filter(dc.C("file.size") <= 10_000).save("small")
```

## Use settings(parallel=True) for Expensive Operations

Always parallelize ML inference, LLM calls, and heavy I/O. Omit for lightweight operations.

```python
# GOOD -- expensive model inference
chain.settings(parallel=8).map(emb=compute_embedding)

# UNNECESSARY -- simple metadata extraction
chain.map(ext=lambda file: file.path.rsplit(".", 1)[-1])
```

## One Signal Per Operation

Each `map()`/`gen()`/`agg()` produces one output signal. Group outputs in a Pydantic model.

```python
from pydantic import BaseModel

class Result(BaseModel):
    label: str
    confidence: float

# GOOD
chain.map(result=classify)

# BAD -- multiple keywords
# chain.map(label=get_label, confidence=get_confidence)
```

## Use Native Analytics, Not Pandas

Use `group_by`, `count`, `sum`, `avg` instead of materializing to pandas.

```python
# GOOD
chain.group_by(avg_size=dc.func.avg("file.size"), partition_by="category")

# BAD
df = chain.to_pandas()
df.groupby("category")["file.size"].mean()
```

## Trailing Slash in Storage Paths

Always append `/` to bucket and prefix URIs.

```python
# GOOD
dc.read_storage("s3://bucket/images/")

# BAD -- may not list directory contents
dc.read_storage("s3://bucket/images")
```

## anon=True for Public Buckets

Pass `anon=True` explicitly or the call stalls or returns 403.

```python
# GOOD
dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)

# BAD -- hangs waiting for credentials
dc.read_storage("gs://datachain-demo/dogs-and-cats/")
```

## Glob Patterns Inside read_storage()

Put glob patterns inside `read_storage()` so they appear in lineage.

```python
# GOOD
dc.read_storage("s3://bucket/images/**/*.jpg")

# BAD -- glob in Python, not tracked
import glob
for f in glob.glob("images/**/*.jpg"):
    ...
```

## Prefer setup() Over Stateful Classes

Use `.setup(x=lambda: init())` instead of class-based `Mapper` when you don't need teardown.

```python
# GOOD
chain.setup(model=lambda: load_model()).map(result=predict)

# OVERKILL for most cases
class MyMapper(Mapper):
    def setup(self): self.model = load_model()
    def process(self, file): return self.model(file.read())
```

## Avoid File Download for Metadata-Only UDFs

Use `params=["file.path"]` to avoid downloading files when you only need path metadata.

```python
# GOOD -- no file download
chain.map(ext=lambda path: path.rsplit(".", 1)[-1], params=["file.path"])

# BAD -- downloads entire file just to read its path
chain.map(ext=lambda file: file.path.rsplit(".", 1)[-1])
```

## Merge, Don't Build Dicts

Read all sources as chains and merge. Never build Python dicts outside the chain.

```python
# GOOD
images = dc.read_storage("s3://bucket/images/")
labels = dc.read_csv("s3://bucket/labels.csv")
labeled = images.merge(labels, on="file.path", right_on="path")

# BAD
labels = {}
for row in csv.reader(open("labels.csv")):
    labels[row[0]] = row[1]
```

## Always Use DataChain for File Access

Never use `os.walk`, `glob.glob`, or `pathlib` for accessing data files. Always use DataChain APIs.

```python
# GOOD
dc.read_storage("s3://bucket/data/")
dc.File.at("s3://bucket/path/to/file.png")

# BAD
import os
for root, dirs, files in os.walk("/data"):
    ...
```

## select_except() After Merge

Drop duplicated columns after merge. Do all merges first, one `select_except()` at end.

```python
# GOOD
result = a.merge(b, on="id").merge(c, on="id").select_except("b.id", "c.id")

# BAD -- select_except between merges can lose columns needed for later merges
result = a.merge(b, on="id").select_except("b.id").merge(c, on="id")
```

## Single File vs Multi-File

Use `dc.File.at()` for one file, `read_storage()` for directories.

```python
# Single file
file = dc.File.at("s3://bucket/specific-file.json")

# Directory listing
chain = dc.read_storage("s3://bucket/data/")
```

## Shared Listing Prefix

Use a common parent prefix for multiple `read_storage()` calls -- you get one listing and cache hits.

```python
# GOOD -- shared prefix, one cached listing
train = dc.read_storage("s3://bucket/dataset/train/")
test = dc.read_storage("s3://bucket/dataset/test/")

# BAD -- two separate listings
train = dc.read_storage("s3://bucket-a/train/")
test = dc.read_storage("s3://bucket-b/test/")
```

## Inline func Expressions

Pass `dc.func.*` directly to `on=`, `partition_by=`. Don't `mutate()` throwaway columns.

```python
# GOOD
chain.group_by(count=dc.func.count(), partition_by=dc.func.path.file_ext("file.path"))

# BAD -- unnecessary intermediate column
chain.mutate(ext=dc.func.path.file_ext("file.path")).group_by(count=dc.func.count(), partition_by="ext")
```
