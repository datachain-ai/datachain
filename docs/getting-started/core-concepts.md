---
title: Core Concepts
---

# Core Concepts

DataChain is built around a small number of ideas. Understanding them makes the entire API predictable.

## Data Memory

Data work produces no persistent memory by default. Every pipeline, every exploration, every labeling session produces knowledge -- and that knowledge evaporates when the script finishes. DataChain changes this: every operation deposits its results as a named, versioned, typed dataset. Memory is the accumulated record of everything the team has done with its data.

Memory compounds over time. Every deposit makes the next operation cheaper, faster, and better-informed. But compounding only works when recall is cheaper than recreation. DataChain's columnar SQL backend makes retrieval fast enough that building on prior work is always the path of least resistance.

## Datasets

The dataset is the atomic unit of memory -- a named, versioned collection of typed records. Everything the system remembers is a dataset. Every pipeline produces one. Every subsequent pipeline starts from one.

```python
import datachain as dc

# Create a versioned dataset
(
    dc.read_storage("s3://bucket/images/", type="image")
    .map(emb=compute_embedding)
    .save("image_embeddings")   # v1.0.0
)

# Build on it later
ds = dc.read_dataset("image_embeddings")
```

Key properties:

- **Immutable.** A saved version never changes. New work produces a new version.
- **Typed.** Schemas are Pydantic models with full Python type support -- nested objects, lists, file references, embeddings.
- **Versioned.** Each `save()` auto-increments the version. Load any version by number or range.
- **Shared.** Datasets are the collaboration primitive -- not files, not Slack messages.

## Chains

A chain is a lazy, composable pipeline. Nothing executes until a terminal operation like `save()`, `show()`, or `to_pandas()` triggers it.

```python
import datachain as dc

chain = (
    dc.read_storage("s3://bucket/data/")
    .filter(dc.C("file.size") > 1000)
    .mutate(ext=dc.func.path.file_ext("file.path"))
    .map(label=classify_file)
    .save("classified_files")
)
```

Lazy evaluation lets the system optimize the full pipeline before executing. Filters push down, data-engine operations compile to SQL, unnecessary computation gets skipped. A chain declares WHAT to do, not HOW -- the same fluent API covers SQL-compiled operations and Python-executed functions.

## Two Execution Engines

DataChain has two execution engines with a clear boundary:

**Memory Engine** (SQL): Runs filter, merge, join, group_by, order_by, mutate, and aggregate operations at warehouse speed (SQLite locally, ClickHouse in Studio). Use for anything expressible without Python.

**Python Data Engine**: Runs `map()`, `gen()`, `agg()` -- operations that need file content, ML models, or LLM calls. Executes in parallel across threads and distributed across machines.

Pydantic bridges the two: Python outputs flatten into columnar storage, and Python expressions transpile to SQL. Users never specify which engine runs what -- the system infers it from the operation type.

```python
import datachain as dc

(
    dc.read_storage("s3://bucket/images/", type="image")
    # Memory Engine: SQL-compiled, warehouse speed
    .filter(dc.C("file.size") > 1000)
    .mutate(ext=dc.func.path.file_ext("file.path"))
    # Python Engine: parallel, distributed
    .settings(parallel=8)
    .map(emb=compute_embedding)
    # Memory Engine: SQL aggregate
    .save("embeddings")
)
```

## Files and Types

DataChain's `File` abstraction bridges object storage and the metadata layer. Files live in cloud storage and are never copied -- the intelligence layer operates by reference.

```python
import datachain as dc

# File objects carry storage coordinates
chain = dc.read_storage("s3://bucket/data/")
# file.source, file.path, file.size, file.etag, ...

# Modality-specific subclasses
images = dc.read_storage("s3://bucket/images/", type="image")   # ImageFile
videos = dc.read_storage("s3://bucket/videos/", type="video")   # VideoFile
audio  = dc.read_storage("s3://bucket/audio/",  type="audio")   # AudioFile
```

## Provenance

Every `save()` automatically records four things: **dependencies** (parent datasets and storage URIs), **source code** (the full script), **author**, and **creation time**. No manual declaration required. This is what makes datasets trustworthy and reproducible -- the team can always trace how any result was produced.
