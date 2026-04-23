---
title: Scaling and Performance
---

# Scaling and Performance

DataChain has a layered optimization architecture. Each layer eliminates a source of waste.

## Parallel Execution

`settings(parallel=N)` runs N threads, each with its own model instance. `parallel=-1` auto-detects CPU cores.

```python
import datachain as dc

chain = (
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, prefetch=10)
    .map(emb=compute_embedding)
    .save("image_embeddings")
)
```

## Distributed Execution

`settings(workers=K)` spreads work across K machines via Studio's BYOC model (10-300 machine Kubernetes clusters). Threads and workers compose: `parallel=8, workers=50` means 8 threads on each of 50 machines.

```python
import datachain as dc

(
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, workers=50, cache=True, prefetch=10)
    .map(emb=compute_embedding)
    .save("image_embeddings")
)
```

## Data Stays in Storage

DataChain's storage-native architecture means files live in cloud storage and are never copied. The File abstraction operates by reference -- only files needed by Python operations are downloaded, and only when their content is actually accessed.

## Async Prefetch

`prefetch=N` downloads N files ahead while the current file is being processed, overlapping network I/O with computation. Tune by file size: ~10-16 for small files, 1 for large files.

## File Cache

`settings(cache=True)` stores downloaded files locally. On the next run, the same files are served from cache.

```python
import datachain as dc

chain = (
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, cache=True)
    .map(emb=clip_embedding)
    .save("image_embeddings")
)
```

Cache identifies files by path, etag, and version. If a file is overwritten on storage, DataChain re-downloads it.

On distributed clusters in Studio, cache can be a shared volume so all machines read from the same cache directory.

## Checkpoints

Checkpoints make pipeline failure recoverable without reprocessing from scratch. DataChain tracks execution state and reuses results of previous runs.

Two levels of checkpointing operate independently:

- **Dataset checkpoints** skip entire `save()` calls when the chain hash matches
- **Python-operation checkpoints** save per-row progress inside `map()` and `gen()` -- if an operation fails mid-execution, only unprocessed rows are recomputed

```bash
# Force full re-run
export DATACHAIN_IGNORE_CHECKPOINTS=1
```

Checkpoints work in script-based execution only -- not in the Python REPL or Jupyter notebooks.

## Delta Updates

`delta=True` processes only new and changed files:

```python
import datachain as dc

chain = (
    dc.read_storage(
        "s3://bucket/data/",
        update=True,
        delta=True,
        delta_on="file.path",
        delta_compare="file.mtime",
    )
    .map(result=process_file)
    .save("processed_data")
)
```

The resulting dataset is always complete -- both previously processed and newly processed records.

### The Error Field Pattern

Standard approach for production pipelines that auto-retry failures:

```python
import datachain as dc

def process_file(file):
    try:
        content = file.read_text()
        result = analyze_content(content)
        return {"content": content, "result": result, "error": ""}
    except Exception as e:
        return {"content": "", "result": "", "error": str(e)}

chain = (
    dc.read_storage(
        "s3://bucket/data/",
        delta=True,
        delta_on="file.path",
        delta_retry="error",
    )
    .map(result=process_file)
    .save("processed_files")
)
```

On each run, DataChain processes newly added files (delta) and re-processes previously failed records (retry). Transient failures resolve automatically.

## Bucket Listing Cache

Storage listings are cached as datasets. Scanning a GCS bucket of 1M files takes minutes once, then queries resolve instantly.

## Vectorized Operations

Data-engine operations (filter, group_by, order_by, mutate, aggregate) run as SQL inside the Memory Engine at warehouse speed. The entire [function library](functions.md) runs natively without Python.

## Dataset Reuse

The most powerful optimization: starting from previously computed results. When a pipeline consumes `read_dataset("embeddings")` instead of recomputing from raw files, it skips all upstream work. This is why `save()` is the default terminal operation.

## Optimization Summary

| Layer | What it eliminates | How to enable |
|---|---|---|
| Data stays in storage | Unnecessary file copies | Automatic (File abstraction) |
| Bucket listing cache | Redundant storage scans | Automatic |
| Async prefetch | I/O idle time | `settings(prefetch=N)` |
| File cache | Redundant downloads | `settings(cache=True)` |
| Checkpoints | Wasted computation on failure | Automatic (scripts only) |
| Delta updates | Reprocessing unchanged data | `delta=True` on read |
| Parallel compute | Sequential bottleneck | `settings(parallel=N)` |
| Distributed compute | Single-machine ceiling | `settings(workers=K)` (Studio) |
| Vectorized ops | Python overhead for SQL-expressible work | Use data-engine operations |
| Dataset reuse | Recomputing upstream work | `save()` + `read_dataset()` |
