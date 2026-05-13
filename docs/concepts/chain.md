---
title: Chain
---

# Chain

A chain is a query that combines Python functions and data operations into a single composable sequence. Python functions run on the [Compute Engine](compute-engine.md): file content reads, ML models, LLM calls, multimodal extraction. Data operations run as SQL at warehouse speed against the [Dataset DB](dataset-db.md): filter, join, aggregate, similarity search. Nothing runs until a terminal operation like `save()`, `show()`, or `to_pandas()`. DataChain is named for this pattern: every query is a chain of operations, and every chain deposits its results into the Dataset DB as a versioned [dataset](datasets.md).

## Lazy Evaluation and Optimization

Lazy evaluation lets the system see the full chain before executing a single row. The transpiler compiles data operations to SQL; filters push down, operations reorder, unnecessary computation gets skipped.

```python
import datachain as dc

chain = (
    dc.read_storage("s3://bucket/images/", type="image")
    .filter(dc.C("file.size") > 1000)
    .mutate(ext=dc.func.path.file_ext("file.path"))
    .settings(parallel=8)
    .map(emb=compute_embedding)
    .save("image_embeddings")
)
```

The `filter` and `mutate` are data operations that compile to SQL and run at warehouse speed. The `map` runs your Python function in parallel. The system decides which engine handles what.

## Atomicity

Results go to staging tables during execution. Only `save()` commits them into the Dataset DB. If the chain fails at any point, no partial results pollute the shared layer. Deposits are all-or-nothing. Combined with [checkpoints](../guide/scaling.md#checkpoints), failed chains resume from where they stopped on the next run, but nothing is visible to other users until the chain succeeds.

## Declarative

A chain declares WHAT to do, not HOW. The same fluent API covers SQL-compiled [data operations](../guide/operations.md) and [Python functions](../guide/python-engine.md). You never specify which operations become SQL and which run Python; the system infers this from the operation type and the Pydantic schema.

## Composable

Every operation returns a new immutable chain. The original is never modified. This lets you branch from any intermediate point:

```python
import datachain as dc

base = (
    dc.read_storage("s3://bucket/images/", type="image")
    .map(emb=clip_embedding)
    .persist()   # cache intermediate result
)

large = base.filter(dc.C("file.size") > 10_000).save("large_images")
small = base.filter(dc.C("file.size") <= 10_000).save("small_images")
```

## Entry Points

Factory functions create a chain from an external source:

```python
import datachain as dc

# Unstructured data from storage
chain = dc.read_storage("s3://bucket/images/", type="image")

# Structured formats
chain = dc.read_csv("s3://bucket/labels.csv")
chain = dc.read_json("gs://bucket/annotations.json", jmespath="images")
chain = dc.read_parquet("s3://bucket/data/*.parquet")

# SQL databases
chain = dc.read_database("SELECT * FROM products", "postgresql://host/db")

# In-memory data
chain = dc.read_pandas(df)
chain = dc.read_hf("beans", split="train")
chain = dc.read_values(scores=[1.2, 3.4, 2.5])

# Previously saved datasets
chain = dc.read_dataset("image_embeddings")
```

## Terminal Operations

Terminal operations trigger execution:

- **Save**: `save("name")` creates a versioned dataset. `persist()` caches intermediate results.
- **Export**: `to_pandas()`, `to_parquet()`, `to_csv()`, `to_json()`, `to_pytorch()`, `to_storage()`, `to_database()`
- **Inspect**: `show()`, `collect()`, `count()`, `sum()`, `avg()`, `min()`, `max()`
