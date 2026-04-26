---
title: Execution Model
---

# Execution Model

DataChain has two execution engines with a total boundary: every operation is either Python or Memory Engine, never both.

## Memory Engine

The Memory Engine is the columnar SQL backend (SQLite locally, ClickHouse in Studio) where filter, merge, join, group_by, order_by, mutate, and vector search run at warehouse speed, scaling to billions of records. If an operation can be expressed without Python, it runs here.

```python
import datachain as dc

(
    dc.read_storage("gs://datachain-demo/")
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

No Python runtime spins up. No rows are deserialized. The query runs at warehouse speed on millions of records.

## Dataset Registry

The Memory Engine is the home of the dataset registry: the queryable system of record for all datasets and their versions. The registry makes every dataset discoverable, joinable with other metadata, and accessible to agents without file-system traversal.

```python
import datachain as dc

# Browse all datasets
for info in dc.datasets().collect("dataset"):
    print(f"{info.name} v{info.version}")

# Inspect a specific dataset
ds = dc.read_dataset("image_embeddings")
ds.print_schema()
print(ds.name, ds.version)
```

## Agent Recall

Agents query the Memory Engine directly. When an agent needs to find existing datasets, filter by schema, join metadata across versions, or run similarity search over embeddings, it submits those queries to the Memory Engine. This is the recall side of the feedback loop: agents consume data context from the [Knowledge Base](knowledge-base.md), then reach into the Memory Engine for the precise operations that context cannot precompute.

## Python Data Engine

The Python Data Engine accepts data from two sources: object storage (S3, GCS, Azure) and databases (Postgres, Snowflake). It executes `map()`, `gen()`, and `agg()` operations, anything that needs file content, ML models, or LLM calls. It runs operations in parallel threads with async I/O and distributed across workers.

```python
import datachain as dc

(
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, workers=50)  # 8 threads on 50 machines
    .map(emb=compute_embedding)
    .save("image_embeddings")
)
```

Every primitive (parallel dispatch, prefetch, batching) is designed for data operations where each row triggers expensive work: an LLM call, a model inference, a file download and parse.

## Pydantic as Bridge

Pydantic is the shared type system that connects Python outputs to the Memory Engine. A single `save()` takes a Python result with its full Pydantic schema and makes it warehouse-queryable.

Types enable transpilation: `filter(dc.C("det.confidence") > 0.9)` compiles to a SQL WHERE clause instead of deserializing every row into Python. Without typed schemas, transpilation is impossible; without transpilation, there is no warehouse speed.

```python
import datachain as dc

chain = dc.read_dataset("llm_responses")

# This traverses nested Pydantic models and runs entirely in SQL
cost = (
    chain.sum("response.usage.prompt_tokens") * 0.000002
    + chain.sum("response.usage.completion_tokens") * 0.000006
)
print(f"Spent ${cost:.2f} on {chain.count()} calls")
```

## The Transpiler

Users operate with Pydantic models and Python expressions. The system transpiles those expressions to SQL and runs them inside the Memory Engine. The result: full SQL power (filter, join, merge, group_by, order_by, windowing functions) without writing or knowing SQL.

For agents, this is critical. Agents generate Python, not SQL. The transpiler means an agent's Python output runs as fast as hand-written SQL; the agent never needs to know that a database exists.

## Local vs Studio

| Aspect | Local | Studio |
|---|---|---|
| Data Engine | SQLite | ClickHouse |
| Python Execution | Threads | Kubernetes clusters |
| Scale | Single machine | 10-300 machines (BYOC) |
| Transpiler | Automatic | Automatic (handles dialect differences) |
