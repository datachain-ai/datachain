---
title: Compute Engine
---

# Compute Engine

The Compute Engine runs heavy Python work over files in object storage: LLM calls, model inference, multimodal extraction, expensive per-row file I/O. Parallel by default, async over network round-trips, distributed across machines through Studio, checkpoint-recoverable so partial progress survives failure. This is the bottom tier of recall economics: the only layer that can answer a question for which no materialised result exists yet.

## Storage Native, Database As Bridge

Object storage is the primary substrate. Files in S3, GCS, or Azure are read by Python through the chain, processed, and the results deposited as typed records into [Dataset DB](dataset-db.md). Database access is a bridge: `read_database()` pulls a query result into the operational layer, `to_database()` writes results back. Appropriate for selective enrichment of warehouse data, not a warehouse-native runtime.

```python
import datachain as dc

(
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, workers=50)  # 8 threads on 50 machines
    .map(emb=compute_embedding)
    .save("image_embeddings")
)
```

Every primitive (parallel dispatch, prefetch, batching) is designed for Python functions where each row triggers expensive work: an LLM call, a model inference, a file download and parse.

## Parallel, Async, Distributed

Per-row costs at AI scale leave no alternative. Sequential execution is economically impossible at dataset scale: a per-row LLM call at one cent over 500,000 documents is several thousand dollars and hours of wall-clock time, and that math compounds with every modality. Parallelism, prefetch, and batching exist because the economics demand it, not as optimisations on top of a sequential model. Threads handle CPU-bound steps; async I/O overlaps storage round-trips with compute; distributed dispatch fans the same chain across a Kubernetes cluster in Studio without changing the Python.

## Checkpoints

Heavy Python compute over files cannot afford to be stateless. dbt can rebuild a warehouse model in seconds of warehouse time; a per-row LLM enrichment over a million files cannot. Checkpoints make recovery a first-class operation: failed runs resume from the last committed batch instead of restarting from raw bytes.

## Pydantic as Bridge

Pydantic is the shared type system that connects Python function outputs to the Dataset DB. A single `save()` takes a Python result with its full Pydantic schema and makes it warehouse-queryable.

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

Users operate with Pydantic models and Python expressions. The transpiler dispatches per-row Python work to the Compute Engine and compiles per-column expressions (filter, join, group_by, similarity) to SQL that runs inside [Dataset DB](dataset-db.md). The result: full SQL power without writing or knowing SQL.

For agents, this is critical. Agents generate Python, not SQL. The transpiler means an agent's Python output runs as fast as hand-written SQL; the agent never needs to know that a database exists. Existing systems collapse to one side of this boundary: Polars and Pandas keep Python expressivity but lose warehouse-speed recall; Snowflake and Snowpark keep warehouse speed but lose Python expressivity. DataChain holds both behind one chain.

## Boundary With the Dataset DB

The Compute Engine produces typed records; the [Dataset DB](dataset-db.md) persists them and serves queries over them. The user writes one chain; the boundary is invisible at the surface and total underneath. Without the Compute Engine, the Dataset DB is empty; without the Dataset DB, Compute Engine results vanish at the end of the session.

## Local vs Studio

| Aspect | Local | Studio |
|---|---|---|
| Dataset DB backend | SQLite | ClickHouse |
| Python execution | Threads | Kubernetes clusters |
| Scale | Single machine | 10-300 machines (BYOC) |
| Transpiler | Automatic | Automatic (handles dialect differences) |
