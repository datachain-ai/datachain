---
title: Multi-Stage Pipelines
---

# Multi-Stage Pipelines

Real production workflows are sequences of stages where each stage is an independent `save()` that produces a versioned dataset. The next stage reads from the previous one with `read_dataset()`. Each stage is independently checkpointed, versionable, and resumable.

## Stage Boundaries

A `save()` followed by `read_dataset()` creates a stage boundary:

```python
import datachain as dc

# Stage 1: Extract chunks from PDFs
dc.read_storage("s3://docs/*.pdf") \
    .gen(chunk=split_pdf) \
    .save("chunks")

# Stage 2: Generate embeddings
dc.read_dataset("chunks") \
    .setup(model=lambda: load_embedding_model()) \
    .settings(parallel=8) \
    .map(emb=embed_chunk) \
    .save("chunk_embeddings")

# Stage 3: Classify with LLM
dc.read_dataset("chunk_embeddings") \
    .setup(client=lambda: create_llm_client()) \
    .settings(parallel=4) \
    .map(category=classify) \
    .save("classified_chunks")
```

If Stage 2 fails at 80% completion, checkpoints preserve completed rows. Fix the bug, re-run, and DataChain resumes from where it left off. Stage 1 is skipped entirely because its chain hash matches.

## Comparative Evaluation

Run two models on the same dataset, then merge the results:

```python
import datachain as dc

# Run two models
dc.read_dataset("chunk_embeddings") \
    .setup(client=lambda: model_a_client()) \
    .map(response_a=run_model_a) \
    .save("model_a_results")

dc.read_dataset("chunk_embeddings") \
    .setup(client=lambda: model_b_client()) \
    .map(response_b=run_model_b) \
    .save("model_b_results")

# Merge and compare
a = dc.read_dataset("model_a_results")
b = dc.read_dataset("model_b_results")
comparison = a.merge(b, on="chunk.text")
comparison.save("model_comparison")
```

Each model run is a versioned dataset with full provenance. The comparison dataset tracks which inputs and model versions produced which results.

## Cost Tracking

Each stage's LLM cost is visible through aggregate analytics on nested token fields, with no deserialization and no Python runtime:

```python
import datachain as dc

chain = dc.read_dataset("model_a_results")
cost = (
    chain.sum("response_a.usage.prompt_tokens") * 0.000002
    + chain.sum("response_a.usage.completion_tokens") * 0.000006
)
print(f"Spent ${cost:.2f}")
```

## Why Stages Matter

Single queries work for simple workflows. Multi-stage pipelines build production systems where each step can be independently monitored, rerun, and compared across versions. The pattern: `save()` to checkpoint, `read_dataset()` to continue. Each stage deposits a versioned conclusion that the next stage builds on.
