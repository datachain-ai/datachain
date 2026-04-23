---
title: Welcome to DataChain
---
# <a class="main-header-link" href="/" ><img style="display: inline-block;" src="/assets/datachain.svg" alt="DataChain"> <span style="display: inline-block;"> DataChain</span></a>

<style>
.md-content .md-typeset h1 { font-weight: bold; display: flex; align-items: center; justify-content: center; gap: 5px; }
.md-content .md-typeset h1 .main-header-link { display: flex; align-items: center; justify-content: center; gap: 8px;
 }
</style>

<p align="center">
  <a href="https://pypi.org/project/datachain/" target="_blank">
    <img src="https://img.shields.io/pypi/v/datachain.svg" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/datachain/" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/datachain" alt="Python Version">
  </a>
  <a href="https://codecov.io/gh/datachain-ai/datachain" target="_blank">
    <img src="https://codecov.io/gh/datachain-ai/datachain/graph/badge.svg?token=byliXGGyGB" alt="Codecov">
  </a>
  <a href="https://github.com/datachain-ai/datachain/actions/workflows/tests.yml" target="_blank">
    <img src="https://github.com/datachain-ai/datachain/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a>
</p>

DataChain is a Python-native data platform that unifies files, database records, and structured formats into versioned, typed datasets with automatic lineage tracking. Every operation deposits results into persistent memory that the team and AI agents build on.

```python
import datachain as dc

(
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, cache=True)
    .map(emb=compute_embedding)
    .save("image_embeddings")   # versioned, named, typed
)

# Later: anyone (or any agent) can build on it
ds = dc.read_dataset("image_embeddings")
```

## Why DataChain

**Data work produces no persistent memory.** Every pipeline, every exploration, every labeling session produces knowledge -- and that knowledge evaporates when the script finishes. DataChain changes this: every `save()` deposits a versioned, typed, lineage-tracked dataset that the next person or agent starts from.

**Python is the center of gravity for AI data work.** DataChain runs Python operations (ML models, LLM calls, file processing) in parallel and distributed, while metadata operations (filter, join, group_by, aggregate) run at warehouse speed in a columnar SQL engine. Pydantic bridges the two -- Python outputs flatten into efficient columnar storage.

**Files stay where they are.** DataChain never copies data from cloud storage. The intelligence layer operates by reference through the File abstraction.

## Key Capabilities

- **Multimodal dataset versioning** -- images, video, audio, text, PDFs as typed, versioned datasets
- **Dual execution engine** -- warehouse-speed SQL for metadata, parallel Python for AI operations
- **Automatic lineage** -- every `save()` records code, dependencies, author, and creation time
- **LLM and model integration** -- parallelize API calls, serialize structured responses, track costs
- **Vector search** -- built-in cosine/euclidean/L2 distance functions at SQL speed
- **Delta updates** -- process only new and changed files on each run
- **Checkpoints** -- resume from failure without reprocessing

## Documentation Guide

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Installation, first pipeline, and the mental model.

    [:octicons-arrow-right-24: Quick Start](quick-start.md) · [Core Concepts](getting-started/core-concepts.md)

-   **Concepts**

    ---

    Why DataChain works the way it does.

    [:octicons-arrow-right-24: Data Memory](concepts/data-memory.md) · [Datasets](concepts/datasets.md) · [Chain](concepts/chain.md) · [Execution Model](concepts/execution-model.md)

-   **Guides**

    ---

    In-depth coverage of each capability.

    [:octicons-arrow-right-24: Reading Data](guide/reading-data.md) · [Operations](guide/operations.md) · [UDFs](guide/udfs.md) · [Datasets](guide/datasets.md) · [Scaling](guide/scaling.md) · [Best Practices](guide/best-practices.md)

-   **Use Cases**

    ---

    Complete workflows for common scenarios.

    [:octicons-arrow-right-24: Unstructured ETL](use-cases/unstructured-data-etl.md) · [LLM Pipelines](use-cases/llm-pipelines.md) · [ML Training](use-cases/ml-training-data.md) · [Analytics](use-cases/multimodal-analytics.md)

-   **API Reference**

    ---

    Auto-generated from docstrings.

    [:octicons-arrow-right-24: DataChain](references/datachain.md) · [Data Types](references/data-types/index.md) · [Functions](references/func.md) · [UDF](references/udf.md)

-   **Studio**

    ---

    Enterprise features: centralized registry, distributed compute, UI.

    [:octicons-arrow-right-24: Studio Guide](studio/index.md)

</div>

## Open Source and Studio

- **[DataChain Open Source](https://github.com/datachain-ai/datachain)**: Python library for versioned, typed datasets with automatic lineage.
- **[DataChain Studio](https://studio.datachain.ai/)**: Centralized dataset registry, ClickHouse-powered analytics, distributed Kubernetes compute, team collaboration, and UI for multimodal data.
