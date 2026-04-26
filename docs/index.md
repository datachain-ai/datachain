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

DataChain is data memory for AI.

You process files with AI -- images, video, sensor data, documents -- and produce embeddings, labels, scores, filtered subsets. DataChain saves every result as a versioned, typed, queryable dataset so the next pipeline or agent builds on what already exists instead of starting over.

Without it, derived data lives in throwaway scripts and local files. Agents recompute what was already done because they cannot see it. With DataChain, every result has a schema, lineage, and a name -- findable by the next person or agent that needs it.

```mermaid
flowchart LR
    A["Source Data\nimages, video, lidar,\naudio, documents"] --> B["Compute\nembeddings, classifications,\nLLM responses, joins"]
    B --> C["Store\nversioned, typed,\nqueryable datasets"]
    C --> D["Discover\nschemas, lineage,\nsamples, context"]
    D -->|"next agent starts here"| B
```

## Get Started

- **[Getting Started: Agents](getting-started/agents.md)** -- agents write DataChain pipelines and build a knowledge graph that makes every subsequent task faster
- **[Getting Started: Python](getting-started/python.md)** -- write DataChain pipelines directly for full control over data processing
- **[Concepts](concepts/index.md)** -- understand Data Memory, Datasets, and the dual engine
