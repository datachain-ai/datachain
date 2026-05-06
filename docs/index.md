# <a class="main-header-link" href="/" ><img style="display: inline-block;" src="/assets/datachain.svg" alt="DataChain"> <span style="display: inline-block;"> DataChain</span></a>

<p align="center" class="subtitle">Memory layer for AI agents working with data</p>

<style>
.md-content .md-typeset h1 { font-weight: bold; display: flex; align-items: center; justify-content: center; gap: 5px; }
.md-content .md-typeset h1 .main-header-link { display: flex; align-items: center; justify-content: center; gap: 8px;
 }
.md-content .md-typeset .subtitle { font-size: 1.2em; color: var(--md-default-fg-color--light); margin-top: -0.5em; }
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

<div style="max-width: 760px; margin: 1.5em auto 2em;">
  <img src="assets/data-memory-layer.svg" alt="DataChain — memory layer for AI agents working with data">
</div>

**Turn messy files in storages into structured, reusable memory that AI agents can query and build on.**

Your data lives across object storage and databases: files in S3, GCS, and Azure, and the structured tables that enrich them. Foundation models made every file modality processable by Python, but agent sessions still restart from raw bytes; embeddings, classifications, and joins recompute every run, and the work fails to compound across sessions.

DataChain is the Python library that builds Data Memory, the operational part of a data context layer over files and Python compute. Agent skills and MCP servers form the data harness into Claude Code, Cursor, and Codex. Every chain you run deposits a typed, versioned dataset; the next pipeline reads it instead of recomputing.

## Why DataChain

Code harnesses gave AI agents the repository for code. Agents over your data need the equivalent: schemas, lineage, and prior conclusions, served through the same agent and the same session. Without Data Memory and the data harness that delivers it, every session over data starts from raw bytes. With them, the agent reads a dataset's summary first, queries its columns when needed, and recomputes only when nothing higher answers. Every `.save()` records source code, lineage, author, and time automatically; the layer is captured by running, not curated by hand.

## Get started

- **[🤖 Agents](getting-started/agents.md)** - knowledge base for Claude Code, Codex, and Cursor
- **[🐍 Python](getting-started/python.md)** - full control over data processing
- **[💡 Concepts](concepts/index.md)** - Data Memory, the Compute Engine, and the Knowledge Base
- **[🧩 Use Cases](use-cases/index.md)** - patterns where the harness changes the work
