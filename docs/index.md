# <a class="main-header-link" href="/" ><img style="display: inline-block;" src="/assets/datachain.svg" alt="DataChain"> <span style="display: inline-block;"> DataChain</span></a>

<p align="center" class="subtitle">The Data Context Layer for unstructured data</p>

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

**Turn files in S3, GCS, and Azure into versioned, typed datasets your agents can query and reason over.**

Your data lives in object storage (millions of images, hours of video, documents) and databases (structured tables). Every chain a teammate or agent runs deposits a typed, versioned dataset into the **Dataset DB**: embeddings, classifications, joins, scores. At scale, those datasets are too expensive to recompute and too scattered to find on demand.

DataChain is the Python library that runs your code over heavy files and tables in parallel and queries the Dataset DB at warehouse speed. Read from S3, GCS, or Azure, run your code, save as a Pydantic-typed dataset; the next pipeline or agent picks up from there. Claude Code, Cursor, and Codex made AI good at code by giving it the repo context. Agents over your data need the same: a **Data Context Layer** with schemas, lineage, and prior conclusions, captured during production, not curated after.

## Get started

- **[🤖 Agents](getting-started/agents.md)** - knowledge base for Claude Code, Codex, and Cursor
- **[🐍 Python](getting-started/python.md)** - full control over data processing
- **[💡 Concepts](concepts/index.md)** - the Dataset DB, the Compute Engine, and the Knowledge Base
- **[🧩 Use Cases](use-cases/index.md)** - patterns where the harness changes the work

<div style="max-width: 680px; margin: 2em auto 0;">
  <img src="assets/architecture.svg" alt="DataChain architecture: AI agents and humans, the Dataset DB at the centre, and object storage">
</div>
