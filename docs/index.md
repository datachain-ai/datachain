# <a class="main-header-link" href="/" ><img style="display: inline-block;" src="/assets/datachain.svg" alt="DataChain"> <span style="display: inline-block;"> DataChain</span></a>

<p align="center" class="subtitle">Data Memory for AI Agents</p>

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

**The model floor is the same for everyone. The context ceiling is yours.**

Millions of images, hours of video, documents, and rows in databases sit in storage as raw bytes. **Data Memory** is what your team and its agents have made of them: typed, versioned datasets - embeddings, classifications, joins, scores, every conclusion an agent or pipeline has already reached. At scale, those conclusions are too expensive to re-derive each session and too scattered across storage, databases, and sidecar files to find on demand. DataChain produces and keeps them, indexed for warehouse-speed recall.

Read files from S3, GCS, or Azure, run your code, save the result as a Pydantic-typed dataset. The next pipeline or agent picks up from there.

## Why Data Memory

Claude Code, Cursor, and Codex made AI good at code by giving it the repo context. Agents over your data need the same: a **data context layer** that describes what datasets exist, what they mean, and what is already computed. **Data Memory is what makes that layer real** - typed, versioned datasets your team produces by running pipelines, surfaced through the Knowledge Base. Without it, the context layer points at nothing and every session starts from zero.

## Get started

- **[🤖 Agents](getting-started/agents.md)** - knowledge base for Claude Code, Codex, and Cursor
- **[🐍 Python](getting-started/python.md)** - full control over data processing
- **[💡 Concepts](concepts/index.md)** - Data Memory, the Python and Query engines, and the Knowledge Base
- **[🧩 Use Cases](use-cases/index.md)** - patterns where the harness changes the work

## Architecture diagram

<div style="max-width: 680px; margin: 2em auto 0;">
  <img src="assets/data-memory.svg" alt="DataChain architecture">
</div>
