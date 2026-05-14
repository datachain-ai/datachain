# <a class="main-header-link" href="/" ><img style="display: inline-block;" src="/assets/datachain.svg" alt="DataChain"> <span style="display: inline-block;"> DataChain</span></a>

<p align="center" class="subtitle">The Context Layer for unstructured data</p>

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

<p align="center" style="font-style: italic; font-size: 1.05em; margin: 1.5em 0;">The Model Floor Is the Same for Everyone. The Context Ceiling Is Yours.</p>

**A Python library that turns files in S3, GCS, and Azure into versioned, typed datasets, queryable at warehouse speed.**

Bytes never leave your storage. Two core components: a **Compute Engine** for distributed Python over files and a **Dataset DB** for warehouse-speed queries over Pydantic-typed records. For agent workflows, two more: a **Knowledge Base** of markdown summaries and an **Agent Harness** (skill + MCP) that plugs all of it into Claude Code, Cursor, and Codex, so they understand your data.

## Get started

- **[🤖 Agents](getting-started/agents.md)** - knowledge base for Claude Code, Codex, and Cursor
- **[🐍 Python](getting-started/python.md)** - full control over data processing
- **[💡 Concepts](concepts/index.md)** - the Dataset DB, the Compute Engine, and the Knowledge Base
- **[🧩 Use Cases](use-cases/index.md)** - patterns where the harness changes the work

<div style="max-width: 680px; margin: 2em auto 0;">
  <img src="assets/architecture.svg" alt="DataChain architecture: AI agents and humans, the Dataset DB at the centre, and object storage">
</div>
