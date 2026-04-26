---
title: Welcome to DataChain
---
# <a class="main-header-link" href="/" ><img style="display: inline-block;" src="/assets/datachain.svg" alt="DataChain"> <span style="display: inline-block;"> DataChain</span></a>

<p align="center" class="subtitle">memory for data agents</p>

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

Agents and people need context to do useful work with data. DataChain is a Python library that reads files from storage, runs your code over them, and saves every result as a Pydantic-typed dataset. The next pipeline or agent picks up where the last one left off.

The **Python Data Engine** handles heavy files: parallel, distributed, async with caching, zero-copy access, and checkpoints. **Data Memory** stores results at SQL speed (SQLite locally, ClickHouse in SaaS). The **Memory Engine** filters, joins, and searches across datasets. The **Knowledge Base** provides data context for Claude Code, Codex, Cursor, custom harnesses, and any LLM they support.

<div class="grid cards" markdown>

-   🤖 __Agents__

    ---

    AI-driven pipelines that build a knowledge graph

    [Get started →](getting-started/agents.md)

-   🐍 __Python__

    ---

    Full control over data processing pipelines

    [Get started →](getting-started/python.md)

-   💡 __Concepts__

    ---

    Data Memory, Datasets, and the dual engine

    [Learn more →](concepts/index.md)

</div>

<div style="max-width: 500px; margin: 2em auto 0;">

![DataChain architecture](assets/data-memory.svg)

</div>
