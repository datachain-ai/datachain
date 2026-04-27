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

DataChain chains Python functions and data operations into composable queries. Python functions process files or data and produce data. Data operations run as SQL at warehouse speed: filter, join, aggregate. Because the system sees the full chain before executing, it applies ten layers of automatic optimization, from no-copy file references to dataset reuse. Every query deposits results into **Data Memory** as a versioned, typed dataset, and the next query starts from what the last one produced. Each session enriches what the next session reads; agents and people build new conclusions on top of prior conclusions instead of re-deriving from raw bytes.

The **Python Data Engine** is the production layer: it runs your Python functions in parallel across threads and machines with async prefetch, file caching, and checkpoints. The **Query Engine** (SQLite locally, ClickHouse in SaaS) is the recall layer: it filters, joins, and searches across datasets at warehouse speed. The **Knowledge Base** is the compilation layer that turns persistent datasets into agent-readable knowledge for Claude Code, Codex, Cursor, custom harnesses, and any LLM they support.

**Get Started**

- [🧭 Why DataChain](why.md) — the problem, who it fits, who it does not
- [🤖 Agents](getting-started/agents.md) — AI-driven with a knowledge base
- [🐍 Python](getting-started/python.md) — Full control over data processing
- [💡 Concepts](concepts/index.md) — Memory, Datasets, and the dual engine
- [🧩 Use Cases](use-cases/index.md) — five patterns where the harness changes the work

<div style="max-width: 680px; margin: 2em auto 0;">
  <img src="assets/data-memory.svg" alt="DataChain architecture">
</div>
