# <a class="main-header-link" href="/" ><img style="display: inline-block;" src="/assets/datachain.svg" alt="DataChain"> <span style="display: inline-block;"> DataChain</span></a>

<p align="center" class="subtitle">Curate training data and explore large media collections</p>

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

**Curate training data and explore large media collections.**

Prepare training and evaluation datasets for Physical AI, robotics and multimodal models from images,
video, audio and sensor recordings in S3, GCS, Azure or local storage. Write Python or give your coding
agent a task; DataChain handles parallel workers, download overlap and checkpoints, and saves typed,
versioned results for the next question.

Use metadata and model outputs to curate selections, investigate failures and find gaps in data coverage.
Inventory millions or billions of objects using storage metadata before reading file contents.
Optional Studio adds shared datasets and job tracking, with or without a compute cluster.

## Get started

- **[🤖 Agents](getting-started/agents.md)** - task-oriented recipes for Claude Code, Codex, and Cursor
- **[🐍 Python](getting-started/python.md)** - full control over data processing
- **[💡 Concepts](concepts/index.md)** - saved datasets, typed schemas and the Compute Engine
- **[🧩 Use Cases](use-cases/index.md)** - file-processing and saved-result workflows

<div style="max-width: 680px; margin: 2em auto 0;">
  <img src="assets/architecture.svg" alt="DataChain architecture: AI agents and humans, the Dataset DB at the centre, and object storage">
</div>
