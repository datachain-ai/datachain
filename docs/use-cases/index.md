---
title: Use Cases
---

# Use Cases

## Data workflows

- **Training and evaluation data:** [extract metadata, curate a selection and feed PyTorch](../guide/file-pipelines.md#public-images).
- **Physical AI and robotics:** [curate LeRobot camera episodes](../guide/file-pipelines.md#robot-demonstrations) or [select MCAP recordings by sensor topics](../guide/file-pipelines.md#catalog-mcap-recordings).
- **Data exploration:** query saved metadata and model outputs to investigate failures, find rare conditions and compare data coverage. See [data operations](../guide/operations.md).
- **Large inventories:** discover objects across S3, GCS and Azure from storage metadata. See [multi-cloud inventory](../guide/file-pipelines.md#inventory-across-clouds).

## Agent workflows

Patterns for building on saved results across sessions, with examples and code.

- [Agent compounding across sessions](agent-compounding.md): each session reads typed datasets the last session produced, instead of re-deriving from raw bytes.
- [Agent anti-patterns over object storage](agent-anti-patterns.md): the things coding agents do wrong on S3 by default, and the chain pattern that replaces each one.
- [Agent as a dataset producer](agent-as-dataset-producer.md): the agent's output is a versioned typed dataset, not a notebook artifact. Other agents and people query it by name.
- [Retroactive agent runs](retroactive-agent-runs.md): a new model drops; re-run it across yesterday's frames without a full recompute, paying only for the new work.
- [Cross-agent reasoning](cross-agent-reasoning.md): Claude Code's saved dataset becomes Cursor's input. The Knowledge Base is the contract.
