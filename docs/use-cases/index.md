---
title: Use Cases
---

# Use Cases

Five patterns where running an agent through DataChain changes the work, instead of just speeding it up. Each page is short: a concrete agent failure mode without the harness, the DataChain pattern that fixes it, and runnable code.

- [Agent compounding across sessions](agent-compounding.md): each session reads typed datasets the last session produced, instead of re-deriving from raw bytes.
- [Agent anti-patterns over object storage](agent-anti-patterns.md): the things coding agents do wrong on S3 by default, and the chain pattern that replaces each one.
- [Agent as a dataset producer](agent-as-dataset-producer.md): the agent's output is a versioned typed dataset, not a notebook artifact. Other agents and people query it by name.
- [Retroactive agent runs](retroactive-agent-runs.md): a new model drops; re-run it across yesterday's frames without a full recompute, paying only for the new work.
- [Cross-agent reasoning](cross-agent-reasoning.md): Claude Code's saved dataset becomes Cursor's input. The Knowledge Base is the contract.
