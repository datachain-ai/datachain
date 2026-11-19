---
title: Cross-Agent Reasoning
---

# Cross-Agent Reasoning

## Without the harness

A teammate runs Claude Code over your S3 bucket on Monday and produces a curated set of high-quality images for fine-tuning. The result lives in their notebook output and a Slack thread. On Wednesday, you open Cursor on the same project for a related task. Cursor cannot see what Claude built. Cursor re-lists the bucket, re-runs the same quality scoring, and rebuilds an equivalent curated set from scratch, with slightly different thresholds because the prompt drifted in the retelling.

## With DataChain

Agents save processing results as typed, versioned datasets. Another agent on the same project inspects those saved datasets before generating code. The dataset name and version (`curated_images@1.0.0`) is the contract between agents. For shared team datasets and generated descriptions, connect [Studio MCP](../studio/user-guide/agents/mcp.md).

Install the skill:

```bash
datachain skill install --target claude
```

A Claude Code session on Monday produces this:

```python
import datachain as dc
from pydantic import BaseModel


class Quality(BaseModel):
    score: float
    reason: str


(
    dc.read_storage("s3://acme/raw-images/", anon=True, type="image")
    .map(quality=score_with_claude_vision)
    .filter(dc.C("quality.score") > 0.85)
    .save("curated_images")
)
```

A Cursor session on Wednesday installs the same skill (`datachain skill install --target cursor`), inspects saved datasets with `dc.datasets()` and `dc.describe("curated_images")`, and finds the existing result. The next prompt becomes:

```text
Train a small classifier on curated_images, holding out 20% for eval.
Save predictions and confusion matrix.
```

Cursor produces a new chain that consumes the prior agent's output by name:

```python
import datachain as dc
from datachain.toolkit import train_test_split

train, evaluate = train_test_split(dc.read_dataset("curated_images"), [0.8, 0.2])

(
    evaluate.map(label=lambda file: file.path.split("/")[-2])
    .map(prediction=run_classifier)
    .save("classifier_predictions")
)
```

The contract works in either direction. A human teammate runs `datachain dataset describe curated_images`, sees the same schema and dependencies, and picks up the work without asking who ran it.

## What this enables

- **No re-derivation across agents.** Claude Code's curated set is Cursor's input. The score column comes with the dataset; nobody re-computes it.
- **Conclusions outlive the agent.** A fine-tuning split saved by one agent is queryable by name months later, after the original session is gone.
- **Humans and agents inspect the same results.** The CLI and Python SDK expose saved dataset metadata; Studio MCP adds shared descriptions and search.

## See also

- [Knowledge Base](../concepts/knowledge-base.md): how datasets become agent-readable pages
- [Knowledge Base guide](../guide/knowledge-base.md): shared descriptions and local dataset inspection
- [Datasets](../concepts/datasets.md): the named, versioned, typed contract
