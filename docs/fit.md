---
title: Where DataChain fits
---

# Where DataChain fits

AI coding agents are stateless on data. Each session starts from raw bytes; embeddings recompute, classifications recompute, joins recompute, and the work fails to compound. The two-orders-of-magnitude gap between recomputing a dataset (~$100), querying a materialised one (~$1), and reading its summary (~$0.01) is what makes Data Memory worth building. DataChain fits where that gap is real for your team.

## When DataChain fits

- Your data is files in object storage (images, video, documents, sensor data) or rows in a database.
- An agent is doing real work over that data.
- You want what the agent produces to outlive the session that produced it.

## When it does not fit

- BI on a curated warehouse with a stable schema: dbt plus a semantic layer.
- Conversation memory for one user across chat sessions: Letta or Mem0.
- File-blob versioning for a small ML repo: DVC.

DataChain is the Python-and-files layer; when the work is not Python over files, the tools above sit closer to the shape of the problem.

## Next

- [Use cases](use-cases/index.md): five patterns where the harness changes the work
- [Agents quickstart](getting-started/agents.md)
