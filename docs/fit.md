---
title: Where DataChain fits
---

# Where DataChain fits

An agent over a folder of files starts from raw bytes every session. It re-lists the bucket, re-downloads the files, re-computes embeddings, re-derives the same conclusions the last session reached. The work disappears when the session ends.

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
