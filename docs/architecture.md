---
title: Architecture
---

# Architecture

<p align="center">
  <img src="../assets/architecture.svg" alt="DataChain architecture" width="700" />
</p>

A **dataset** is the unit of work: a named, versioned result of a pipeline step like `pets_embeddings@1.0.0`. Every `.save()` registers one.

DataChain is split into two parts, matching the two containers in the diagram.

**Python Library** runs and queries data:

- **Python Data Engine** runs your Python over heavy files and tables in parallel, with async prefetch and checkpoints.
- **Data Memory** is the typed, versioned dataset registry. Every chain deposits its results here.
- **Query Engine** filters, joins, and runs similarity search across Data Memory at warehouse speed.

**Skill and MCP** serves agents:

- **Knowledge Base** is a structured reflection of Data Memory enriched by LLMs: markdown files agents read before generating code. Always accurate because it's derived.
