---
title: Architecture
---

# Architecture

<p align="center">
  <img src="../assets/architecture.svg" alt="DataChain architecture: AI agents and humans, the Dataset DB at the centre, and object storage" width="700" />
</p>

A **dataset** is the unit of work: a named, versioned result of a pipeline step like `pets_embeddings@1.0.0`. Every `.save()` registers one.

DataChain has three components. Together they implement the data harness as a single Python library plus an agent-facing skill and MCP layer. The user writes one [chain](concepts/chain.md); the boundaries are invisible at the surface and total underneath.

- **[Compute Engine](concepts/compute-engine.md)** runs heavy Python work over files in object storage: LLM calls, model inference, multimodal extraction, expensive per-row I/O. Parallel by default, async over network round-trips, distributed across machines through Studio, checkpoint-recoverable. This is the only component that can answer a question for which no materialised result exists yet.
- **[Dataset DB](concepts/dataset-db.md)** holds the typed, versioned datasets every chain deposits and serves filter, join, group_by, and similarity-search operations at warehouse speed over them. The dataset registry, provenance, and warehouse-speed querying are properties of the Dataset DB itself, not separate services next to it.
- **[Knowledge Base](concepts/knowledge-base.md)** compiles the Dataset DB into agent-readable pages: schema, lineage, previews, links. Reaches Claude Code, Cursor, and Codex through the [Skill and MCP](concepts/skill-and-mcp.md) layer, architecturally separate from the Python library that holds the Compute Engine and the Dataset DB. Agents read here first; they query the Dataset DB when summaries are not enough; they trigger Compute Engine work only when nothing higher answers.

The transpiler dispatches per-row Python to the Compute Engine and per-column expressions (filter, join, group_by, similarity) to the Dataset DB. Pydantic is the shared type system: chains carry Python types into the Compute Engine and the same types out of the Dataset DB, with no separate schema registration.
