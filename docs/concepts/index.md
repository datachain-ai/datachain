---
title: Concepts
---

# Concepts

DataChain is built around a small number of ideas. Understanding them makes the entire API predictable. Start with the Dataset DB and Datasets, then explore deeper topics as needed.

- [Dataset DB](dataset-db.md): the typed-record store, composed of versioned, typed datasets, queryable at warehouse speed; the operational half of the Data Context Layer for files and Python
- [Datasets](datasets.md): the atom of the Dataset DB: named, versioned, typed, immutable; the unit of persistence, sharing, compounding, and reasoning
- [Chain](chain.md): query combining Python and SQL execution in one composable chain; lazy, optimized, atomic
- [Files and Types](files-and-types.md): the File abstraction, modality types, annotation types, and the type system
- [Compute Engine](compute-engine.md): heavy Python work over files in object storage; parallel, async, distributed, checkpoint-recoverable; the only layer that produces what does not yet exist
- [Knowledge Base](knowledge-base.md): the compilation layer that turns persistent datasets into agent-readable knowledge; derived from the Dataset DB via LLM enrichments
- [Skill and MCP](skill-and-mcp.md): the delivery surface that reaches Claude Code, Cursor, and Codex; agents read context here while pipelines write through the Python library
