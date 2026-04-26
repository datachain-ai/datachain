---
title: Concepts
---

# Concepts

DataChain is built around a small number of ideas. Understanding them makes the entire API predictable. Start with Data Memory and Datasets, then explore deeper topics as needed.

- [Data Memory](data-memory.md): the accumulated record of everything the team has done with its data, composed of versioned, typed datasets
- [Datasets](datasets.md): the atomic unit: named, versioned, typed, immutable; the unit of persistence, sharing, compounding, and reasoning
- [Chain](chain.md): query combining Python and SQL execution in one composable chain; lazy, optimized, atomic
- [Files and Types](files-and-types.md): the File abstraction, modality types, annotation types, and the type system
- [Execution Model](execution-model.md): the dual-engine architecture: Python Data Engine + Memory Engine; dataset registry
- [Knowledge Base](knowledge-base.md): data context layer in the Skill and MCP container; derived from Data Memory via LLM enrichments
