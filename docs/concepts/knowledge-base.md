---
title: Knowledge Base
---

# Knowledge Base

The Knowledge Base is the data context layer over [Data Memory](data-memory.md): per-dataset and per-bucket files carrying schemas, lineage summaries, previews, and links. Humans browse it in Obsidian. Agents pre-read it before generating code.

## Skill and MCP Layer

The Knowledge Base lives in the DataChain Skill and MCP layer, architecturally separate from the Python Library that holds the Python Data Engine, Data Memory, and Memory Engine. The Python Library runs pipelines and stores results. The Skill and MCP layer serves data context to agents via a different protocol. This separation reflects the two audiences: pipelines need a compute engine, agents need a context surface.

## Why Data Context Matters

Without a shared, navigable surface, humans and agents work in parallel and diverge. Onboarding collapses to finding the person who knows the person. The Knowledge Base makes the team's accumulated memory visible across people and time, not just queryable, but discoverable.

## Agents Need Context Before They Act

Agents without data context produce wrong answers, not slow ones. An agent hallucinates columns, recomputes what has already been computed, or joins on columns with matching names but different meaning. A human who does not know will at least ask. An agent will not. The Knowledge Base is the data context the agent reads before generating code; it turns "the dataset exists" into "the agent found it and built on it."

The relationship is bidirectional. Agents consume data context from the Knowledge Base, and they submit tasks back to it: updating enrichments, adding new dataset descriptions, refining the context that the next agent session will read.

## Derived From Data Memory

The Knowledge Base is derived from [Data Memory](data-memory.md) via LLM enrichments, never maintained as a separate system. Derivation flows in one direction: Data Memory to Knowledge Base. Accuracy does not depend on human maintenance. Unlike catalogs that drift, the Knowledge Base is a function of the data, not a parallel system.

## What It Contains

Each dataset in the Knowledge Base carries:

- **Schema**: column names, types, nested structure
- **Lineage**: what produced this dataset, what it depends on
- **Session context**: when it was last updated, by whom
- **Previews**: sample rows and statistics
- **Links**: connections to related datasets
