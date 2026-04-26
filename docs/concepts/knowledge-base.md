---
title: Knowledge Base
---

# Knowledge Base

The Knowledge Base is the navigable surface over [Data Memory](data-memory.md): per-dataset and per-bucket files carrying schemas, session context, lineage summaries, previews, and links. Humans browse it. Agents pre-read it before generating code.

## Why Navigable Context Matters

Without a shared, navigable surface, humans and agents work in parallel and diverge. Onboarding collapses to finding the person who knows the person. The Knowledge Base makes the team's accumulated memory visible across people and time, not just queryable, but discoverable.

## Agents Need Context Before They Act

Agents without navigable context produce wrong answers, not slow ones. An agent hallucinates columns, recomputes what has already been computed, or joins on columns with matching names but different meaning. A human who does not know will at least ask. An agent will not. The Knowledge Base is what the agent reads before generating code; it turns "the dataset exists" into "the agent found it and built on it."

## Always Derived, Never Maintained Separately

The Knowledge Base is always derived from the Memory Engine, never maintained as a separate system. Derivation flows in one direction: Memory Engine to Knowledge Base. Accuracy does not depend on human maintenance. Unlike catalogs that drift, the Knowledge Base is a function of the engine, not a parallel system.

## What It Contains

Each dataset in the Knowledge Base carries:

- **Schema**: column names, types, nested structure
- **Lineage**: what produced this dataset, what it depends on
- **Session context**: when it was last updated, by whom
- **Previews**: sample rows and statistics
- **Links**: connections to related datasets
