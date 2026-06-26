---
title: Knowledge Base
---

# Knowledge Base

The Knowledge Base is the compilation layer that turns persistent typed datasets into agent-readable knowledge. It reads accumulated [datasets](datasets.md) from [Dataset DB](dataset-db.md) and emits typed pages (one per dataset, one per bucket listing) that the next agent session reads as premises before generating its first line of code. Each page carries schema, lineage summary, previews, and links. Humans browse the same pages in Obsidian; the compilation step exists for agents.

## Skill and MCP Layer

The Knowledge Base reaches agents through the [Skill and MCP](skill-and-mcp.md) layer, architecturally separate from the [Compute Engine](compute-engine.md) and the [Dataset DB](dataset-db.md) that the Python library exposes. Pipelines run through compute and memory; agents read context through Skill and MCP. This separation reflects the two audiences: pipelines need a compute engine, agents need a context surface.

## Compilation Is What Makes the Store Readable

A raw Dataset DB holds billions of typed records across thousands of dataset versions. An agent cannot reason over that directly any more than a CPU can execute uncompiled source. The Knowledge Base compiles the store into the form the agent's reasoning step consumes: dataset pages that summarise schema, lineage, and prior conclusions in finite tokens. Without this step, three people build three versions of the same dataset because nobody can see what already exists; an agent hallucinates a column that a human classified yesterday.

## Agents Need Context Before They Act

Agents without data context produce wrong answers, not slow ones. An agent hallucinates columns, recomputes what has already been computed, or joins on columns with matching names but different meaning. A human who does not know will at least ask. An agent will not. The Knowledge Base is the data context the agent reads before generating code; it turns "the dataset exists" into "the agent found it and built on it."

The relationship is bidirectional. Agents consume data context from the Knowledge Base and deposit reasoning artefacts back: updated enrichments, new dataset descriptions, schema annotations, query patterns. Each agent interaction enriches what the next agent session compiles from.

## Derived From the Dataset DB

The Knowledge Base is derived from the [Dataset DB](dataset-db.md) via LLM enrichments, never maintained as a separate system. Derivation flows in one direction: Dataset DB to Knowledge Base. Accuracy does not depend on human maintenance. Unlike catalogs that drift, the Knowledge Base is a function of the data, not a parallel system.

## Compiled, Not Retrieved

Compiled knowledge replaces RAG retrieval. Instead of pulling raw chunks from a vector store at query time, the system compiles knowledge into structured pages the next agent reads as a premise. Agents that pull raw chunks from datasets hallucinate columns, joins, and meaning; agents that read compiled pages do not. The Knowledge Base extends this pattern from personal markdown to team typed data. Pages come from the Dataset DB rather than from markdown sources. Schema and lineage are part of the page rather than absent. Compute history is captured automatically rather than reconstructed by hand.

## What It Contains

Each dataset in the Knowledge Base carries:

- **Schema**: column names, types, nested structure
- **Lineage**: what produced this dataset, what it depends on
- **Session context**: when it was last updated, by whom
- **Previews**: sample rows and statistics
- **Links**: connections to related datasets
