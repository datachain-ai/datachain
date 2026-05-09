---
title: Skill and MCP
---

# Skill and MCP

The Skill and MCP layer is the read side of DataChain: the surface agents reach to discover datasets, browse schema, and query Data Memory. It is split into two delivery shapes that map directly to the OSS-vs-Studio split. The **Skill** is the OSS local path: it ships with `pip install datachain`, installs into the agent's harness, and reads the local filesystem. The **MCP server** is the Studio path: it runs in front of a centralized Memory Server, speaks Model Context Protocol over HTTP, and serves a shared dataset registry to teams. Same purpose on both — agents read what the Python library wrote — different deployment shape.

## Skill vs MCP at a Glance

| | Skill | MCP server |
|---|---|---|
| Ships with | `pip install datachain` (OSS) | Studio (hosted) or self-hosted Memory Server |
| Read source | Local `dc-knowledge/` + `.datachain/db` | Centralized ClickHouse + shared Knowledge Base |
| Transport | Filesystem (the agent's own Read/Bash tools) | Model Context Protocol over HTTP |
| Authentication | None | Studio auth tokens; namespace + project permissions |
| Audience | Solo developer, single machine | Team sharing Data Memory across people and machines |
| Scale | One project's `.datachain/db` | Billions of records, thousands of dataset versions |

The Skill is always present; it is what makes the agent literate in DataChain. The MCP server is added when datasets cross machines.

## The Skill (OSS)

The Skill is the agent-side instruction package shipped inside the open-source DataChain library. Installing it places skill files into the agent harness's directory (for Claude Code, that is `~/.claude/skills/datachain/`; Cursor and Codex have analogous slots).

```bash
pip install datachain
datachain skill install --target claude     # also: --target cursor, --target codex
```

The Skill teaches the agent four things:

- **Where the Knowledge Base lives.** The agent reads `dc-knowledge/` before generating code, so it knows which datasets exist and what they contain. Reads happen with the agent's own filesystem tools; no network call.
- **How to introspect Data Memory.** When the Knowledge Base does not answer the question, the Skill instructs the agent to issue a structured query rather than recompute from raw files. In OSS mode, that query runs against `.datachain/db` (the local SQLite registry). In Studio mode, the same query is routed through the MCP server.
- **When to materialize a new dataset.** The Skill encodes the rule that non-trivial intermediate results land in Data Memory as a versioned dataset, so the next session sees them. Without this discipline, the work fails to compound.
- **How to write idiomatic chains.** The Skill ships codegen hints: import patterns (`import datachain as dc`, never `from datachain import ...`), terminal-operation choices, naming conventions, the `.save()` discipline that keeps lineage clean, and the patterns that compile to SQL versus those that fall back to Python.

The Skill is version-pinned with the SDK. After `pip install -U datachain`, reinstall the Skill so agent-side instructions match the library surface:

```bash
pip install -U datachain
datachain skill install --target claude --force
```

In OSS-only deployment, the Skill is the entire read side. There is no DataChain process running between the agent and the data; the Skill teaches the agent how to read the project's own files. This is a feature, not a limitation: a developer working alone gets agent-aware data work without standing up any service.

## The MCP Server (Studio / Memory Server)

When Data Memory crosses machines — a team sharing datasets, an agent service running headless, a centralized registry of every dataset every team ever produced — the filesystem path stops working. Datasets live in a hosted ClickHouse, the registry indexes thousands of versions across teams, and the Knowledge Base is regenerated on the server side. The MCP server is the read surface for that world.

Model Context Protocol is the open standard for agent tools and resources, with native client support in Claude Code, Cursor, and Codex. DataChain's MCP server runs in Studio (managed) or as a self-hosted Memory Server alongside ClickHouse, and the Skill is configured to route its read calls through it instead of the local filesystem.

```bash
# In Studio: agent points its skill at the team's MCP endpoint
datachain skill install --target claude --mcp https://team.studio.datachain.ai/mcp
```

The server answers three classes of request:

- **Registry listings.** "What datasets exist matching `quant.prod.*`?" returns names, versions, owners, namespaces, and timestamps without scanning the underlying rows. This is the most-frequent call: the agent's first action on every session.
- **Schema introspection.** "What columns does `image_embeddings@1.2.0` have, and what types?" returns the Pydantic schema as structured JSON, including nested types and optional fields. Agents use this to type-check the chain before they generate code.
- **Bounded queries against Data Memory.** Filter, join, group_by, and similarity search over typed records, returning previews, counts, or small result sets. The MCP server enforces row and time limits so a query can never accidentally fan out to a multi-hour scan; large result sets are surfaced as a hint to materialize a new dataset through the Python library instead.

Heavy Python compute is intentionally out of scope. The MCP server does not run `map()`, does not invoke LLMs, does not read raw files. That is the [Compute Engine](compute-engine.md)'s job, dispatched through the Python library, not over MCP. Keeping compute and context apart is what lets the MCP surface stay sub-second and safe to call by default.

Authentication and permissions follow Studio's model: tokens scope to a team, namespaces partition datasets by project, and per-namespace ACLs determine which agents can read what. A read through the MCP server is governed by the same access controls a human pulling up the Studio UI would face.

## When You Need Each

| Scenario | What you install |
|---|---|
| Solo developer, single machine, datasets in `.datachain/db` | Skill alone |
| Small team, shared bucket, each developer runs locally | Skill alone (each dev's own `.datachain/db`) |
| Team sharing Data Memory in Studio | Skill + MCP (Skill ships in `pip install`, MCP hosted by Studio) |
| Self-hosted production deployment | Skill + self-hosted Memory Server with MCP front-end |
| Headless agent service in production | MCP (no human-facing skill harness; agent connects directly to MCP) |

A team's path is typically: start OSS with the Skill, hit the wall when datasets need to be shared across people or machines, move Data Memory to Studio (or a self-hosted Memory Server), and reconfigure the Skill to read through MCP. The Skill keeps its job — agent-side codegen and discipline — and the MCP server takes over registry reads and Data Memory queries.

## Boundary With the Python Library

The library writes; the Skill and MCP layer reads. The two surfaces share Data Memory in the middle but execute on different code paths:

| Direction | Surface | What runs |
|---|---|---|
| Write | Python library (`pip install datachain`) | `read_storage()`, `map()`, `save()`, Compute Engine, transpiler, checkpoint recovery |
| Read (local) | Skill | Filesystem reads of `dc-knowledge/` and `.datachain/db` |
| Read (remote) | MCP server | Registry listings, schema introspection, bounded queries against shared Data Memory |

A pipeline run that produces a new dataset does not automatically refresh the Knowledge Base; a deliberate skill command (`Build a knowledge base for my current datasets` in OSS, automatic recompilation in Studio) regenerates it. This split is intentional: pipeline runs should be cheap to retry, and Knowledge Base regeneration is its own step.

## Why Separation

A unified surface that mixed compute and context would force every agent request through the same code path that runs heavy Python. Latency budgets diverge: a Knowledge Base lookup is sub-millisecond, an LLM-call pipeline is minutes. Permissions diverge: an agent reading dataset summaries is safe by default, an agent triggering a 300-machine compute run is not. Failure modes diverge: a Knowledge Base read returning stale content is recoverable, a half-finished pipeline run is not.

Keeping the two surfaces apart lets each evolve under its own constraints. The Python library can add new operations and storage backends without changing the agent contract. The Skill and MCP layer can add new tools, transports, and clients without re-architecting compute. The OSS-vs-Studio split falls out of the same logic: local filesystem is the right transport for one developer, MCP is the right transport for a team.

## See Also

- [Knowledge Base](knowledge-base.md) — the compiled content delivered through this layer
- [Agents quickstart](../getting-started/agents.md) — installation walkthrough
- [Knowledge Base guide](../guide/knowledge-base.md) — operational details, regeneration, browsing
- [Studio overview](../studio/index.md) — when and how Data Memory moves from OSS-local to centralized
