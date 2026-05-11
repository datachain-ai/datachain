---
title: Skill and MCP
---

# Skill and MCP

Skill and MCP are the two ways an agent reaches DataChain. The **Skill** ships with `pip install datachain`, lives on the developer's machine, and uses local files for reads and a single machine for compute. **MCP** is the Studio surface: a hosted endpoint backed by a centralized registry and attached BYOC clusters that scale compute across thousands of machines. Same chain code on either side; the difference is reach.

## Skill vs MCP

| | Skill (OSS) | MCP (Studio) |
|---|---|---|
| Ships with | `pip install datachain` | Studio |
| Reads | Local `dc-knowledge/` + `.datachain/db` (SQLite) | Centralized data warehouse + shared Knowledge Base |
| Compute | Local machine, parallel threads | Attached BYOC clusters: dozens to thousands of CPU/GPU machines |
| Scale | One project's SQLite registry | Billions of typed records (LAION-scale corpora) |
| Access control | Filesystem only | Teams, namespaces, per-team ACLs |
| Audience | Solo dev; small teams syncing via Git | Teams; large datasets; headless agent services |

## The Skill (OSS)

```bash
pip install datachain
datachain skill install --target claude     # also: cursor, codex
```

The Skill is the agent-side instruction package. Installing it places files into the agent harness's directory (for Claude Code: `~/.claude/skills/datachain/`). The Skill teaches the agent where the Knowledge Base is, how to introspect Data Memory, when to materialize a new dataset, and how to write idiomatic chains.

Reads target two local sources:

- **`dc-knowledge/`** holds compiled markdown pages: schema, lineage, previews, links. The agent reads this first.
- **`.datachain/db`** is the SQLite dataset registry: every named, versioned dataset and its provenance. The agent queries this when the Knowledge Base summary is not enough.

Compute runs locally. `map()`, `gen()`, and `agg()` execute on the developer's machine through the in-process [Compute Engine](compute-engine.md) with parallel threads and async prefetch. Single-machine reach.

A small team can extend the Skill beyond solo work by **synchronizing both folders** (`dc-knowledge/` and `.datachain/db`) through Git or any shared filesystem. That gives the team a shared read surface without standing up a service. The Skill is enough until the registry stops fitting in SQLite, the team needs row-level access control, or compute outgrows one machine.

## MCP (Studio)

When the team outgrows file sync, Data Memory moves into Studio and the agent reads through MCP, the open Model Context Protocol with native client support in Claude Code, Cursor, and Codex. The Skill stays installed; it routes registry queries, schema introspection, and Data Memory reads through the MCP endpoint instead of the local filesystem. To set this up for your team, get in touch at [datachain.ai](https://datachain.ai).

Studio adds four capabilities the OSS path does not have:

- **Distributed compute.** `map()` / `gen()` / `agg()` dispatch to attached BYOC clusters: dozens to thousands of CPU and GPU machines. Same chain code; the fleet runs the work.
- **Billion-record registry.** A full-scale data warehouse replaces SQLite. Filter, join, group_by, and similarity search run at warehouse speed over billions of typed records (LAION-scale corpora).
- **Teams and access control.** Users belong to teams; datasets sit in namespaces; per-team ACLs determine who reads and writes what. The MCP surface enforces the same model as the Studio UI.
- **BYOC.** Compute clusters and storage stay in your cloud account. Studio orchestrates without holding the data.

## When You Need Each

| Scenario | Install |
|---|---|
| Solo developer | Skill |
| Small team syncing `dc-knowledge/` + `.datachain/db` via Git | Skill |
| Team needing access control across users and projects | Skill + MCP |
| Billion-row datasets or distributed compute across many machines | Skill + MCP |
| Headless production agent service | MCP |

## See Also

- [Knowledge Base](knowledge-base.md): the compiled content delivered through this layer
- [Agents quickstart](../getting-started/agents.md): installation walkthrough
- [Studio overview](../studio/index.md): managed deployment, BYOC, cluster setup
