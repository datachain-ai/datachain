---
title: Skill and MCP
---

# Skill and MCP

The **skill** ships with `pip install datachain` and teaches agents to inspect saved results, process files and save new datasets. It guides local SDK workflows and, when connected, shared team workflows through **Studio MCP**. Local use needs no Studio account. Studio is optional and currently invite-only.

## Local SDK and Studio MCP

| | Local SDK (OSS) | MCP (Studio) |
|---|---|---|
| Ships with | `pip install datachain` | Studio |
| Reads | Local `.datachain/db` (SQLite); optional `dc-knowledge/` | Centralized data warehouse + shared Knowledge Base |
| Compute | Local machine, parallel threads | Jobs with or without a cluster; clusters scale across dozens or hundreds of CPU/GPU machines |
| Scale | One project's SQLite registry | Billions of typed records (LAION-scale corpora) |
| Access control | Filesystem only | Teams, namespaces, per-team ACLs |
| Audience | Local data processing and saved results | Teams; large datasets; headless agent services |

## The Skill (OSS)

```bash
pip install datachain
datachain skill install --target claude     # also: cursor, codex
```

The Skill is the agent-side instruction package. Installing it places files into the agent harness's directory (for Claude Code: `~/.claude/skills/datachain/`). The default skill teaches the agent to inspect the Dataset DB, process data files, save expensive results and query them later.

For a markdown Knowledge Base, explicitly install the optional knowledge skill and its core dependency:

```bash
datachain skill install knowledge --target claude
```

That workflow reads two local sources:

- **`dc-knowledge/`** holds compiled markdown pages: schema, lineage, previews, links. The agent reads this first.
- **`.datachain/db`** is the SQLite dataset registry: every named, versioned dataset and its provenance. The agent queries this when the Knowledge Base summary is not enough.

Compute runs locally. `map()`, `gen()`, and `agg()` execute on the developer's machine through the in-process [Compute Engine](compute-engine.md) with parallel threads and async prefetch. Single-machine reach.

Local datasets remain available across sessions. For a shared team catalog, see Studio MCP below.

## MCP (Studio)

Studio provides shared datasets and jobs through MCP, the Model Context Protocol supported by Claude Code, Cursor and Codex. The skill guides the agent to inspect shared results and relevant jobs before computing, then choose an available execution option. Shared results and job tracking work without a cluster; clusters are available for larger workloads. Generated knowledge descriptions help discover existing work; inspect dataset versions and producing code to verify it. See [MCP Server](../studio/user-guide/agents/mcp.md) for connection instructions for users with Studio access.

Studio adds four capabilities the OSS path does not have:

- **Distributed compute.** Optional BYOC clusters run data processing across dozens or hundreds of CPU and GPU machines using the same DataChain code.
- **Billion-record registry.** A full-scale data warehouse replaces SQLite. Filter, join, group_by, and similarity search run at warehouse speed over billions of typed records (LAION-scale corpora).
- **Teams and access control.** Users belong to teams; datasets sit in namespaces; per-team ACLs determine who reads and writes what. The MCP surface enforces the same model as the Studio UI.
- **BYOC.** Compute clusters and storage stay in your cloud account. Studio orchestrates without holding the data.

## When You Need Each

| Scenario | Install |
|---|---|
| Solo developer | Skill |
| Team needing access control across users and projects | Skill + MCP |
| Billion-row datasets or distributed compute across many machines | Skill + MCP |
| Headless production agent service | MCP |

## See Also

- [Knowledge Base](knowledge-base.md): the compiled content delivered through this layer
- [Agents quickstart](../getting-started/agents.md): installation walkthrough
- [Studio overview](../studio/index.md): managed deployment, BYOC, cluster setup
