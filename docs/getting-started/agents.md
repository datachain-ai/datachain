---
title: Getting Started with Agents
---

# Getting Started with Agents

Install DataChain and its coding-agent skill to process data files and reuse saved
results across sessions.

## Installation

=== "pip"

    ```bash
    pip install datachain
    ```

=== "uv"

    ```bash
    uv add datachain
    ```

Install the skill for the current project, then start a new agent session:

```bash
datachain skill install --target claude --local
# Other targets: cursor, codex, copilot, pi
```

Omit `--local` to install for your user account. The default `datachain` skill
includes recipes for inventories, media metadata, batched inference, video and recording readers,
saved-result queries, updates and interrupted runs.

For shared team datasets and compute, the skill can also guide agents using
[Studio MCP](../studio/user-guide/agents/mcp.md). Studio is optional and currently invite-only;
local workflows need no Studio account.

## Your First Agent Task

Describe the task in your own words:

```text
Inventory our deliveries across S3, GCS and Azure.
Report file counts and bytes by source, folder and extension without downloading contents.
```

| Task | DataChain pattern |
|---|---|
| One-off inventory | `list_storage()` and one aggregation pass |
| Media metadata | A typed file's `get_info()` inside `map()`, with `prefetch=0` |
| Batched inference | A function with `setup()` and `batch_map(..., batch=32)` |
| Video and recordings | Use `VideoFile.get_frames` or a Python MCAP reader in map/gen; retain source files and timestamps |
| Saved results | Inspect datasets and recorded code, then query `read_dataset()` |
| File updates | A compatible pipeline with `read_storage(..., update=True, delta=True)` |
| Interrupted runs | Rerun the script to reuse checkpointed work |

For questions about saved results or provenance, start from recorded datasets and
code. Check external sources when current freshness is part of the request.
Read the skill's installed `SDK.md` for API details and its `SKILL.md` for the
model-change rebuild rule.

A small shell command is sufficient for questions it answers directly. Use DuckDB
or pandas for ready-made tables.

## Knowledge Base

[Studio MCP](../studio/user-guide/agents/mcp.md) provides shared dataset descriptions and knowledge search. Locally, agents inspect saved datasets with `dc.datasets()`, `dc.describe()` and `dc.read_dataset()`.

Installing `datachain` replaces the retired `core`, `knowledge` and `jobs` skills in the selected target and scope, including their companion instructions. Existing datasets and `dc-knowledge/` files are preserved.

## Next Steps

- [Concepts](../concepts/index.md): the Dataset DB, datasets and execution
- [Knowledge Base guide](../guide/knowledge-base.md): shared dataset descriptions
- [Guides](../guide/index.md): in-depth coverage of each capability
