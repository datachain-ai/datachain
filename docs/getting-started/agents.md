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
includes recipes for inventories, media headers, batched inference, video frames,
saved-result queries, updates and interrupted runs.

## Your First Agent Task

Describe the task in your own words:

```text
Inventory our deliveries across S3, GCS and Azure.
Report file counts and bytes by source, folder and extension without downloading contents.
```

| Task | DataChain pattern |
|---|---|
| One-off inventory | `list_storage()` and one aggregation pass |
| Media headers | A typed file's `get_info()` inside `map()`, with `prefetch=0` |
| Batched inference | `BatchMapper` with `setup()`, then `batch_map(..., batch=32)` |
| Video frames | Stream chunks from `VideoFile.sample_frames()` inside a generator |
| Saved results | Inspect datasets and recorded code, then query `read_dataset()` |
| File updates | A compatible pipeline with `read_storage(..., update=True, delta=True)` |
| Interrupted runs | Rerun the script to reuse checkpointed work |

For questions about saved results or provenance, start from recorded datasets and
code. Check external sources when current freshness is part of the request.
Read the skill's installed `SDK.md` for API details and its `SKILL.md` for the
model-change rebuild rule.

A small shell command is sufficient for questions it answers directly. Use DuckDB
or pandas for ready-made tables.

## Optional Knowledge Base

To generate and maintain markdown summaries in `dc-knowledge/`, explicitly install
the `knowledge` skill, which also installs its `core` dependency:

```bash
datachain skill install knowledge --target claude --local
```

Then ask the agent to build a knowledge base for your saved datasets. See the
[Knowledge Base guide](../guide/knowledge-base.md) for generation and browsing.
The default skill does not require a knowledge base.

Installing the default `datachain` skill replaces `core` and `knowledge`
installations from earlier releases. Optional skills explicitly installed with
this version remain installed. To use only the task-oriented skill, remove those explicitly in
that target and scope:

```bash
datachain skill uninstall core,knowledge --target claude --local
```

## Next Steps

- [Concepts](../concepts/index.md): the Dataset DB, datasets and execution
- [Knowledge Base guide](../guide/knowledge-base.md): optional markdown summaries
- [Guides](../guide/index.md): in-depth coverage of each capability
