---
title: Knowledge Base
---

# Knowledge Base

Studio's Knowledge Base adds generated descriptions to saved datasets and indexed storages. Agents read and search these descriptions through [Studio MCP](../studio/user-guide/agents/mcp.md); people browse them in Studio.

See the [Studio Knowledge Base guide](../studio/user-guide/agents/knowledge-base.md) to configure a provider, generate descriptions and manage access.

## Local datasets

The `datachain` skill inspects saved datasets directly with `dc.datasets()`, `dc.describe()` and `dc.read_dataset()`. It does not generate or maintain a local markdown knowledge base.

```bash
pip install datachain
datachain skill install --target claude --local
```

Ask the agent to find existing datasets and inspect their schemas and producing code before starting new work. See [Getting Started with Agents](../getting-started/agents.md).
