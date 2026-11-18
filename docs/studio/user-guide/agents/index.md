# Agents

An agent connected to Studio works with the same datasets, storages, and jobs as the
people on the team, through MCP instead of the UI or the SDK. It answers from the team's
data, sees what has already been computed, and builds on it instead of redoing it.
The pieces below are configured in **Team settings → AI features**.

- **[Knowledge Base](knowledge-base.md)** - How an agent knows what the team already
  has: an enriched description of every dataset and storage.
- **[MCP Server](mcp.md)** - Connect Claude Code, Claude Desktop, Codex, Cursor, and other
  MCP clients to the team's data, knowledge base, and jobs.
- **[MCP Tools](mcp-tools.md)** - What a connected agent can do: the tool list, limits,
  and example prompts.

## Getting started

1. [Turn on agents and add a provider](knowledge-base.md#set-up).
2. [Generate the knowledge base](knowledge-base.md#generate-pages).
3. [Install the skill and connect an agent](mcp.md#install-the-skill), then try a prompt.
