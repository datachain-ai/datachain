---
title: MCP Server
---

# MCP Server

Studio exposes your team's datasets, storages, knowledge base, and jobs over
[MCP (Model Context Protocol)](https://modelcontextprotocol.io). Any MCP client can
connect. The agent acts with your Studio permissions and can run DataChain jobs on the
team's clusters. See [Tools](mcp-tools.md) for what it can do.

## Prerequisites

1. **Team settings → AI features → Enable AI** is on. No LLM provider is needed.
   Knowledge search and the knowledge index require generated pages; the enrichment
   tools generate them using the team's provider and budget. See
   [Knowledge Base](knowledge-base.md).
2. Copy the MCP URL from **Team settings → AI features → MCP server**. It looks like
   `https://studio.datachain.ai/api/mcp/<team>`.

## Connect

=== "Claude Code"

    ```bash
    claude mcp add --transport http --scope user studio https://studio.datachain.ai/api/mcp/<team>
    ```

    Start Claude Code, run `/mcp`, select **studio**, choose **Authenticate**, and
    approve access in the browser. To share the setup through a repository, commit
    `.mcp.json` at its root; it holds no credentials.

    ```json
    {
      "mcpServers": {
        "studio": {
          "type": "http",
          "url": "https://studio.datachain.ai/api/mcp/<team>"
        }
      }
    }
    ```

=== "Claude Desktop and claude.ai"

    1. **Settings → Connectors → Add custom connector**.
    2. Name: `Studio`. URL: the MCP URL. Leave the advanced fields empty.
    3. Click **Add**, then **Connect**, and approve access in the browser.
    4. In a chat, open the tools menu (the **+** button) and make sure **Studio** is
       enabled.

    On Team and Enterprise plans an organization owner adds the connector once; each
    member then connects it. Custom connectors require a paid plan.

=== "Codex"

    ```bash
    codex mcp add studio --url https://studio.datachain.ai/api/mcp/<team>
    codex mcp login studio
    ```

    Approve access in the browser. Codex CLI, the Codex IDE extension, and the ChatGPT
    desktop app share this configuration.

=== "Cursor"

    Add to `~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` in a repository:

    ```json
    {
      "mcpServers": {
        "studio": {
          "url": "https://studio.datachain.ai/api/mcp/<team>"
        }
      }
    }
    ```

    Open **Settings → MCP** and click **Needs login** next to **studio**.

=== "VS Code"

    1. Open the Command Palette and run **MCP: Add Server**.
    2. Select **HTTP**, enter the MCP URL and the name `studio`, and choose **Global** or
       **Workspace**.
    3. Run **MCP: List Servers**, select **studio**, click **Start Server**, and approve
       access in the browser.

=== "Other"

    Follow your client's instructions for a remote server and use the MCP URL. The server
    speaks Streamable HTTP and supports OAuth with dynamic client registration; request
    the scopes `DATASETS JOBS`. If the client cannot sign in through a browser, use an
    [access token](#access-tokens).

Then ask the agent:

```prompt
What datasets does my team have in Studio?
```

A connected client calls the `list_datasets` tool and answers with dataset names from
Studio. If no tool call appears in the transcript, the client did not use the server.

## Authentication

Interactive clients sign in with OAuth: the client opens the Studio sign-in page, you
approve once, and the client keeps the session.

### Access tokens

For CI, scripts, or agents without a browser, create a token under **Personal settings
→ Tokens** with the **DATASETS** and **JOBS** scopes, the token's **Write** role, and an
expiration. Keep the token in an environment variable and reference it from the config.

=== "Claude Code"

    `.mcp.json` at the project root; `${STUDIO_TOKEN}` is expanded from the environment.

    ```json
    {
      "mcpServers": {
        "studio": {
          "type": "http",
          "url": "https://studio.datachain.ai/api/mcp/<team>",
          "headers": { "Authorization": "Bearer ${STUDIO_TOKEN}" }
        }
      }
    }
    ```

=== "Codex"

    ```bash
    export STUDIO_TOKEN=<token>
    codex mcp add studio --url https://studio.datachain.ai/api/mcp/<team> --bearer-token-env-var STUDIO_TOKEN
    ```

=== "Cursor"

    ```json
    {
      "mcpServers": {
        "studio": {
          "url": "https://studio.datachain.ai/api/mcp/<team>",
          "headers": { "Authorization": "Bearer ${env:STUDIO_TOKEN}" }
        }
      }
    }
    ```

## Permissions and safety

- The agent sees what your Studio account can see on that team, further limited by the
  token's scopes. See [Security & Permissions](../teams/permissions.md).
- `run_job` executes code on the team's compute cluster. Keep your client's per-call
  confirmation on for `run_job`, `cancel_job`, and `enrich_all`.
- To revoke access, delete the token under **Personal settings → Tokens**. Turning off
  **Enable AI** disconnects every client of the team.

## Troubleshooting

- **401, or the client keeps asking to authenticate**: sign in again from the client.
- **403 "No access to team"**: your account is not a member of the team in the URL, or
  the token was created for another team.
- **403 "AI features are disabled for this team"**: turn on **Enable AI**.
