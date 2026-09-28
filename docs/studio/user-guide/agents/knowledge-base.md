---
title: Knowledge Base
---

# Knowledge Base

The Knowledge Base is how an agent knows what the team already has and what has
already been computed, so it builds on that instead of redoing it. Every dataset and
indexed storage in the team gets an enriched description: what the data contains, its
schema, sample rows, version history, and the code that produced it. Agents read it
through the [MCP server](mcp.md); people read it in the Studio UI.

Producing a description is called *enrichment*. It runs when someone asks for it and
uses an LLM provider account that the team supplies. For the concept behind it, see
[Knowledge Base](../../../concepts/knowledge-base.md).

## Set up

In **Team settings → AI features** (Admin role):

1. **Add provider**: choose Anthropic or OpenAI, paste the API key, pick a default
   model, and check **Set as active**. Pricing is filled in for known models and is
   required before enrichment runs.
2. Turn on **Enable AI for this team**.
3. Check the **Monthly budget**. Enrichment stops when the month's estimated spend
   reaches it; the default is $200.

Only enrichment uses the provider. Browsing data and running jobs through MCP do not.

### What leaves Studio

For each page, Studio sends the provider one snapshot of the dataset: its description
and attributes, schema, up to five preview rows, and for each version the query script,
dependencies, and what changed since the previous version. For a storage it sends the
listing totals. Nothing else goes to the provider.

## Generate pages

- **One dataset**: open the dataset's menu in the sidebar and choose **Dataset
  knowledge**, then click **Enrich**. The tab shows the page, its status, and when it
  was last enriched; **Copy** copies it as Markdown.
- **One storage**: open the storage's menu in the **Storages** section and choose
  **Storage knowledge**.
- **The whole team**: in **Team settings → AI features → Knowledge base**, click
  **Update knowledge base** (Admin role). It enriches every dataset and storage that has
  no page or whose page is outdated, shows progress, and can be canceled.

!!! note
    Currently pages are refreshed manually: **Re-enrich** on a dataset, or
    **Update knowledge base** for the team. When a new dataset version completes, the
    coverage bar in team settings counts its page as outdated. Automatic refresh on new
    dataset versions is coming.

A page shows one of these statuses: `pending`, `collecting`, and `enriching` while it
is being generated; `ready` when stored; `stale` when the budget was reached or a step
timed out; `failed` with the reason. A previous page stays readable after a failure.

## Permissions

Roles and grants are explained in [Security & Permissions](../teams/permissions.md).
Admins can do everything below without grants.

| Action | Requires |
|---|---|
| Read a dataset's page | Viewer or Editor role and a `read` grant on the dataset |
| Enrich one dataset | Editor role and a `write` grant on the dataset |
| Read or enrich a storage's page | Viewer or Editor role to read, Editor role to enrich |
| Manage providers, the **Enable AI** toggle, the budget, and team-wide updates | Admin role |

## Troubleshooting

- **Provider credentials missing or invalid**: set an active provider with a working
  key under **Team settings → AI features**.
- **Budget message, status `stale`**: raise the monthly budget or wait for the next
  month.
- **Nothing to enrich yet**: the dataset has no complete version. Wait for its job to
  finish.
- **Any other failure**: check the error shown in the tab and click **Retry**. If it
  persists, contact your team admin.
