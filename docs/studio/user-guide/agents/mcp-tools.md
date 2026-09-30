---
title: MCP Tools
---

# MCP Tools

Tools exposed by the Studio [MCP server](mcp.md). The last column is the token scope and
role a tool needs; OAuth sessions get both scopes.

| Tool | What it does | Access |
|---|---|---|
| `search_knowledge_base` | Full-text search over the generated knowledge of datasets and storages | Datasets: read |
| `list_datasets` | List datasets that have a complete version, filtered by namespace and project | Datasets: read |
| `get_dataset` | Return a dataset's schema, versions, knowledge page, and up to 5 preview rows | Datasets: read |
| `get_row_content` | Read the file a dataset row points to, as an image or text | Datasets: read |
| `list_storages` | List indexed storage roots | Datasets: read |
| `get_storage` | Return an indexed storage's listing statistics and knowledge page | Datasets: read |
| `get_knowledge_base` | Return a Markdown index of enriched datasets and storages | Datasets: read |
| `enrich_dataset` | Generate the knowledge page for one dataset, using the team's provider and budget | Datasets: write |
| `enrich_all` | Generate pages for every dataset and storage that lacks a current one | Datasets: admin |
| `get_datachain_guide` | Return the DataChain SDK rules a `run_job` script must follow | Jobs: read |
| `list_clusters` | List the team's compute clusters, default first | Jobs: read |
| `run_job` | Submit a Python DataChain script as a Studio job | Jobs: write |
| `get_job` | Return a job's settings, status, progress, errors, and produced dataset versions | Jobs: read |
| `get_job_logs` | Return a job's most recent log lines | Jobs: read |
| `cancel_job` | Cancel an active job | Jobs: write |

## Limits

- `get_dataset` previews at most 5 rows and truncates each cell to 200 characters.
- `get_row_content` reads files from `s3://`, `gs://`, `az://`, and `hf://` only, up to
  4 MiB; JPG, PNG, GIF, and WebP come back as images, UTF-8 files as text.
- `search_knowledge_base` returns at most 50 matches per call; `list_datasets` at most
  200 datasets per page.
- Requests are rate limited per user and per IP address; a limited request gets HTTP
  429. Wait and retry.
- Knowledge search and the knowledge index cover only datasets and storages with a
  generated page; `enrich_dataset` and `enrich_all` generate pages using the team's
  provider and budget.

## Example prompts

```prompt
What datasets do we have about product images? Show the schema of the largest one.
```

```prompt
Find storages with audio files and summarize what the knowledge base says about them.
```

The loop that makes an agent and Studio worth combining: a question comes in, no
existing column answers it, the agent runs a pass over the files, saves the result as a
dataset, and the next person who asks gets it from the store.

```prompt
Do we already have embeddings for the oxford-pets images?
```

```prompt
Which pets in s3://dc-readme/oxford-pets-micro/ were photographed outdoors?
```

```prompt
Why did job <job-id> fail? Show the relevant log lines.
```
