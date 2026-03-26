---
name: datachain-graph
description: Use when you need to discover, understand, or navigate DataChain datasets in the current project. Apply this skill to answer questions about what datasets exist, their schemas, row counts, versions, and relationships. Generates and maintains a knowledge base at .datachain/graph/ (JSON data + AI-enriched markdown) so dataset context is always available to the AI without running queries each time.
triggers:
  - "what datasets exist"
  - "show me the schema"
  - "list datasets"
  - "datachain graph"
  - "update the graph"
  - "refresh dataset docs"
---

You are now loaded with the datachain-graph skill. Maintain a dataset knowledge base at `.datachain/graph/`. Each dataset has a `.json` file (structured data, source of truth) and a `.md` file (AI-generated human-readable summary). Follow the 4-step flow below exactly.

---

## Resolving {skill_dir}

| Tool | Local (per-project) | Global (user-wide) |
|------|--------------------|--------------------|
| Claude Code | `.claude/skills/dc-graph` | `~/.claude/skills/dc-graph` |
| Cursor | `.cursor/skills/dc-graph` | `~/.cursor/skills/dc-graph` |
| Codex | `.codex/skills/dc-graph` | `~/.codex/skills/dc-graph` |

Use the local path if the skill directory exists under the project root; otherwise fall back to the global path.

---

## Step 1 — Sync

Sync the dataset list from DB, diff against the existing graph, save the plan to disk, and update the index.

```bash
python3 {skill_dir}/scripts/plan.py [--studio] > .datachain/graph/.plan.json
```

- Do **NOT** add `--studio` unless the user explicitly requests it (e.g. "update from Studio", "refresh Studio datasets").
- If the output contains `"up_to_date": true` → print "Graph is up to date." and stop.
- If the output contains `"warnings"` → report them to the user after the update summary.
- If the script fails → report the error and stop gracefully.

Then update the index:

```bash
python3 {skill_dir}/scripts/render_index.py --plan .datachain/graph/.plan.json --output .datachain/graph/index.md
```

Review `.plan.json` to understand what's new. Datasets with `status` of `"new"` or `"stale"` need processing in Step 2. Datasets with `"ok"` are skipped entirely.

---

## Step 2 — Save Data

For each dataset where `status != "ok"` in the plan, fetch its data and save as JSON:

```bash
python3 {skill_dir}/scripts/dataset_all.py <name> \
  --plan .datachain/graph/.plan.json \
  --output .datachain/graph/<file_path>.json
```

- `<name>` and `<file_path>` come from the plan's `datasets[]` entries. `file_path` is extensionless — append `.json` here.
- With `--plan` and `--output`, `dataset_all.py` merges new version data with the existing `.json` file (if any), preserving versions not being re-fetched.
- Without `--plan`/`--output`, it prints JSON to stdout (useful for debugging).
- Do **not** modify the output — it is deterministic and complete.
- If the output contains `"warnings"` → collect and report them in Step 4.

---

## Step 3 — Enrich

For each dataset processed in Step 2, generate a human-readable markdown summary from the JSON data.

1. Read the enrichment prompt at `{skill_dir}/prompts/enrich.md`.
2. For each dataset, read `.datachain/graph/<file_path>.json`.
3. Following the prompt instructions, write `.datachain/graph/<file_path>.md`.

The prompt defines the exact markdown structure: dataset description, stats, schema, preview, and version history with AI-optimized summaries. Skip this step only if the user requests raw output only.

---

## Step 4 — Report

```
Graph updated: <N> datasets total, <M> updated, <K> unchanged.
```

Include any warnings collected from Steps 1-2.
