---
name: datachain-graph
description: Use when you need to discover, understand, or navigate DataChain datasets in the current project. Apply this skill to answer questions about what datasets exist, their schemas, row counts, versions, and relationships. Generates and maintains a Markdown knowledge base at .datachain/graph/ so dataset context is always available to the AI without running queries each time.
triggers:
  - "what datasets exist"
  - "show me the schema"
  - "list datasets"
  - "datachain graph"
  - "update the graph"
  - "refresh dataset docs"
---

You are now loaded with the datachain-graph skill. Your job is to maintain a Markdown knowledge base of all DataChain datasets in this project. Follow the 5-phase flow below exactly.

---

## Resolving {skill_dir}

Before running any script, determine `{skill_dir}` based on the AI tool in use:

| Tool | Local (per-project) | Global (user-wide) |
|------|--------------------|--------------------|
| Claude Code | `.claude/skills/dc-graph` | `~/.claude/skills/dc-graph` |
| Cursor | `.cursor/skills/dc-graph` | `~/.cursor/skills/dc-graph` |
| Codex | `.codex/skills/dc-graph` | `~/.codex/skills/dc-graph` |

Use the local path if the skill directory exists under the project root; otherwise fall back to the global path.

---

## 5-Phase Flow

### Phase 1 — Check Staleness

Run:
```
python3 {skill_dir}/scripts/dc_extract.py --db-mtime
```

This prints an ISO-8601 UTC timestamp (e.g. `2024-01-15T10:30:00Z`) representing the last modification time of the `.datachain/db*` files.

Read `.datachain/graph/index.md` (if it exists) and extract the `db_last_updated` field from its YAML frontmatter.

- If the timestamps match → print "Graph is up to date." and stop.
- If `.datachain/graph/index.md` does not exist, or the timestamps differ → proceed to Phase 2.

### Phase 2 — Extract Dataset List

Run:
```
python3 {skill_dir}/scripts/dc_extract.py --list
```

Output is JSON:
```json
{"datasets": [{"name": "...", "version": "...", "num_objects": 123, "status": 4, "namespace": "...", "project": "...", "created_at": "...", "updated_at": "..."}]}
```

Datasets in namespace `system` or project `listing` are already filtered out by the script.

### Phase 3 — Diff Against Existing Graph

For each dataset returned in Phase 2:
1. Derive the filename: replace `.` and `/` with `_`, lowercase → `{name_slug}.md`
2. Read `.datachain/graph/datasets/{name_slug}.md` (if it exists)
3. Extract `latest_version` and `num_objects` from its YAML frontmatter
4. Mark the dataset as **stale** if either field differs from the `--list` output, or if the file does not exist

### Phase 4 — Write

**Always rewrite `index.md`:**

Create `.datachain/graph/index.md` with YAML frontmatter:
```yaml
---
db_last_updated: <value from Phase 1>
generated_at: <current UTC timestamp>
dataset_count: <N>
---
```

Body: a table listing all datasets with columns: Name, Version, Objects, Updated.
Each name is a wikilink: `[[name_slug]]`.

**For each stale dataset**, run:
```
python3 {skill_dir}/scripts/dc_extract.py --dataset <name>
```

Output:
```json
{"name": "...", "schema": {"col": "type", ...}, "preview": [{...}, ...]}
```

Write `.datachain/graph/datasets/{name_slug}.md`:
```markdown
---
name: <name>
latest_version: <version>
num_objects: <num_objects>
updated_at: <updated_at>
---

# <name>

## Schema

| Column | Type |
|--------|------|
| col    | type |

## Preview

| col1 | col2 | ... |
|------|------|-----|
| val  | val  | ... |
```

All files use Obsidian-compatible wikilinks (`[[name_slug]]`) and YAML frontmatter.

### Phase 5 — Report

Print a summary:
```
Graph updated: <N> datasets total, <M> updated, <K> unchanged.
Output: .datachain/graph/
```

---

## Notes

- The script is standalone — no CLI args beyond the three flags.
- `status` field is a raw integer from the DB (4 = complete). Map to human-readable strings in the Markdown if desired: `{1: "pending", 2: "running", 3: "failed", 4: "complete"}`.
- Never read `.datachain/db` directly — always go through `dc_extract.py`.
- If `dc_extract.py` fails (DataChain not installed, no DB), report the error and stop gracefully.
