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

You are now loaded with the datachain-graph skill. Maintain a Markdown knowledge base of all DataChain datasets at `.datachain/graph/`. Follow the 3-step flow below exactly.

---

## Resolving {skill_dir}

| Tool | Local (per-project) | Global (user-wide) |
|------|--------------------|--------------------|
| Claude Code | `.claude/skills/dc-graph` | `~/.claude/skills/dc-graph` |
| Cursor | `.cursor/skills/dc-graph` | `~/.cursor/skills/dc-graph` |
| Codex | `.codex/skills/dc-graph` | `~/.codex/skills/dc-graph` |

Use the local path if the skill directory exists under the project root; otherwise fall back to the global path.

---

## Step 1 — Check & Plan

```
python3 {skill_dir}/scripts/graph.py --plan [--studio]
```

- Do **NOT** add `--studio` unless the user's message contains an explicit request to update from Studio (e.g. "update from Studio", "refresh Studio datasets", "include Studio datasets"). An empty datasets list or `"studio_available": true` in the plan output does **not** justify adding `--studio`.
- If `"up_to_date": true` → print "Graph is up to date." and stop.
- If the output contains a `"warnings"` key → report those warnings to the user after the update summary.
- If `graph.py` fails → report the error and stop gracefully.

The output tells you exactly what to do: `datasets[].status` is `"ok"` (skip), `"stale"` (update), or `"new"` (create). `file_path` is the exact path to write, relative to `.datachain/graph/`. `versions_to_fetch` lists which versions need a new or refreshed sub-section.

---

## Step 2 — Fetch & Write

**For each dataset where `status != "ok"`**, run one call for all its versions:

```
python3 {skill_dir}/scripts/graph.py --dataset-all <name>
```

Pass `<name>` exactly as it appears in the plan output.

Write (or overwrite) `.datachain/graph/{file_path}` using the format below. For versions already in `file_versions` and not in `versions_to_fetch`, carry over their existing `### version` sub-sections verbatim.

### Dataset file format

```markdown
---
name: <name>
latest_version: <latest_version>
num_objects: <num_objects of latest version>
updated_at: <updated_at of latest version>
---

# <name>

## Schema

<signal>: <type>
  <field>: <type>     ← only if fields is not null
...

## Preview

| col1 | col2 | ... |
|------|------|-----|
| ...  | ...  | ... |

## Query Script

```python
<query_script>
```

## Changes

<human-readable summary of changes vs previous version>

## Dependencies

| Dataset | Version | Type |
|---------|---------|------|
| [[dep_slug]] | version | type |

## Version History

### <version> — <updated_at date> (<num_objects> objects)

<human-explained summary: what changed vs previous version, or "Initial version.">

**Previous script** (if `changes.script_changed` is true — show old code then new code):
```python
<changes.previous_script>
```

**Query Script:**
```python
<query_script>
```

**Dependencies:**
| Dataset | Version | Type |
|---------|---------|------|
| dep | version | type |

**Dependency changes:**
- added: `dep` v1.0.0
- removed: `dep` v1.0.0
- updated: `dep` 1.0.0 → 1.0.1
```

**Section rules:**
- Omit `## Schema` and `## Preview` if both are null (Studio dataset with no local access).
- Omit `## Query Script` if `query_script` is null.
- Omit `## Changes` if `changes` is null (first version).
- Omit `## Dependencies` (top-level section) if the latest version's `dependencies` array is empty. Populate it from `versions[-1].dependencies`.
- In `## Changes`: describe in plain prose. If `script_changed` is true, show what changed between `previous_script` and `query_script` in prose. List added/removed/updated deps.
- In each version history sub-section:
  - Omit `**Previous script**` block if `changes` is null or `changes.script_changed` is false.
  - Omit `**Query Script**` if `query_script` is null.
  - Omit `**Dependencies**` table if `dependencies` is empty for that version.
  - Omit `**Dependency changes**` if `deps_added`, `deps_removed`, and `deps_updated` are all empty.
- Version History: newest-first. Write sub-sections for `versions_to_fetch` only; carry over existing sub-sections verbatim for versions in `file_versions` but not `versions_to_fetch`.
- If `history_complete` is false, append at the end of Version History: `_(Older versions exist but were not included in this update.)_`
- Wikilinks use `[[path|display]]` syntax — path uses `/`-separated slugs, display uses the dot-separated dataset name. Local datasets: `[[name_slug|name]]`. Studio datasets: `[[namespace/project/bare_name_slug|namespace.project.bare_name]]`.

**Always rewrite `index.md`** (`.datachain/graph/index.md`):

```markdown
---
db_last_updated: <db_last_updated from plan>   ← omit if not present in plan output
generated_at: <current UTC timestamp>
local_dataset_count: <count of local datasets>
studio_dataset_count: <count of studio datasets>   ← omit if 0
---

| Name | Source | Version | Objects | Updated |
|------|--------|---------|---------|---------|
| [[path\|display]] | local/studio | version | num_objects | updated_at date |
```

Include all datasets from the plan (all statuses). `status` integer from the DB maps to: `{1: "pending", 2: "running", 3: "failed", 4: "complete"}` — use human-readable strings if you display it.

---

## Step 3 — Report

```
Graph updated: <N> datasets total, <M> updated, <K> unchanged.
```
