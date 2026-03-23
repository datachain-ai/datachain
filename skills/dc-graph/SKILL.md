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
python3 {skill_dir}/scripts/graph.py --db-mtime
```

This prints an ISO-8601 UTC timestamp (e.g. `2024-01-15T10:30:00Z`) representing the last modification time of the `.datachain/db*` files.

Read `.datachain/graph/index.md` (if it exists) and extract the `db_last_updated` field from its YAML frontmatter.

- If the timestamps match → print "Graph is up to date." and stop.
- If `.datachain/graph/index.md` does not exist, or the timestamps differ → proceed to Phase 2.

### Phase 2 — Extract Dataset List

Run:
```
python3 {skill_dir}/scripts/graph.py --list
```

Output is JSON:
```json
{"datasets": [{"name": "...", "version": "...", "num_objects": 123, "status": 4, "namespace": "...", "project": "...", "created_at": "...", "updated_at": "..."}]}
```

**One entry per dataset×version** — a dataset with three versions appears three times with the same `name` but different `version`, `num_objects`, and `updated_at`. Datasets in namespace `system` or project `listing` are already filtered out by the script.

### Phase 3 — Diff Against Existing Graph

Group the `--list` entries by `name`. For each unique dataset name:
1. Derive the filename: replace `.` and `/` with `_`, lowercase → `{name_slug}.md`. **Never include a version in the filename.**
2. Identify `latest_version` as the highest version among all entries for this dataset.
3. Read `.datachain/graph/datasets/{name_slug}.md` (if it exists).
4. Extract `latest_version` and `num_objects` from its YAML frontmatter.
5. Collect the set of versions already present as `### version` sub-sections in the Version History.
6. Mark the dataset as **stale** if any of:
   - The file does not exist
   - `latest_version` or `num_objects` (for the latest version) differs from the file's frontmatter
   - Any version from `--list` is missing from the Version History sub-sections

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

**For each stale dataset**, run one script call **per version that is missing from the Version History**, plus always one call for `latest_version` (even if it already has an entry, to refresh the top-level metadata). Run calls oldest-version-first so `changes` fields chain correctly.

```
python3 {skill_dir}/scripts/graph.py --dataset <name>@<version>
```

Run this once per version that needs a new or updated sub-section. **Do not call for versions already present verbatim in the history unless they are the latest.**

Output:
```json
{
  "name": "...",
  "schema": {
    "file": {"type": "File", "fields": null},
    "meta": {"type": "ImageMeta", "fields": {"width": "int", "height": "int", "mode": "str"}},
    "ratio": {"type": "str", "fields": null}
  },
  "preview": {
    "columns": ["file.path", "file.size", "meta.width", "meta.height", "ratio"],
    "rows": [["dogs-and-cats/cat.1.jpg", 16880, 123, 456, "1:1"]]
  },
  "query_script": "dc.read_csv(...).save('image_ratio')",
  "changes": {
    "previous_version": "1.0.0",
    "script_changed": true,
    "previous_script": "dc.read_csv(...).save('image_ratio')",
    "deps_added": [],
    "deps_removed": [],
    "deps_updated": [
      {"name": "raw_images", "version_from": "1.0.0", "version_to": "1.0.1",
       "script_changed": true, "previous_script": "...", "current_script": "..."}
    ]
  },
  "dependencies": [
    {
      "name": "...", "version": "...", "type": "...",
      "dependencies": [{"name": "...", "version": "...", "type": "..."}]
    }
  ]
}
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

```
file: File          ← no fields (well-known, see dc-core)
meta: ImageMeta
  width: int
  height: int
  mode: str
ratio: str
```

Render schema: for each signal, print `{signal}: {type}`, then if `fields` is not null indent each field as `  {name}: {type}`.

## Preview

| file.path | file.size | meta.width | meta.height | ratio |
|-----------|-----------|------------|-------------|-------|
| ...       | ...       | ...        | ...         | ...   |

Render preview: use `columns` as table headers, `rows` as table rows.

## Query Script

```python
dc.read_csv(...).save('image_ratio')
```

Omit the `## Query Script` section if `query_script` is null or empty.

## Changes

Omit this section if `changes` is null (first version of the dataset).

When present, write a human-readable summary relative to `previous_version`:
- If `script_changed` is true: describe what changed between `previous_script` and the current `query_script`
- For each entry in `deps_added`: note the new dependency
- For each entry in `deps_removed`: note the removed dependency
- For each entry in `deps_updated`: note the version bump, and if `script_changed` is true describe the script difference between `previous_script` and `current_script`

Example:

```
Compared to 1.0.0:
- Query script: switched from CSV reader to Parquet reader
- `raw_images` updated 1.0.0 → 1.0.1 (filter condition tightened)
```

## Dependencies

| Dataset | Version | Type |
|---------|---------|------|
| [[dep_name_slug]] | version | type |
```

Omit the `## Dependencies` section if `dependencies` is empty.
For each top-level dependency, list it as a wikilink row. If it has child dependencies, add them as indented sub-rows or a nested list below the table.

## Version History

### 1.0.1 — 2026-03-20 (12400 objects)

Script switched from CSV reader to Parquet reader.
`raw_images` updated 1.0.0 → 1.0.1 (filter condition tightened).

**Query Script:**
```python
dc.read_parquet(...).save('image_ratio')
```

**Dependency changes:**
- `raw_images` bumped 1.0.0 → 1.0.1

### 1.0.0 — 2026-03-10 (11000 objects)

Initial version.

**Query Script:**
```python
dc.read_csv(...).save('image_ratio')
```
```

Each `### {version}` sub-section heading format: `### {version} — {updated_at} ({num_objects} objects)`

Sub-section contents (in order):
1. **Human-explained summary** (plain prose paragraph): derived from the `changes` field — describe what changed vs the previous version. For the first version write "Initial version." If `changes` is null on a non-first version write "No recorded changes."
2. **Query Script** (fenced code block, label `**Query Script:**`): omit if `query_script` is null.
3. **Dependency changes** (bulleted list, label `**Dependency changes:**`): list deps added, removed, or version-bumped. Omit entirely if no dependency changes.

**Version History update rule:**

The goal is **one sub-section per known version, newest first, with no gaps**.

- Collect all versions from the `--list` output for this dataset (there is one entry per version).
- Collect all `### version` sub-sections already in the file (if it exists).
- For every version that has no sub-section yet (including versions older than the current file), run `--dataset name@version` and write a new sub-section from the script output.
- Carry over all existing sub-sections verbatim — do not re-fetch versions already in the file unless they are `latest_version`.
- Always re-run `--dataset name@latest_version` and replace (or write) its sub-section with fresh data.
- Write all sub-sections sorted newest-first.

**Handling leftover versioned files:** If old versioned files (e.g. `flower-large_1_0_0.md`) exist in `.datachain/graph/datasets/`, delete them when rewriting the dataset to the canonical unversioned filename.

All files use Obsidian-compatible wikilinks (`[[name_slug]]`) and YAML frontmatter.

### Phase 5 — Report

Print a summary:
```
Graph updated: <N> datasets total, <M> updated, <K> unchanged.
Output: .datachain/graph/
```

---

## Notes

- The script is standalone — flags: `--db-mtime`, `--list`, `--dataset <name>` or `--dataset <name@version>`.
- `status` field is a raw integer from the DB (4 = complete). Map to human-readable strings in the Markdown if desired: `{1: "pending", 2: "running", 3: "failed", 4: "complete"}`.
- Never read `.datachain/db` directly — always go through `graph.py`.
- If `graph.py` fails (DataChain not installed, no DB), report the error and stop gracefully.
- `dependencies` in `--dataset` output is best-effort: if unavailable it is an empty list — do not error.
- `query_script` in `--dataset` output is best-effort: if unavailable or empty it is null — omit the section.
- `changes` in `--dataset` output is null for the first version and best-effort otherwise — omit the section when null.
