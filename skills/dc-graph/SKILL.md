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

This prints either:
- An ISO-8601 UTC timestamp (e.g. `2024-01-15T10:30:00Z`) — local DB mtime
- The string `studio` — no local DB, but Studio is authenticated; staleness is determined per-dataset in Phase 3

**If the output is `studio`:** skip the timestamp comparison entirely and always proceed to Phase 2.

**Otherwise:** read `.datachain/graph/index.md` (if it exists) and extract the `db_last_updated` field from its YAML frontmatter.
- If the timestamps match → print "Graph is up to date." and stop.
- If `.datachain/graph/index.md` does not exist, or the timestamps differ → proceed to Phase 2.

### Phase 2 — Extract Dataset List

Run:
```
python3 {skill_dir}/scripts/graph.py --list
```

This returns **local datasets only**. Studio datasets are **not** included unless the user explicitly requests a Studio sync (e.g. "sync Studio datasets", "update graph from Studio"). In that case, also run:
```
python3 {skill_dir}/scripts/graph.py --list-studio
```
and merge both outputs, deduplicating by `(name, version)`.

Output is JSON:
```json
{"datasets": [{"name": "...", "version": "...", "num_objects": 123, "status": 4, "namespace": "...", "project": "...", "source": "local", "created_at": "...", "updated_at": "..."}]}
```

**One entry per dataset×version** — a dataset with three versions appears three times with the same `name` but different `version`, `num_objects`, and `updated_at`. Datasets in namespace `system` or project `listing` are already filtered out by the script.

### Phase 3 — Diff Against Existing Graph

Group the `--list` entries by `name`. For each unique dataset name:

1. **Derive the file path** from the `name` field:
   - If `name` contains `.` and splits into exactly 3 parts on the first two dots (Studio dataset, format `namespace.project.bare_name`):
     - `bare_name_slug` = `bare_name` lowercased, `.` → `_`
     - File: `.datachain/graph/datasets/{namespace}/{project}/{bare_name_slug}.md`
   - Otherwise (local dataset):
     - `name_slug` = `name` lowercased, `.` → `_`
     - File: `.datachain/graph/datasets/{name_slug}.md`
   - **Never include a version in the filename.**
   - The dot-separated `name` is used everywhere in Markdown content (frontmatter, headings, wikilink display text). The `/`-separated path is used only for file system paths and wikilink routing.

2. Identify `latest_version` as the highest version among all entries for this dataset.
3. Read the file at the derived path (if it exists).
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
db_last_updated: <value from Phase 1>   # omit this field if Phase 1 returned "studio"
generated_at: <current UTC timestamp>
local_dataset_count: <N>
studio_dataset_count: <M>   # omit if no Studio datasets were synced this run
---
```

Body: a table listing all datasets with columns: Name, Source, Version, Objects, Updated.
- `Source` is `local` or `studio` (from the `source` field in `--list` / `--list-studio` output).
- Each name is a wikilink with separate path and display text (`[[path|display]]`):
  - Studio dataset: `[[namespace/project/bare_name_slug|namespace.project.bare_name]]`
  - Local dataset: `[[name_slug|name]]`

Example:
```
| Name | Source | Version | Objects | Updated |
|------|--------|---------|---------|---------|
| [[flower-large\|flower-large]] | local | 1.0.1 | 2014 | 2026-03-20 |
| [[vladimir/default/300k\|vladimir.default.300k]] | studio | 1.0.5 | 300000 | 2026-03-18 |
```

**For each stale dataset**, run one script call **per version that is missing from the Version History**, plus always one call for `latest_version` (even if it already has an entry, to refresh the top-level metadata). Run calls oldest-version-first so `changes` fields chain correctly.

```
python3 {skill_dir}/scripts/graph.py --dataset <name>@<version>
```

Pass `<name>` exactly as it appears in the `--list` / `--list-studio` output: dot-separated (`namespace.project.bare_name`) for Studio datasets, plain for local datasets. Run once per version that needs a new or updated sub-section. **Do not call for versions already present verbatim in the history unless they are the latest.**

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

Write to the path derived in Phase 3:
- Studio dataset: `.datachain/graph/datasets/{namespace}/{project}/{bare_name_slug}.md`
- Local dataset: `.datachain/graph/datasets/{name_slug}.md`

Create parent directories as needed.
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

Render preview: use `columns` as table headers, `rows` as table rows. Omit the `## Preview` section entirely if `preview` is null (data not locally accessible).

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

- Collect all versions from the `--list` / `--list-studio` output for this dataset (one entry per version).
- Collect all `### version` sub-sections already in the file (if it exists).
- For every version that has no sub-section yet, run `--dataset name@version` and write a new sub-section from the script output.
- Carry over all existing sub-sections verbatim — do not re-fetch versions already in the file unless they are `latest_version`.
- Always re-run `--dataset name@latest_version` and replace (or write) its sub-section with fresh data.
- Write all sub-sections sorted newest-first.

**Incomplete history note:** If the `--list-studio` output for a dataset does not start from version `1.0.0` (i.e. older versions may exist that were not included in this sync), append this note at the very end of the Version History section:

```
_(Older versions exist but were not included in this sync. Run a full Studio sync to populate the complete history.)_
```

Do **not** add this note for local datasets — their full history is always available.

**Schema and Preview for Studio datasets:** `schema` and `preview` fields in the `--dataset` output may be `null` for Studio-only datasets (data is not accessible locally). In that case omit the `## Schema` and `## Preview` sections entirely — do not write placeholder text or "not available" messages.

All files use Obsidian-compatible wikilinks with `[[path|display]]` syntax — path uses `/`-separated slugs for routing, display text uses the dot-separated dataset name.

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
- **Studio datasets are opt-in**: `--list` always returns local datasets only. Use `--list-studio` only when the user explicitly requests a Studio sync.
- **Studio dataset names**: `--list-studio` emits `name` as `namespace.project.bare_name` (dot-separated). Dots are used in all Markdown content (frontmatter `name:`, headings, wikilink display text). File paths and wikilink routing use `/`-separated slugs derived by splitting on the first two dots: `namespace/project/bare_name_slug.md`. Pass the dot-separated name directly to `--dataset`.
- **Studio metadata availability**: `schema` and `preview` in `--dataset` output are `null` when dataset rows aren't locally accessible. `query_script` and `changes` come from the Studio API and are always populated when the dataset exists. `dependencies` are unavailable for Studio datasets (local metastore only).
