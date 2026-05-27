# Dataset Enrichment Prompt

Generate a human-readable markdown summary for a DataChain dataset from its JSON data file.

## Input

JSON fields:

- `name`, `source` (`"local"` / `"studio"`)
- `description` (optional) — free-form description from `.save(description=...)`. May be `null`.
- `attrs` (optional) — list of string tags from `.save(attrs=[...])`. The CAST keys: `cast:<layer>`, `scope:<bucket|directory|sample|onetime>`, `source:<slug>`. Empty list `[]` if no tags set.
- `versions[]` — ordered oldest-first. Each has:
  - `version`, `uuid`, `records`, `updated`
  - `schema` (latest version has full schema; older may be `{}`)
  - `preview` (latest version only) — `{columns, rows}` plus optional `file_url_prefix` (HTTPS URL prefix for clickable file links)
  - `summary` (latest version only, may be `null`) — `{overview, sampled, sample_size, columns}` where `columns[col].line` is a pre-formatted statistical summary
  - `query_script` — Python code that produced this version (may be `null`)
  - `changes` — diff vs previous (`null` for first version): `script_changed`, `previous_script`, `deps_added`, `deps_removed`, `deps_updated`
  - `dependencies[]` — upstream datasets/listings with `name`, `version`, `type`, optional `file_path`

## Output Format

```markdown
---
name: {name}
last_version: {version}
last_version_uuid: {uuid from latest version}
updated: {updated}
records: {records}
is_local: {true if source == "local" else false}
known_versions: [{comma-separated version strings}]
cast_layer: {container | asset | sense | task, or empty}
cast_scope: {bucket | directory | sample | onetime, or empty}
cast_source: {bucket slug for L1-L3, task slug for L4, or empty}
---

# {dataset_name}

{One short paragraph: what this dataset contains and how it is produced.
Optimised for reuse — name data types, transformations, and what it is helpful for.
Dependency names are not necessary here (covered in Dependencies section).
If the input JSON has a non-null `description`, prefer it as the lead sentence.}

## Session Context

{Include ONLY when session context exists. Two cases:

1. **Re-enrichment**: if the existing `.md` has a `## Session Context` section,
   preserve it here verbatim. Do not paraphrase or rewrite.
2. **New dataset created during an agentic session**: 1-3 sentences on WHY this
   dataset was created — the analytical goal, the investigation, the user's
   motivation. For a CAS layer built during the session, add one line on why
   the layer is reusable beyond the current task.

Omit entirely if no existing section AND no meaningful session to describe,
or if the dataset was recovered from the DB without conversation context,
or if the "why" is already obvious from the description above.}

## Dependencies

List dependencies as clickable links when `file_path` is present:
`[{name}]({file_path}.md)`. Otherwise, just the name.

## Preview

Render the Markdown table **from `preview.columns` and `preview.rows`
verbatim**. Do NOT rename columns, drop columns, reorder columns, or
inject derived columns. The saved schema is the contract; the preview
shows that schema's rows as they are.

- Show every row in `preview.rows` (never omit for size — truncate
  values instead).
- Long list / vector columns: show the first 2-3 elements + `…` + length,
  e.g. `[0.0132, -3.34e-3, …] (768)`.
- **Clickable file paths — mandatory when `preview.file_url_prefix` is
  set.** For every column whose name ends in `.path` or equals `path`,
  wrap each cell value in `[<value>]({file_url_prefix}/<value>)`. This
  applies to **every** such column, including secondary file columns
  on Task rows that joined multiple sources (e.g. a `right_*.path` from
  a merge). If a column was renamed for display, the rule still applies
  — the underlying source field type, not the display label, determines
  clickability.

  Transformation recipe:

  ```
  Cell value:     <relative/file/path.ext>
  Wrapped:        [<relative/file/path.ext>]({file_url_prefix}/<relative/file/path.ext>)
  ```

Omit this section entirely if `preview` is null.

## Schema

```
{column}: {type}
  {nested_field}: {type}
```

Use the latest version's schema. Show nested fields indented under their parent.
Add a brief comment after each field explaining what it represents.

## Stats

If the latest version has a `summary` field, render per-field statistical summaries
mirroring the Schema nesting. If `summary` is null, omit this section.

```
{column}                # {summary.columns[col_path].line}
  {nested_field}        # {summary.columns[col_path].line}
```

- Do NOT repeat types (they are in Schema).
- Parent signal names appear as group headers without a comment.
- Omit fields whose `line` is empty.
- If `summary.sampled` is true, add after the block: `_Stats based on a random sample of {sample_size}._`

Example:

```
file
  size                  # 32KB - 165KB, p50=90KB
info
  width                 # 375 - 600, p50=480, p95=590
label                   # cat 55%, dog 40%, bird 5%
```

# Versions

One subsection per version, newest first.

### {version} — {date} ({records} records)

- **Latest version**: 1-2 sentences on current state, then the full `query_script` in a `python` block (mandatory when `query_script` is not null).
- **Older versions with `changes.script_changed == true`**: 1-2 sentences on what changed, plus the full `query_script`.
- **Older versions with `changes.script_changed == false`**: 1-2 sentences on what changed (dep updates, refresh, etc.). No code block.
- **Initial version** (no changes): "Initial version." plus `query_script` if available.
- If `query_script` is null for all versions, omit code blocks entirely.

## Guidelines

- **Be concise.** Each version summary is 1-2 sentences max.
- **Omit empty sections.** If preview is null, skip Preview. If schema is empty, skip Schema. If no session context, skip Session Context.
- **Session context is verbatim on re-enrichment.** Do not paraphrase, merge with description, or rewrite.
- **No duplication.** Description = what the dataset contains. Session context = why it was created.
- **No dependency tables in version summaries.** Only mention a dependency if it was added, removed, or significantly changed.
- **Human-readable timestamps:** `YYYY-MM-DD HH:MM:SS` (no `T`, no `Z`).
- **No functional change?** Write "Data refreshed; no functional changes."
- **`known_versions` lists every version string**, comma-separated inside brackets.
- **CAST frontmatter resolution order:**
  1. If the existing `.md` has `cast_layer` / `cast_scope` / `cast_source` in frontmatter, **preserve verbatim** (same rule as Session Context).
  2. Otherwise, read `attrs`. `cast:<layer>` → `cast_layer`; `scope:<scope>` → `cast_scope`; `source:<slug>` → `cast_source`.
  3. If `attrs` has no CAST tags, fall back to the name prefix: `l1_…` → container; `l2_…` → asset; `l3_…` → sense. (Task uses prefix-free names; no name-prefix rule.) Leave other CAST fields empty.
  4. If neither encodes a CAST layer, leave all fields empty.
- **Conflict resolution.** If name prefix and `attrs cast:<layer>` disagree, prefer `attrs` silently — `attrs` is authoritative.
