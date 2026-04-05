# Dataset Enrichment Prompt

Generate a human-readable one short paragraph summary for a DataChain dataset from its JSON data file.

## Input

Read the JSON file at the path provided. It contains:

- `name`: dataset name
- `source`: `"local"` or `"studio"`
- `versions[]`: array ordered oldest-first, each with:
  - `version`, `uuid`, `records`, `updated`
  - `schema`: column definitions (latest version has full schema; older versions may have `{}`)
  - `preview`: `{columns, rows}` sample data (latest version only). May include `file_url_prefix` — an HTTPS URL prefix (e.g., `https://bucket.s3.amazonaws.com`) for building clickable file links.
  - `summary`: statistical summary (latest version only, may be `null`). Contains:
    - `overview`: one-line dataset summary (e.g., `"10.2K items · 5 cols · 2.3 GB · JPEG · ~480×400"`)
    - `sampled`: boolean — whether stats are based on a random sample
    - `sample_size`: number of sampled rows (only present if `sampled` is true)
    - `columns`: dict of `{col_path: {type, category, line, stats}}` where `line` is a pre-formatted one-line statistical summary
  - `query_script`: Python code that produced this version (may be `null`)
  - `changes`: diff vs previous version (`null` for first version)
    - `script_changed`: boolean
    - `previous_script`: the prior version's script (if changed)
    - `deps_added`, `deps_removed`, `deps_updated`: dependency change lists
  - `dependencies[]`: upstream datasets/listings with `name`, `version`, `type`, and optional `file_path` (relative link target for listings)

## Output Format

Write a markdown file with this structure:

```
---
name: {name}
last_version: {version}
last_version_uuid: {uuid from the latest version}
updated: {updated}
records: {records}
is_local: {true if source is "local", false if "studio"}
known_versions: [{comma-separated list of all version strings, e.g. 1.0.0, 1.0.1}]
---

# {dataset_name}

{AI-generated description: one short paragraph explaining what this dataset contains
and how it is produced. Be specific — mention data types and transformations.
It should be optimized for dataset reusage - how this dataset is helpful.
Dependency names are not necessary here since they are presented in another section.}

## Dependencies

List dependencies as clickable links when `file_path` is present:
`[{name}]({file_path}.md)`. Otherwise, just the name.

## Preview

{Markdown table from preview.columns and preview.rows. Show all rows provided.
If preview is null, omit this section entirely.
If `preview.file_url_prefix` is present and a column ends with `.path` or is named `path`,
make those cells clickable: `[value]({file_url_prefix}/{value})`.}

## Schema

```
{column}: {type}
  {nested_field}: {type}
```

Use the latest version's schema. Show nested fields indented under their parent.
Provide meaning of the columns as a comment.

## Stats

If the latest version has a `summary` field, render per-field statistical summaries.
If `summary` is null, omit this section entirely.

```
{column}                # {summary.columns[col_path].line}
  {nested_field}        # {summary.columns[col_path].line}
```

- Mirror the same nesting as the Schema section (same grouping, same order).
- Do NOT repeat types — they are already shown in Schema.
- Show parent signal names as group headers (no comment needed for parents).
- For each leaf field, use the pre-formatted `line` from `summary.columns[col_path]` as the comment after `#`.
- If a field has no stats (empty `line`), omit it from Stats.
- If `summary.sampled` is true, add a note after the block: `_Stats based on a random sample of {sample_size}._`

Example:

```
file
  size                  # 32KB - 165KB, p50=90KB
info
  width                 # 375 - 600, p50=480, p95=590
  height                # 313 - 500, p50=420, p95=490
  format                # JPEG 100%
label                   # cat 55%, dog 40%, bird 5%
```

# Versions

{One subsection per version, newest first.}

### {version} — {date} ({records} records)

{For the latest version: 1-2 sentences describing the current state — what the
dataset produces and its key characteristics.}

```python
{query_script}
```

{For older versions with changes.script_changed == true: 1-2 sentences summarizing
what changed. Include the full query_script in a python code block.}

{For older versions with changes.script_changed == false: 1-2 sentences describing
what changed (dependency updates, data refresh, etc.). Do NOT include code — the
reader can refer to the nearest version above that shows code.}

{For the initial version (no changes): "Initial version." followed by the
query_script in a python block if available.}
```

## Guidelines

- **Be concise.** Each version summary is 1-2 sentences maximum.
- **Infer purpose.** Read the query_script to understand what the dataset does. Name transformations, filters, and computations — not implementation details.
- **Compress old versions.** The reader wants to understand evolution at a glance. Do not reproduce raw diffs, full dependency tables, or scripts.
- **Omit empty sections.** If preview is null, skip ## Preview. If schema is empty, skip ## Schema.
- **No dependency tables in version summaries.** Only mention a dependency if it was added, removed, or significantly changed.
- **Code inclusion rules:**
  - **Latest version:** ALWAYS include the full `query_script` in a ```python block. This is mandatory when query_script is not null.
  - **Older versions:** Include `query_script` ONLY when `changes.script_changed` is true. Otherwise, describe changes textually.
  - **Initial version:** Include `query_script` if available.
  - **Code reconstructibility:** When omitting code for a version, the textual summary must describe what changed clearly enough that combined with the nearest version that shows code, any version's code can be reconstructed.
  - If `query_script` is null for all versions, omit code blocks entirely.
- **Human-readable timestamps.** Format all timestamps as `YYYY-MM-DD HH:MM:SS` (no `T`, no `Z`).
- **If nothing meaningful changed** between versions (no script change, no dep changes), write "Data refreshed; no functional changes."
- **Frontmatter `known_versions`** must list every version string from the input, comma-separated inside brackets. This field is used by tooling to detect which versions are documented.
