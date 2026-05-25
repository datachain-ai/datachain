# Dataset Enrichment Prompt

Generate a human-readable one short paragraph summary for a DataChain dataset from its JSON data file.

## Input

Read the JSON file at the path provided. It contains:

- `name`: dataset name
- `source`: `"local"` or `"studio"`
- `description` (optional): the dataset's free-form description string from the DataChain record (set via `.save(description=...)`). May be `null` if no description was provided.
- `attrs` (optional): a list of string tags from the DataChain record (set via `.save(attrs=[...])`). The skill encodes CASE methodology fields here: `case:<layer>`, `scope:<bucket|directory|sample|onetime>`, `source:<slug>`, `parent:<dataset-name>` (the parent key may appear multiple times). Empty list `[]` is the default when no tags were set.
- `session_context` (optional): not present in the JSON. If the dataset already has an enriched `.md` file, check it for a `## Session Context` section — this is session-level context about why the dataset was created, preserved across re-enrichments.
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
case_layer: {container | asset | sense | experiment, or empty if not a CASE dataset}
case_scope: {bucket | directory | sample | onetime, or empty}
case_source: {bucket slug for L1-L3, task slug for L4, or empty}
case_parents: [{comma-separated list of upstream CASE dataset names, or empty}]
---

# {dataset_name}

{AI-generated description: one short paragraph explaining what this dataset contains
and how it is produced. Be specific — mention data types and transformations.
It should be optimized for dataset reusage - how this dataset is helpful.
Dependency names are not necessary here since they are presented in another section.
If the input JSON has a non-null `description` field, prefer it as the lead sentence — it
is the human-curated one-liner set by the pipeline author at `.save()` time.}

## Session Context

{Include this section ONLY when session context exists. Two cases:

1. **Re-enrichment**: If the existing `.md` file has a `## Session Context` section,
   preserve it here verbatim. Do not paraphrase or rewrite.

2. **New dataset created during an agentic session**: If you are enriching a dataset
   that was just created in this session and the session provides meaningful context
   about WHY this dataset was created — the analytical goal, the investigation that
   led to it, the user's motivation — write 1-3 sentences here.

   For a CAS layer (Container / Asset / Sense) built during the session, add a
   one-line "Layer rationale" sentence stating why this layer is reusable beyond the
   current task. Example: "Built as a Sense layer alongside the user's similarity
   query; the same embeddings will answer every future visual-similarity question on
   this bucket."

Omit this section entirely if:
- There is no existing session context AND no meaningful session to describe
- The dataset was recovered from the operational DB without a conversation
- The "why" is already obvious from the description above}

## Dependencies

List dependencies as clickable links when `file_path` is present:
`[{name}]({file_path}.md)`. Otherwise, just the name.

## Preview

{Markdown table from preview.columns and preview.rows. Show all rows provided.
If preview is null, omit this section entirely.
NEVER omit the preview table because values are large. Truncate instead.

**Long list/vector columns (e.g., embeddings):** Show the first 2-3 elements followed by `…` and the length. Example: `[0.0132, -3.34e-3, …] (768)`. This keeps the table readable while confirming the data exists and its dimensionality.

**IMPORTANT: If `preview.file_url_prefix` is present, ALL cells in columns ending with `.path` or named `path` MUST be clickable links: `[value]({file_url_prefix}/{value})`.**

Example with `file_url_prefix = "https://my-bucket.s3.amazonaws.com"`:
`| [images/cat.jpg](https://my-bucket.s3.amazonaws.com/images/cat.jpg) | 32,362 |`}

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
- **Omit empty sections.** If preview is null, skip ## Preview. If schema is empty, skip ## Schema. If no session context exists, skip ## Session Context.
- **Session context is verbatim on re-enrichment.** If the existing `.md` has a `## Session Context` section, preserve it unchanged — do not paraphrase, merge with description, or rewrite.
- **Session context is optional.** Only add it for datasets created during an agentic session when the session provides meaningful "why/how" context. Do not fabricate context. Do not add it during routine knowledge-base refreshes.
- **No duplication with description.** Description = what the dataset contains. Session context = why it was created, what session/investigation led to it.
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
- **CASE frontmatter fields are resolved in this order:**
  1. If the existing `.md` already has `case_layer` / `case_scope` / `case_source` / `case_parents` in its frontmatter, **preserve them verbatim** during re-enrichment (same rule as `## Session Context`).
  2. Otherwise, read the `attrs` list from the input JSON. If it contains a `case:<layer>` tag, use that as `case_layer`. Apply the same for `scope:<scope>` → `case_scope`, `source:<slug>` → `case_source`, and all `parent:<name>` entries → `case_parents` (comma-separated list).
  3. If `attrs` carries no CASE tags, fall back to the dataset name. If the name matches `l1_container_…` / `l2_asset_…` / `l3_sense_…`, set `case_layer` to container / asset / sense respectively. There is no name-prefix rule for Experiment (Experiment outputs use natural prefix-free names); they fall through to step 4. Leave `case_scope`, `case_source`, `case_parents` empty.
  4. If neither `attrs` nor the name encodes a CASE layer, leave all four fields empty. The KB index will render the dataset under "Experiment Dataset" as the catch-all.
- **Conflict resolution.** If the name prefix and the `attrs` `case:<layer>` tag disagree, prefer `attrs` silently — `attrs` is authoritative.
