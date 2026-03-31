# Dataset Enrichment Prompt

Generate a human-readable markdown summary for a DataChain dataset from its JSON data file.

## Input

Read the JSON file at the path provided. It contains:

- `name`: dataset name
- `source`: `"local"` or `"studio"`
- `versions[]`: array ordered oldest-first, each with:
  - `version`, `num_objects`, `updated_at`
  - `schema`: column definitions (latest version has full schema; older versions may have `{}`)
  - `preview`: `{columns, rows}` sample data (latest version only)
  - `query_script`: Python code that produced this version (may be `null`)
  - `changes`: diff vs previous version (`null` for first version)
    - `script_changed`: boolean
    - `previous_script`: the prior version's script (if changed)
    - `deps_added`, `deps_removed`, `deps_updated`: dependency change lists
  - `dependencies[]`: upstream datasets/listings with `name`, `version`, `type`, and optional `file_path` (relative link target for listings)

## Output Format

Write a markdown file with this structure:

```
# {dataset_name}

{AI-generated description: 1-3 sentences explaining what this dataset contains
and how it is produced. Infer purpose from schema fields, query_script logic,
and dependency names. Be specific — mention data types and transformations.}

**Source data:** [{json_filename}]({json_filename})

## Stats

- **Latest version:** {version}
- **Objects:** {num_objects}
- **Updated:** {updated_at or "N/A"}
- **Source:** {source}

## Dependencies

List dependencies as clickable links when `file_path` is present:
`[{name}]({file_path}.md)`. Otherwise, just the name.

## Schema

```
{column}: {type}
  {nested_field}: {type}
```

Use the latest version's schema. Show nested fields indented under their parent.
Provide meaning of the columns as a comment.

## Preview

{Markdown table from preview.columns and preview.rows. Show all rows provided.
If preview is null, omit this section entirely.}

# Versions

{One subsection per version, newest first.}

### {version} — {date} ({num_objects} objects)

{For the latest version: 1-2 sentences describing the current state — what the
dataset produces and its key characteristics.}

{For older versions with changes: 1-2 sentences summarizing what changed.
Focus on meaning — "Added ratio classification logic" not "script_changed: true".
Mention dependency changes only if significant (new data source, version bump).}

{For the initial version (no changes): just "Initial version."}
```

## Guidelines

- **Be concise.** Each version summary is 1-2 sentences maximum.
- **Infer purpose.** Read the query_script to understand what the dataset does. Name transformations, filters, and computations — not implementation details.
- **Compress old versions.** The reader wants to understand evolution at a glance. Do not reproduce raw diffs, full dependency tables, or scripts.
- **Omit empty sections.** If preview is null, skip ## Preview. If schema is empty, skip ## Schema.
- **No dependency tables in version summaries.** Only mention a dependency if it was added, removed, or significantly changed.
- **No script blocks in version summaries.** The full scripts live in the JSON for reference — the markdown is a summary, not a data dump.
- **Human-readable timestamps.** Format all timestamps as `YYYY-MM-DD HH:MM:SS` (no `T`, no `Z`).
- **If nothing meaningful changed** between versions (no script change, no dep changes), write "Data refreshed; no functional changes."
- **Source data link.** `{json_filename}` is the basename of the JSON input file (e.g. `my_dataset.json`). The link lets readers access the raw structured data.
