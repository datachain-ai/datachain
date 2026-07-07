# Bucket Enrichment Prompt

Generate a human-readable markdown overview for a cloud storage bucket from its JSON data file. The audience is data practitioners.

## Input

JSON fields:

- Identity: `uri`, `scheme`, `bucket`, `prefix`, `anon` (true/false/null)
- Listing: `scanned`, `listing_uuid`, `listing_created`, `listing_finished`
- Aggregates: `total_files`, `total_size_bytes`, `max_depth`
- `extensions[]`: file-type breakdown with counts, bytes, percentages
- `directories[]`: per-dir path, file count, bytes, depth
- `size_distribution`: min, max, median, p10, p90, empty_count
- `time_range`: oldest, newest
- `samples{}`: per-extension content samples with type-specific metadata
- `file_url_prefix` (optional): HTTPS URL prefix for clickable links
- `sampled` (optional): if true, totals reflect a subset; pass through to frontmatter
- `warnings` (optional): collection diagnostics. Surface only completeness caveats (e.g. a timed-out scan); never echo raw internal error strings

## Output Format

```markdown
---
uri: {uri}
bucket: {bucket}
prefix: {prefix}
anon: {true | false | omit if null}
uuid: {listing_uuid}
scanned: {scanned}
files: {total_files}
size: {human-readable}
sampled: {true | omit if false}
---

# {bucket}{" / " + prefix if prefix else ""}

{1-3 sentences: what this bucket contains and what it is likely used for.
Infer from directory structure, file types, naming patterns, and sample content.
Name the modalities and likely use case.}

{If prefix is set, add: "**Note:** This is a subdirectory (`{prefix}`) within `{bucket}`."}

## Quick Stats

- **Total files:** {total_files, comma-formatted}
- **Total size:** {human-readable}
- **File types:** {top 3 extensions with counts}
- **Date range:** {oldest} to {newest}
- **Access:** {"Public (use `anon=True`)" if anon=true; "Authenticated" if anon=false; omit if null}
- **Listing:** {freshness message — see below}

Listing freshness message (the keys are absent on sampled overviews):
- `listing_finished` missing or null → "Listing timestamp unavailable"
- otherwise → "bucket file list as of {listing_finished}; may be stale (new/changed files since then aren't included). Re-scan with `dc.read_storage(\"{uri}\", update=True)`"

## Directory Structure

Render a human-readable tree from `directories[]`:

```
{uri}
├── {dir1}/                  {files:>10,} files  ({human_size})
│   ├── {subdir1}/           {files:>10,} files  ({human_size})
│   └── {subdir2}/           {files:>10,} files  ({human_size})
└── {dir2}/                  {files:>10,} files  ({human_size})
```

- Show depth-1 and depth-2 as the primary structure; collapse deeper paths into one entry (`misc/deep/nested/`).
- Right-align file counts and sizes.
- If there are more directories than shown, add: "(N additional directories not shown)".
- After the tree, add one sentence naming the organizational pattern (e.g. "train/val splits with image and label subdirectories").

## File Types

| Extension | Files | Size | % Files | Description |
|-----------|------:|-----:|--------:|-------------|
| {ext}     | {count,} | {human_size} | {pct}% | {inferred description from samples} |

Describe each extension from its samples — name the contents concretely (dimensions for images, columns for tabular, duration/codec for media).

## Samples

For each extension with samples, show representative examples.

**File-path clickability.** If `file_url_prefix` is present, ALL file paths in sample tables MUST be clickable: `[path/to/file]({file_url_prefix}/{path/to/file})`.

### {ext} — {type_detected}

- Images: table with path, size, dimensions, format.
- Structured (Parquet/CSV/JSON): column list; if a snippet is available, show it in a code block.
- Text: first few lines in a code block.
- Audio/video: duration, codec, sample rate, channels.

## Data Quality

Include this section ONLY when there's something notable:
- Empty files (`empty_count > 0`): "⚠ {N} empty files (0 bytes)".
- Size outliers: if `p90/p10 > 100`, note wide size variation.
- If `max_bytes` is >100× the median, note outliers.

Omit the section entirely when nothing is notable.

```

## Guidelines

- **Be concise.** Each section is scannable in seconds.
- **Infer purpose** from directory names, file patterns, and sample content. Name the likely use case.
- **Human-readable numbers.** Comma separators (10,000) and human-readable sizes (3.2 GB).
- **Omit empty sections.** If `time_range` is empty, skip date info. If no quality issues, skip Data Quality.
- **No raw JSON dumps.** Markdown is a summary, not a data dump.
- **`uri` is not a clickable URL.** Storage URIs (`s3://`, `gs://`, `az://`) are not browsable — show as plain text.
- **Listing freshness is critical.** Always include the Listing line — show the timestamp when available, otherwise "Listing timestamp unavailable".
- **Human-readable timestamps:** `YYYY-MM-DD HH:MM:SS` (no `T`, no `Z`).
- **Structure only, no pipelines.** Do not propose decompositions, slices, example queries, or recommended pipelines — the right shape depends on the question being asked.
- **Sampled mode.** If input has `sampled: true`, totals reflect a subset; carry through to frontmatter, state prominently in the body, and link to the underlying dataset (`dataset_name`) so readers can query actual file rows.
- **Completeness caveats.** If `warnings` shows the scan was partial or timed out, state it plainly. Never reproduce raw internal error strings from `warnings` verbatim — they are diagnostics, not reader-facing.
