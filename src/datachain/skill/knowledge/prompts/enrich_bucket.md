# Bucket Enrichment Prompt

Generate a human-readable markdown overview for a cloud storage bucket from its JSON data file. The audience is data practitioners who build models, extract insights, or process data.

## Input

Read the JSON file at the path provided. It contains:

- `uri`: the full storage URI
- `scheme`: storage scheme (s3, gs, az)
- `bucket`: bucket name
- `prefix`: subdirectory prefix (empty string if whole bucket)
- `anon`: whether the bucket requires anonymous access (`true`, `false`, or `null` if unknown)
- `scanned`: when this scan was performed
- `listing_uuid`: unique identifier for the listing
- `listing_created`, `listing_expires`, `listing_expired`: listing freshness
- `total_files`, `total_size_bytes`: aggregate counts
- `max_depth`: deepest directory nesting level
- `extensions[]`: file type breakdown with counts, bytes, percentages
- `directories[]`: directory breakdown with path, file count, bytes, depth
- `size_distribution`: min, max, median, p10, p90, empty_count
- `time_range`: oldest and newest file timestamps
- `samples{}`: per-extension content samples with type-specific metadata
- `file_url_prefix` (optional): HTTPS URL prefix for building clickable file links

## Output Format

Write a markdown file with this structure:

```
---
uri: {uri}
bucket: {bucket}
prefix: {prefix}
anon: {anon — true, false, or omit if null}
uuid: {listing_uuid}
scanned: {scanned}
files: {total_files}
size: {human-readable size}
---

# {bucket}{" / " + prefix if prefix else ""}

{AI-generated description: 1-3 sentences explaining what this bucket contains
and what it is likely used for. Infer purpose from directory structure, file types,
naming patterns, and sample content. Be specific — mention data modalities,
organizational patterns, and likely use cases.}

{If prefix is set, add: "**Note:** This is a subdirectory (`{prefix}`) within
the `{bucket}` bucket."}

## Quick Stats

- **Total files:** {total_files, comma-formatted}
- **Total size:** {human-readable}
- **File types:** {top 3 extensions with counts}
- **Date range:** {oldest} to {newest}
- **Access:** {if anon is true: "Public (use `anon=True` in `read_storage()`)", if false: "Authenticated", if null: omit this line}
- **Listing:** {freshness message — see below}

{Listing freshness message:
- If listing_expired is false: "Bucket listing from {listing_created} (valid until {listing_expires})"
- If listing_expired is true: "Bucket listing from {listing_created} (**expired** — refresh with `dc.read_storage(\"{uri}\", update=True)`)"
- If listing_created is null: "Listing timestamp unavailable"}

## Directory Structure

```
{uri}
├── {dir1}/                  {files:>10,} files  ({human_size})
│   ├── {subdir1}/           {files:>10,} files  ({human_size})
│   └── {subdir2}/           {files:>10,} files  ({human_size})
└── {dir2}/                  {files:>10,} files  ({human_size})
```

{Build a human-readable tree from the directories array. Rules:
- Show depth-1 and depth-2 directories as the primary structure.
- For deeper paths, collapse intermediate levels: show as "misc/deep/nested/" not three separate entries.
- Right-align file counts and sizes for readability.
- If there are more directories than shown, add a note: "(N additional directories not shown)"
- Highlight organizational patterns in 1 sentence after the tree: e.g., "Data is organized into train/val splits with separate image and label subdirectories."}

## File Types

| Extension | Files | Size | % Files | Description |
|-----------|------:|-----:|--------:|-------------|
| {ext}     | {count,} | {human_size} | {pct}% | {AI-inferred description from samples} |

{For each extension, use the samples data to describe what these files contain.
E.g., ".jpg" with width/height samples → "JPEG images, mostly 640×480"
E.g., ".json" with snippet → "JSON annotation files with bbox labels"
E.g., ".parquet" with columns → "Parquet files with columns: id, embedding, label"}

## Samples

{For each extension that has samples, show representative examples.}

**IMPORTANT: If `file_url_prefix` is present in the JSON, ALL file paths in sample tables MUST be clickable links using the format `[path/to/file]({file_url_prefix}/{path/to/file})`.**

Example with `file_url_prefix = "https://my-bucket.s3.amazonaws.com"`:
`| [images/cat.jpg](https://my-bucket.s3.amazonaws.com/images/cat.jpg) | 102.4 KB |`

### {ext} — {type_detected}

{For images: show a table with path, size, dimensions, format.}
{For structured: show column names. If snippet available, show a formatted code block.}
{For text: show first few lines in a code block.}
{For audio/video: show duration, codec, sample rate, etc.}

## Data Quality

{Only include this section if there are notable quality observations:
- Empty files (empty_count > 0): "⚠ {N} empty files (0 bytes)"
- Size outliers: if p90/p10 ratio > 100, note wide size variation
- If max_bytes is very large relative to median (>100x), note outliers
- If nothing notable, omit this section entirely.}

```

## Guidelines

- **Be concise.** Each section should be scannable in seconds.
- **Infer purpose.** Read directory names, file patterns, and sample content to understand what the data is for. Name the likely use case.
- **Human-readable numbers.** Use comma separators (10,000) and human-readable sizes (3.2 GB).
- **Omit empty sections.** If time_range is empty, skip date info. If no quality issues, skip Data Quality.
- **No raw JSON dumps.** The markdown is a summary, not a data dump.
- **Do not make `uri` values clickable.** Storage URIs like `s3://`, `gs://`, `az://` are not browsable URLs — show them as plain text.
- **Listing freshness is critical.** Users need to know if they're looking at stale data. Always show the listing timestamp.
- **Human-readable timestamps.** Format all timestamps as `YYYY-MM-DD HH:MM:SS` (no `T`, no `Z`).
- **Structure only, no pipelines.** Do not propose decompositions, slices, example queries, or recommended pipelines. The right shape depends on the question being asked — let the agent decide at runtime.
- **Sampled.** If the input has `sampled: true`, totals reflect a subset, not full enumeration. Carry `sampled: true` to frontmatter, state prominently in the body that the document describes a sampled overview, and link to the underlying dataset (`dataset_name` field) so readers can query the actual file rows.
