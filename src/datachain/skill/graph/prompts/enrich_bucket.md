# Bucket Enrichment Prompt

Generate a human-readable markdown overview for a cloud storage bucket from its JSON data file. The audience is data practitioners who build models, extract insights, or process data.

## Input

Read the JSON file at the path provided. It contains:

- `uri`: the full storage URI
- `scheme`: storage scheme (s3, gs, az)
- `bucket`: bucket name
- `prefix`: subdirectory prefix (empty string if whole bucket)
- `scanned_at`: when this scan was performed
- `listing_created_at`, `listing_expires_at`, `listing_expired`: listing freshness
- `total_files`, `total_size_bytes`: aggregate counts
- `max_depth`: deepest directory nesting level
- `extensions[]`: file type breakdown with counts, bytes, percentages
- `directories[]`: directory breakdown with path, file count, bytes, depth
- `size_distribution`: min, max, median, p10, p90, empty_count
- `time_range`: oldest and newest file timestamps
- `samples{}`: per-extension content samples with type-specific metadata

## Output Format

Write a markdown file with this structure:

```
---
uri: {uri}
bucket: {bucket}
prefix: {prefix}
total_files: {total_files}
total_size: {human-readable size}
scanned_at: {scanned_at}
---

# {bucket}{" / " + prefix if prefix else ""}

**Source data:** [{json_filename}]({json_filename})

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
- **Listing:** {freshness message — see below}

{Listing freshness message:
- If listing_expired is false: "Bucket listing from {listing_created_at} (valid until {listing_expires_at})"
- If listing_expired is true: "Bucket listing from {listing_created_at} (**expired** — refresh with `dc.read_storage(\"{uri}\", update=True)`)"
- If listing_created_at is null: "Listing timestamp unavailable"}

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

## Getting Started

```python
import datachain as dc

# List all files
dc.read_storage("{uri}").show()

# Filter by type
dc.read_storage("{uri}", type="image").show()
```

{Add 1-2 more DataChain snippets tailored to the actual content:
- If images found: show image processing example
- If structured data: show read_csv/read_json/read_parquet example
- If train/val splits detected: show filtering by split
Always use the actual URI from the JSON.}
```

## Guidelines

- **Be concise.** Each section should be scannable in seconds.
- **Infer purpose.** Read directory names, file patterns, and sample content to understand what the data is for. Name the likely use case.
- **Human-readable numbers.** Use comma separators (10,000) and human-readable sizes (3.2 GB).
- **Omit empty sections.** If time_range is empty, skip date info. If no quality issues, skip Data Quality.
- **No raw JSON dumps.** The markdown is a summary, not a data dump.
- **Practical Getting Started.** The code snippets should work copy-paste. Use the actual URI and file types from the data.
- **Listing freshness is critical.** Users need to know if they're looking at stale data. Always show the listing timestamp.
- **Source data link.** `{json_filename}` is the basename of the JSON input file (e.g. `datachain_demo.json`). The link lets readers access the raw structured data.
