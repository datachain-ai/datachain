# Delta Processing

Delta processing allows you to optimize the creation of new dataset versions by calculating and processing only the differences (deltas) between dataset versions. This is especially useful when working with large datasets that change incrementally over time.

## Overview

When working with large datasets that receive regular updates, reprocessing the entire dataset each time is inefficient. The `delta=True` flag in DataChain enables incremental processing, where only new or changed records are processed.

## Upgrading existing delta datasets

**Upgrading does not automatically clean stale results saved by older delta
updates.** Existing datasets may still contain rows for deleted files or old
revisions of overwritten files. If the source version has already been recorded
as processed, another delta run can leave those rows untouched.

To bring an existing result into sync:

1. Rerun the same pipeline once with `delta=False`. For `read_storage`, also keep
   `update=True` to refresh the source listing. Keep the same `.save(...)` name.
2. After that run succeeds, restore `delta=True` for subsequent runs.

**This is a full-source rebuild and can rerun expensive UDFs over every current
source record.** It saves a new result version; previous versions remain
unchanged. Newly created results do not need this rebuild.

## How Delta Processing Works

When delta processing is enabled:

1. DataChain calculates the difference between the latest version of your source dataset and the version used to create the most recent version of your result dataset
2. Only the records that are new or modified in the source dataset are processed
3. All result rows belonging to modified or deleted source records are removed, even when a filter rejects the replacement or a generator yields no rows
4. The newly processed rows are combined with the retained results for unchanged source records

For a fixed row-local pipeline using `map`, `gen`, and filters, the saved result contains the same rows as a full non-delta recompute of the current source. UDFs run only on added and modified source records, plus records explicitly selected by `delta_retry`. Nondeterministic UDF values can differ from a full recompute because unchanged records retain their saved results.

A generator's entire output for a source record is replaced or removed together. Every output row must retain the source key fields, or their counterparts named by `delta_result_on`. A generator must yield these fields explicitly. The same requirement applies when using `select` or renaming fields.

Use `update=True` with `read_storage` to refresh the storage index. A deletion-only refresh can create a new result version without calling the UDF. Source changes that produce no output still advance the saved dependencies, so subsequent runs do not process those changes again. If neither source records nor retry inputs change, the existing result version is reused.

## Usage

Delta processing can be enabled through the `delta` parameter when using `read_storage()` or `read_dataset()`:

```python
import datachain as dc

# Process only new or changed files
chain = (
    dc.read_storage(
        "data/",
        update=True,  # Refresh the storage index
        delta=True,  # Enable delta processing
        delta_on="file.path",  # Field that uniquely identifies records
        delta_compare="file.last_modified",  # Field to check for changes
    )
    .map(result=process_function)
    .save(name="processed_data")
)
```

## Parameters

- **delta**: Boolean flag to enable delta processing
- **delta_on**: Field(s) that uniquely identify rows in the source dataset. The default is `("file.path", "file.etag", "file.version")`. An overwrite changes this revision key: the old revision's results are removed and the new revision is processed. For explicit identity-based matching, use e.g. `("file.source", "file.path")` with `delta_compare=("file.etag", "file.version")`. Include the source when paths can collide across storages.
- **delta_result_on**: Field(s) in the resulting dataset that correspond to `delta_on` fields (if they have different names)
- **delta_compare**: Field(s) used to check if a record has been modified. By default, all non-key source fields are compared. An explicit list must cover every source field whose changes can affect your output or filters; ignored changes cannot trigger recomputation.

Storage listings with `versions=True` may contain historical object revisions, but `read_storage` returns current objects by filtering `file.is_latest`. The default revision key and comparison fields detect both new revisions and transitions out of the current listing. With `versions=False`, the listing contains only current objects; overwritten revisions disappear from the source and their results are removed. When processing a versioned listing directly with `read_dataset`, retain `file.version` in the key if historical revisions should remain distinct.

## Example: Keeping File Results in Sync

```python
import datachain as dc
import time


def process_file(file):
    """Process a file and return results."""
    content = file.read_text()
    # Simulate processing time
    time.sleep(0.1)
    return {"content": content, "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")}


# Process only new or modified files
chain = (
    dc.read_storage(
        "data/",
        update=True,  # Update the storage index
        delta=True,  # Process new and changed files; remove deleted files' results
        delta_on="file.path",  # Files are identified by their paths
    )
    .map(result=process_file)
    .save(name="processed_files")
)

print(f"Processed {chain.count()} files")
```

## Combining with Retry Processing

Delta processing can be combined with [retry processing](./retry.md) to create a powerful workflow that both:

1. Processes only new or changed records (delta)
2. Reprocesses records with errors or that are missing (retry)

## Using Delta with Restricted Methods

By default, delta updates cannot be combined with the following methods:

1. `merge`
2. `union`
3. `subtract`
4. `diff`
5. `file_diff`
6. `distinct`
7. `agg`
8. `group_by`

These methods can depend on unchanged records as well as changed ones, so replaying only the changed inputs is not always sufficient.

If you still need to use these methods together with delta, you can override this restriction by setting the additional flag:

```python
delta_unsafe = True
```

If more than one delta-enabled source participates in the same composed query, set `delta_unsafe=True` on every participating delta source.

Multiple delta sources require this opt-in and a shared result key. DataChain uses partial replay for joins on that key when the delta sources change the same keys. Independently changing sources, non-key or full joins, and cross-record operations such as `distinct`, aggregation, window functions, `limit`, and `offset` fall back to recomputing the full current query. A changed non-delta input also triggers a full recompute.

**A full recompute can rerun UDFs on unchanged records.** `delta_unsafe=True` keeps the existing key requirements and enables these compositions; it does not guarantee incremental execution. Use a fixed pipeline and retain the configured source-key values in its output.
