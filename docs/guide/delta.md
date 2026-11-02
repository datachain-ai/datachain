# Delta Processing

Delta processing allows you to optimize the creation of new dataset versions by calculating and processing only the differences (deltas) between dataset versions. This is especially useful when working with large datasets that change incrementally over time.

## Overview

When working with large datasets that receive regular updates, reprocessing the entire dataset each time is inefficient. The `delta=True` flag in DataChain enables incremental processing, where only new or changed records are processed.

## How Delta Processing Works

When delta processing is enabled:

1. DataChain calculates the difference between the latest version of your source dataset and the version used to create the most recent version of your result dataset
2. Only the records that are new or modified in the source dataset are processed
3. All result rows belonging to modified or deleted source records are removed, even when `.filter(...)` excludes the replacement or `.gen(...)` produces no replacement rows
4. The newly processed rows are combined with the retained results for unchanged source records

For a fixed row-local pipeline using `map`, `gen`, and filters, the saved result contains the same rows as a full non-delta recompute of the current source. UDFs run only on added and modified source records, plus records explicitly selected by `delta_retry`. Nondeterministic UDF values can differ from a full recompute because unchanged records retain their saved results.

All rows produced by `.gen(...)` for one source record are replaced or removed together. The function must yield the source key fields, or their counterparts named by `delta_result_on`, in every output row. Preserve those fields when using `select` or renaming them.

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
- **delta_on**: Field(s) that uniquely identify source records. The default is `("file.path", "file.etag", "file.version")`; overwriting a file replaces the results for its previous revision. Include `file.source` if paths can collide across storages.
- **delta_result_on**: Field(s) in the resulting dataset that correspond to `delta_on` fields (if they have different names)
- **delta_compare**: Field(s) used to check if a record has been modified. By default, all non-key source fields are compared. An explicit list must cover every source field whose changes can affect your output or filters; ignored changes cannot trigger recomputation.

`read_storage` processes current files, even when the storage keeps older versions. When reading a versioned listing directly with `read_dataset`, keep `file.version` in any custom `delta_on` to distinguish historical revisions.

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

Use stable, unique, non-null delta keys. Repeated keys are unsupported and can process the same source record more than once; null keys can cause extra processing when source versions change.

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

When combining multiple delta sources, use the same result key for all of them and set `delta_unsafe=True` on each source. Different result keys are unsupported.

With `delta_unsafe=True`, these combinations can still process only changed records:

- `merge` or `subtract` with an input that has no delta source, such as a lookup dataset, when the delta chain is on the left side and the merge is not `full`.
- Multiple delta sources that share a result key and change the same keys, combined with `union` or merged on that key.

## When a full recompute is needed

Some operations need the full current source, so DataChain may rerun UDFs even for unchanged records. These include aggregation, full merges, independently changing delta sources, and changes to a saved lookup dataset. Window functions, `limit`, and `offset` also require a full recompute and can be used without `delta_unsafe=True`.

Inline inputs such as `read_values`, single-file readers such as `read_csv`, and `persist()` results are treated as fixed; changes to them do not trigger recomputation. Save an input as a dataset first if it can change.

## Upgrading existing delta datasets

**Upgrading does not automatically clean stale results saved by older delta updates.** Existing datasets may still contain rows for deleted files or old revisions of overwritten files. If the source version has already been recorded as processed, another delta run can leave those rows untouched.

To bring an existing result into sync:

1. Rerun the same pipeline once with `delta=False`. For `read_storage`, also keep `update=True` to refresh the source listing. Keep the same `.save(...)` name.
2. After that run succeeds, restore `delta=True` for subsequent runs.

**This is a full-source rebuild and can rerun expensive UDFs over every current source record.** It saves a new result version; previous versions remain unchanged. Newly created results do not need this rebuild.
