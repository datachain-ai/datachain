# Delta Processing

Delta processing allows you to optimize the creation of new dataset versions by calculating and processing only the differences (deltas) between dataset versions. This is especially useful when working with large datasets that change incrementally over time.

## Overview

When working with large datasets that receive regular updates, reprocessing the entire dataset each time is inefficient. The `delta=True` flag in DataChain enables incremental processing, where only new or changed records are processed.

## How Delta Processing Works

When delta processing is enabled:

1. DataChain calculates the difference between the latest version of your source dataset and the version used to create the most recent version of your result dataset
2. Only the records that are new or modified in the source dataset are processed
3. These processed records are then merged with the latest version of your result dataset

This optimization significantly reduces processing time and resource usage for incremental updates.

## Usage

Delta processing can be enabled through the `delta` parameter when using `read_storage()` or `read_dataset()`:

```python
import datachain as dc

# Process only new or changed files
chain = (
    dc.read_storage(
        "data/",
        delta=True,                    # Enable delta processing
        delta_on="file.path",          # Field that uniquely identifies records
        delta_compare="file.mtime"     # Field to check for changes
    )
    .map(result=process_function)
    .save(name="processed_data")
)
```

## Parameters

- **delta**: Boolean flag to enable delta processing
- **delta_on**: Field(s) that uniquely identify rows in the source dataset
- **delta_result_on**: Field(s) in the resulting dataset that correspond to `delta_on` fields (if they have different names)
- **delta_compare**: Field(s) used to check if a record has been modified

## Example: Processing New Files Only

```python
import datachain as dc
import time

def process_file(file):
    """Process a file and return results."""
    content = file.read_text()
    # Simulate processing time
    time.sleep(0.1)
    return {
        "content": content,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# Process only new or modified files
chain = (
    dc.read_storage(
        "data/",
        update=True,                   # Update the storage index
        delta=True,                    # Process only new files
        delta_on="file.path"           # Files are identified by their paths
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

These methods are restricted because they may produce **unexpected results** when used with delta processing. Delta runs the chain only on a subset of rows (new and changed records), while methods like `distinct`, `agg`, or `group_by` are designed to operate on the entire dataset.

Similarly, combining delta with methods like `merge`, `union`, `subtract`, `diff`, or `file_diff` may produce inconsistent results because the operation is being applied to replayed delta inputs rather than to the full logical datasets.

If you still need to use these methods together with delta, you can override this restriction by setting the additional flag:

```python
delta_unsafe=True
```

If more than one delta-enabled source participates in the same composed query, set `delta_unsafe=True` on every participating delta source.

`delta_unsafe=True` is an advanced option. Use it only when you know the participating delta sources are updated in a consistent way and the result of replaying only the changed rows will still match the result of recomputing the full query.
