# Retry Processing

The retry functionality in DataChain allows for selective reprocessing of records that either have errors or are missing from the final result dataset. This powerful feature works alongside the existing `delta` functionality, providing both incremental processing of new data and reprocessing of problematic records.

## Overview

When working with large datasets, two common scenarios arise:

1. **Processing new data incrementally**: You only want to process new data that has been added since the last run (handled by the `delta=True` functionality)
2. **Reprocessing problematic records**: Some records may have failed processing or produced errors that need to be fixed and reprocessed

The retry functionality addresses the second scenario while still supporting the first.

## How It Works

The retry functionality allows you to:

1. **Identify error records**: Records in the result dataset with a non-NULL value in a specified error field
2. **Identify missing records**: Records that exist in the source dataset but are missing from the result dataset
3. **Reprocess only these records**: Applying your full processing chain only to these records that need attention

## Usage

`retry` can be enabled by specifying `delta_retry`. It is enabled only when `delta` is enabled.

```python
import datachain as dc
from datachain import C

chain = (
    dc.read_storage(
        "path/to/data/",
        # Enable delta processing to handle only new files (and retries in this case)
        delta=True,
        # Field(s) that uniquely identify records in the source dataset
        delta_on="id",
        # Controls which records to reprocess:
        # - String: field name indicating errors when not empty
        # - True: retry missing records from result dataset
        # - False/None: no retry processing
        delta_retry="error",
    )
    .map(result=process_function)  # Your processing function
    .save(name="processed_data")  # Save results
)
```

## Example: Processing Files with Error Handling

```python
import datachain as dc
from datachain import C


def process_file(file):
    """Process a file - may occasionally fail with an error."""
    try:
        # Your processing logic here
        content = file.read_text()
        result = analyze_content(content)
        return {
            "content": content,
            "result": result,
            "error": "",  # No error
        }
    except Exception as e:
        # Log the error and return it in the result
        return {
            "content": "",
            "result": "",
            "error": str(e),  # Store the error message
        }


# Process files with with retry functionality
chain = (
    dc.read_storage(
        "data/",
        delta=True,  # Process only new files
        delta_on="file.path",  # Files are identified by their paths
        delta_retry="error",  # Retry records with errors in "error" field
    )
    .map(result=process_file)
    .save(name="processed_files")
)

# Show records with errors that will be retried next time
error_records = chain.filter(C("error") != None)
if not error_records.empty:
    print("Records with errors that will be retried:")
    error_records.show()
```

## Combining Delta and Retry

The real power comes when combining both delta and retry processing:

1. **Delta processing**: Only process new or modified records
2. **Retry processing**: Reprocess any records that had errors previously

When both are enabled, DataChain will:

1. First identify records that need to be retried (based on errors or missing records)
2. Then apply delta processing to that subset
3. Process only the resulting records, avoiding unnecessary reprocessing of unchanged data

This provides the most efficient way to maintain a dataset that is always up-to-date and free of processing errors.
