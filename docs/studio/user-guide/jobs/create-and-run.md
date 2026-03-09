# Create and Run Jobs

Write and execute DataChain scripts directly in Studio to process data from your connected storage.

## Prerequisites

- Connected storage (S3, GCS, Azure Blob Storage, or other supported storage)
- Storage credentials configured in account settings
- Access to DataChain Studio workspace

## Writing Your Script

### 1. Access the Editor

In DataChain Studio, open the code editor through `Data` tab in the topbar to write your DataChain script. You'll see connected storages listed in the left sidebar.

### 2. Write DataChain Code

Write your data processing script using DataChain operations:

```python
import datachain as dc

# Process data from connected storage
dc.read_storage("gs://datachain-demo").save("datachain-demo")
```

### Basic Operations Example

```python
from datachain import DataChain

# Read from storage and process
dc = (
    DataChain.from_storage("s3://my-bucket/images/")
    .filter(lambda file: file.size > 1000)
    .map(lambda file: {"path": file.path, "size": file.size})
    .save("processed_images")
)

print(f"Processed {len(dc)} files")
```

### Working with Multiple Storages

```python
from datachain import DataChain

# Access different connected storages
source_data = DataChain.from_storage("s3://source-bucket/data/")
reference_data = DataChain.from_storage("gs://reference-bucket/metadata/")

# Process and combine
result = source_data.join(reference_data, on="id").save("combined_data")
```

## Configuring Run Settings

Click the run settings button to configure your job execution parameters.

### Python Version

Select the Python version for your job environment:
- Python 3.12 (recommended)
- Python 3.11
- Python 3.10

### Workers

Set the number of parallel workers for data processing:
- **1 worker**: Sequential processing (default)
- **2-10 workers**: Parallel processing for larger datasets
- More workers increase throughput but consume more resources

### Priority

Set job queue priority:
- **1-10**: Higher numbers = higher priority in the job queue
- **5**: Default priority
- Use higher priority for time-sensitive jobs

### Requirements.txt

Specify additional Python packages needed for your job:

```text
pandas==2.0.0
pillow>=9.0.0
requests
torch==2.0.1
```

### Environment Variables

Set environment variables for your script:

```text
AWS_REGION=us-east-1
BATCH_SIZE=1000
LOG_LEVEL=INFO
MODEL_VERSION=v2.1
```

### Override Credentials

By default, jobs use team credentials for storage access. You can override with:
- **Using team defaults**: Use configured team credentials
- **Custom credentials**: Select specific credentials for this job

### Attached Files

Upload additional files needed by your job (currently disabled in standard plan).

## Running Your Job

### Submit for Execution

1. Write your DataChain script in the editor
2. Click the run settings button (gear icon)
3. Configure Python version, workers, and priority
4. Add any required packages or environment variables
5. Click `Apply settings`
6. Click the run button to execute

Your job will be queued and executed with the specified configuration.

### Execution Process

1. **QUEUED**: Job enters the execution queue based on priority
2. **INIT**: Python environment is set up with specified version and requirements
3. **RUNNING**: Your DataChain script executes with configured workers
4. **COMPLETE**: Results are saved and available in the data table

## Viewing Results

After job completion:

### Data Table

Results appear in the data table below your script:
- View processed files and their properties
- Sort and filter results
- Examine file paths, sizes, and metadata
- Download data if needed

### Saved Datasets

Access saved datasets by name:
```python
# Later access to saved results
saved_dc = DataChain.from_dataset("processed_images")
```

## Common Patterns

### Processing Images

```python
from datachain import DataChain

dc = (
    DataChain.from_storage("s3://images/")
    .filter(lambda file: file.path.endswith(('.jpg', '.png')))
    .map(lambda file: {
        "path": file.path,
        "size": file.size,
        "extension": file.path.split('.')[-1]
    })
    .save("image_catalog")
)
```

### Data Quality Checks

```python
from datachain import DataChain

dc = (
    DataChain.from_storage("gs://data-lake/")
    .filter(lambda file: file.size > 0)  # Non-empty files
    .filter(lambda file: file.modified_at > "2024-01-01")  # Recent files
    .save("validated_data")
)
```

### Batch Processing

```python
from datachain import DataChain

# Process data in batches
for batch in DataChain.from_storage("s3://large-dataset/").batch(1000):
    processed = batch.map(transform_function)
    print(f"Processed batch of {len(processed)} files")
```

## Troubleshooting

### Common Issues

#### Package Import Errors
- Add missing packages to `requirements.txt`
- Verify package names and versions are correct
- Check for compatible package versions

#### Storage Access Errors
- Verify storage credentials are configured
- Check storage paths are correct and accessible
- Ensure team has necessary permissions

#### Memory Errors
- Reduce batch size in your processing
- Increase number of workers to distribute load
- Process data in smaller chunks

#### Timeout Errors
- Optimize your processing code
- Reduce amount of data being processed
- Consider splitting into multiple jobs

### Debugging Tips

1. **Start Simple**: Test with small data samples first
2. **Check Logs**: Review job logs in the monitor tab
3. **Verify Storage**: Ensure connected storage is accessible
4. **Test Locally**: Test scripts locally when possible
5. **Use Print Statements**: Add logging to track progress

## Next Steps

- Learn how to [monitor running jobs](monitor-jobs.md)
- Set up [team collaboration](../team-collaboration.md)
- Explore [DataChain operations](../../../references/datachain.md)
