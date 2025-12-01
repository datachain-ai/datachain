# Monitor Jobs

Track your DataChain job execution in real-time with Studio's monitoring interface.

## Job Status Bar

At the top of the Studio interface, you'll see the current job status:

### Status Display
- **Workers**: Shows active/total workers (e.g., "2 / 10 workers busy")
- **Tasks**: Displays running tasks count (e.g., "2 tasks")
- **Execution Time**: Shows how long the job has been running

### Job States
- 🟡 **QUEUED**: Waiting in the execution queue
- 🔵 **INIT**: Setting up environment and dependencies
- 🟢 **RUNNING**: Actively processing data
- ✅ **COMPLETE**: Successfully finished
- ❌ **FAILED**: Encountered an error
- ⚫ **CANCELED**: Stopped by user

## Real-time Logs

### Logs Tab

Click the "Logs" tab to view real-time execution output:

```
Running job 7897833d-080c-464f-978b-59316886099a in cluster 'default'
Using cached virtualenv

Listing gs://datachain-demo: 269981 objects [00:16, 16568.50 objects/s]
```

### Log Information
- **Job ID and Cluster**: Shows which cluster is running your job
- **Environment Status**: Indicates if using cached virtualenv or installing fresh
- **Timestamped Entries**: Real-time progress updates
- **Error Messages**: Stack traces for debugging failures
- **Data Statistics**: Files processed and rows handled
- **Performance Metrics**: Execution timing information

## Dependencies Tab

View data lineage and dataset dependencies:

### Dataset Lineage
The Dependencies tab shows a visual graph of data flow:

- **Output Dataset**: Your saved dataset (e.g., `@amritghimire.default.datachain-demo@v1.0.0`)
  - Shows version number
  - Displays creator and timestamp
  - Indicates verification status

- **Source Storage**: Connected storage sources (e.g., `gs://datachain-demo/`)
  - Shows storage path
  - Displays who added the storage
  - Links to original data source

- **Data Flow**: Visual arrows showing how data flows from source to output

This helps you:
- Understand data lineage and provenance
- Track which storages were used
- Verify dataset versions
- Debug data pipeline issues

## Diagnostics Tab

View detailed job execution timeline and diagnostics:

### Job Summary

At the top, see the overall job status:

```
✓ Job complete: 00:07:30
```

- **Execution Time**: Total duration (hours:minutes:seconds)
- **Status Icon**: Checkmark for success, X for failure

### Execution Details

Key job information:

- **Started**: Start timestamp with timezone (e.g., `2025-10-18 07:48:27 GMT+5:45`)
- **Finished**: Completion timestamp
- **Compute Cluster**: Which cluster ran the job (e.g., `default`)
- **Job ID**: Unique identifier for the job (e.g., `7897833d-080c-464f-978b-59316886099a`)

### Execution Timeline

Detailed breakdown of each execution phase:

```
✓ Waiting in queue          2s
✓ Starting a worker         15s
✓ Initializing job           3s
✓ Installing dependencies    0s
✓ Waking up data warehouse   29s
✓ Running query           2m 35s
```

Each phase shows:
- **Checkmark**: Indicates successful completion
- **Phase Name**: What the system was doing
- **Duration**: Time spent in that phase

### Understanding Phase Durations

- **Waiting in queue**: Time before resources became available
- **Starting a worker**: Worker initialization and allocation
- **Initializing job**: Setting up job environment
- **Installing dependencies**: Installing Python packages from requirements.txt
- **Waking up data warehouse**: Activating data processing infrastructure
- **Running query**: Actual data processing time

This breakdown helps identify bottlenecks and optimize job performance.

## Data Results

### Data Tab

View processed results in the data table:

- **Row Count**: Shows processed rows (e.g., "20 of 270,345 rows")
- **Columns**: File paths, sizes, and metadata
- **Sorting**: Click column headers to sort
- **Filtering**: Use filters to find specific data
- **Pagination**: Navigate through large result sets

### Files Tab

Browse processed files:

- File paths and names
- File sizes and types
- Metadata and attributes
- Quick preview capabilities

## Job Controls

### Stop Job

Click the stop button to cancel a running job:
- Job will transition to CANCELING state
- Current operations complete gracefully
- Resources are cleaned up

## Monitoring Job Progress

### Progress Indicators

Track your job execution:

- **Rows Processed**: Current progress through dataset
- **Processing Rate**: Files or records per second
- **Time Elapsed**: How long the job has been running
- **Estimated Completion**: Projected finish time (when available)

### Resource Usage

Monitor resource consumption:

- **Workers Active**: Number of parallel workers processing data
- **Memory Usage**: RAM consumption during processing
- **Storage I/O**: Data read/write operations

## Troubleshooting

### Common Issues

#### Job Stuck in QUEUED
- Check worker availability in status bar
- Verify team hasn't exceeded resource quotas
- Review job priority settings

#### INIT Failures
- Check requirements.txt for invalid packages
- Verify package versions are compatible
- Review error messages in Logs tab

#### RUNNING Failures
- Examine stack trace in Logs tab
- Verify storage credentials are valid
- Check storage paths are accessible
- Review error messages for specific issues

#### Storage Access Errors
- Verify credentials in account settings
- Check storage bucket permissions
- Ensure storage path exists
- Test storage connection separately

### Debugging Workflow

1. **Check Diagnostics Tab**: Review job completion status and execution timeline
2. **Identify Bottleneck**: Look for phases with unusually long durations:
   - Long "Starting a worker" time → Check cluster availability
   - Long "Installing dependencies" → Review requirements.txt
   - Long "Waking up data warehouse" → Contact support
   - Long "Running query" → Optimize DataChain code
3. **Open Logs Tab**: Look for error messages and stack traces
4. **Check Dependencies Tab**: Verify data sources are connected correctly
5. **Test with Subset**: Try with smaller data sample
6. **Contact Support**: Provide Job ID from Diagnostics tab

## Performance Optimization

### Analyzing Execution Timeline

Use the Diagnostics tab to identify optimization opportunities:

#### Quick Queue Times (< 2m)
✓ Good - Your jobs are getting resources quickly

#### Long Worker Start (> 5m)
Possible causes:
- High cluster demand
- Cold start of compute resources

#### Slow Dependency Installation (> 3m)
Optimization tips:
- Pin package versions in requirements.txt
- Minimize number of dependencies

#### Extended Data Warehouse Wake (> 2m)
This is infrastructure initialization. If consistently slow:
- Keep warehouse warm with regular jobs
- Contact support for dedicated warehouse

#### Long Running Query Time
Optimize your DataChain code:
- Filter data early to reduce volume
- Use efficient DataChain operations
- Increase worker count for large datasets
- Batch operations appropriately

### Monitoring Best Practices

- **Compare Job Runs**: Check Diagnostics across multiple runs to spot trends
- **Track Phase Durations**: Note which phases take longest
- **Use Job ID**: Reference Job ID when reporting issues
- **Review Logs**: Check for warnings about performance

## Next Steps

- Set up [webhook notifications](../../webhooks.md) for job status updates
- Configure [team collaboration](../team-collaboration.md) for shared job access
- Explore [DataChain operations](../../../references/datachain.md) for optimization
- Review [account settings](../account-management.md) for credentials
