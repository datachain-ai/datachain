# Jobs

DataChain Studio allows you to run DataChain scripts directly in the cloud, processing data from connected storage. Write your code in the Studio editor and execute it with configurable compute resources.

## Key Features

- **[Create and Run](create-and-run.md)** - Write and execute DataChain scripts in Studio
- **[Monitor Jobs](monitor-jobs.md)** - Track job progress, view logs, and analyze results

## How Jobs Work

Jobs in DataChain Studio let you execute data processing workflows:

### Direct Script Execution
- Write DataChain code directly in the Studio editor
- Execute scripts against connected storage (S3, GCS, Azure)
- Results saved automatically

### Configurable Compute
- Select Python version for your environment
- Configure number of workers for parallel processing
- Set job priority for queue management
- Specify custom requirements and environment variables

## Job Lifecycle

### 1. Write Script
Write your DataChain code in the Studio editor using connected storage sources.

### 2. Configure Settings
Set Python version, workers, priority, and any required dependencies or environment variables.

### 3. Execute
Submit the job to run on Studio's compute infrastructure with your specified configuration.

### 4. Monitor
View real-time logs, progress, and results as your job executes.

### 5. Review Results
Access processed data through the Studio interface, with datasets saved automatically.

## Job States

- **QUEUED**: Job is waiting in the execution queue
- **INIT**: Job environment is being initialized
- **RUNNING**: Job is actively processing data
- **COMPLETE**: Job finished successfully
- **FAILED**: Job encountered an error
- **CANCELED**: Job was stopped by user

## Getting Started

1. Connect your storage sources (S3, GCS, Azure)
2. Write DataChain code in the Studio editor
3. Configure job settings (Python version, workers, priority)
4. Run your job and monitor execution
5. View results in the data table

## Next Steps

- Learn how to [create and run jobs](create-and-run.md)
- Explore [job monitoring capabilities](monitor-jobs.md)
- Set up [webhooks](../../webhooks.md) for job notifications
- Configure [team collaboration](../team-collaboration.md) for shared access
