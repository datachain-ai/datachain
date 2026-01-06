# job run

Run a job in Studio.

## Synopsis

```usage
usage: datachain job run [-h] [-v] [-q] [--team TEAM] [--env-file ENV_FILE]
                         [--env ENV [ENV ...]]
                         [--cluster CLUSTER] [--workers WORKERS]
                         [--files FILES [FILES ...]]
                         [--python-version PYTHON_VERSION]
                         [--repository REPOSITORY]
                         [--req-file REQ_FILE] [--req REQ [REQ ...]]
                         [--priority PRIORITY]
                         [--start-time START_TIME] [--cron CRON]
                         [--no-wait]
                         file
```

## Description

This command runs a job in Studio using the specified query file. You can configure various aspects of the job including environment variables, Python version, dependencies, and more. When using --start-time or --cron, the job is scheduled to run but won't start immediately. (can be seen in the Tasks tab in UI)

## Arguments

* `file` - Query file to run

## Options

* `--team TEAM` - Team to run job for (default: from config)
* `--env-file ENV_FILE` - File with environment variables for the job
* `--env ENV` - Environment variables in KEY=VALUE format
* `--cluster CLUSTER` - Compute cluster to run the job on
* `--credentials-name CREDENTIALS_NAME` - Name of the credentials to use for the job
* `--workers WORKERS` - Number of workers for the job
* `--files FILES` - Additional files to include in the job
* `--python-version PYTHON_VERSION` - Python version for the job (e.g., 3.10, 3.11, 3.12, 3.13)
* `--repository REPOSITORY` - Repository URL to clone before running the job
* `--req-file REQ_FILE` - Python requirements file
* `--req REQ` - Python package requirements
* `--priority PRIORITY` - Priority for the job in range 0-5. Lower value is higher priority (default: 5)
* `--start-time START_TIME` - Time to schedule the task in YYYY-MM-DDTHH:mm format or natural language.
* `--cron CRON` - Cron expression for the cron task.
* `--no-wait` - Do not wait for the job to finish.
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Run a basic job:
```bash
datachain job run query.py
```

2. Run a job with specific team and Python version:
```bash
datachain job run --team my-team --python-version 3.11 query.py
```

3. Run a job with environment variables and requirements:
```bash
datachain job run --env-file .env --req-file requirements.txt query.py
```

4. Run a job with multiple workers and additional files:
```bash
datachain job run --workers 4 --files utils.py config.json query.py
```

5. Run a job with inline environment variables and package requirements:
```bash
datachain job run --env API_KEY=123 --req pandas numpy query.py
```

6. Run a job with a repository (will be cloned in the job working directory):
```bash
datachain job run --repository https://github.com/datachain-ai/datachain query.py

# To specify a branch / revision:
datachain job run --repository https://github.com/datachain-ai/datachain@main query.py

# Git URLs are also supported:
datachain job run --repository git@github.com:datachain-ai/datachain.git@main query.py
```

7. Run a job with higher priority
```bash
datachain job run --priority 2 query.py
```

8. Run a job in a specific cluster
```bash
# Get the cluster id using following command
datachain job clusters
# Use the id  of an active clusters from above
datachain job run --cluster 1 query.py
```

9. Run a job with specific credentials
```bash
datachain job run --credentials-name my-aws-credentials query.py
```

10. Schedule a job to run once at a specific time
```bash
# Run job tomorrow at 3pm
datachain job run --start-time "tomorrow 3pm" query.py

# Run job in 2 hours
datachain job run --start-time "in 2 hours" query.py

# Run job on Monday at 9am
datachain job run --start-time "monday 9am" query.py

# Run job at a specific date and time
datachain job run --start-time "2024-01-15 14:30:00" query.py
```

11. Schedule a recurring job using cron expression
```bash
# Run job daily at midnight
datachain job run --cron "0 0 * * *" query.py

# Run job every Monday at 9am
datachain job run --cron "0 9 * * 1" query.py

# Run job every hour
datachain job run --cron "0 * * * *" query.py

# Run job every month
datachain job run --cron "@monthly" query.py
```

12. Schedule a recurring job with a start time
```bash
# Start the cron job after tomorrow 3pm
datachain job run --start-time "tomorrow 3pm" --cron "0 0 * * *" query.py
```

13. Start the job and do not wait for the job to complete
```bash
# Do not follow or tail the logs from Studio.
datachain job run query.py --no-wait
```

## Notes

* Closing the logs command (e.g., with Ctrl+C) will only stop displaying the logs but will not cancel the job execution
* To cancel a running job, use the `datachain job cancel` command
* The job will continue running in Studio even after you stop viewing the logs
* You can get the list of compute clusters using `datachain job clusters` command.
* When using `--start-time` or `--cron` options, the job is scheduled as a task and will not show logs immediately. The job will be executed according to the schedule.
* The `--start-time` option supports natural language parsing using the [dateparser](https://dateparser.readthedocs.io/en/latest/) library, allowing flexible time expressions like "tomorrow 3pm", "in 2 hours", "monday 9am", etc.
* Cron expressions follow the standard format: minute hour day-of-month month day-of-week (e.g., "0 0 * * *" for daily at midnight) or Vixie cron-style “@” keyword expressions.
* Following options for Vixie cron-style expressions are supported:
    * @midnight
    * @hourly
    * @daily
    * @weekly
    * @monthly
    * @yearly
    * @annually
