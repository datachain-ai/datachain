# job ls

List jobs in Studio.

## Synopsis

```usage
usage: datachain job ls [-h] [-v] [-q] [--status STATUS] [--team TEAM] [--limit LIMIT] [-e]
```

## Description

This command lists jobs in Studio. You can filter jobs by their status, specify a team, and limit the number of jobs returned. By default, it shows the 20 most recent jobs.


## Options

* `--status STATUS` - Status to filter jobs by
* `--team TEAM` - Team to list jobs for (default: from config)
* `--limit LIMIT` - Limit the number of jobs returned (default: 20)
* `-e`, `--extended` - Show extra job details, such as the compute cluster
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Status options

You will be able to filter the job with following status:

* `CREATED` - Job has been created but not yet scheduled
* `SCHEDULED` - Job is scheduled to run at a future time
* `QUEUED` - Job is in the queue waiting to be executed
* `INIT` - Job is initializing and preparing to run
* `RUNNING` - Job is currently executing
* `COMPLETE` - Job has finished successfully
* `FAILED` - Job has failed during execution
* `CANCELING_SCHEDULED` - A scheduled job is being canceled
* `CANCELING` - A running job is being canceled
* `CANCELED` - Job has been canceled
* `ACTIVE` - Job is in active state.
* `INACTIVE` - Job is in inactive state.

Note: The following statuses are considered active jobs:

* `CREATED`
* `SCHEDULED`
* `QUEUED`
* `INIT`
* `RUNNING`
* `CANCELING_SCHEDULED`
* `CANCELING`


## Examples

1. List all jobs (default limit of 20):
```bash
datachain job ls
```

2. List jobs for a specific team:
```bash
datachain job ls --team my-team
```

3. List jobs with a specific status:
```bash
datachain job ls --status complete
```

4. List more jobs by increasing the limit:
```bash
datachain job ls --limit 50
```

5. List jobs with verbose output:
```bash
datachain job ls -v
```

6. List jobs with extra details, including the compute cluster they ran on:
```bash
datachain job ls --extended
```

## Notes

* The default limit of 20 jobs helps manage the output size and performance
* Jobs are typically listed in reverse chronological order (newest first)
* Use the `--status` filter to find jobs in specific states (e.g., running, completed, failed)
