# job ls

List jobs in Studio.

## Synopsis

```usage
usage: datachain job ls [-h] [-v] [-q] [--status STATUS] [--team TEAM] [--limit LIMIT] [-e]
```

## Description

This command lists jobs in Studio. You can filter jobs by their status, specify a team, and limit the number of jobs returned. By default, it shows the 20 most recent jobs.

Every job shows its ID, name, status, creation time and author. `--extended` adds
the compute cluster it ran on and a breakdown of its stages - see
[Extended output](#extended-output).


## Options

* `--status STATUS` - Status to filter jobs by
* `--team TEAM` - Team to list jobs for (default: from config)
* `--limit LIMIT` - Limit the number of jobs returned (default: 20)
* `-e`, `--extended` - Show the compute cluster and per-stage timings ([details](#extended-output))
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Extended output

`--extended` adds two columns, and asks Studio for the per-stage timings that fill
the second:

| Column | Meaning |
|--------|---------|
| `Cluster` | Name of the compute cluster the job ran on. [`datachain job clusters`](clusters.md) describes that cluster - its region, machine and capacity |
| `Stages` | One line per stage, as `<label>: <duration>` |

A job's stages are `waiting` (queued, before anything was provisioned),
`requesting_workers`, `preparation`, `virtualenv` (installing dependencies),
`downloading_files`, `dw_wake_up` (waking the data warehouse) and `running_query`
(the script itself). Which of them a job has depends on when it ran - they were
added over time - and on how far it got, so a missing stage means unknown, not
zero.

```
+--------------------------------------+----------+----------+-----------------+------------------------------+
| ID                                   | Name     | Status   | ...   | Cluster  | Stages                       |
+======================================+==========+==========+=================+==============================+
| 0502eef6-a32e-45fa-8e3b-d20ec0abbcf0 | daily    | COMPLETE | ...   | prod     | Waiting in queue: 4s         |
|                                      |          |          |       |          | Downloading files: 1h 5m     |
|                                      |          |          |       |          | Installing dependencies: 2m  |
|                                      |          |          |       |          | Running query: 9m 12s        |
+--------------------------------------+----------+----------+-----------------+------------------------------+
```

A stage reads `running` while it is still going, and `-` when its length was never
recorded - it never started, or the job stopped without closing it.

Comparing `Waiting in queue` against `Running query` is how you tell a slow job
from a job that sat waiting for a worker.

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
* `--extended` costs an extra round trip for the stage data, so plain `job ls` stays
  the quicker way to check what is running
* Studio records more per job than this table shows - the exit code, error message,
  Python version, requirements, worker count, metrics and the cluster's UUID among
  them. [`datachain job logs`](logs.md) shows a job's output, and the
  `StudioClient.get_jobs()` API returns all of it
