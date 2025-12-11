# pipeline remove-job

Remove specific job from a paused pipeline in Studio

## Synopsis

```usage
usage: datachain pipeline remove-job [-h] [-v] [-q] [-t TEAM] name job_id
```

## Description

This commands allows users to drop a specific job from the pipeline before it runs. This operation is restricted to PAUSED pipeline and PENDING jobs, that have not yet started running, only. You can't modify a pipeline that is actively running or already completed.
When you remove a job from the pipeline, the graph of the pipeline is automatically repaired.


## Arguments

* `name` - Name of the pipeline
* `job_id` - ID of the job to remove


## Options

* `-t TEAM, --team TEAM` - Team the pipeline belongs to (default: from config)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Example

### Command

```bash
datachain pipeline remove-job burry-user faa8ef11-ad9d-4a83-8b1d-b41fecc6b0e9
```

## Notes
* You can run `datachain pipeline status` to see the list of jobs and its ids to pass here.
* You can run `datachain pipeline pause` to pause the pipeline if it is running to remove the jobs from the pipeline.
