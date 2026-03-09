# pipeline remove-job

Remove specific job from a pipeline in Studio

## Synopsis

```usage
usage: datachain pipeline remove-job [-h] [-v] [-q] [-t TEAM] name job_id
```

## Description

This command removes a specific job from a pipeline in Studio. This operation can only be performed on PAUSED pipelines, and only PENDING jobs (jobs that have not yet started running) can be removed. You cannot remove jobs from pipelines that are actively running or already completed.

When you remove a job from the pipeline, the pipeline graph is automatically updated to reflect the change.


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
* You can run `datachain pipeline status` to see the list of jobs and their IDs.
* You can run `datachain pipeline pause` to pause a running pipeline so you can remove jobs from it.
