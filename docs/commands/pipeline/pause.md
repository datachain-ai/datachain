# pipeline pause

Pause the running pipeline in Studio

## Synopsis

```usage
usage: datachain pipeline pause [-h] [-v] [-q] [-t TEAM] name
```

## Description

This command pauses the currently running pipeline in Studio. When a pipeline is paused, any job that is part of the pipeline that is already running will continue running, but new jobs are not triggered once the jobs are completed.

## Argument


## Arguments

* `name` - Name of the pipeline

## Options

* `-t TEAM, --team TEAM` - Team of the pipeline (default: from config)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.


## Example

### Command

```bash
datachain pipeline pause rathe-kyat
```
