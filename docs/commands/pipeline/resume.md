# pipeline resume

Resume the running pipeline in Studio

## Synopsis

```usage
usage: datachain pipeline resume [-h] [-v] [-q] [-t TEAM] name
```

## Description

This command resumes the currently paused pipeline in Studio. When a pipeline is resumed, it checks for the jobs that has all its dependencies met and has not run yet. Such jobs will start running.


## Argument


## Arguments

* `name` - Name of the pipeline

## Options

* `-t TEAM, --team TEAM` - Team to resume the pipeline (default: from config)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.


## Example

### Command

```bash
datachain pipeline resume rathe-kyat
```
