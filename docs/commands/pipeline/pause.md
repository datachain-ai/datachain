# pipeline pause

Pause the running pipeline in Studio

## Synopsis

```usage
usage: datachain pipeline pause [-h] [-v] [-q] [-t TEAM] name
```

## Description

This command pauses a running pipeline in Studio. When a pipeline is paused, any currently running jobs will continue to completion, but new jobs will not be started even when their dependencies are met.

## Arguments

* `name` - Name of the pipeline

## Options

* `-t TEAM, --team TEAM` - Team the pipeline belongs to.(default: from config)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet


## Example

### Command

```bash
datachain pipeline pause rathe-kyat
```
