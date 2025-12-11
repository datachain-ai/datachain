# pipeline resume

Resume a paused pipeline in Studio

## Synopsis

```usage
usage: datachain pipeline resume [-h] [-v] [-q] [-t TEAM] name
```

## Description

This command resumes a paused pipeline in Studio. When a pipeline is resumed, it identifies jobs whose dependencies are met and that haven't run yet, then starts those jobs.


## Arguments

* `name` - Name of the pipeline

## Options

* `-t TEAM, --team TEAM` - Team of the pipeline (default: from config)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet


## Example

### Command

```bash
datachain pipeline resume rathe-kyat
```
