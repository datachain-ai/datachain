# pipeline create

Create a pipeline to update a dataset in Studio.

## Synopsis

```usage
usage: datachain pipeline create [-h] [-v] [-q]
                               [-V VERSION]
                               [-t TEAM]
                               dataset
```

## Description

This command creates a pipeline in Studio that will update the specified dataset. The pipeline automatically includes all necessary jobs to update the dataset based on its dependencies. If no version is specified, the latest version of the dataset is used.

The pipeline is created in paused state. Use `datachain pipeline resume` to start pipeline execution.

The dataset name can be provided in fully qualified format (e.g., `@namespace.project.name`) or as a short name. If using a short name, Studio uses the default project and namespace.

## Arguments

* `dataset` - Name of the dataset. Can be a fully qualified name (e.g., `@namespace.project.name`) or a short name.

## Options

* `-V VERSION, --version VERSION` - Dataset version to create the pipeline for (default: latest version)
* `-t TEAM, --team TEAM` - Team to create the pipeline for (default: from config)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Examples

1. Create a pipeline for a dataset using a fully qualified name:
```bash
datachain pipeline create "@amritghimire.default.final_result" --version "1.0.9"
```

2. Create a pipeline using a short dataset name:
```bash
datachain pipeline create "final_result" --version "1.0.9"
```

3. Create a pipeline for the latest version of a dataset:
```bash
datachain pipeline create "@amritghimire.default.final_result"
```
