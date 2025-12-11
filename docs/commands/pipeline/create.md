# pipeline create

Create a pipeline to update a dataset in Studio.

## Synopsis

```usage
usage: datachain pipeline create [-h] [-v] [-q]
                               [-V VERSION]
                               [-r] [-n NAMESPACE]
                               [-p PROJECT]
                               [-t TEAM]
                               dataset
```

## Description

This command creates a pipeline in Studio that will update the specified dataset. The pipeline automatically includes all necessary jobs to update the dataset based on its dependencies. If no version is specified, the latest version of the dataset is used.

The dataset name can be provided in fully qualified format (e.g., `@namespace.project.name`) or as a short name. When using a short name, you can optionally specify the namespace and project separately using the `--namespace` and `--project` options. If not specified, default values from your configuration will be used.

## Arguments

* `dataset` - Name of the dataset. Can be a fully qualified name (e.g., `@namespace.project.name`) or a short name.

## Options

* `-V VERSION, --version VERSION` - Dataset version to create the pipeline for (default: latest version)
* `-r, --review` - Create the pipeline in paused state for review before execution
* `-n NAMESPACE, --namespace NAMESPACE` - Dataset namespace (only needed when using short dataset names)
* `-p PROJECT, --project PROJECT` - Dataset project (only needed when using short dataset names)
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

4. Create a pipeline in review mode (paused for review before execution):
```bash
datachain pipeline create "final_result" --review
```

## Notes

* When using `--review`, the pipeline is created in a paused state in Studio. You can open Studio to review the pipeline configuration and resume it when ready.
* The pipeline automatically includes all jobs needed to update the dataset based on its dependencies.
