# pipeline create

Create a pipeline to update datasets in Studio.

## Synopsis

```usage
usage: datachain pipeline create [-h] [-v] [-q]
                               [-t TEAM]
                               dataset [dataset ...]
```

## Description

Creates a pipeline in Studio to update the specified datasets. The pipeline automatically includes all necessary jobs to update the datasets based on their dependencies.

Each dataset name can optionally include a version suffix (e.g., `dataset@1.0.9`). If no version is specified, the latest version is used.

The pipeline is created in paused state for review. Use `datachain pipeline resume` to start execution.

Dataset names can be provided in fully qualified format (e.g., `@namespace.project.name`) or as a short name. Short names use the default project and namespace from Studio.

## Arguments

* `dataset [dataset ...]` - Dataset name(s). Can be fully qualified (e.g., `@namespace.project.name`) or short names. Optionally include version suffix: `name@version`. Multiple datasets can be specified.

## Options

* `-t TEAM, --team TEAM` - Team to create the pipeline for (default: from config)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Examples

1. Create a pipeline for a single dataset with a specific version:
```bash
datachain pipeline create "@amritghimire.default.final_result@1.0.9"
```

2. Create a pipeline for multiple datasets:
```bash
datachain pipeline create "@amritghimire.default.final_result@1.0.9" "final_result_new" "final_result_updated@1.0.9"
```
This creates a pipeline that updates:
- Version `1.0.9` of `@amritghimire.default.final_result`
- Latest version of `final_result_new` (using default namespace and project)
- Version `1.0.9` of `final_result_updated` (using default namespace and project)

3. Create a pipeline for a dataset using the latest version:
```bash
datachain pipeline create "@amritghimire.default.final_result"
```
