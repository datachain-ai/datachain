# pipeline create

Create an update for dataset dependency in Studio

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

This command creates an pipeline for a dataset dependency. The pipeline is determined based on the dependency for the dataset in Studio and is created accordingly. The dataset name can be a fully qualified name including the namespace and project, or a regular name, in which case the explicitly defined namespace and project will be used if they are set; otherwise, default values will be applied.

## Arguments

* `dataset` - Name of the dataset (can be a fully qualified name including the namespace and project or regular name)

## Options

* `-V VERSION, --version VERSION` - Version of the dataset (default: latest)
* `-r, --review` - Review the pipeline before creating. (Opens in paused state)
* `-n NAMESPACE, --namespace NAMESPACE` - Namespace of the dataset
* `-p PROJECT, --project PROJECT` - Project of the dataset
* `-t TEAM, --team TEAM` - Team to run job for (default: from config)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Run a create with fully qualified dataset name
```bash
datachain pipeline create "@amritghimire.default.final_result" --version "1.0.9"
```

2. Specify namespace, project separately
```bash
datachain pipeline create "final_result" --namespace "@amritghimire" --project "default" --version "1.0.9"
```

3. Select the latest dataset version
```bash
datachain pipeline create "@amritghimire.default.final_result"
```

## Notes
* If you passed review mode, the pipeline will be in a paused state in Studio. You can open Studio to review the pipeline and resume the pipeline after changes.
