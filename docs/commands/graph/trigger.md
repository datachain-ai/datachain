# graph trigger

Trigger an update for dataset dependency in Studio

## Synopsis

```usage
usage: datachain graph trigger [-h] [-v] [-q]
                               [-V VERSION]
                               [-r] [-n NAMESPACE]
                               [-p PROJECT]
                               [-t TEAM]
                               dataset
```

## Description

This command triggers an execution graph for a dataset dependency. Execution graph will figure out based on the dependency for the dataset in studio and triggered accordingly. The dataset name, which can be a fully qualified name including the namespace and project. Alternatively, it can be a regular name, in which case the explicitly defined namespace and project will be used if they are set; otherwise, default values will be applied.

## Arguments

* `dataset` - Name of the dataset (can be a fully qualified name including the namespace and project or regular name)

## Options

* `-V VERSION, --version VERSION` - Version of the dataset (default: latest)
* `-r, --review` - Review the execution graph before triggering. (Opens in paused state)
* `-n NAMESPACE, --namespace NAMESPACE` - Namespace of the dataset
* `-p PROJECT, --project PROJECT` - Project of the dataset
* `--team TEAM` - Team to run job for (default: from config)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Run a trigger with fully qualified dataset name
```bash
datachain graph trigger "@amritghimire.default.final_result" --version "1.0.9"
```

2. Specify namespace, project separately
```bash
datachain graph trigger "final_result" --namespace "@amritghimire" --project "default" --version "1.0.9"
```

3. Select the latest dataset version
```bash
datachain graph trigger "@amritghimire.default.final_result"
```

## Notes
* If you passed review mode, the execution graph will be in paused state in Studio. You can open Studio to review the graph and resume the graph after changes.
