# pipeline status

Get the status of a pipeline from Studio

## Synopsis

```usage
usage: datachain pipeline status [-h] [-v] [-q] [-t TEAM] name
```

## Description

This command fetches the latest status of a pipeline along with the status of its jobs from Studio.

## Arguments

* `name` - Name of the pipeline

## Options

* `-t TEAM, --team TEAM` - Team to run job for (default: from config)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

#### Command

```bash
datachain pipeline status rusty-vise
```

#### Sample output

```
Name: rusty-vise
Status: PAUSED
Progress: 0/7 jobs completed

Job Runs:
+------------------------------+----------+--------------------------------------+
| Name                         | Status   | Job ID                               |
+==============================+==========+======================================+
| Untitled-2.py                | COMPLETE  | 775eda9e-4290-4aac-a801-e038e4a068d6 |
+------------------------------+----------+--------------------------------------+
| 04_version_specific.py       | COMPLETE  | 24070cca-e36f-4c84-aeb7-f40bd0c6a466 |
+------------------------------+----------+--------------------------------------+
| 05_final_aggregation.py      | PENDING  | e576391b-fbdf-4a4a-a935-bd918c5282f3 |
+------------------------------+----------+--------------------------------------+
| 01_base_datasets.py          | COMPLETE  | 577f7273-b4b8-4334-967f-cea600fee8b0 |
+------------------------------+----------+--------------------------------------+
| 02_alternate_process.py      | COMPLETE  | f75bc0da-f540-4342-a480-e028af732fa9 |
+------------------------------+----------+--------------------------------------+
| 02_alternate_process.py-1.py | COMPLETE  | 2809d04d-c60c-4945-bcd6-7de0b75978ac |
+------------------------------+----------+--------------------------------------+
| 03_separate_merge.py         | COMPLETE  | d03a5633-e0b3-4b91-bb06-f4f127f07d74 |
+------------------------------+----------+--------------------------------------+
```
