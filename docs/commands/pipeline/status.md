# pipeline status

Get the status of an pipeline from Studio

## Synopsis

```usage
usage: datachain pipeline status [-h] [-v] [-q] [-t TEAM] name
```

## Description

This command fetches the latest status of an pipeline along with the status of its jobs from Studio.

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
datachain pipeline status rathe-kyat
```

#### Sample output

```
Name: rathe-kyat
Status: RUNNING
Progress: 8/16 jobs completed

Job Runs:
+-------------------------+--------------+--------------------------------------+
| Name                    | Status       | Job ID                               |
+=========================+==============+======================================+
| DependencyTracking11.py | PENDING      | 7a12136c-b10c-4bb5-a88b-adef1554e53f |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking12.py | PENDING      | 910c5464-d1ad-4631-b231-e43aa0b4f2c2 |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking13.py | PENDING      | 50cebd33-d25f-4955-af78-8e40b3b0e2a9 |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking14.py | PENDING      | 272e8c2c-0d20-4fd9-8f89-b46025f4622f |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking15.py | PENDING      | c43b8ea6-0193-4240-be0c-923ea91bf7bb |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking1.py  | COMPLETE     | c2301b20-744f-4981-8c51-d8f1b01f1925 |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking2.py  | COMPLETE     | a9cc1c9f-61fd-4d68-be8e-356866fa848b |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking3.py  | COMPLETE     | f8fcf5b7-1c02-466c-8f31-671ea6324b1c |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking4.py  | COMPLETE     | 49590870-75a5-4b89-ad40-2fa88f344e39 |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking10.py | PENDING      | 1ac68ffd-8d4d-4241-8958-084d06de59c7 |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking8.py  | PENDING      | 9547ca44-db18-4c19-a744-d0583ac33d67 |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking5.py  | COMPLETE     | fe1fd73a-904d-4ae6-bb3f-869369c9c536 |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking5.py  | COMPLETE     | e9c05ba7-dec4-4ea8-834b-0a31b5af0061 |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking9.py  | COMPLETE     | 7f67e740-9459-43c9-a203-d68b1553f08f |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking7.py  | COMPLETE     | dc3ae78c-d15b-4920-8320-66c203f91f58 |
+-------------------------+--------------+--------------------------------------+
| DependencyTracking7.py  | PROVISIONING | b7b338f8-6c0d-4c84-87f8-a0349fd96c98 |
+-------------------------+--------------+--------------------------------------+
```
