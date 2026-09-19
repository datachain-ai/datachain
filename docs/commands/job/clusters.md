# job clusters

List compute clusters in Studio.

## Synopsis

```usage
usage: datachain job clusters [-h] [-v] [-q] [--team TEAM] [--json]
```

## Description

This command lists the compute clusters your team can run jobs on, with the machine each one provisions and how busy it is. Use it to pick a cluster for [`datachain job run --cluster`](run.md), or to check spare capacity before submitting.

Retired clusters are not listed.

## Options

* `--team TEAM` - Team to list clusters for (default: from config)
* `--json` - Print the cluster list as JSON
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Picking a cluster

List what your team has, then run on one by name:

```bash
datachain job clusters
datachain job run analysis.py --cluster prod-cluster
```

Omit `--cluster` and the job runs on the team's default.

## Output

An excerpt - the full table also carries Cloud Provider, Compute Class, Disk Request, Job Quota and Is Default:

```
+--------------------------------------+--------------+----------+-------------+-----------------+-------------------+
| UUID                                 | Name         | Status   | Region      | Instance Type   | Busy/Active/Max   |
+======================================+==============+==========+=============+=================+===================+
| 550e8400-e29b-41d4-a716-446655440000 | prod-cluster | ACTIVE   | us-west-2   | m5.xlarge       | 2/4/8             |
+--------------------------------------+--------------+----------+-------------+-----------------+-------------------+
| 6f1c2d90-8a71-4a3e-9f22-0b5d4e7c1a88 | gpu-a100     | INACTIVE | us-central1 | a2              | 0/0/16            |
+--------------------------------------+--------------+----------+-------------+-----------------+-------------------+
```

| Column | Meaning |
|--------|---------|
| `UUID` | Identifies the cluster. Names can be reused; this cannot |
| `Name` | Pass this to `datachain job run --cluster` |
| `Status` | `ACTIVE` and `MODIFYING` clusters accept jobs; `INACTIVE` and `FAILED` do not |
| `Cloud Provider` | `AWS`, `GCP`, `AZ` or `NB` |
| `Region` | Where the cluster runs |
| `Instance Type` | Worker machine type or family |
| `Compute Class` | Worker class, such as `Performance` or `gpu` |
| `Disk Request` | Requested temporary storage per worker (allocated capacity may differ) |
| `Busy/Active/Max` | Workers running jobs / started / allowed |
| `Job Quota` | Configured worker limit |
| `Is Default` | The cluster a job runs on when `--cluster` is omitted |

A `-` means the cluster does not set that field. It never means zero.

## All fields

`--json` prints everything the cluster has, for scripting:

```console
$ datachain job clusters --json
[
  {
    "uuid": "550e8400-e29b-41d4-a716-446655440000",
    "name": "prod-cluster",
    "status": "ACTIVE",
    "cloud_provider": "AWS",
    "cloud_credentials": "aws-creds",
    "is_active": true,
    "default": true,
    "max_workers": 8,
    "active_workers": 4,
    "busy_workers": 2,
    "cloud_region": "us-west-2",
    "instance_type": "m5.xlarge",
    "compute_class": "gpu",
    "disk_size": "100Gi",
    "job_quota": 8
  }
]
```

## Examples

1. List all clusters for the default team:
```bash
datachain job clusters
```

2. List clusters for a specific team:
```bash
datachain job clusters --team my-team
```

3. Find the default cluster's instance type, for scripting:
```bash
datachain job clusters --json | jq -r '.[] | select(.default) | .instance_type'
```

## Notes

* **Working out a cost.** Provider, region and machine information can help identify a rate. Estimating a job's cost also requires its resource usage and the applicable compute and storage rates, none of which this command reports.
