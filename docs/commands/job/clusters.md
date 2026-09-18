# job clusters

List compute clusters in Studio.

## Synopsis

```usage
usage: datachain job clusters [-h] [-v] [-q] [--team TEAM] [--json]
```

## Description

This command lists the compute clusters your team can run jobs on, with the machine
each one provisions and how busy it is. Use it to pick a cluster for
[`datachain job run --cluster`](run.md), to check spare capacity before submitting,
or to work out what a job costs.

Retired clusters are not listed.

## Options

* `--team TEAM` - Team to list clusters for (default: from config)
* `--json` - Print the full cluster list as JSON, with every field
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Output

```
+------+--------------+----------+------------------+-------------+-----------------+-----------------+--------+-------------------+-------------+--------------+
|   ID | Name         | Status   | Cloud Provider   | Region      | Instance Type   | Compute Class   | Disk   | Busy/Active/Max   | Job Quota   | Is Default   |
+======+==============+==========+==================+=============+=================+=================+========+===================+=============+==============+
|    1 | prod-cluster | ACTIVE   | AWS              | us-west-2   | m5.xlarge       | gpu             | 100Gi  | 2/4/8             | 8           | True         |
+------+--------------+----------+------------------+-------------+-----------------+-----------------+--------+-------------------+-------------+--------------+
```

| Column | Meaning |
|--------|---------|
| `ID` | Numeric cluster id. Every cluster also has a `uuid` - see [All fields](#all-fields) |
| `Name` | Pass this to `datachain job run --cluster` |
| `Status` | `ACTIVE` and `MODIFYING` clusters accept jobs; `INACTIVE` and `FAILED` do not |
| `Cloud Provider` | `AWS`, `GCP`, `AZ` or `NB` |
| `Region` | Where the cluster runs, e.g. `us-west-2` |
| `Instance Type` | Machine type or family a worker runs on, e.g. `m5.xlarge` |
| `Compute Class` | Node class a worker is scheduled onto, e.g. `Performance` or `gpu`. This is not a purchase model - spot capacity is configured separately, so it does not tell you whether the rate is spot or on-demand |
| `Disk` | Disk a worker gets, e.g. `100Gi` |
| `Busy/Active/Max` | Workers assigned to jobs / provisioned / the cap |
| `Job Quota` | Configured limit on the cluster's workers, which the cluster reports live as its max workers. It is not a number of jobs each worker runs |
| `Is Default` | The cluster a job runs on when `--cluster` is omitted |

A `-` means the cluster does not configure that field, so Studio has no value to
report. It never means zero.

## All fields

The table above is a readable summary. `--json` prints every field:

```console
$ datachain job clusters --json
[
  {
    "id": 1,
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

Three of these never appear in the table:

| Field | Meaning |
|-------|---------|
| `uuid` | The cluster's stable identifier, and what a job's `compute_cluster_uuid` points at. Use it to tell which cluster a job ran on - names can be reused, and the numeric `id` is not durable |
| `cloud_credentials` | Name of the cloud credentials the cluster provisions with, or `null` |
| `is_active` | True while the cluster accepts jobs - the same thing `status` says |

## Examples

1. List all clusters for the default team:
```bash
datachain job clusters
```

2. List clusters for a specific team:
```bash
datachain job clusters --team my-team
```

3. Get every field as JSON:
```bash
datachain job clusters --json
```

4. Find the default cluster's instance type:
```bash
datachain job clusters --json | jq -r '.[] | select(.default) | .instance_type'
```

## Notes

* Cluster names are what `datachain job run --cluster` expects
* To price a job, find the cluster it ran on -
  [`datachain job ls --extended`](ls.md) shows it by name, and the jobs API also
  carries a `compute_cluster_uuid` that joins to a cluster's `uuid` - then take the
  rate for that `instance_type` in that `cloud_region` from your cloud provider's
  price list. Whether the cluster runs spot or on-demand capacity is not reported
