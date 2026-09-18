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
+--------------------------------------+--------------+----------+------------------+-----------+-----------------+-----------------+----------------+-------------------+-------------+--------------+
| UUID                                 | Name         | Status   | Cloud Provider   | Region    | Instance Type   | Compute Class   | Disk Request   | Busy/Active/Max   |   Job Quota | Is Default   |
+======================================+==============+==========+==================+===========+=================+=================+================+===================+=============+==============+
| 550e8400-e29b-41d4-a716-446655440000 | prod-cluster | ACTIVE   | AWS              | us-west-2 | m5.xlarge       | gpu             | 100Gi          | 2/4/8             |           8 | True         |
+--------------------------------------+--------------+----------+------------------+-----------+-----------------+-----------------+----------------+-------------------+-------------+--------------+
```

| Column | Meaning |
|--------|---------|
| `UUID` | Identifies the cluster. Names can be reused; this cannot |
| `Name` | Pass this to `datachain job run --cluster` |
| `Status` | `ACTIVE` and `MODIFYING` clusters accept jobs; `INACTIVE` and `FAILED` do not |
| `Cloud Provider` | `AWS`, `GCP`, `AZ` or `NB` |
| `Region` | Where the cluster runs, e.g. `us-west-2` |
| `Instance Type` | The machine a worker runs on, e.g. `m5.xlarge` |
| `Compute Class` | The kind of machine a worker gets, e.g. `Performance` or `gpu`. It does not tell you whether you are paying spot or on-demand rates |
| `Disk Request` | Temporary storage a worker asks for, e.g. `100Gi`. Not the size of its disk, and no basis for a storage bill |
| `Busy/Active/Max` | Workers running jobs / started / allowed |
| `Job Quota` | How many workers the cluster is allowed, as configured. It is not jobs per worker |
| `Is Default` | The cluster a job runs on when `--cluster` is omitted |

A `-` means the cluster does not set that field. It never means zero.

## All fields

The table is a summary. `--json` prints everything the cluster has:

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

* To work out what a job cost, find the cluster it ran on with
  [`datachain job ls --extended`](ls.md), then price that cluster's instance type in
  its region from your cloud provider's rates. Spot and on-demand are billed very
  differently and are not reported here, so check which one the cluster uses
