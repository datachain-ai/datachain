# dataset describe

Show what is recorded about a saved local dataset version.

## Synopsis

```usage
usage: datachain dataset describe [-h] [-v] [-q] [--version VERSION] [--json] name
```

## Description

Show what is recorded about a saved dataset version: schema, row count, versions,
the datasets and storage it was built from, the datasets built from it, and the script
that saved it, plus recorded producing jobs. All available complete versions are listed,
even when describing an older version. Rows and storage are not read. Describe is
local only: Studio is not contacted.

Recorded lineage says which versions were used, not whether the files or models
behind them have changed since. Storage that a UDF opens itself, and imported modules,
environment variables or model weights, are not recorded.

## Arguments

* `name` - Dataset name, optionally with namespace and project (`dev.team.labels`) and version (`labels@1.0.0`)

## Options

* `--version VERSION` - Version to describe (default: latest complete version)
* `--json` - Print the description as JSON
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## JSON output

`--json` prints the same description as JSON. `depends_on` lists the versions the
dataset was built from directly. Each reference matches a `ref` in `lineage`, which
lists every ancestor version once with its own `depends_on`. References retain the
exact version identity without separate duplicate `name` and `version` fields.
Storage references contain the URI, exact listing version, and listing kind.
`num_objects` is the row count saved with that entire version, not its contribution
to the described result.
`removed` marks a version deleted after it was used, and a `null` reference is a
dependency whose record no longer exists. `dependents` lists the versions built
directly from this one.

Storage entries also include `listing_kind`: `all versions` or `current only`.
Each version's `job_id` refers to an entry in `jobs`, which contains the recorded
DataChain job metadata. A job record may be unavailable even when its ID is known.

```json
{
  "depends_on": ["local.local.inventory@1.0.0"],
  "lineage": [
    {
      "ref": "local.local.inventory@1.0.0",
      "type": "dataset",
      "num_objects": 2,
      "depends_on": ["file:///data/photos/@1.0.0 (all versions)"]
    },
    {
      "ref": "file:///data/photos/@1.0.0 (all versions)",
      "type": "storage",
      "listing_kind": "all versions",
      "num_objects": 2,
      "depends_on": []
    }
  ]
}
```

The excerpt omits timestamps, status, job IDs, and other description fields.

## Examples

1. Describe the latest complete version of a dataset:
```bash
datachain dataset describe joined
```

Example output (job details omitted):
```
local.local.joined@2.0.0
Rows: 2
Created: 2026-11-01T10:02:00+00:00
Versions: 1.0.0, 2.0.0
Schema:
  file: File@v1
    source: str
    path: str
    size: int
    version: str
    etag: str
    is_latest: bool
    last_modified: datetime
    location: Union[dict, list[dict], NoneType]
  value: Optional[int]
Sources: file:///data/photos/@1.0.0 (all versions)
Depends on: local.local.inventory@1.0.0, local.local.sizes@2.0.0
Lineage:
  local.local.inventory@1.0.0
    depends on: file:///data/photos/@1.0.0 (all versions)
  file:///data/photos/@1.0.0 (all versions)
  local.local.sizes@2.0.0
Dependents: none recorded
Script:
import datachain as dc

sizes = dc.read_dataset("sizes", version="2.0.0")
dc.read_dataset("inventory").merge(sizes, on="file.size", right_on="value").save("joined", version="2.0.0")
```

2. Describe a specific version:
```bash
datachain dataset describe joined@1.0.0
```

3. Print the description as JSON:
```bash
datachain dataset describe local.local.joined --version 1.0.0 --json
```

## Python API

`dc.describe_dataset()` returns a `DatasetDescription` object. Version metadata uses
`DatasetInfo`, as in `dc.datasets()`; storage dependencies use `ListingInfo`, as in
`dc.listings()`. `schema` is a `SignalSchema`, and `job` is the selected version's
recorded `Job`, or `None` when unavailable. `jobs` maps job IDs to the recorded jobs
for the listed versions and dependencies.

```python
import datachain as dc

info = dc.describe_dataset("joined")
print(info.selected_version.num_objects)
print(info.available_versions)
info.schema.print_tree()
if info.job:
    print(info.job.id, info.job.created_at, info.job.finished_at)
```

Use `dc.describe_dataset("joined@1.0.0")` to select an exact version, and
`info.to_dict()` for the JSON-compatible representation used by the CLI.
