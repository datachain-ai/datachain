# dataset describe

Show what is recorded about a saved local dataset version.

## Synopsis

```usage
usage: datachain dataset describe [-h] [-v] [-q] [--version VERSION] [--json] name
```

## Description

Show what is recorded about a saved dataset version: schema, row count, versions,
the datasets and storage it was built from, the datasets built from it, and the script
that saved it. Rows and storage are not read. Describe is local only: Studio is not
contacted.

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
recorded dataset identity, so current-only and versioned listings remain distinct.
Storage `name` values are decoded URIs for display. `rows` is the row count saved with that exact version.
`removed` marks a version deleted after it was used, and a `null` reference is a
dependency whose record no longer exists. `dependents` lists the versions built
directly from this one.

Storage entries also include `listing_kind`: `all versions` or `current only`.

```json
"depends_on": ["local.local.inventory@1.0.0", "local.local.sizes@2.0.0"],
"lineage": [
  {"ref": "local.local.inventory@1.0.0", "name": "local.local.inventory", "version": "1.0.0", "type": "dataset", "rows": 2,
   "removed": false, "created_at": "2026-11-01T10:00:00+00:00",
   "depends_on": ["system.listing.lst__file:///data/photos/@1.0.0"]},
  {"ref": "system.listing.lst__file:///data/photos/@1.0.0", "name": "file:///data/photos/", "version": "1.0.0", "type": "storage", "listing_kind": "all versions", "rows": 2,
   "removed": false, "created_at": "2026-11-01T09:59:58+00:00", "depends_on": []},
  {"ref": "local.local.sizes@2.0.0", "name": "local.local.sizes", "version": "2.0.0", "type": "dataset", "rows": 1,
   "removed": false, "created_at": "2026-11-01T10:01:00+00:00", "depends_on": []}
]
```

## Examples

1. Describe the latest complete version of a dataset:
```bash
datachain dataset describe joined
```

Example output:
```
local.local.joined@1.0.0
Rows: 2
Created: 2026-11-01T10:02:00+00:00
Versions: 1.0.0
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
Sources: file:///data/photos/
Depends on: local.local.inventory@1.0.0, local.local.sizes@2.0.0
Lineage:
  local.local.inventory@1.0.0 (2 rows)
    depends on: file:///data/photos/ (all versions)@1.0.0
  file:///data/photos/ (all versions)@1.0.0 (2 rows)
  local.local.sizes@2.0.0 (1 rows)
Dependents: none recorded
Script:
import datachain as dc

sizes = dc.read_dataset("sizes", version="2.0.0")
dc.read_dataset("inventory").merge(sizes, on="file.size", right_on="value").save("joined")
```

2. Describe a specific version:
```bash
datachain dataset describe joined@1.0.0
```

3. Print the description as JSON:
```bash
datachain dataset describe local.local.joined --version 1.0.0 --json
```
