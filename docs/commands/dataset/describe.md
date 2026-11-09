# dataset describe

Show a dataset's schema, versions, direct dependencies, saved script and jobs. Reads metadata only.

```bash
datachain dataset describe joined
datachain dataset describe joined@1.0.0
datachain dataset describe joined --json
datachain dataset describe dev.project.joined
datachain dataset describe dev.project.joined --team my-team
```

Use `name@version` to select a version; otherwise the latest complete version is selected. All available complete versions are listed.

## Text output

```text
Name: joined@2.0.0
Rows: 2
Created: 2026-11-01 02:02:00 PST
Versions: 1.0.0, 2.0.0
Schema:
  value: int
Dependencies:
  s3://bucket/photos/@1.0.0
Dependents: none recorded
Job: job-1
Job_status: COMPLETE
Job_created: 2026-11-01 02:02:00 PST
Job_finished: 2026-11-01 02:02:02 PST
Workers: 1
Python: 3.12
Script:
build()
```

`Dependencies` lists direct inputs; `Dependents` lists datasets built from the selected version. Storage is shown as `URI@listing-version`. Dates use your local time zone.

## JSON output

JSON includes canonical names, stable `dataset_uuid` and `version_uuid` identifiers, and UTC timestamps. The dataset UUID stays the same across versions and renames within a catalog. Unavailable sections are `null`.

[Full JSON example](describe-example.json), including version, dependency and job metadata.

## Python

`dc.describe()` returns a `DatasetDescription` with `DatasetInfo` versions, a typed schema and recorded `Job` objects.

```python
import datachain as dc

info = dc.describe("joined@1.0.0")
print(info.dataset_uuid, info.version_uuid)
print(info.available_versions)
info.schema.print_tree()
if info.job:
    print(info.job.id, info.job.finished_at)
```

Local metadata is used first, with Studio fallback for missing remote datasets or versions. Use `--team` or `dc.describe(..., team="my-team")` to read Studio directly.

Studio includes the selected version's job details and every version's job ID. Exact dependencies, dependents and job parameters are unavailable through its public API.
