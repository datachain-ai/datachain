# dataset describe

Show a dataset's schema, versions, sources, lineage, saved script and jobs. Reads metadata only.

```bash
datachain dataset describe joined
datachain dataset describe joined@1.0.0
datachain dataset describe joined --json
datachain dataset describe dev.project.joined
```

Use `name@version` to select a version; otherwise the latest complete version is selected. All available complete versions are listed.

## Text output

```text
Name: local.local.joined@2.0.0
Rows: 2
Created: 2026-11-01 10:02:00 UTC
Versions: 1.0.0, 2.0.0
Schema:
  value: int
Sources: s3://bucket/photos/@1.0.0
Lineage:
  local.local.joined@2.0.0
    depends_on: s3://bucket/photos/@1.0.0
  s3://bucket/photos/@1.0.0
Dependents: none recorded
Job: job-1
Job_status: COMPLETE
Job_created: 2026-11-01 10:02:00 UTC
Job_finished: 2026-11-01 10:02:02 UTC
Workers: 1
Python: 3.12
Script:
build()
```

`Lineage` shows the dataset's inputs and their ancestors. `Dependents` lists datasets built from the selected version. Storage is shown as `URI@listing-version`.

## JSON output

`name` and `depends_on` use canonical dataset names with versions. Unavailable sections are `null`.

[Full JSON example](describe-example.json), including version, lineage and job metadata.

## Python

`dc.describe_dataset()` returns a `DatasetDescription` with `DatasetInfo` versions, a typed schema and recorded `Job` objects.

```python
import datachain as dc

info = dc.describe_dataset("joined@1.0.0")
print(info.available_versions)
info.schema.print_tree()
if info.job:
    print(info.job.id, info.job.finished_at)
```

Non-local namespaces are read from Studio. Its public API does not provide exact lineage, dependents or job parameters; jobs require read permission.
