# dataset describe

Show a dataset version's schema, direct dependencies and dependents, saved script and producing job. Reads metadata only.

```bash
datachain dataset describe joined
datachain dataset describe joined@1.0.0
datachain dataset describe joined --json
datachain dataset describe dev.project.joined
datachain dataset describe dev.project.joined --studio
```

Details describe one version: the latest complete version, or an exact version selected with `name@version`. `Versions` lists all available complete versions.

## Text output

```text
Name: joined@2.0.0
Rows: 2
Source: local
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

`dc.describe()` returns a `DatasetDescription` with `DatasetInfo` versions, a typed schema and the selected version's `Job`.

```python
import datachain as dc

info = dc.describe("joined@1.0.0")
print(info.dataset_uuid, info.version_uuid)
print(info.available_versions)
info.schema.print_tree()
if info.job:
    print(info.job.id, info.job.finished_at)
```

Datasets in the `local` namespace use the local catalog. Other namespaces use a pulled copy first, then Studio if the dataset or version hasn't been pulled. A copy lists only pulled versions. `--studio` (`studio=True`) skips the copy. Studio uses the team set by `datachain auth team` or `DATACHAIN_STUDIO_TEAM`. Inside Studio jobs, reads always use Studio. `Source` identifies the catalog used.

Studio's public API provides no dependencies or dependents, and job `params` is empty. Inside Studio jobs, dependencies are available; dependents are not. Pulling does not copy lineage. Unavailable sections show `unavailable` (`null` in JSON), missing inputs show `unavailable dependency`, and jobs without details show `Job: <id> (unavailable)`.
