# Handling Local Database Migrations (CLI)

When using the DataChain CLI, datasets are stored in a local SQLite database located at:

```
.datachain/db
```

Unlike the SaaS version (Studio), the CLI does **not** support automatic database migrations. This means that after upgrading the DataChain CLI, the local database schema may become incompatible with the updated codebase.

## Schema Mismatch Detection

The CLI automatically checks for schema compatibility. If a mismatch is detected, you’ll see an error like:

```
OutdatedDatabaseSchemaError: You have an old version of the database schema. Please refer to the documentation for more information.
```

This typically happens after upgrading the CLI to a newer version.

## How to Fix It

The recommended fix is to **delete the local database** and let the CLI recreate it. To avoid losing datasets, you should **export them before removing the database**.

Before deleting the file, we strongly recommend making a backup of your current database:

```bash
cp .datachain/db .datachain/db.backup
```

This allows you to recover data manually if needed later.

---

## Exporting and Re-Importing All Local Datasets

**Important:** Exporting datasets must be done **before upgrading** to a new DataChain version. Export with the old version to avoid the `OutdatedDatabaseSchemaError` during export. After deleting the database file, upgrade/install the new DataChain version.

### Step 1: Export All Datasets to Parquet

Export all datasets into a folder named `exported_datasets` (created if it doesn't exist). Each dataset will be saved to a file in the format:

```
<dataset_name>.<dataset_version>.parquet
```

Example: `metrics.1.0.1.parquet`

```python
import os
import datachain as dc

export_dir = "exported_datasets"
os.makedirs(export_dir, exist_ok=True)

# dc.datasets() returns a chain of DatasetInfo objects
for ds_info in dc.datasets(column="dataset").to_values("dataset"):
    ds = dc.read_dataset(ds_info.name, version=ds_info.version)
    filename = f"{ds_info.name}.{ds_info.version}.parquet"
    filepath = os.path.join(export_dir, filename)
    ds.to_parquet(filepath)
```

### Step 2: Delete Local Database

Make sure you've backed it up (see above), then:

```bash
rm .datachain/db
```

### Step 3: Re-import All Datasets from Parquet (In Correct Version Order)

To avoid import errors due to semantic versioning constraints, datasets must be imported in ascending order by version for each dataset name.

```python
import os
import datachain as dc
from packaging.version import Version

import_dir = "exported_datasets"

# Gather all dataset files
datasets = []

for fname in os.listdir(import_dir):
    if not fname.endswith(".parquet"):
        continue
    base = fname[:-8]  # remove '.parquet'
    name, version = base.split('.', 1)  # split on first dot
    filepath = os.path.join(import_dir, fname)
    datasets.append((name, Version(version), filepath))

# Sort by dataset name and then by version ascending
datasets.sort(key=lambda x: (x[0], x[1]))

# Import datasets in order
for name, version, filepath in datasets:
    dc.read_parquet(filepath).save(name, version=str(version))
```

**Note:** While exporting and importing datasets to Parquet files preserves the datasets and their data, some metadata - such as dataset dependencies - will **not** be preserved. This information will be lost during this process.

---

## Notes

- This limitation only applies to the **CLI**, which uses a local SQLite database.
- The **Studio (SaaS)** version handles all schema migrations automatically - no manual steps are required.
- The CLI only supports the default namespace/project: `local.local`.

---

This export/import workflow is the recommended way to preserve your datasets during local CLI upgrades that involve database schema changes.
