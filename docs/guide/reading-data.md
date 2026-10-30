---
title: Reading Data
---

# Reading Data

DataChain reads data from many sources and formats through a family of `read_*` entry points. Each returns a lazy chain: nothing executes until a terminal operation triggers it.

## Storage Files

`read_storage()` connects to any supported storage provider and returns a chain of File objects. It is the primary entry point for unstructured data: images, video, audio, PDFs, text. Always include a trailing slash in bucket and prefix URIs.

```python
import datachain as dc

images = dc.read_storage("s3://bucket/images/**/*.jpg", type="image")
videos = dc.read_storage("gs://bucket/clips/", type="video")
all_files = dc.read_storage("az://container/data/")
```

The `type=` parameter selects the right File subclass (`ImageFile`, `VideoFile`, `AudioFile`, `TextFile`). Without it, files are plain `File` objects with binary access.

The chain runs over a saved listing of the storage, a snapshot that later runs reuse. A saved listing of the URI, or of a prefix above it, younger than 4 hours is reused; otherwise the storage is listed and saved, and `update=True` lists it again. `dc.listings()` shows the saved listings. On a versioned bucket the chain holds current objects; `versions=False` lists faster and leaves old versions out of the snapshot.

For public buckets, pass `anon=True` explicitly; for explicit keys or an S3-compatible storage such as MinIO, pass `client_config`:

```python
chain = dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)

minio = {
    "endpoint_url": "https://minio.example.com",
    "key": "access-key",
    "secret": "secret-key",
}
chain = dc.read_storage("s3://my-bucket/data/", client_config=minio)
```

To get the objects of a storage as rows in Python, without a chain or a saved snapshot, use `dc.list_storage()`: see [Listing Storage](listing-storage.md).

## Structured Formats

DataChain reads CSV, JSON, and Parquet directly from storage. These entry points parse the format and produce chains with typed columns.

```python
import datachain as dc

# CSV -- auto-detects delimiter, headers, column types
labels = dc.read_csv("s3://bucket/labels.csv")
labels = dc.read_csv("s3://bucket/csvs/", delimiter=";")

# JSON and JSONL -- with optional JMESPath for nested structures
meta = dc.read_json("gs://bucket/annotations.json", jmespath="images")
captions = dc.read_json("gs://bucket/coco.json", jmespath="annotations")

# Parquet -- supports glob patterns and Hive partitioning
data = dc.read_parquet("s3://bucket/data/*.parquet")
data = dc.read_parquet("s3://bucket/202{1..4}/{yellow,green}-{01..12}.parquet")
```

JMESPath is powerful for real-world JSON formats like COCO, where images, annotations, and categories live under different top-level keys. Each `read_json()` call with a different `jmespath` extracts one array, and you merge them together on shared IDs.

For complex JSON, auto-generate a Pydantic model from a sample file:

```python
from datachain.lib.meta_formats import gen_datamodel_code

code = gen_datamodel_code("s3://bucket/data.json", jmespath="images")
```

### Selecting and naming CSV and Parquet columns

For `read_csv()`, `read_parquet()`, and `parse_tabular()`, use `columns=` to select input columns by name and set their order. Names may be written as in the file or as DataChain normalizes them: `"ZIP Code"` and `"zip_code"` refer to the same column. Unknown or repeated columns raise an error.

`output=` defines the result's names and Python types **by position**. A dictionary uses insertion order; CSV and `parse_tabular()` also accept a Pydantic model (field order) or a list of names (types inferred). Output names do not select or match input names. Supply exactly one output field for each column read: the selected columns if `columns` is provided, or all input columns if it is omitted. Omit `output` to infer names and types.

With an `output` dictionary or model and no `columns`, the columns come from the first file by path, and every file is read by those names. Partition columns (from folders like `year=2024/`) and a pandas index saved as a column can be left out of `output` when they come last.

For an ordinary Parquet file with columns `a, b, c`:

```python
data = dc.read_parquet("s3://bucket/data.parquet", columns=["c"])
data = dc.read_parquet(
    "s3://bucket/data.parquet",
    columns=["c", "a"],
    output={"x": float, "y": float},
)  # x receives c; y receives a

data = dc.read_parquet(
    "s3://bucket/data.parquet", output={"c": float, "b": float, "a": float}
)  # Renames a, b, c to c, b, a by position
```

An `output={"c": float}` without `columns` supplies only one field for three columns and raises an error. Replace partial `output` schemas with `columns=["c"]`; add `output={"c": float}` only if you also need to specify the result type. The same rules apply to CSV files.

CSV's `column_types=` controls parsing before `output` is applied. Its keys are input names, before normalization or renaming, and it does not select columns. For example, preserve the leading zero in `02134`:

```python
labels = dc.read_csv(
    "s3://bucket/labels.csv",
    columns=["zip_code"],
    column_types={"ZIP Code": "string"},
    output={"postal_code": str},
)
```

For CSV without a header, use generated input names `f0`, `f1`, etc. when selecting columns, then optionally rename them:

```python
data = dc.read_csv(
    "s3://bucket/no-header.csv",
    header=False,
    columns=["f2", "f0"],
    output=["c", "a"],
)
```

With `header=False` and no `columns`, `output` is required and supplies names for every input column.

Parquet exported by DataChain restores its saved schema and does not support `columns=`. Read it without `columns` and use `.select(...)` on the chain. With `columns` or `output`, read it separately from other Parquet files.

## SQL Databases

`read_database()` connects to any SQLAlchemy-compatible database:

```python
import datachain as dc

# Basic query
records = dc.read_database("SELECT * FROM experiments", "sqlite:///local.db")

# Parameterized query -- prevents SQL injection
chain = dc.read_database(
    "SELECT * FROM products WHERE category = :cat",
    "postgresql://host/db",
    params={"cat": "electronics"},
)

# Full enrichment pattern: query -> enrich with LLM -> save as dataset
(
    dc.read_database("SELECT id, name, raw_text FROM articles", "postgresql://host/db")
    .settings(parallel=8)
    .map(summary=generate_summary)
    .save("article_summaries")
)
```

Supported databases include PostgreSQL, MySQL, SQLite, DuckDB, Snowflake, and anything else SQLAlchemy supports. Schema inference is automatic.

## In-Memory Sources

For data already in Python:

```python
import pandas as pd
import datachain as dc

# From pandas
df = pd.DataFrame({"path": paths, "label": labels})
chain = dc.read_pandas(df)

# From HuggingFace Hub
chain = dc.read_hf("beans", split="train")
chain = dc.read_hf("beans", split="train", streaming=True, limit=100)

# HuggingFace datasets as storage URIs
chain = dc.read_storage("hf://datasets/mozilla-foundation/common_voice_17_0/audio/en")

# From Python values
chain = dc.read_values(scores=[1.2, 3.4, 2.5])

# From explicit records with schema
records = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
chain = dc.read_records(records, schema={"name": str, "age": int})
```

## Merging External Metadata

The most common real-world pattern: files live in storage, metadata lives in a sidecar format. Read both sources as chains and merge on a shared key.

### JSON Sidecars

Each image has a matching `.json` file with annotations:

```python
import datachain as dc

images = dc.read_storage("gs://bucket/dogs-and-cats/*jpg", anon=True)
meta = dc.read_json("gs://bucket/dogs-and-cats/*json", column="meta", anon=True)

images_id = images.map(id=lambda file: file.path.split(".")[-2])
annotated = images_id.merge(meta, on="id", right_on="meta.id")
```

### COCO Annotations

One JSON file contains multiple arrays, merged by ID:

```python
import datachain as dc

images = dc.read_storage("gs://bucket/coco2017/images/val/")
meta = dc.read_json(
    "gs://bucket/coco2017/annotations/captions_val2017.json", jmespath="images"
)
captions = dc.read_json(
    "gs://bucket/coco2017/annotations/captions_val2017.json", jmespath="annotations"
)

images_meta = images.merge(meta, on="file.path", right_on="images.file_name")
captioned = images_meta.merge(captions, on="images.id", right_on="annotations.image_id")
```

### CSV Labels

```python
import datachain as dc

files = dc.read_storage("gs://bucket/data/")
labels = dc.read_csv("gs://bucket/labels.csv")
labeled = files.merge(labels, on="file.path", right_on="path")
```

## Storage Providers

| Provider | URI Format |
|---|---|
| AWS S3 | `s3://bucket-name/path/` |
| Google Cloud Storage | `gs://bucket-name/path/` |
| Azure Blob Storage | `az://container-name/path/` |
| HuggingFace Hub | `hf://dataset-name` |
| Local Filesystem | `./path/to/data` or `file://path` |

Each provider uses standard credential locations by default. For non-default configurations, use `client_config`:

```python
import datachain as dc

# S3-compatible (MinIO, Ceph, Cloudflare R2, etc.)
chain = dc.read_storage(
    "s3://my-bucket/data/",
    client_config={
        "endpoint_url": "https://minio.example.com",
        "key": "access-key",
        "secret": "secret-key",
    },
)
```

Cross-provider workflows work transparently:

```python
import datachain as dc

(
    dc.read_storage("s3://source-bucket/raw/", type="image")
    .settings(parallel=8)
    .map(embedding=compute_embedding)
    .save("processed_images")
)

dc.read_dataset("processed_images").to_storage("gs://dest-bucket/output/")
```
