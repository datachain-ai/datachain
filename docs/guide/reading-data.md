---
title: Reading Data
---

# Reading Data

DataChain reads data from many sources and formats through a family of `read_*` entry points. Each returns a lazy chain: nothing executes until a terminal operation triggers it.

## Storage Files

Two functions read a storage. `dc.listing()` lists it: the objects of buckets, prefixes or single objects, as rows to iterate, collect or turn into a table. `dc.read_storage()` lists it and saves the listing as a dataset: a snapshot of the storage that chains filter, process and join, that later runs reuse, and that records which files a result was computed from. Both take the same URIs: one or a list, with glob patterns (`*`, `**`, `?`, `{a,b}`, `{1..9}`) and `recursive=False` for the first level only. Always include a trailing slash in bucket and prefix URIs.

### Listing a Storage

```python
import datachain as dc

total = sum(f.size for f in dc.listing("s3://bucket/episodes/"))
images = dc.listing("s3://bucket/images/**/*.{jpg,png}").to_objects(type="image")
```

Rows are plain tuples with `File`'s columns (`f.path`, `f.size`, `f.etag`, `f.last_modified`, ...) and its path helpers (`f.name`, `f.parent`, `f.get_file_ext()`), which build a `File` on each call: on large listings, prefer the columns, such as `f.path.endswith(".jpg")`. Rows don't read files; `to_objects()` gives `File` objects that do. A listing is listed once: iterate it, or take it whole with one of

```python
rows = dc.listing(uri).to_list()
table = dc.listing(uri).to_arrow()  # pyarrow.Table; to_pandas() for a DataFrame
images = dc.listing(uri).to_objects(type="image")  # ImageFile objects that read files
chain = dc.listing(uri).save()  # the snapshot read_storage(uri) reuses
```

Call `dc.listing()` again to list the storage again. A listing reads only what its URI selects: with `recursive=False`, or a pattern with a fixed start such as `s3://bucket/logs/2024-*`, it reads that level or those keys, not the whole prefix (S3 and GCS).

### Versioned Buckets

On a bucket with versioning enabled (S3, GCS, Azure), a listing has every version of every object by default, the current one with `is_latest`, so counts and sums over its rows include old versions. Pass `versions=False` to list current objects only, which is also faster:

```python
total = sum(f.size for f in dc.listing("s3://versioned-bucket/data/", versions=False))
```

`read_storage()` returns current objects either way; there `versions=False` only makes listing faster and leaves old versions out of the snapshot. A snapshot from `save()` is reused by `read_storage()` with the same `versions`:

```python
dc.listing("s3://versioned-bucket/data/", versions=False).save()
chain = dc.read_storage("s3://versioned-bucket/data/", versions=False)  # reuses it
```

A URI of a single object gives its current version either way. Buckets without versioning, local files and HTTP storages have one kind of listing, and `versions` changes nothing there.

### Saving a Storage Snapshot

`read_storage()` returns a chain of `File` objects over a saved listing. A saved listing of the URI, or of a prefix above it, younger than 4 hours is reused; otherwise the storage is listed and saved, and `update=True` lists it again. A glob lists its base, the part before the first pattern, and filters the snapshot by it; the chain holds current objects, and `versions=True` (the default) keeps every version in the snapshot.

```python
images = dc.read_storage("s3://bucket/images/**/*.jpg", type="image")
videos = dc.read_storage("gs://bucket/clips/", type="video")
all_files = dc.read_storage("az://container/data/")
```

The `type=` parameter selects the right File subclass (`ImageFile`, `VideoFile`, `AudioFile`, `TextFile`). Without it, files are plain `File` objects with binary access. `dc.listing_datasets()` shows the saved listings.

### Credentials

By default a storage is reached with the provider's standard credentials (AWS profile and environment variables, `gcloud` login, Azure environment). Both `dc.listing()` and `dc.read_storage()` take the same two parameters to change that.

A public bucket, without credentials:

```python
lst = dc.listing("gs://datachain-demo/dogs-and-cats/", anon=True)
```

With `anon=None`, the default, a bucket that needs no credentials is detected and read anonymously.

Explicit keys, or an S3-compatible storage such as MinIO:

```python
minio = {
    "endpoint_url": "https://minio.example.com",
    "key": "access-key",
    "secret": "secret-key",
}
lst = dc.listing("s3://my-bucket/data/", client_config=minio)
chain = dc.read_storage("s3://my-bucket/data/", client_config=minio)
```

Files from `to_objects()` read with the credentials of their listing, so nothing else is needed to open them:

```python
lst = dc.listing("s3://my-bucket/data/", client_config=minio)
for img in lst.to_objects(type="image"):
    print(img.path, img.read().size)
```

Rows from iterating a listing do not read files; use `to_objects()` for that.

### Sessions

A session holds datasets, the saved snapshots among them. Listing a storage and reading its files do not use one, so a session matters only for `save()` and `read_storage()`. Without one, both use the default session: nothing to set up.

To save snapshots in a session of your own, open it with `with`; everything inside uses it, including its credentials:

```python
with dc.Session("ingest", client_config=minio):
    rows = dc.listing("s3://my-bucket/data/").to_list()  # lists with minio
    chain = dc.listing("s3://my-bucket/data/").save()  # saved in "ingest"
    same = dc.read_storage("s3://my-bucket/data/")  # reuses that snapshot
```

Or pass it as `session=`:

```python
session = dc.Session("ingest", client_config=minio)
chain = dc.listing("s3://my-bucket/data/", session=session).save()
```

A session passed as `session=` brings its own credentials, and `client_config` is then ignored; `anon` still applies.

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
