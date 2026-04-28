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

For public buckets, pass `anon=True` explicitly:

```python
chain = dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
```

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
meta = dc.read_json("gs://bucket/coco2017/annotations/captions_val2017.json", jmespath="images")
captions = dc.read_json("gs://bucket/coco2017/annotations/captions_val2017.json", jmespath="annotations")

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

# S3-compatible (MinIO, Ceph, etc.)
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
