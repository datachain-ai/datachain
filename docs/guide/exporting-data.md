---
title: Exporting Data
---

# Exporting Data

Terminal operations trigger chain execution and produce output. Export methods get data out of DataChain into formats that downstream tools consume.

## DataFrames and Files

```python
import datachain as dc

chain = dc.read_storage("s3://bucket/data/").filter(dc.C("file.size") > 0)

# To pandas DataFrame
df = chain.to_pandas()

# To Parquet -- columnar format, efficient for large datasets
chain.to_parquet("output/results.parquet")

# To CSV / JSON / JSONL
chain.to_csv("output/results.csv")
chain.to_json("output/results.json")

# Back to storage -- writes files to a new location
chain.to_storage("s3://bucket/output/", signal="file")
```

## Export Placement Strategies

When exporting to storage, the `placement` parameter controls how file paths are constructed:

- **`filename`** -- retains only the original filename, discards directories
- **`filepath`** -- preserves the relative directory structure
- **`fullpath`** -- prefixes paths with the storage host
- **`etag`** -- uses the file ETag with original extension (guarantees uniqueness)

```python
chain.to_storage("./local-copy/", signal="file", placement="filepath")
```

## PyTorch DataLoader Integration

DataChain exports directly to PyTorch datasets:

```python
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
import datachain as dc

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

ds = (
    dc.read_storage("gs://bucket/images/", type="image", anon=True)
    .map(label=lambda name: name.split(".")[0], params=["file.path"])
    .select("file", "label")
    .to_pytorch(
        transform=processor.image_processor,
        tokenizer=processor.tokenizer,
    )
)

loader = DataLoader(ds, batch_size=16)
```

## Train/Test Split

```python
from datachain.toolkit import train_test_split

train, test = train_test_split(chain, [0.7, 0.3])
train, test, val = train_test_split(chain, [0.7, 0.2, 0.1])
```

Each split is a full chain that can be saved, exported, or fed to `to_pytorch()`.

## Extracting Results in Python

```python
# Single column as a flat list
scores = chain.to_values("score")          # -> [0.9, 0.7, 0.3]

# Multiple columns as tuples
rows = chain.to_list("file", "label")      # -> [(File, "cat"), (File, "dog")]
```

For processing chain results, prefer `map()`/`gen()` over extracting and looping -- they preserve parallelism and lineage.

## Writing to Databases

`to_database()` writes chain results to any SQLAlchemy-compatible database:

```python
import datachain as dc

# Basic export
chain.to_database("results_table", "postgresql://host/db")

# Round-trip: database -> enrich -> write back
(
    dc.read_database("SELECT id, text FROM reviews", "postgresql://host/db")
    .settings(parallel=8)
    .map(sentiment=classify_sentiment)
    .to_database("review_sentiments", "postgresql://host/db")
)

# Conflict handling
chain.to_database(
    "products",
    "postgresql://host/db",
    on_conflict="update",
    conflict_columns=["id"],
)

# Column mapping
chain.to_database(
    "users",
    engine,
    column_mapping={"user.name": "name", "user.email": "email", "internal_id": None},
)
```

Rows are written in batches (default 10,000). The `on_conflict` parameter controls duplicate handling: `"ignore"` skips, `"update"` overwrites. `column_mapping` renames columns and can exclude columns by mapping them to `None`.
