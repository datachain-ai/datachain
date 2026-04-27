---
title: Data Engine Operations
---

# Data Engine Operations

Data operations execute directly in the Memory Engine (SQLite locally, ClickHouse in Studio). They never spin up Python runtimes, never download files, and scale to millions or billions of records.

**The rule: if it can be expressed as a data operation, it should be.** Never materialize a chain with `to_pandas()` or `to_list()` just to run aggregation or grouping in Python; use native operations instead.

## Native vs Non-Native

**Non-native** (pulls data into Python, iterates manually):

```python
import datachain as dc

files = dc.read_storage("gs://datachain-demo/").to_list()
totals = {}
for f, in files:
    ext = f.path.rsplit(".", 1)[-1].lower()
    total_bytes, file_count = totals.get(ext, (0, 0))
    totals[ext] = (total_bytes + f.size, file_count + 1)
```

**Native** (runs entirely in the Memory Engine):

```python
import datachain as dc

(
    dc.read_storage("gs://datachain-demo/")
    .filter(dc.C("file.size") > 0)
    .group_by(
        count=dc.func.count(),
        total=dc.func.sum(dc.C("file.size")),
        partition_by=dc.func.path.file_ext(dc.C("file.path")),
    )
    .order_by("total", descending=True)
    .show()
)
```

## Aggregate Analytics on Nested Objects

The Memory Engine reaches into Pydantic models serialized in the database, including deeply nested fields:

```python
import datachain as dc

chain = dc.read_dataset("llm_responses")

cost = (
    chain.sum("response.usage.prompt_tokens") * 0.000002
    + chain.sum("response.usage.completion_tokens") * 0.000006
)
print(f"Spent ${cost:.2f} on {chain.count()} calls")
```

`response.usage.prompt_tokens` traverses two levels of Pydantic nesting. The sum runs entirely in SQL: no Python runtime, no deserialization.

## Operations Catalog

Every operation returns a new chain. Chains are immutable and composable.

### Reading and Combining

- **merge**: join two chains on shared keys (like SQL JOIN)
- **union**: vertical concatenation (also the `|` operator)
- **subtract**: set difference, rows in left chain not in right
- **distinct**: deduplicate rows

```python
import datachain as dc

images = dc.read_storage("s3://bucket/images/", type="image")
labels = dc.read_json("s3://bucket/labels/*json", column="meta")
annotated = images.merge(labels, on="id", right_on="meta.id")
```

### Filtering and Selecting

- **filter**: keep rows matching a condition
- **select**: keep only named columns
- **select_except**: drop named columns
- **limit** / **offset**: pagination
- **sample**: random subset

```python
import datachain as dc

chain = dc.read_storage("s3://bucket/data/")
chain.filter(dc.C("score") > 0.9)
chain.filter((dc.C("meta.inference.class_") == "cat") & (dc.C("meta.confidence") > 0.93))
chain.filter(dc.C("detection.label").contains("person"))
```

### Sorting and Grouping

- **order_by**: sort by columns (ascending or descending)
- **group_by**: partition rows and apply aggregate functions
- **shuffle**: randomize row order

```python
import datachain as dc

chain.group_by(
    avg_size=dc.func.avg("file.size"),
    count=dc.func.count(),
    partition_by="category",
)

chain.order_by("score", descending=True)
```

### Mutate

`mutate` adds or replaces columns using native expressions that run at warehouse speed. It accepts `dc.func.*` expressions, column arithmetic, and column comparisons:

```python
import datachain as dc

chain = dc.read_storage("s3://bucket/images/").mutate(
    ext=dc.func.path.file_ext("file.path"),
    parent=dc.func.path.parent("file.path"),
)

chain.mutate(
    total=dc.C("price") * dc.C("quantity"),
    label=dc.func.case(
        (dc.C("score") > 0.9, "high"),
        (dc.C("score") > 0.5, "medium"),
        else_="low",
    ),
    is_large=dc.C("file.size") > 1_000_000,
)
```

Mutate does NOT accept lambdas or Python callables; use `map()` for those. See the [mutate vs map comparison](#mutate-vs-map) below.

### Comparing

- **diff**: compare two chains and surface differences (A/D/M/S status)
- **file_diff**: compare file-level changes between chains

### Configuring

- **settings**: set `parallel`, `cache`, `namespace`, `project`, and other execution parameters
- **setup**: run initialization code on each worker before processing
- **chunk**: split chain into fixed-size chunks for batch processing

### Terminal Operations

- **Persist**: `save` (named dataset), `persist` (anonymous cache)
- **Export**: `to_pandas`, `to_parquet`, `to_csv`, `to_json`, `to_jsonl`, `to_pytorch`, `to_storage`, `to_database`
- **Inspect**: `show`, `collect`, `count`, `sum`, `avg`, `min`, `max`

## Mutate vs Map

| Aspect      | `mutate`                                    | `map`                                        |
| ----------- | ------------------------------------------- | -------------------------------------------- |
| Runs in     | Memory Engine (SQL, data operation)         | Python runtime (Python function)             |
| Speed       | Warehouse speed, vectorized                 | Per-record Python execution                  |
| Accepts     | `dc.func.*`, column arithmetic, comparisons | Any Python callable                          |
| File access | No                                          | Yes (can read file content)                  |
| Parallelism | Automatic (engine-level)                    | Requires `.settings(parallel=True)`          |
| Use when    | Deriving columns from existing data         | Processing file content, calling models/LLMs |

## Complete Examples

### Merging Files with Metadata

The most common pattern: files in storage, annotations in a sidecar format. Read both, derive a join key, merge, filter, and export:

```python
import datachain as dc

images = dc.read_storage("gs://bucket/images/*jpg", anon=True)
meta = dc.read_json("gs://bucket/images/*json", column="meta", anon=True)

images_id = images.map(id=lambda file: file.path.split(".")[-2])
annotated = images_id.merge(meta, on="id", right_on="meta.id")

high_conf = annotated.filter(
    (dc.C("meta.inference.confidence") > 0.93)
    & (dc.C("meta.inference.class_") == "cat")
)
high_conf.to_storage("high-confidence-cats/", signal="file")
```

### LLM Cost Tracking

Compute API costs across thousands of calls using aggregate analytics on nested Pydantic fields, no deserialization, no Python runtime:

```python
import datachain as dc

chain = dc.read_dataset("llm_responses")

cost = (
    chain.sum("response.usage.prompt_tokens") * 0.000002
    + chain.sum("response.usage.completion_tokens") * 0.000006
)
print(f"Spent ${cost:.2f} on {chain.count()} calls")
```
