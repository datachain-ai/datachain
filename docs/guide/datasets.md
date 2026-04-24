---
title: Datasets
---

# Working with Datasets

Datasets are the primary output and collaboration mechanism in DataChain. This guide covers creating, managing, versioning, and sharing datasets.

## Creating Datasets

### save() -- Named, Versioned, Shared

`save("name")` creates an immutable, versioned dataset in the registry. It records author, dependencies, and code automatically.

```python
import datachain as dc

embeddings = (
    dc.read_storage("s3://bucket/images/", type="image")
    .map(emb=clip_embedding)
    .save("image_embeddings")
)
```

**`save()` is the default terminal operation** -- not `show()` or `to_list()`. Any chain that runs a Python operation should end in `save("name")`. This makes pipelines reusable instead of producing throwaway output.

### persist() -- Anonymous, Script-Local

`persist()` creates a hidden dataset for caching within a script. Not in the registry, not versioned, not shared.

```python
import datachain as dc

base = (
    dc.read_storage("s3://bucket/images/", type="image")
    .map(emb=clip_embedding)
    .persist()
)

large = base.filter(dc.C("file.size") > 10_000).save("large_images")
small = base.filter(dc.C("file.size") <= 10_000).save("small_images")
```

## Versioning

Every `save()` auto-increments the version:

```python
import datachain as dc

dc.read_values(x=[1, 2, 3]).save("experiment")        # v1.0.0
dc.read_values(x=[1, 2, 3, 4]).save("experiment")      # v1.0.1
dc.read_values(x=[1, 2, 3, 4, 5]).save("experiment")   # v1.0.2
```

Read specific versions:

```python
ds = dc.read_dataset("experiment")                       # latest
ds = dc.read_dataset("experiment", version="1.0.0")      # exact
ds = dc.read_dataset("experiment", version=">=1.0,<3.0") # PEP 440 range
ds = dc.read_dataset("experiment", version=2)             # legacy int
```

Control version bumps:

```python
import datachain as dc

chain = dc.read_values(x=[1, 2, 3])
chain.save("experiment")                              # 1.0.0
chain.save("experiment", update_version="patch")      # 1.0.1
chain.save("experiment", update_version="minor")      # 1.1.0
chain.save("experiment", update_version="major")      # 2.0.0
```

## Namespaces

Two-level namespace system: `team.project.name`.

```python
import datachain as dc

# Save with explicit namespace
dc.read_storage("s3://bucket/images/", type="image") \
    .map(emb=clip_embedding) \
    .save("quant.prod.image_embeddings")

# Read with fully qualified name
ds = dc.read_dataset("quant.prod.image_embeddings")
```

Override defaults with environment variables:

```bash
export DATACHAIN_NAMESPACE="quant.prod"
```

Or per-chain:

```python
import datachain as dc

dc.read_values(scores=[1.2, 3.4, 2.5]) \
    .settings(namespace="dev", project="analytics") \
    .save("metrics")
```

Dataset URI scheme: `ds://namespace.project.name@v1.2.3`.

## Comparing Versions

`diff()` compares two chains row by row:

```python
import datachain as dc

v1 = dc.read_dataset("experiment", version="1.0.0")
v2 = dc.read_dataset("experiment", version="2.0.0")

changes = v2.diff(v1, on="id", compare=["score", "label"])
changes.show()
```

Status values: **A** (added), **D** (deleted), **M** (modified), **S** (same).

`file_diff()` compares file-level changes:

```python
import datachain as dc

old = dc.read_storage("s3://bucket/images/", version="2025-01-01")
new = dc.read_storage("s3://bucket/images/", update=True)

changes = new.file_diff(old)
added = changes.filter(dc.C("diff_status") == "A")
```

## Management

```python
import datachain as dc

# List all datasets
for info in dc.datasets().collect("dataset"):
    print(f"{info.name} v{info.version}")

# Inspect schema
ds = dc.read_dataset("image_embeddings")
ds.print_schema()

# Delete, move
dc.delete_dataset("old_experiment")
dc.delete_namespace("scratch")
dc.move_dataset("draft_results", "prod.validated_results")

# Add description and attributes
dc.read_storage("s3://bucket/images/", type="image") \
    .map(emb=clip_embedding) \
    .save("image_embeddings",
          description="CLIP embeddings for product images",
          attrs=["NLP", "location=US"])
```

## Metrics and Parameters

Attach quantitative metrics to dataset versions:

```python
import datachain as dc

results = (
    dc.read_dataset("training_data")
    .map(prediction=run_model)
    .save("predictions")
)

dc.metrics.set("accuracy", 0.95)
dc.metrics.set("f1_score", 0.91)

accuracy = dc.metrics.get("accuracy")
```

Read parameters from the environment:

```python
import datachain as dc

learning_rate = dc.param("learning_rate", 0.001)
batch_size = dc.param("batch_size", 32)
```

Parameters are captured in lineage so the exact configuration is always recoverable.

## CLI Commands

```bash
datachain dataset ls                    # list all datasets
datachain dataset rm <name>             # delete
datachain dataset edit <name>           # modify metadata
datachain dataset pull <name>           # pull remote dataset to local
datachain show <name>                   # display schema and info
datachain gc                            # garbage-collect orphans
```

CLI commands accept `--studio` or `--local` flags.

## Studio Sync

`read_dataset()` auto-falls-back to Studio when a dataset is not found locally. Use the CLI to pull datasets explicitly:

```bash
datachain dataset pull quant.prod.embeddings --studio
```
