---
title: Provenance
---

# Provenance

The provenance store records how every dataset was produced: what code ran, what inputs it consumed, who ran it, and when. Without provenance, datasets are opaque results that the team cannot verify, reproduce, or confidently build on.

## Captured Automatically at Every save()

The provenance store records four things for each dataset:

1. **Dependencies**: parent datasets with versions, storage URIs for every input
2. **Source code**: the full script, stored verbatim
3. **Author**: the person or service account that ran the script
4. **Creation time**

None of this requires manual declaration. It is captured from code and execution as a structural consequence of the operation.

## Dataset Registry

The dataset registry is the queryable system of record for all datasets and their versions. It lives inside the Memory Engine, making every dataset discoverable, joinable with other metadata, and accessible to agents without file-system traversal.

```python
import datachain as dc

# Browse all datasets
for info in dc.datasets().collect("dataset"):
    print(f"{info.name} v{info.version}")

# Inspect a specific dataset
ds = dc.read_dataset("image_embeddings")
ds.print_schema()
print(ds.name, ds.version)
```

## Why Provenance Matters

[Data Memory](data-memory.md) compounds only when deposits are trustworthy. If the next person cannot see what code made a dataset, what inputs it consumed, or who ran it, they start over, and compounding breaks. Provenance is the reason those downstream consumers can build forward instead of rebuilding from scratch.

## Metrics and Parameters

DataChain supports attaching metrics and reading parameters alongside provenance:

```python
import datachain as dc

results = (
    dc.read_dataset("training_data")
    .map(prediction=run_model)
    .save("predictions")
)

dc.metrics.set("accuracy", 0.95)
dc.metrics.set("f1_score", 0.91)

learning_rate = dc.param("learning_rate", 0.001)
```

Metrics are recorded in the registry alongside provenance. Parameters are captured in lineage so the exact configuration that produced a dataset is always recoverable.
