---
title: Datasets
---

# Datasets

A dataset is a unit of reasoning, not just storage. It is the atom of [Data Memory](data-memory.md): a named, versioned collection of typed records. Every query produces one; every subsequent query starts from one. Each dataset encodes a conclusion the team or an agent reached, and the next consumer starts from that conclusion as a settled fact.

## Immutability Is the Foundation of Trust

A dataset version, once saved, never changes. New work produces a new version, never overwrites the old. This is what makes datasets safe to build on; the next person or agent knows that the thing they read is the same thing the author saved.

```python
import datachain as dc

dc.read_values(x=[1, 2, 3]).save("experiment")        # v1.0.0
dc.read_values(x=[1, 2, 3, 4]).save("experiment")      # v1.0.1

ds = dc.read_dataset("experiment", version="1.0.0")     # exact version
ds = dc.read_dataset("experiment")                       # latest
```

## Typed by Construction

Schemas are Pydantic models, not SQL DDL. A dataset carries all data as typed columns: file references, embeddings, annotations, scores, nested objects, database fields. A Physical AI dataset might have 1000+ columns across sensor measurements, multi-level annotations, and LLM response objects. This is what makes datasets fundamentally different from flat SQL tables; the type system matches the complexity of the data.

```python
from pydantic import BaseModel
import datachain as dc

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]

(
    dc.read_storage("s3://bucket/frames/", type="image")
    .map(det=run_detector)   # returns Detection
    .save("detections")
)
```

## Types as Code, Not Configuration

Pydantic models are the same Python objects the Compute Engine operates on and the columnar schemas Data Memory stores. The schema is the code; there is no mapping layer between Python types and warehouse columns, no separate registration step, no parallel YAML inventory to maintain. YAML-declared schemas in semantic-layer products drift the moment work moves faster than maintenance. Every save deposits typed records the engines already understand; every load returns the same Python objects the next operation consumes.

## Two-Level Data Model

DataChain has a two-level data model: the dataset **name** (durable identity) and the dataset **version** (operational unit). Each `save()` creates a new version with semver auto-increment. The name groups all versions into a logical unit.

```python
import datachain as dc

chain = dc.read_values(x=[1, 2, 3])
chain.save("experiment")                              # 1.0.0
chain.save("experiment", update_version="patch")      # 1.0.1
chain.save("experiment", update_version="minor")      # 1.1.0
chain.save("experiment", update_version="major")      # 2.0.0
```

Lineage references track specific versions, not names; the registry records exactly which version was consumed.

## Sharing Through Datasets

The dataset is the collaboration primitive. When one person saves a dataset, the next person starts from it, not from a re-run, not from a shared folder, not from a Slack message with a path. Provenance makes each dataset self-describing. The knowledge base makes it discoverable.

```python
import datachain as dc

# Load someone else's dataset and continue
enriched = (
    dc.read_dataset("scan_classifications")
    .filter(dc.C("diagnosis") == "anomaly")
    .map(severity=score_severity, params=["file"])
    .save("anomaly_severity")
)
```

## Namespaces

Datasets are organized by a two-level namespace system: `team.project.name`. The default namespace is `@<username>.default.<name>`.

```python
import datachain as dc

# Save with explicit namespace
dc.read_storage("s3://bucket/images/", type="image") \
    .map(emb=clip_embedding) \
    .save("quant.prod.image_embeddings")

# Read with fully qualified name
ds = dc.read_dataset("quant.prod.image_embeddings")
```

Override the default namespace with `DATACHAIN_NAMESPACE` environment variable or `.settings(namespace=, project=)`.

## Comparing Versions

`diff()` compares two chains row by row:

```python
import datachain as dc

v1 = dc.read_dataset("experiment", version="1.0.0")
v2 = dc.read_dataset("experiment", version="2.0.0")

changes = v2.diff(v1, on="id", compare=["score", "label"])
changes.show()
```

Each row carries a status: **A** (added), **D** (deleted), **M** (modified), or **S** (same). `file_diff()` compares file-level changes between storage scans.

## Reasoning

Each dataset encodes a conclusion: what the team or an agent found, proved, or disproved. A dataset of classified documents is not "rows with a label column"; it is the claim "these documents have been classified by this model with this confidence", typed and versioned and backed by provenance that makes it verifiable. The next query consumes that claim as a premise rather than re-deriving it.

This is what makes [Data Memory](data-memory.md) composable. The next person or agent starts from a conclusion as a settled fact and builds forward. Agents depend on this property structurally: an agent that receives a dataset treats it as an established fact in its reasoning chain. Without this, every agent interaction starts from raw bytes and re-derives everything.
