---
title: Getting Started
---

# Getting Started

## Installation

=== "pip"

    ```bash
    pip install datachain
    ```

=== "uv"

    ```bash
    uv add datachain
    ```

## What Is DataChain

DataChain is a Python-native data platform for AI teams. It unifies files, database records, and structured formats into versioned, typed datasets with automatic lineage tracking. Every operation deposits results into persistent memory that the team builds on.

```python
import datachain as dc

(
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, cache=True)
    .map(emb=compute_embedding)
    .save("image_embeddings")
)
```

This single pipeline reads images from S3, computes embeddings in parallel, and saves a versioned dataset. The next person (or agent) loads it with `dc.read_dataset("image_embeddings")` and builds on it.

## Import Convention

Always import DataChain as a module:

```python
import datachain as dc
```

Access everything through the `dc.*` prefix: `dc.read_storage()`, `dc.Column()`, `dc.func.*`, `dc.File`, `dc.ImageFile`, etc.

For annotation types and custom models:

```python
from datachain import model      # BBox, Pose, Segment, etc.
from pydantic import BaseModel   # for custom types
```

## Next Steps

- [Quick Start](../quick-start.md) -- practical examples with code
- [Core Concepts](core-concepts.md) -- the mental model behind DataChain
- [Guides](../guide/index.md) -- in-depth coverage of each capability
