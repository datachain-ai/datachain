---
title: Vector Search
---

# Vector Search

Vector search is a built-in capability in DataChain using `cosine_distance()`, `euclidean_distance()`, and `l2_distance()`. Create a vector (list of floats) and run searches against it.

## Full Example: CLIP Embedding + Similarity Search

```python
import numpy as np
import datachain as dc
from PIL import Image
from sentence_transformers import SentenceTransformer

SRC = "gs://datachain-demo/dogs-and-cats/*jpg"
QUERY_LOCAL_IMG = "mycat.jpg"
model = SentenceTransformer("clip-ViT-B-32")

def clip_embedding(file: dc.ImageFile) -> list[float]:
    img = file.read().convert("RGB")
    emb = model.encode(img).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.tolist()

# 1) Build embeddings for all images in storage
ds = (
    dc.read_storage(SRC, type="image", anon=True)
    .map(emb=clip_embedding)
    .persist()
)

# 2) Compute embedding for a local query image
query_img = Image.open(QUERY_LOCAL_IMG).convert("RGB")
query_emb = model.encode(query_img).astype(np.float32)
query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-12)
query_emb = query_emb.tolist()

# 3) Similarity search (Top-10 closest images)
top10 = (
    ds
    .mutate(dist=dc.func.cosine_distance(dc.C("emb"), query_emb))
    .order_by("dist")
    .limit(10)
    .select("file.path", "dist")
)
top10.show()
```

The pattern: compute embeddings via `map()` (Python), then search via `mutate` + `order_by` + `limit` (Memory Engine SQL). The similarity search runs at warehouse speed.

## Model Drift Detection

Vector distances work as analytical tools. Compare embeddings per record to reveal drift or disagreement:

```python
import datachain as dc

chain.mutate(drift=dc.func.cosine_distance(dc.C("emb_a"), dc.C("emb_b"))) \
    .filter(dc.C("drift") > 0.3) \
    .order_by("drift", descending=True)
```

This runs entirely in the Memory Engine -- no Python, no deserialization.
