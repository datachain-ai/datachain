---
title: Agent Compounding Across Sessions
---

# Agent Compounding Across Sessions

## Without the harness

A coding agent over a folder of files starts every session from the same place: it lists the bucket, downloads the files, and re-runs the same enrichment the last session ran. Ten sessions over the same data run ten embedding jobs, ten classification passes, ten quality filters, all producing equivalent outputs that vanish when each session ends. The model gets faster and cheaper every quarter; the work the agent does on your data does not compound.

## With DataChain

Every chain you run deposits its result as a versioned typed dataset into the Dataset DB. The next session reads that dataset by name with full schema and lineage, and starts from the conclusion the last session reached. Capability over your data rises with usage rather than staying flat.

Below, three sessions build on each other. Each session saves a named dataset; the next session reads it as a premise.

```python
import datachain as dc
from PIL import Image
from sentence_transformers import SentenceTransformer

# Session 1: index the bucket, derive a breed from the filename, save typed records
(
    dc.read_storage("s3://dc-readme/oxford-pets-micro/", anon=True, type="image")
    .filter(dc.C("file.path").glob("*/images/*.jpg"))
    .map(breed=lambda file: file.path.split("/")[-1].rsplit("_", 1)[0])
    .save("pets")
)

# Session 2: read the saved dataset and add CLIP embeddings as a new column
encoder = SentenceTransformer("clip-ViT-B-32")

def embed(file: dc.ImageFile) -> list[float]:
    return encoder.encode(file.read()).tolist()

(
    dc.read_dataset("pets")
    .settings(parallel=8, cache=True)
    .map(emb=embed)
    .save("pet_embeddings")
)

# Session 3: read the embeddings, find nearest neighbours to a target image
target_emb = encoder.encode(Image.open("fiona.jpg")).tolist()

(
    dc.read_dataset("pet_embeddings")
    .filter(dc.C("breed") != "cocker_spaniel")
    .mutate(distance=dc.func.cosine_distance("emb", target_emb))
    .order_by("distance")
    .limit(10)
    .save("shiba_candidates")
)
```

Sessions 2 and 3 do not re-list the bucket and do not re-decode the JPEGs. They read the typed records the prior session produced, with the embedding column already present and a breed column the agent inferred from the path on the first pass.

## What this enables

- **Reusable conclusions.** A breed label or an embedding computed once survives every later session. The hundredth query runs in an environment qualitatively richer than the tenth.
- **Cheap iteration.** A new filter, a new threshold, or a new target image runs in seconds against `pet_embeddings`, not minutes against the bucket.
- **Discoverable work.** The Knowledge Base compiles `pets`, `pet_embeddings`, and `shiba_candidates` into pages your next agent reads as context before it generates code, so the next session does not duplicate work it could find.

## See also

- [Dataset DB](../concepts/dataset-db.md): why persistent typed datasets are the layer agents reason over
- [Knowledge Base](../concepts/knowledge-base.md): how saved datasets become agent-readable context
- [Datasets](../concepts/datasets.md): the unit of persistence
