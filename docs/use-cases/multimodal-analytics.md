---
title: Multimodal Analytics
---

# Multimodal Analytics

DataChain brings warehouse-grade analytics to unstructured and multimodal data -- cross-modal joins, metadata enrichment, and aggregate analytics all run at SQL speed.

## Cross-Modal Joins

Join files from storage with metadata from any format:

```python
import datachain as dc

# COCO-format annotations: one JSON with multiple arrays
images = dc.read_storage("gs://bucket/coco2017/images/val/")
meta = dc.read_json("gs://bucket/coco2017/annotations/captions_val2017.json", jmespath="images")
captions = dc.read_json("gs://bucket/coco2017/annotations/captions_val2017.json", jmespath="annotations")

images_meta = images.merge(meta, on="file.path", right_on="images.file_name")
captioned = images_meta.merge(captions, on="images.id", right_on="annotations.image_id")
captioned.save("captioned_images")
```

## Aggregate Analytics at Warehouse Speed

Run analytics on nested Pydantic objects without deserialization:

```python
import datachain as dc

chain = dc.read_dataset("llm_responses")

# Cost per model across thousands of calls
cost = (
    chain.sum("response.usage.prompt_tokens") * 0.000002
    + chain.sum("response.usage.completion_tokens") * 0.000006
)
print(f"Spent ${cost:.2f} on {chain.count()} calls")

# Group by file type
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

## Object Detection Pipeline

```python
import datachain as dc
from datachain import model
from pydantic import BaseModel

class YoloPose(BaseModel):
    bbox: model.BBox
    pose: model.Pose
    confidence: float

chain = (
    dc.read_storage("s3://frames/", type="image")
    .map(detections=run_yolo, output={"detections": list[model.BBox]})
    .map(poses=run_pose_estimator, output={"poses": list[model.Pose]})
    .save("annotated_frames")
)
```

## Similarity Search and Drift Detection

```python
import datachain as dc

# Top-10 similar images
ds = dc.read_dataset("image_embeddings")
top10 = (
    ds
    .mutate(dist=dc.func.cosine_distance(dc.C("emb"), query_embedding))
    .order_by("dist")
    .limit(10)
    .select("file.path", "dist")
)
top10.show()

# Model drift: compare two embedding columns
chain.mutate(drift=dc.func.cosine_distance(dc.C("emb_v1"), dc.C("emb_v2"))) \
    .filter(dc.C("drift") > 0.3) \
    .order_by("drift", descending=True)
```

## Window Functions for Ranking

```python
import datachain as dc

w = dc.func.window(partition_by="category", order_by="score")
chain.mutate(
    rank=dc.func.rank().over(w),
    row_num=dc.func.row_number().over(w),
    top_path=dc.func.first("file.path").over(w),
)
```

## Database Round-Trip

Enrich database records with AI, then write results back:

```python
import datachain as dc

(
    dc.read_database("SELECT id, text FROM reviews", "postgresql://host/db")
    .settings(parallel=8)
    .map(sentiment=classify_sentiment)
    .to_database("review_sentiments", "postgresql://host/db",
                 on_conflict="update", conflict_columns=["id"])
)
```
