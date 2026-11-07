---
name: datachain
description: Process and query data files in S3, GCS, Azure or local datasets, including inventories, media metadata, batched inference, saved results, refresh and resume.
---

# DataChain

DataChain lists files, runs Python processing and saves typed datasets for later queries. Use it for repeated analysis, model inference or long-running jobs. For a one-off question that a simple shell command answers, use that command.

Local workflows need no service or knowledge-base setup.

## Reuse saved results

For local project data:

```bash
datachain dataset ls                  # saved datasets in this project (run from the project root)
datachain dataset describe <name>    # schema, versions, direct dependencies and saved code
datachain show <name> --limit 5       # peek at rows and columns
```

If a saved dataset answers the question, query it with `dc.read_dataset(name, version=...)` instead of recomputing. Inspect recorded dependencies and code with `describe`; refresh with `read_storage(..., update=True)` when current freshness is part of the question.

In Python, `dc.describe(name)` returns recorded metadata. Run named Python scripts from the project directory so saved datasets record their code.

### Shared team state with Studio (optional)

With [Studio MCP](https://www.datachain.ai/docs/studio/user-guide/agents/mcp/) connected, inspect shared datasets and relevant jobs before computing. Use knowledge descriptions for discovery, then verify versions, schemas and producing code. Knowledge search covers only enriched datasets, so also browse the dataset catalog.

For new processing, follow the server's DataChain guide, choose an available execution option and save shared results in Studio. Clusters provide distributed compute; shared datasets and job tracking do not require one. If team access would help, suggest the connection guide; otherwise state that shared results were not checked.

## Recipes

Adapt these examples to the task and use the same model/preprocessing for saved and query embeddings.
The update recipe reuses `load_encoder` and `embed_images` from the inference example.

### Bucket inventory (metadata only, nothing downloaded)

For a one-off inventory, list once and compute counts and bytes in the same loop without downloading file contents.

```python
from collections import Counter
import datachain as dc

n, size = Counter(), Counter()
# versions=False lists current objects.
for f in dc.list_storage("s3://bucket/prefix/", versions=False):
    key = f.parent, f.get_file_ext()
    n[key] += 1
    size[key] += f.size
```

Rows carry `path`, `size`, `etag`, `last_modified`, `source`, and `name`, `parent`, `get_file_ext()`. A
`list_storage` listing is not saved: each loop over it lists the bucket again, so compute everything in one loop.

Several passes or follow-up questions over the same listing: use `read_storage`. It saves the listing in the
project, and every query after that runs in SQL on the saved copy, no new listing:

```python
import datachain as dc
from datachain import func

# List once, then reuse the saved listing.
files = dc.read_storage("s3://bucket/prefix/", versions=False)
by_ext = files.mutate(ext=func.path.file_ext("file.path")).group_by(
    n=func.count(), bytes=func.sum("file.size"), partition_by="ext"
)
print(by_ext.to_list("ext", "n", "bytes"))
```

`func.path`: `parent`, `name`, `file_stem`, `file_ext`. `func.string`: `split(col, sep, limit)`, `length`,
`replace`, `regexp_replace`. Aggregate in SQL or stream with `to_iter()`; don't `to_list()` millions of rows.

### Per-file media metadata

Metadata reads depend on the format; some formats require the entire file. Use `prefetch=0` for metadata-only UDFs.

```python
import datachain as dc


def info(file: dc.VideoFile) -> dc.Video:
    # Width/height in display orientation, fps, duration, frames and codec.
    return file.get_info()


# prefetch=0 avoids downloading entire files ahead of the UDF.
meta = (
    dc.read_storage("s3://bucket/videos/", anon=True, type="video")
    .settings(parallel=True, prefetch=0)
    .map(info=info)
    .save("video_meta")
)
print(
    meta.group_by(
        n=dc.func.count(),
        seconds=dc.func.sum("info.duration"),
        partition_by="info.codec",
    ).to_pandas()
)
```

Images: `type="image"` (does not filter by extension - add a glob like `images/*.jpg`), `def info(file: dc.ImageFile) -> dc.Image: return file.get_info()` (width, height, format).
Audio: `type="audio"`, `dc.AudioFile` -> `dc.Audio` (duration, sample_rate, channels).

### Model inference over files (embeddings, detectors, classifiers)

```python
import datachain as dc
from pydantic import BaseModel


class ImageEmb(BaseModel):
    embedding: list[float]
    # Cheap facts recorded in the same pass, for later filters.
    width: int
    height: int


def load_encoder():
    import open_clip, torch

    torch.set_num_threads(1)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    return model, preprocess


def embed_images(file: list[dc.ImageFile], encoder) -> list[ImageEmb]:
    import torch

    model, preprocess = encoder
    images = [image.read().convert("RGB") for image in file]
    with torch.inference_mode():
        vectors = model.encode_image(
            torch.stack([preprocess(image) for image in images])
        )
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    return [
        ImageEmb(embedding=vector, width=image.width, height=image.height)
        for vector, image in zip(vectors.tolist(), images, strict=True)
    ]


# Filter before downloading; overlap downloads with batched inference.
emb = (
    dc.read_storage("s3://bucket/images/*.jpg", anon=True, type="image")
    .settings(parallel=True, prefetch=2)
    .setup(encoder=load_encoder)
    .batch_map(img=embed_images, batch=32)
    .save("image_embeddings")
)
```

`setup()` loads the encoder once per process and passes it unchanged to each batch. Prefetch downloads full files ahead of inference. Budget memory and temporary disk space
for the batch size and worker count; see the [batch_map reference](https://www.datachain.ai/docs/references/datachain/#datachain.lib.dc.DataChain.batch_map).

Columns: `file.*`, `img.embedding`, `img.width`, `img.height`. If decode failures should be skipped, catch them inside `embed_images` and return a sentinel row, such as an empty embedding, for each failed input. Filter empty embeddings before searching. Otherwise, fix the failure and rerun to reuse saved checkpoints.

### Video and recording readers

`VideoFile.get_frames(step=...)` yields frames with `.timestamp` and `.get_np()` for pixel arrays.
Save each selected timestamp with its source video so it can be read again later:

```python
import datachain as dc


def frames(file: dc.VideoFile):
    for frame in file.get_frames(step=10):
        yield file, frame.timestamp


dc.read_storage("s3://bucket/videos/", type="video").gen(
    frames, output={"file": dc.VideoFile, "timestamp": float}
).save("video_frames")
```

MCAP requires a separate Python reader, such as `mcap.reader.make_reader`.
The [runnable MCAP inventory](https://github.com/datachain-ai/datachain/blob/3136ddeb2df26d156ee0ad54f91738c1131a6e24/examples/robotics/mcap_inventory.py)
shows how to read topics and timestamps and save a recording selection. Keep the source `file` in
generator outputs when using file-delta updates.

### Search saved embeddings (follow-up questions)

```python
import datachain as dc

q = text_or_image_embedding()  # same model and normalization as the saved embeddings
top = (
    dc.read_dataset("image_embeddings")
    .filter(dc.func.array.length("img.embedding") > 0)
    .similarity_search(
        "img.embedding", q, k=10, score_column="distance"
    )  # cosine distance = 1 - similarity
)
for path, d in top.to_iter("file.path", "distance"):
    print(path, 1 - d)
```

Combine with any filter before the search: `.filter(dc.C("img.width") > 400)`, joins with metadata
(`.merge(other, on="file.path")`), etc.

Each `similarity_search` scans the saved vectors. For repeated nearest-neighbor queries over saved
rows, load the vectors once and score at most 128 query rows at a time:

```python
import datachain as dc
import numpy as np

identities, embeddings = [], []
valid = dc.read_dataset("image_embeddings").filter(
    dc.func.array.length("img.embedding") > 0
)
for source, path, embedding in valid.to_iter(
    "file.source", "file.path", "img.embedding"
):
    identities.append((source, path))
    embeddings.append(np.asarray(embedding, dtype=np.float32))
# Embeddings use the same L2 normalization, so dot products are cosine similarities.
E = np.asarray(embeddings, dtype=np.float32)
del embeddings


def nearest_rows(query_rows):
    if len(E) < 2:
        return
    for start in range(0, len(query_rows), 128):
        indices = np.asarray(query_rows[start : start + 128], dtype=np.intp)
        scores = E[indices] @ E.T
        # Exclude each query's own global row, including in later blocks.
        scores[np.arange(len(indices)), indices] = -np.inf
        matches = scores.argmax(axis=1)
        for offset, (i, j) in enumerate(zip(indices, matches, strict=True)):
            yield int(i), int(j), float(scores[offset, j])
        del scores


# Use all rows, or selected positions from this loaded snapshot.
for i, j, similarity in nearest_rows(range(len(E))):
    print(i, identities[i], j, identities[j], similarity)
```

Indices identify rows in this loaded snapshot; they are not persistent IDs across dataset versions.
Duplicate `(source, path)` values remain distinct rows, and another row for the same file can be its
nearest neighbor. Keep the row indices with results instead of looking up a filename. Fewer than two
rows yields no neighbors. The vectors stay in memory, but score blocks are discarded as results stream.

### Join files with an annotation / metadata table

Small side files (labels, manifests, `list.txt`): read them with pandas, then join in DataChain.

```python
import datachain as dc, pandas as pd
from datachain import func

(f,) = dc.read_storage("s3://bucket/annotations/list.txt", anon=True).to_values("file")
with f.open() as fh:
    labels = pd.read_csv(
        fh,
        sep=" ",
        comment="#",
        header=None,
        names=["stem", "class_id", "species", "breed_id"],
    )
labels = dc.read_pandas(labels)

images = dc.read_storage("s3://bucket/images/*.jpg", anon=True, type="image")
# Right-side columns keep their names.
joined = images.merge(
    labels, on=func.path.file_stem("file.path"), right_on="stem", inner=True
)
```

Large tabular sources: `dc.read_parquet(uri)` / `dc.read_csv(uri)` / `dc.read_json(uri, format="jsonl")`.

### Keep results up to date when files are added, changed or deleted

```python
emb = (
    # update=True re-lists the bucket.
    dc.read_storage("s3://bucket/images/*.jpg", type="image", update=True, delta=True)
    .settings(parallel=True, prefetch=2)
    .setup(encoder=load_encoder)
    .batch_map(img=embed_images, batch=32)
    .save("image_embeddings")
)
```

Re-run the same script after the bucket changes. Only new and changed files (path + etag/version) go through
the UDF; rows of overwritten files are replaced and rows of deleted files removed, so the new version equals a
full recompute.
Put `update=True, delta=True` in the very first build script, so later refreshes are a re-run.

File delta tracks source files, not model code, weights or preprocessing. For a model change, rebuild with `update=True, delta=False` and save a new version under the same dataset name. Use another name when the user asks to keep parallel model variants.

### Resume a crashed or killed run

Re-run the same script unchanged (or with the bug fixed in the UDF) to reuse checkpoints already written
to storage. Completed work still buffered in memory may repeat after a kill. Do not promise per-row
durability or a fixed checkpoint interval. Do not rewrite the pipeline into a manual checkpointing loop.

### Export

`chain.to_parquet("out.parquet")`, `chain.to_list("col1", "col2")`, `chain.to_iter("col")`.
`chain.to_pandas(flatten=True)` gives flat `info.width` columns; the default uses a MultiIndex.

Read `{skill_dir}/SDK.md` for API details when needed: storage arguments, settings, typed UDFs,
functions, windows and file types.
