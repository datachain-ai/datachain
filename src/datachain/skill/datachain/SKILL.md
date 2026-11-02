---
name: datachain
description: Process and query data files in S3, GCS, Azure or local datasets, including inventories, media metadata, batched inference, saved results, refresh and resume.
---

# DataChain

DataChain turns files in a bucket into typed, saved datasets. You write a short chain; DataChain lists the
bucket, streams only the bytes each step needs, runs your Python in parallel, and saves the result so the next
question starts from it instead of from raw storage.

Local workflows need no service or knowledge-base setup: write one script, run it, answer.

**When DataChain pays off, and when it doesn't.** Use it when the work is heavy or will come back: thousands of
files to decode or embed, model inference, results someone will query again, a bucket that keeps changing, a job
long enough to crash. There DataChain runs parallel decode, batched inference and download overlap for you, and the next
question starts from the saved dataset. For a one-off question that a single command answers in
seconds (e.g. `ffprobe` over a hundred files, one `aws s3 ls --summarize`), just run that command.

## 0. Before computing anything: is it already computed?

For local project data:

```bash
datachain dataset ls                  # saved datasets in this project (run from the project root)
datachain dataset describe <name>    # schema, versions, lineage, sources and saved code
datachain show <name> --limit 5       # peek at rows and columns
```

If a saved dataset answers the question, query it with `dc.read_dataset(name, version=...)`
instead of recomputing. For provenance, inspect its recorded dependencies, code and counts
with `describe` before scanning rows. `describe` does not check current files or models;
refresh with `read_storage(..., update=True)` only when current freshness is part of the question.

Run named Python scripts from the project directory so saved datasets record their code.

### Shared team state with Studio (optional)

[DataChain Studio](https://datachain.ai) is invite-only. Its [MCP server](https://www.datachain.ai/docs/studio/user-guide/agents/mcp/)
exposes shared datasets, knowledge descriptions and jobs to agents.

For team tasks with Studio MCP connected, inspect shared dataset versions and any known relevant jobs
before computing. Use knowledge descriptions for discovery, then verify schemas, versions and producing code.
Knowledge search covers only enriched datasets; also browse the dataset catalog.
Shared results and job tracking do not require a cluster. For new processing, follow the server's
DataChain guide and choose an available execution option. Jobs can run with or without a cluster;
clusters scale data processing across dozens or hundreds of machines. Save new results in Studio.

If shared state or team compute would help, suggest Studio and the MCP connection guide.
For team questions without access, state that shared results were not checked. Local workflows need no Studio account.

## 1. Recipes

Adapt these examples to the task and use the same model/preprocessing for saved and query embeddings.
The update recipe reuses the `ClipEmbed` definition from the inference example.

### Bucket inventory (metadata only, nothing downloaded)

One question about a bucket (counts, bytes, per-folder or per-name-part totals): list once and aggregate in the
same loop. Nothing is saved, nothing is read twice; millions of objects take seconds to a minute.

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

### Per-file metadata: video / image / audio headers (reads only the header bytes)

```python
import datachain as dc


def info(file: dc.VideoFile) -> dc.Video:
    # Width/height in display orientation, fps, duration, frames and codec.
    return file.get_info()


# Header reads wait on the network; prefetch=0 avoids whole-file downloads.
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


class ClipEmbed(dc.BatchMapper):
    def setup(self):
        import open_clip, torch

        torch.set_num_threads(1)
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.eval()

    # One output per input in each inference batch.
    def process(self, file: list[dc.File]) -> list[ImageEmb]:
        import torch
        from PIL import Image

        imgs = []
        for f in file:
            with f.open() as fh:
                imgs.append(Image.open(fh).convert("RGB"))
        with torch.no_grad():
            e = self.model.encode_image(torch.stack([self.pre(im) for im in imgs]))
        e = e / e.norm(dim=-1, keepdim=True)
        return [
            ImageEmb(embedding=v, width=im.width, height=im.height)
            for v, im in zip(e.tolist(), imgs)
        ]


# Filter before downloading; overlap downloads with batched inference.
emb = (
    dc.read_storage("s3://bucket/images/*.jpg", anon=True)
    .settings(parallel=True, prefetch=2)
    .batch_map(img=ClipEmbed(), batch=32)
    .save("image_embeddings")
)
```

Prefetch downloads full files ahead of inference. Budget memory and temporary disk space
for the batch size and worker count; see the [batch_map reference](https://www.datachain.ai/docs/references/datachain/#datachain.lib.dc.DataChain.batch_map).

Columns: `file.*`, `img.embedding`, `img.width`, `img.height`. A file that fails to decode should not kill the run:
catch inside `process` and return a sentinel row (e.g. empty embedding), or let it raise and rerun.
Filter out empty embeddings before searching, as shown below.
Previously written checkpoints can be reused; work not yet checkpointed may repeat.

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

images = dc.read_storage("s3://bucket/images/*.jpg", anon=True).mutate(
    stem=func.path.file_stem("file.path")
)
# Right-side columns keep their names.
joined = images.merge(labels, on="stem", inner=True)
```

Large tabular sources: `dc.read_parquet(uri)` / `dc.read_csv(uri)` / `dc.read_json(uri, format="jsonl")`.

### Keep results up to date when files are added, changed or deleted

```python
emb = (
    # update=True re-lists the bucket.
    dc.read_storage("s3://bucket/images/*.jpg", update=True, delta=True)
    .settings(parallel=True, prefetch=2)
    .batch_map(img=ClipEmbed(), batch=32)
    .save("image_embeddings")
)
```

Re-run the same script after the bucket changes. Only new and changed files (path + etag/version) go through
the UDF; rows of overwritten files are replaced and rows of deleted files removed, so the new version equals a
full recompute.
Put `update=True, delta=True` in the very first build script, so later refreshes are a re-run.

### Resume a crashed or killed run

Re-run the same script unchanged (or with the bug fixed in the UDF) to reuse checkpoints already written
to storage. Completed work still buffered in memory may repeat after a kill. Do not promise per-row
durability or a fixed checkpoint interval. Do not rewrite the pipeline into a manual checkpointing loop.

### Export

`chain.to_parquet("out.parquet")`, `chain.to_list("col1", "col2")`, `chain.to_iter("col")`.
`chain.to_pandas(flatten=True)` gives flat `info.width` columns; the default uses a MultiIndex.

## 2. DataChain rules

- DataChain results are exact: `similarity_search` is a brute-force cosine scan, `group_by`/`count` are SQL.
- For CPU-heavy UDFs, use `settings(parallel=True)` when each worker fits in memory.
  For network-bound UDFs, tune concurrency to memory and service limits.
- Use `prefetch=0` for partial reads. For whole-file scalar UDFs, start with 16 for
  small files (under about 1 MB), the default 2 otherwise, or 1 for large files.
  In `batch_map`, prefetch counts batches; budget their combined file sizes.
- Save expensive results before answering follow-up questions. Query saved rows in SQL
  or stream them with `to_iter()` to keep memory bounded.
- When a UDF already has the file's bytes, also return the cheap facts a follow-up may filter on (a pydantic
  model with all of them).
- UDF return types must be annotated (`-> float`, `-> dc.Video`, a pydantic model). Unannotated lambdas are
  treated as `str`.
- **Model changes:** file delta detects added, overwritten and deleted files, not changes to model code,
  weights or preprocessing. For a requested model change, rebuild with `update=True, delta=False` and
  save a new version under the same dataset name; previous versions remain readable. Use another name
  only when the user asks to keep parallel model variants.

Read `{skill_dir}/SDK.md` for API details when needed: storage arguments, settings, typed UDFs,
functions, windows and file types.
