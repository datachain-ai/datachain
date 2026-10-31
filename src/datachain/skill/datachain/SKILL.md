---
name: datachain
description: Process and query data files in S3, GCS, Azure or local datasets, including inventories, media metadata, batched inference, saved results, refresh and resume.
---

# DataChain

DataChain turns files in a bucket into typed, saved datasets. You write a short chain; DataChain lists the
bucket, streams only the bytes each step needs, runs your Python in parallel, and saves the result so the next
question starts from it instead of from raw storage.

Work directly. No setup, no knowledge base, no bucket enlistment: write one script, run it, answer.

**When DataChain pays off, and when it doesn't.** Use it when the work is heavy or will come back: thousands of
files to decode or embed, model inference, results someone will query again, a bucket that keeps changing, a job
long enough to crash. There DataChain runs parallel decode, batched inference and download overlap for you, and the next
question starts from the saved dataset. For a one-off question that a single command answers in
seconds (e.g. `ffprobe` over a hundred files, one `aws s3 ls --summarize`), just run that command.

## 0. Before computing anything: is it already computed?

```bash
datachain dataset ls                  # saved datasets in this project (run from the project root)
datachain dataset describe <name>    # schema, versions, lineage, sources and saved code
datachain show <name> --limit 5       # peek at rows and columns
```

Start with `dataset ls` and `dataset describe --json`; for provenance, check recorded code and counts there before scanning result rows.
Save expensive results under meaningful names and run named Python scripts so `query_script` captures their code.
`describe` reports recorded provenance; it does not check current files or models.
For saved-result and provenance questions, use `dc.read_dataset(name, version=...)` and recorded dependencies/code.
Do not rerun build/refresh scripts or `read_storage(..., update=True)` just to answer; check sources only when current freshness is part of the question.

For questions about row values (embeddings, metadata, labels), query the saved dataset with
`dc.read_dataset("<name>")`. Reuse its rows rather than recomputing them.

Run scripts from the project directory: `cd /work && python scripts/x.py`. Inside a Git repository,
DataChain uses the nearest `.datachain/` containing a `db` or `config` file, stopping at the nearest
repository root; without existing state, it uses that root. Outside Git, it uses the current directory.
`DATACHAIN_DIR` and `DATACHAIN_ROOT_DIR` override discovery.

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
`replace`, `regexp_replace`. Aggregate in SQL first; don't `to_list()` millions of rows of a saved listing.

### Per-file metadata: video / image / audio headers (reads only the header bytes)

```python
import datachain as dc


def info(file: dc.VideoFile) -> dc.Video:
    # Width/height in display orientation, fps, duration, frames and codec.
    return file.get_info()


# Header reads wait on the network; prefetch=0 avoids whole-file downloads.
meta = (
    dc.read_storage("s3://bucket/videos/", anon=True, type="video")
    .settings(parallel=64, prefetch=0)
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

`prefetch=2` counts batches here: each worker may download two batches concurrently.

Columns: `file.*`, `img.embedding`, `img.width`, `img.height`. A file that fails to decode should not kill the run:
catch inside `process` and return a sentinel row (e.g. empty embedding), or let it raise and re-run - finished
rows are kept (see Resume).

### Video frames (one row per sampled frame, with source video and timestamp)

Install `datachain[video]` and the model dependencies (`torch`, `open_clip_torch`) for this recipe.

```python
from collections.abc import Iterator
from itertools import islice
import datachain as dc
from pydantic import BaseModel


class FrameEmb(BaseModel):
    index: int
    # Actual timestamp of the decoded frame, in seconds.
    pts: float
    embedding: list[float]


class EmbedFrames(dc.Generator):
    def setup(self):
        import open_clip, torch

        torch.set_num_threads(1)
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.eval()

    def process(self, file: dc.VideoFile) -> Iterator[tuple[dc.VideoFile, FrameEmb]]:
        import torch
        from PIL import Image

        frames = file.sample_frames(fps=1.0)
        index = 0
        while chunk := list(islice(frames, 32)):
            with torch.no_grad():
                e = self.model.encode_image(
                    torch.stack([self.pre(Image.fromarray(f.get_np())) for f in chunk])
                )
            e = e / e.norm(dim=-1, keepdim=True)
            for f, v in zip(chunk, e.tolist()):
                yield file, FrameEmb(index=index, pts=f.timestamp, embedding=v)
                index += 1
            del chunk, f


frames = (
    dc.read_storage("s3://bucket/videos/*.mov", anon=True, type="video")
    # Whole videos are needed for decoding.
    .settings(parallel=True, prefetch=4)
    .gen(EmbedFrames(), output={"file": dc.VideoFile, "frame": FrameEmb})
    .save("video_frame_embeddings")
)
```

Columns: `file.*`, `frame.index`, `frame.pts`, `frame.embedding`.
Keep the top-level `file` output for source identities and compatible file-delta updates.
Consume or embed sampled pixels inside the generator; saved frame references do not retain decoded buffers.

### Search saved embeddings (follow-up questions)

```python
import datachain as dc

q = text_or_image_embedding()  # same model and normalization as the saved embeddings
top = (
    dc.read_dataset("image_embeddings").similarity_search(
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

rows = list(
    dc.read_dataset("image_embeddings").to_iter(
        "file.source", "file.path", "img.embedding"
    )
)
identities = [(source, path) for source, path, _ in rows]
# Embeddings use the same L2 normalization, so dot products are cosine similarities.
E = np.asarray([embedding for _, _, embedding in rows], dtype=np.float32)
del rows


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

Re-run the same script unchanged (or with the bug fixed in the UDF). Rows the UDF already finished are reused
from checkpoints; only the rest is computed. Do not rewrite the pipeline into a manual checkpointing loop.

### Export

`chain.to_parquet("out.parquet")`, `chain.to_list("col1", "col2")`, `chain.to_iter("col")`.
`chain.to_pandas(flatten=True)` gives flat `info.width` columns; the default uses a MultiIndex.

## 2. DataChain rules

- DataChain results are exact: `similarity_search` is a brute-force cosine scan, `group_by`/`count` are SQL.
- Inventories and one-off totals: `list_storage` + one loop. Processing files: one `read_storage` per bucket
  prefix; filter with a glob in the URI (`.../*.jpg`, `.../**/*.mov`) or `.filter(...)` on `file.path` before
  any `map`.
- `settings(parallel=True)` (one process per CPU of the container quota; never a hard-coded count) for CPU-heavy UDFs (decoding, models).
  Network-bound UDFs (header reads, small range reads, API calls) wait, not compute: `parallel=64, prefetch=0`.
- Header-only reads (`prefetch=0`) pay off for large files (videos, audio, multi-MB images). Many small files
  (thumbnails, captions, JSON sidecars under ~1 MB) are cheaper to read whole with `prefetch=16`. Read related
  small files in one pass: join them first (e.g. image + caption by `func.path.file_stem`), then one `map`.
- `.save(name)` every expensive result, then filter and aggregate downstream from the saved dataset in SQL
  (`filter`/`group_by`/`order_by`), no Python loop needed. Don't `to_list()`/`to_pandas()` millions of rows.
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
