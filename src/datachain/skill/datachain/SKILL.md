---
name: datachain
description: Use for any work on files in object storage or local folders (S3, GCS, Azure, file://) - bucket inventories, counting/summing over many objects, image/video/audio metadata, embeddings, model or LLM labeling, similarity search, refreshing results after new files land, resuming a crashed run, and answering follow-up questions from earlier results. DataChain does the parallel I/O, keeps results as queryable datasets, and reprocesses only what changed.
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

<!-- practices:start -->
## Practices for heavy file work

- **Reuse before computing.** Look at what earlier sessions left in the project (saved results, scripts, logs).
  Reuse a result only if it was computed from the same files (same path + etag/version) with the same code and
  model; otherwise it is stale for that configuration. Answer questions about saved results or their provenance
  from those results and recorded code, without refreshing. For current-state questions, check source identities
  cheaply; refresh only confirmed changes or an explicitly requested rebuild. If freshness is unknown, say so.
- **Persist every expensive result** (embeddings, model outputs, extracted metadata) on all rows, keyed by file
  path + etag/version, together with the model and preprocessing that produced it. Answer follow-ups from it.
- **Read only what the question needs.** Explore a large bucket on one prefix or a small sample (`aws s3 ls` of one
  prefix), never by listing all of it just to look. List a bucket once and aggregate in the same pass. For large media read
  only the header bytes when metadata is enough (some containers keep it at the end: seek, don't assume a prefix); read small files whole. Filter by path before downloading.
  Read related small files (image + caption) in one pass, joined first on an explicit, validated key.
- **Use the machine.** CPU-bound work (decoding, models): one process per CPU (fewer if each loads a large model and memory is
  short), one torch thread and single-threaded decoders per process (PyAV `thread_count=1`, `cv2.setNumThreads(1)`). Count CPUs from the container's quota (`/sys/fs/cgroup/cpu.max`), not
  `os.cpu_count()`, which may report the host; more processes than the quota only adds throttling and memory.
  Network-bound work (header reads, API calls): many concurrent requests. Overlap downloads with compute. Batch
  model inference (e.g. 32 inputs per forward pass). Load models once per worker, in eval mode, without
  gradients. Bound queued downloads and inference work so memory stays flat.
- **Media correctness.** Sample video frames by timestamp, not by frame index (variable frame rate); apply
  rotation metadata; record the real timestamp of each frame. Normalize embeddings the same way for index and query.
- **Long jobs survive crashes.** Write results incrementally, keyed, so a killed run loses little; publish
  checkpoint parts atomically (one writer per result directory) and on re-run skip finished keys, never
  producing duplicates. A file that fails to decode is recorded with its error, not allowed to kill the run.
- **Updates equal a full recompute.** Bind reads to the listed identity (etag/version) and join file-derived
  results on path + etag/version. When files change, diff by (path, etag/version): process new and
  overwritten files, drop rows of deleted files. A requested model or preprocessing update requires recomputing
  the affected results; keep historical versions available for questions about earlier reports.
- **Run jobs to completion.** A job that may outlive one shell call (tool calls time out, often at 10 min) is launched
  in one call and waited for in later ones:
  `nohup python job.py > job.log 2>&1 & echo $! > job.pid` then, per call, `timeout 540 tail --pid=$(cat job.pid) -f /dev/null`,
  repeated until the PID is gone; then check the job's exit status (have it write `job.done` at the end), not a
  log line (progress bars rewrite lines). Never `pgrep -f` (it matches itself). Don't relaunch a running job.
  Do the next step only after the run finished. Test on a small slice first.
- **Match the process to the job.** Time the small slice and estimate the full run. If it takes a couple of
  minutes or less, run it directly in the foreground and check the result; background jobs, checkpoints and
  resume logic pay off only for long runs.
- **Check cheaply.** Use cheap independent checks (counts, keys, a few sampled rows); avoid recomputing a
  whole result in another tool. Filter, join and aggregate saved tables in a query engine before collecting
  rows into Python. Record cheap facts (size, resolution, duration) in the same pass that already reads the file.
- **Know when not to build anything.** A one-off question a single command answers in seconds (`ffprobe` on a
  few files, `aws s3 ls --summarize`): just run the command.
<!-- practices:end -->

# DataChain: the same practices, built in

## 0. Before computing anything: is it already computed?

```bash
datachain dataset ls                  # saved datasets in this project (run from the project root)
datachain dataset describe <name>    # schema, versions, lineage, sources and saved code
datachain show <name> --limit 5       # peek at rows and columns
```

Start with `dataset ls` and `dataset describe --json`; for provenance, check recorded code and counts there before scanning result rows.
Save expensive results under meaningful names and run named Python scripts so `query_script` captures their code.
`describe` reports recorded provenance; it does not check current files or models. Delta does not detect model/UDF changes: on a requested update, rebuild the affected results.
For saved-result and provenance questions, use `dc.read_dataset(name, version=...)` and recorded dependencies/code.
Do not rerun build/refresh scripts or `read_storage(..., update=True)` just to answer; check sources only when current freshness is part of the question.

For questions about row values (embeddings, metadata, labels), query the saved dataset with
`dc.read_dataset("<name>")`. Reuse its rows rather than recomputing them.

Datasets live in `.datachain/` of the project directory (DataChain uses the nearest `.datachain/` in the current
directory or its parents). Run scripts from the project directory: `cd /work && python scripts/x.py`.

## 1. Recipes

Every recipe is a complete script. Adjust the URI and fields; keep the structure.

### Bucket inventory (metadata only, nothing downloaded)

One question about a bucket (counts, bytes, per-folder or per-name-part totals): list once and aggregate in the
same loop. Nothing is saved, nothing is read twice; millions of objects take seconds to a minute.

```python
from collections import Counter
import datachain as dc

n, size = Counter(), Counter()
for f in dc.list_storage(
    "s3://bucket/prefix/", versions=False
):  # versions=False: current objects, faster
    key = f.path.split("/")[1], f.get_file_ext()  # any keys from the path or name
    n[key] += 1
    size[key] += f.size
```

Rows carry `path`, `size`, `etag`, `last_modified`, `source`, and `name`, `parent`, `get_file_ext()`. A
`list_storage` listing is not saved: each loop over it lists the bucket again, so compute everything in one loop.

Several passes or follow-up questions over the same listing: use `read_storage`. It saves the listing in the
project, and every query after that runs in SQL on the saved copy, no new listing:

```python
from datachain import func

files = dc.read_storage(
    "s3://bucket/prefix/", versions=False
)  # listed once, then reused
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


def info(file: dc.VideoFile) -> dc.Video:  # always annotate the return type
    return (
        file.get_info()
    )  # width/height in display orientation, fps, duration, frames, codec


meta = (
    dc.read_storage("s3://bucket/videos/", anon=True, type="video")
    .settings(
        parallel=64, prefetch=0
    )  # header reads wait on the network: many more processes than CPUs;
    # prefetch=0: read headers only, never download whole files
    .map(info=info)
    .save("video_meta")
)
print(
    meta.group_by(
        n=dc.func.count(), hours=dc.func.sum("info.duration"), partition_by="info.codec"
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
    width: int  # cheap facts recorded in the same pass, for later filters
    height: int


class ClipEmbed(dc.BatchMapper):
    def setup(self):  # once per worker process
        import open_clip, torch

        torch.set_num_threads(1)  # parallelism comes from settings(parallel=True)
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.eval()

    def process(
        self, file: list[dc.File]
    ) -> list[ImageEmb]:  # batch=32: lists in, one output per input
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


emb = (
    dc.read_storage(
        "s3://bucket/images/*.jpg", anon=True
    )  # glob filters at listing time
    .settings(
        parallel=True, prefetch=2
    )  # a process per CPU (container quota); prefetch overlaps downloads
    .batch_map(img=ClipEmbed(), batch=32)  # batched forward passes
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
import datachain as dc
from pydantic import BaseModel


class FrameEmb(BaseModel):
    video: (
        str  # gen() output replaces the input columns, so carry provenance explicitly
    )
    second: int  # the sampled target time t = 0, 1, 2, ... s
    pts: float  # actual timestamp of the decoded frame (first frame at/after t)
    embedding: list[float]


class EmbedFrames(dc.Generator):
    def setup(self):
        import open_clip, torch

        torch.set_num_threads(1)
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.eval()

    def process(
        self, file: dc.VideoFile
    ) -> Iterator[
        tuple[dc.VideoFile, FrameEmb]
    ]:  # the annotation defines the output schema
        import torch
        from PIL import Image

        frames = list(
            file.sample_frames(fps=1.0)
        )  # k-th frame: first frame at/after t=k s; VFR-safe, upright
        for i in range(0, len(frames), 32):  # batched forward passes
            chunk = frames[i : i + 32]
            with torch.no_grad():
                e = self.model.encode_image(
                    torch.stack([self.pre(Image.fromarray(f.get_np())) for f in chunk])
                )
            e = e / e.norm(dim=-1, keepdim=True)
            for k, (f, v) in enumerate(zip(chunk, e.tolist()), start=i):
                yield (
                    file,
                    FrameEmb(video=file.path, second=k, pts=f.timestamp, embedding=v),
                )


frames = (
    dc.read_storage("s3://bucket/videos/*.mov", anon=True, type="video")
    .settings(
        parallel=True, prefetch=4
    )  # prefetch>0: whole videos are needed for decoding
    .gen(EmbedFrames(), output={"file": dc.VideoFile, "frame": FrameEmb})
    .save("video_frame_embeddings")
)
```

Columns: `file.*`, `frame.video`, `frame.second`, `frame.pts`, `frame.embedding`.
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

Each `similarity_search` is one full scan (seconds per call at 100k rows). For more than a few queries (nearest
neighbour of every hit, all-pairs, many text prompts), read the vectors once and use one matrix product:

```python
import numpy as np

paths, embs = zip(
    *dc.read_dataset("image_embeddings").to_iter("file.path", "img.embedding")
)
E = np.asarray(embs, dtype=np.float32)  # rows are L2-normalized: dot = cosine
hits = [paths.index(p) for p in top_paths]
S = E[hits] @ E.T
S[np.arange(len(hits)), hits] = -np.inf  # exclude each item itself
nearest = [(paths[j], float(S[i, j])) for i, j in enumerate(S.argmax(1))]
```

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
joined = images.merge(
    labels, on="stem", inner=True
)  # right-side columns keep their names
```

Large tabular sources: `dc.read_parquet(uri)` / `dc.read_csv(uri)` / `dc.read_json(uri, format="jsonl")`.

### Keep results up to date when files are added, changed or deleted

```python
emb = (
    dc.read_storage(
        "s3://bucket/images/*.jpg", update=True, delta=True
    )  # update=True re-lists the bucket
    .settings(parallel=True, prefetch=2)
    .batch_map(img=ClipEmbed(), batch=32)
    .save("image_embeddings")
)
```

Re-run the same script after the bucket changes. Only new and changed files (path + etag/version) go through
the UDF; rows of overwritten files are replaced and rows of deleted files removed, so the new version equals a
full recompute. Existing results built before the delta replacement fix need a one-time
full rebuild under the same name with `update=True, delta=False`, then delta may be restored.
Put `update=True, delta=True` in the very first build script, so later refreshes are a re-run.

### Resume a crashed or killed run

Re-run the same script unchanged (or with the bug fixed in the UDF). Rows the UDF already finished are reused
from checkpoints; only the rest is computed. Do not rewrite the pipeline into a manual checkpointing loop.

### Export

`chain.to_parquet("out.parquet")`, `chain.to_pandas()`, `chain.to_list("col1", "col2")`,
`chain.to_iter("col")`. Nested columns flatten as `info.width`; in pandas they become `info.width` columns.

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
- **What `delta` tracks:** file changes (new, overwritten by etag/version, deleted). It does not track your code:
  when the model, its weights or the preprocessing change, save to a new dataset name (e.g. `frames_openai`)
  instead of re-running delta into the old one, or old-model rows are kept.
- Resume after a crash: re-run the same script; finished rows come from checkpoints. Don't hand-roll
  checkpointing.

See `reference.md` (same folder) for the API reference: read_storage arguments, settings, UDF classes,
func catalog, file types, delta details.
