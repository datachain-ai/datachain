---
title: Reading Data
---

# Reading Data

DataChain reads data from many sources and formats through a family of `read_*` entry points. Each returns a lazy chain: nothing executes until a terminal operation triggers it.

## Storage Files

`read_storage()` connects to any supported storage provider and returns a chain of File objects. It is the primary entry point for unstructured data: images, video, audio, PDFs, text. Always include a trailing slash in bucket and prefix URIs.

```python
import datachain as dc

images = dc.read_storage("s3://bucket/images/**/*.jpg", type="image")
videos = dc.read_storage("gs://bucket/clips/", type="video")
all_files = dc.read_storage("az://container/data/")
```

The `type=` parameter selects the right File subclass (`ImageFile`, `VideoFile`, `AudioFile`, `TextFile`). Without it, files are plain `File` objects with binary access.

The chain runs over a saved listing of the storage, a snapshot that later runs reuse. A saved listing of the URI, or of a prefix above it, younger than 4 hours is reused; otherwise the storage is listed and saved, and `update=True` lists it again. `dc.listings()` shows the saved listings. On a versioned bucket the chain holds current objects; `versions=False` lists faster and leaves old versions out of the snapshot.

For public buckets, pass `anon=True` explicitly; for explicit keys or an S3-compatible storage such as MinIO, pass `client_config`:

```python
chain = dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)

minio = {
    "endpoint_url": "https://minio.example.com",
    "key": "access-key",
    "secret": "secret-key",
}
chain = dc.read_storage("s3://my-bucket/data/", client_config=minio)
```

To get the objects of a storage as rows in Python, without a chain or a saved snapshot, use `dc.list_storage()`: see [Listing Storage](listing-storage.md).

## Structured Formats

DataChain reads CSV, JSON, and Parquet directly from storage. These entry points parse the format and produce chains with typed columns.

```python
import datachain as dc

# CSV -- auto-detects delimiter, headers, column types
labels = dc.read_csv("s3://bucket/labels.csv")
labels = dc.read_csv("s3://bucket/csvs/", delimiter=";")

# JSON and JSONL -- with optional JMESPath for nested structures
meta = dc.read_json("gs://bucket/annotations.json", jmespath="images")
captions = dc.read_json("gs://bucket/coco.json", jmespath="annotations")

# Parquet -- supports glob patterns and Hive partitioning
data = dc.read_parquet("s3://bucket/data/*.parquet")
data = dc.read_parquet("s3://bucket/202{1..4}/{yellow,green}-{01..12}.parquet")
```

Timestamps are read as Python `datetime`, which holds microseconds. Digits beyond microseconds are dropped, for example in the nanosecond timestamps that pandas 2 writes to Parquet by default.

JMESPath is powerful for real-world JSON formats like COCO, where images, annotations, and categories live under different top-level keys. Each `read_json()` call with a different `jmespath` extracts one array, and you merge them together on shared IDs.

For complex JSON, auto-generate a Pydantic model from a sample file:

```python
from datachain.lib.meta_formats import gen_datamodel_code

code = gen_datamodel_code("s3://bucket/data.json", jmespath="images")
```

### Selecting and naming CSV and Parquet columns

For `read_csv()`, `read_parquet()`, and `parse_tabular()`, use `columns=` to select input columns by name and set their order. Names may be written as in the file or as DataChain normalizes them: `"ZIP Code"` and `"zip_code"` refer to the same column. Unknown or repeated columns raise an error.

`output=` defines the result's names and Python types **by position**. A dictionary uses insertion order; CSV and `parse_tabular()` also accept a Pydantic model (field order) or a list of names (types inferred). Output names do not select or match input names. Supply exactly one output field for each column read: the selected columns if `columns` is provided, or all input columns if it is omitted. Omit `output` to infer names and types.

With an `output` dictionary or model and no `columns`, the columns come from the first file by path, and every file is read by those names. Partition columns (from folders like `year=2024/`) and a pandas index saved as a column can be left out of `output` when they come last.

For an ordinary Parquet file with columns `a, b, c`:

```python
data = dc.read_parquet("s3://bucket/data.parquet", columns=["c"])
data = dc.read_parquet(
    "s3://bucket/data.parquet",
    columns=["c", "a"],
    output={"x": float, "y": float},
)  # x receives c; y receives a

data = dc.read_parquet(
    "s3://bucket/data.parquet", output={"c": float, "b": float, "a": float}
)  # Renames a, b, c to c, b, a by position
```

An `output={"c": float}` without `columns` supplies only one field for three columns and raises an error. Replace partial `output` schemas with `columns=["c"]`; add `output={"c": float}` only if you also need to specify the result type. The same rules apply to CSV files.

CSV's `column_types=` controls parsing before `output` is applied. Its keys are input names, before normalization or renaming, and it does not select columns. For example, preserve the leading zero in `02134`:

```python
labels = dc.read_csv(
    "s3://bucket/labels.csv",
    columns=["zip_code"],
    column_types={"ZIP Code": "string"},
    output={"postal_code": str},
)
```

For CSV without a header, use generated input names `f0`, `f1`, etc. when selecting columns, then optionally rename them:

```python
data = dc.read_csv(
    "s3://bucket/no-header.csv",
    header=False,
    columns=["f2", "f0"],
    output=["c", "a"],
)
```

With `header=False` and no `columns`, `output` is required and supplies names for every input column.

#### Parquet written by DataChain

A Parquet file written by `to_parquet()` keeps its DataChain schema, so it reads back with the same signals and models. There, `columns=` picks signals as `select()` does:

```python
class Point(dc.DataModel):
    x: int
    y: int


dc.read_values(a=[1], p=[Point(x=2, y=3)]).to_parquet("points.parquet")

dc.read_parquet("points.parquet", columns=["p", "a"])  # signals p and a
dc.read_parquet("points.parquet", columns=["p.x"])  # p as a model with only x
```

As with ordinary Parquet and CSV files above, `output=` gives fields by position: one field for each picked signal, in `columns` order, or for each top-level signal when `columns` is omitted. A model field may declare only some of the stored fields:

```python
class PointX(dc.DataModel):
    x: int


dc.read_parquet("points.parquet", columns=["p", "a"], output={"q": PointX, "n": int})
# q = PointX(x=2), n = 1
```

Each field must have the shape of the signal it reads: a model for a model, optional where the signal can be None, and a union of the same types for a union. A mismatch raises an error before anything is read.

A signal inside an optional model or a union is None wherever that parent doesn't hold it, so it reads into an optional field. For `value: Text | Image | None`:

```python
dc.read_parquet("docs.parquet", columns=["value.Text"], output={"t": Text | None})
# t is None where value holds an Image or is None
```

With `columns=` or `output=`, one read can't mix these files with ordinary Parquet files. With `source=True`, the default, the reader adds its own `source`, which replaces a `source` stored by an earlier read. A stored `source` you pick with `columns=` is kept instead, and if the name `source` is taken, the reader's is named `c0_source`.

## WebDataset

`dc.read_webdataset()` reads WebDataset tar shards without extracting them, a row per sample: the image member is `wds.file`, and the other members (`.txt`, `.json`, `.cls`, ...) become columns, parsed once while the shards are indexed. A `spec=` model, such as `WDSLaion` for LAION-style shards, gives the JSON member typed fields.

**Questions about captions and metadata run as SQL, without reading the images.**

```python
import datachain as dc
from datachain.lib.webdataset_laion import WDSLaion

shards = "gs://datachain-demo/datacomp-small/shards/000000[0-5]*.tar"
samples = dc.read_webdataset(shards, spec=WDSLaion, anon=True)
```

```python
# captions of 3+ words, images of at least 200 px with aspect ratio under 3
w, h = "wds.json.original_width", "wds.json.original_height"
words = dc.func.array.length(dc.func.string.split("wds.txt", " "))
short_side, long_side = dc.func.least(w, h), dc.func.greatest(w, h)
good = samples.filter((words > 2) & (short_side > 200) & (long_side < 3 * short_side))
```

```python
# join a Parquet table of CLIP scores and keep the best-matching captions
metadata = "gs://datachain-demo/datacomp-small/metadata/0020f*.parquet"
scores = dc.read_parquet(metadata, anon=True)
matched = good.merge(scores, on="wds.json.uid", right_on="uid")
best = matched.filter(dc.C("clip_l14_similarity_score") > 0.3)
```

**Questions about the pixels run your Python code on the images.** Images are read from their shards directly, without extracting them. Samples are processed in parallel, and the results become columns you can filter with SQL again:

```python
def contrast(file: dc.File) -> int:
    low, high = file.as_image_file().read().convert("L").getextrema()
    return high - low
```

```python
scored = best.settings(parallel=8).map(contrast=contrast, params=["wds.file"])
flat = scored.filter(dc.C("contrast") < 40)  # nearly blank images
```

## MCAP Robotics Logs

`dc.read_mcap()` reads [MCAP](https://mcap.dev) logs (ROS 2, Foxglove). It reads only the summary section at the end of each file (about 1 MB), so a row per log, an [`McapFile`](../references/data-types/mcapfile.md) with its topics, message counts and duration, is ready in seconds even for terabytes of logs.

**Questions about the summary run as SQL and never touch the files.** For example, find logs where joint states were dropped, or count hours of data per robot:

```python
import datachain as dc

# joint states are logged at 15 Hz; find logs that recorded less
joints = dc.read_mcap("s3://bucket/logs/", topics="/joint_states")
hz = dc.C("summary.messages") / dc.C("summary.duration")
dropped = joints.mutate(hz=hz).filter(dc.C("hz") < 14)
```

```python
# hours of data and number of logs per robot folder
inventory = (
    dc.read_mcap("s3://bucket/logs/")
    .mutate(robot=dc.func.path.parent("mcap.path"))
    .group_by(
        logs=dc.func.count(),
        hours=dc.func.sum("summary.duration") / 3600,
        partition_by="robot",
    )
)
```

**Questions about message content run your Python code on the messages.** With `duration=`, `read_mcap` gives a row per time window, an [`McapSlice`](../references/data-types/mcapfile.md). `messages()` reads only the chunks of that window and topic, and `decoder_factories` decodes them into ROS 2 or Protobuf objects (with `DecoderFactory` from `mcap_ros2.decoder` or `mcap_protobuf.decoder`, packages `mcap-ros2-support` and `mcap-protobuf-support`). The windows are processed in parallel, and the results become columns you can filter with SQL again:

```python
from mcap_ros2.decoder import DecoderFactory


def max_joint_speed(mcap: dc.McapSlice) -> float:
    messages = mcap.messages(decoder_factories=[DecoderFactory()])
    return max((abs(v) for m in messages for v in m.decoded.velocity), default=0.0)
```

```python
windows = dc.read_mcap("s3://bucket/logs/", topics="/joint_states", duration=10)
speeds = windows.settings(parallel=8).map(speed=max_joint_speed)
fast = speeds.filter(dc.C("speed") > 2.0)
```

Times are nanoseconds.

## LeRobot Datasets

`dc.read_lerobot()` reads a [LeRobot](https://github.com/huggingface/lerobot) v3 dataset as a row per episode: its metadata (`episode_index`, `length`, `tasks`, ...) and, under `videos`, a `VideoFragment` per camera that points into the shared mp4. Only the metadata is read, so the rows are ready in seconds; the videos are not downloaded. Column names and camera keys are normalized like `read_parquet` columns, e.g. `observation.image` becomes `videos.observation_image`.

**Questions about episodes and frames run as SQL, without touching the videos.**

```python
import datachain as dc

episodes = dc.read_lerobot("hf://datasets/lerobot/pusht")
fps = 10  # "fps" in meta/info.json

# hours of data, and how many episodes are longer than 20 s
hours = episodes.sum("length") / fps / 3600
long_ones = episodes.filter(dc.C("length") > 20 * fps).count()
```

```python
# short episodes that reached a high reward, from the per-frame table
frames = dc.read_parquet("hf://datasets/lerobot/pusht/data/")
max_reward = dc.func.max("next_reward")
rewards = frames.group_by(reward=max_reward, partition_by="episode_index")
episodes = episodes.merge(rewards, on="episode_index")
good = episodes.filter((dc.C("reward") > 0.9) & (dc.C("length") < 150))
```

**Questions about the video run your Python code on the frames of each episode.** Decoding a fragment seeks to that episode's span of the shared mp4 instead of reading the whole file. Episodes are processed in parallel, and the results become columns you can filter with SQL again:

```python
def motion(cam: dc.VideoFragment) -> float:
    start, end = round(cam.start * fps), round(cam.end * fps)
    sampled = cam.video.get_frames(start, end, fps)  # a frame per second
    frames = [f.get_np().astype("float32") for f in sampled]
    diffs = [abs(a - b).mean() for a, b in zip(frames, frames[1:])]
    return float(sum(diffs) / len(diffs)) if diffs else 0.0
```

```python
camera = "videos.observation_image"
scored = good.settings(parallel=8).map(motion=motion, params=[camera])
stuck = scored.filter(dc.C("motion") < 1.0)  # the scene barely changes
```

Many episodes share one mp4. A function that takes a fragment seeks to that episode's span instead of reading the whole file; with `settings(cache=True)` each mp4 downloads once and is reused by the other episodes.

## SQL Databases

`read_database()` connects to any SQLAlchemy-compatible database:

```python
import datachain as dc

# Basic query
records = dc.read_database("SELECT * FROM experiments", "sqlite:///local.db")

# Parameterized query -- prevents SQL injection
chain = dc.read_database(
    "SELECT * FROM products WHERE category = :cat",
    "postgresql://host/db",
    params={"cat": "electronics"},
)

# Full enrichment pattern: query -> enrich with LLM -> save as dataset
(
    dc.read_database("SELECT id, name, raw_text FROM articles", "postgresql://host/db")
    .settings(parallel=8)
    .map(summary=generate_summary)
    .save("article_summaries")
)
```

Supported databases include PostgreSQL, MySQL, SQLite, DuckDB, Snowflake, and anything else SQLAlchemy supports. Schema inference is automatic.

## In-Memory Sources

For data already in Python:

```python
import pandas as pd
import datachain as dc

# From pandas
df = pd.DataFrame({"path": paths, "label": labels})
chain = dc.read_pandas(df)

# From HuggingFace Hub
chain = dc.read_hf("beans", split="train")
chain = dc.read_hf("beans", split="train", streaming=True, limit=100)

# HuggingFace datasets as storage URIs
chain = dc.read_storage("hf://datasets/mozilla-foundation/common_voice_17_0/audio/en")

# From Python values
chain = dc.read_values(scores=[1.2, 3.4, 2.5])

# From explicit records with schema
records = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
chain = dc.read_records(records, schema={"name": str, "age": int})
```

## Merging External Metadata

The most common real-world pattern: files live in storage, metadata lives in a sidecar format. Read both sources as chains and merge on a shared key.

### JSON Sidecars

Each image has a matching `.json` file with annotations:

```python
import datachain as dc

images = dc.read_storage("gs://bucket/dogs-and-cats/*jpg", anon=True)
meta = dc.read_json("gs://bucket/dogs-and-cats/*json", column="meta", anon=True)

images_id = images.map(id=lambda file: file.path.split(".")[-2])
annotated = images_id.merge(meta, on="id", right_on="meta.id")
```

### COCO Annotations

One JSON file contains multiple arrays, merged by ID:

```python
import datachain as dc

images = dc.read_storage("gs://bucket/coco2017/images/val/")
meta = dc.read_json(
    "gs://bucket/coco2017/annotations/captions_val2017.json", jmespath="images"
)
captions = dc.read_json(
    "gs://bucket/coco2017/annotations/captions_val2017.json", jmespath="annotations"
)

images_meta = images.merge(meta, on="file.path", right_on="images.file_name")
captioned = images_meta.merge(captions, on="images.id", right_on="annotations.image_id")
```

### CSV Labels

```python
import datachain as dc

files = dc.read_storage("gs://bucket/data/")
labels = dc.read_csv("gs://bucket/labels.csv")
labeled = files.merge(labels, on="file.path", right_on="path")
```

## Storage Providers

| Provider | URI Format |
|---|---|
| AWS S3 | `s3://bucket-name/path/` |
| Google Cloud Storage | `gs://bucket-name/path/` |
| Azure Blob Storage | `az://container-name/path/` |
| HuggingFace Hub | `hf://dataset-name` |
| Local Filesystem | `./path/to/data` or `file://path` |

Each provider uses standard credential locations by default. For non-default configurations, use `client_config`:

```python
import datachain as dc

# S3-compatible (MinIO, Ceph, Cloudflare R2, etc.)
chain = dc.read_storage(
    "s3://my-bucket/data/",
    client_config={
        "endpoint_url": "https://minio.example.com",
        "key": "access-key",
        "secret": "secret-key",
    },
)
```

Cross-provider workflows work transparently:

```python
import datachain as dc

(
    dc.read_storage("s3://source-bucket/raw/", type="image")
    .settings(parallel=8)
    .map(embedding=compute_embedding)
    .save("processed_images")
)

dc.read_dataset("processed_images").to_storage("gs://dest-bucket/output/")
```
