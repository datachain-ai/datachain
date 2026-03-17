# ![DataChain](docs/assets/datachain.svg) DataChain

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/datachain-ai/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/datachain-ai/datachain)
[![Tests](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/datachain-ai/datachain)

---

**Object storage knows nothing about your data. DataChain fixes that.**

Built for ML engineers and researchers. Your data is sensor recordings, video, images and point clouds — not rows and columns.

Works with S3, GCS, Azure Blob, and local storage.


```bash
pip install datachain
```

DataChain has three layers. They build on each other.
1. **Library** — A Python library that adds what S3 lacks: resumable pipelines, typed multimodal schemas, dataset versioning with lineage. This is where you start.
2. **Agent Skills** — Skills that patch Claude Code, Codex or Cursor to write DataChain-native code by default, plus a knowledge graph your agents read to find the right dataset without you specifying it.
3. **Studio** — Centralized datasets with team access control, cloud execution and UI for data and lineage. Same scripts you run locally, at scale.

---

## The Problem

Object storage gives agents nothing but file paths. No schema. No version history. No memory of what's already been processed. Just keys.

Ask Claude Code or Cursor to build a pipeline over 500K sensor recordings and it produces this:

```python
import boto3

s3 = boto3.client("s3")
objects = s3.list_objects_v2(Bucket="my-bucket", Prefix="recordings/")

for obj in objects["Contents"]:
    file = download(obj["Key"])         # no delta: reprocesses all 500K every run
    embedding = run_model(file)         # no checkpoints: crash = start over
    save_result(obj["Key"], embedding)  # no lineage: result is unattributable
                                        # no parallelism: sequential downloads
```

This is not a prompt engineering problem. It's an abstraction problem — and it compounds the longer you leave it.

---

## 1. DataChain Library

### 1.1. Your Pipeline Should Survive a Crash

S3 was designed for storage, not pipelines. It has no concept of what changed, where you stopped, or what produced a result. Every ML team builds the same workarounds from scratch — and they're all wrong in the same ways.

Four comments. Four problems to solve before your pipeline is production-ready. DataChain solves all four.

```python
import datachain as dc

(
    dc.read_storage("s3://my-bucket/recordings/**/*.jpg", update=True, delta=True)
    .settings(parallel=8, pre_fetch=4)
    .map(embedding=embed_file)
    .save("embeddings")
)
```

DataChain stores dataset and bucket state locally: a SQLite database in `.datachain/db`. If processing fails or new files are added to the bucket — just rerun. DataChain picks up from the last checkpoint and skips files already processed.

```
$ python embed.py
Processing 5,000 new files (495,000 unchanged)...
Done. embeddings @ v0.0.2
```

Parallel. Cached. Versioned. Only new files processed. Failed files requeued.

### 1.2. Dataset diff and filters

Every `.save()` snapshots which files, which metadata, and which code produced it. `.diff()` gives an exact file-level manifest between any two versions. Cross-modal joins work as dataset operations.

```python
>>> df = dc.file_diff("embeddings@0.0.1", "embeddings@0.0.2").to_pandas()
>>> print(df)
....
```

Built-in metadata filtering:
```python
>>> df_sensor_a = dc.read_dataset("embeddings@0.0.2").filter(C("source") == "sensor-a")
>>> df_sensor_a.show()
...
```

---

### 1.3. Your Schema Should Match Your Data

In real settings, there is never one source, one schema, one format. LiDAR arrives as point clouds. Camera as JPEG frames. Radar as CSVs. Each with its own timestamp format, sensor ID convention, and metadata structure.

SQL tables can't model this. You end up flattening rich structures into strings, losing type safety, and writing joins that break every time a vendor changes their schema.

DataChain uses Pydantic models instead — define the shape of each source once, and the rest is typed, validated, and mergeable:

```python
from datachain import File
from pydantic import BaseModel

class LidarMeta(BaseModel):
    file_stem: str
    timestamp: float
    n_points: int
    sensor_id: str

def parse_lidar(file: File) -> LidarMeta:
    stem = file.get_stem()
    return LidarMeta(
        file_stem=stem,
        timestamp=float(stem.split("_")[1]),
        n_points=extract_n_points(file),
        sensor_id=file.parent.name,
    )

lidar = dc.read_storage("s3://my-bucket/lidar/**/*.pcd").map(meta=parse_lidar)
camera = dc.read_storage("s3://my-bucket/camera/**/*.jpg").map(meta=parse_camera)

lidar.merge(camera, on=("meta.file_stem", "meta.timestamp")).save("lidar-camera-aligned")
```

Each source keeps its native structure. The merge works across schemas. Add a radar source tomorrow — define a `RadarMeta` model and merge it in. Nothing else changes.


---

## 2. Agent Skills

### 2.1. Your Agent Should Write This Code First

Installs a DataChain skill into Claude Code or Cursor — patches the agent's context so it generates DataChain-native code by default instead of raw SDK loops.

```bash
datachain skills install --target claude dc-core
```
```
✓ dc-core installed → Claude Code (.claude/skills/datachain.md)
```

Same prompt. Different output.

```
Prompt: "Get all jpgs from s3://my-bucket/recordings/ and compute embeddings"
```

| | Output |
|---|---|
| **Without skill** | boto3 loop, no checkpoints, no delta |
| **With skill** | `update=True, delta=True, parallel=8` — the Layer 1 pipeline |

No more correcting agents after the fact. DataChain becomes their default abstraction.

---

### 2.2. Your Agent Should Already Know Your Data

```bash
datachain skills install --target claude dc-graph
```

A side effect of working through DataChain. You don't configure it. Every `.save()` silently writes lineage, source code, library versions, schemas, and diffs into `.datachain/graph/` — one Markdown file per dataset, all versions in one place.

```bash
$ tree .datachain/graph
    index.md              # all datasets with descriptions and stats
    datasets/
      embeddings-v1.md    # schema, all versions, lineage, diffs, source code
      raw-recordings.md
      filtered-scenes.md
```

Then something shifts. Instead of specifying datasets, you describe what you need:

> *"Find scenes from outdoor captures with high model confidence and run similarity search against this reference embedding."*

```
Agent reads: .datachain/graph/index.md
→ scanning 14 datasets for outdoor captures + confidence scores + embeddings...
→ match: embeddings-v1 — 12,000 rows, confidence: float, embedding: list[float]
→ produced by: embed_pipeline.py @ v0.3.1, torch==2.1.0

Agent reads: .datachain/graph/datasets/embeddings-v1.md
→ confirmed schema, version v0.0.4
```

```python
(
    dc.read_dataset("embeddings-v1")
    .filter(C("confidence") > 0.9)
    .mutate(dist=func.cosine_distance(C("embedding"), reference))
    .order_by("dist")
    .limit(50)
    .save("similar-outdoor-scenes")
)
```

The agent found the right dataset, read its schema, and generated correct code — without you specifying the dataset name, column types, or which pipeline produced it.

Claude Code, Cursor, and Codex all read from the same `.datachain/graph/` — shared via Git, always current. Multiple agents, one growing data brain. Operations build knowledge. Knowledge improves operations.

**The graph doesn't depreciate. It compounds.**

---

## 3. DataChain Studio

At team scale, Studio adds the infrastructure layer the open source library doesn't have. The same script you run locally runs in the cloud unchanged:

```bash
datachain auth login
datachain job run --workers 8 --cluster gpu-pool embed.py
```
```
✓ Job submitted → studio.datachain.ai/jobs/1042
Processing 5,000 new files (495,000 unchanged)...
Done. embeddings@v0.0.2
```

Schedule recurring jobs with `--cron`. Monitor logs, lineage, and dataset versions in the UI.

- **Centralized collaboration** — data, code, and dependencies in one place
- **Data lineage UI** — sources and all derivative datasets, visualized
- **Multimodal data viewer** — browse images, video, PDFs, DICOM, NIfTI directly in UI
- **Scalable compute (Bring Your Own Cloud)** — create clusters of 100s of CPU or GPU machines
- **Access control** — SSO, team collaboration, self-hosting (AWS AMI or Kubernetes)

[studio.datachain.ai](https://studio.datachain.ai)

## Contributing

Contributions are very welcome. To learn more, see the [Contributor Guide](https://docs.
datachain.ai/contributing).

## Community and Support

- [Report an issue](https://github.com/datachain-ai/datachain/issues) if you encounter any problems
- [Docs](https://docs.datachain.ai/)
- [Email](mailto:support@datachain.ai)
- [Twitter](https://twitter.com/datachain_ai)
