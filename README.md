# ![DataChain](docs/assets/datachain.svg) DataChain - Data Context Layer for Object Storage

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/datachain-ai/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/datachain-ai/datachain)
[![Tests](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/datachain-ai/datachain)

**Coding agents write great code but fall apart with data - they can't see what's in your buckets, what's already computed, or how datasets relate. DataChain fixes that.**

```bash
pip install datachain
datachain skill install --target claude   # or --target cursor, --target codex
```

Built for large-scale unstructured data - millions of images, video, audio, sensor streams, documents - stored in S3, GCS, or local filesystem.

## Extend coding agents with data

Claude Code (Codex, Cursor, etc) isn't just a chat interface with a shell - it's a harness that gives the LLM repo context, dedicated tools, and persistent memory. That's what makes it good.

DataChain extends that harness to data. The agent that understands your codebase now also understands your storages and datasets: schemas, dependencies, what's already computed, what's mid-run, and what changed since last time.

```
┌──────────────────────┐             ┌──────────────────────┐
│     Claude Code      │─── skill ──▶│      DataChain       │
├──────────────────────┤             ├──────────────────────┤
│  git + commits       │             │  datasets + versions │
│  Prompt caching      │             │  data lineage graph  │
│  file tree           │             │  schemas + types     │
├──────────────────────┤             ├──────────────────────┤
│  Grep / Glob / LSP   │             │  async · parallel    │
│  session memory      │             │  execution state     │
└──────────────────────┘             └──────────────────────┘
          │                                     │
       codebase                           object storage
     (git + files)                    (S3, GCS, AZ, local FS)
```

## 1. Examples

Three prompts to the agent. Each one builds on the last - without you thinking about it.

Use your favorite coding agent:

```bash
claude # --dangerously-skip-permissions
```

### 1.1 Create dataset

Type prompt:

```prompt
Build a dataset from s3://dc-readme/oxford-pets-micro/images, including image width and height (in pixels), and cache the images locally for future processing
```

This generates a small script with efficient code that can scalle to millions of images.
It creats a dataset in the in internal DB with pointers to files in storage but wihout duplicating data.

Now you can point to it as dataset `oxford_pets_micro_images@1.0.0`.
Dataset is a new abstraction that your data context and your team is operating with.

<details>
  <summary>Generated: `build_pets_dataset.py`</summary>
```python
import datachain as dc


def get_info(file: dc.ImageFile) -> dc.Image:
    return file.get_info()


# Read images, extract dimensions, cache locally, and save
# JPEGs ~100KB avg → prefetch = 4MB/100KB = 40
ds = (
    dc.read_storage(
        "s3://dc-readme/oxford-pets-micro/images/**/*.jpg",
        type="image",
        anon=True,
    )
    .settings(prefetch=40, cache=True)
    .map(info=get_info)
    .save("oxford_pets_micro_images")
)

ds.show(5)
print(f"\nTotal images: {ds.count()}")
```
</details>

### 1.2 Datasets knowledge base

Now, the dataset as well as the bucket in your knowledge base and you'r agent is using agentic search to get all the context about it:
```bash
$ tree dc-knowledge
dc-knowledge
├── buckets
│   └── s3
│       └── dc_readme.md
├── datasets
│   └── oxford_pets_micro_images.md
└── index.md
```

You can browse all the datasets and buckets. `index.md` is an entry point. It's human readbale MD format with dataset summary.
The direcotry structure is optimize for agentic search, so your agent doesn't need any extra instruction to navigate on it.

You can open it in any text editor or Obsidian (it slightly optimized for it including wikilinks):

![Visualize data knowledge base](docs/assets/obsidian_single.gif)

This become more and more useful with many datasets that you work on. Agents start bringing insides that you didn't expect and they also generate much better code based on the coding paterns they see in your datasets (since code is also part of this lineage in the knowledge base).

### 1.3 Efficient data quirying

```prompt
How many images in the pets dataset wider than 500 pixels
```

Agent finds context about "the pet" dataset from the knowledge base and answer the question.

```
⏺ Skill(/datachain-knowledge)
  ⎿  Successfully loaded skill

⏺ Let me first check the existing graph knowledge base and then query the dataset.
...

⏺ 3 images are wider than 500px.
```

But the answer is not based on the knowledge base - it does not have it - it's based on operational DB that contains the dataset.
Technicaly in `.datachain/db` file.

It generates super efficient, vectorized code that can run against 100s millions files size dataset:

```prompt
what code did you use to answer the question? save it as a file query_size.py
```

```python
import datachain as dc

count = dc.read_dataset("oxford_pets_micro_images").filter(dc.C("info.width") > 500).count()
print(count)
```

### 1.4 Make it more realistic

Type prompt:

```prompt
Extend the pets dataset by including all possible metadata from annotation/ dir in the bucket and extract breeds from filename suffix
```

Ouptut:

```
⏺ Bash(python3 -c "
      import datachain as dc…)
...
```

Generated: `ingest.py`

```python

```


### 1.4 Similarity search and beyond

Prepare data. Get an image (or use your own):

```copy
datachain cp s3://datachain-usw2-main-dev/oxford-pets/images/Abyssinian_113.jpg my_cat.jpg
```

Prompt:

```
Find the 3 dogs most similar to my_cat.jpg but not Abyssinian breed
```

No embeddings exist yet. The agent notices, generates the embedding pipeline (you can specify what embedding to use), runs it, registers the result, then does the search.

Output:

```
⏺ Results — Similarity Search:

  The 3 images most similar to my_cat.jpg (excluding Abyssinian breed), ranked by CLIP
  cosine distance:

  ┌──────┬──────────────────────────┬───────────────────┬──────────┐
  │ Rank │          Image           │       Breed       │ Distance │
  ├──────┼──────────────────────────┼───────────────────┼──────────┤
  │ 1    │ British_Shorthair_25.jpg │ British Shorthair │ 0.201    │
  ├──────┼──────────────────────────┼───────────────────┼──────────┤
  │ 2    │ Russian_Blue_140.jpg     │ Russian Blue      │ 0.201    │
  ├──────┼──────────────────────────┼───────────────────┼──────────┤
  │ 3    │ Russian_Blue_76.jpg      │ Russian Blue      │ 0.205    │
  └──────┴──────────────────────────┴───────────────────┴──────────┘

  The pipeline used CLIP ViT-B-32 embeddings + cosine distance across all 962
  non-Abyssinian images. The script is saved at find_similar.py.

  Graph updated: 1 dataset (1 updated, 0 unchanged), 1 bucket (1 scanned, 0 unchanged).
```

Generated: `find_similar.py`

```python
import datachain as dc
import open_clip
import torch
from PIL import Image

# Step 1: Compute embeddings for all images
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", "laion2b_s34b_b79k")
model.eval()


def embed(file: dc.ImageFile) -> list[float]:
    img = preprocess(file.read().convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].tolist()

ds = (
    dc.read_dataset("pet_images_annotated")
    .settings(cache=True)
    .map(emb=embed)
    .save("pet_embeddings")
)

# Step 2: Get reference embedding
ref_emb = (
    dc.read_dataset("pet_embeddings")
    .filter(dc.C("file.path").glob("*Abyssinian_113*"))
    .collect("emb")
)
ref = list(ref_emb)[0]

# Step 3: Find top 20 similar, excluding Abyssinian
(
    dc.read_dataset("pet_embeddings")
    .filter(~dc.C("ann.breed_name").glob("Abyssinian*"))
    .mutate(dist=dc.func.cosine_distance(dc.C("emb"), ref))
    .order_by("dist")
    .limit(20)
    .select("file", "ann.breed_name", "ann.species", "dist")
    .show()
)
```



### 1.4 Datasets and embedding reusage

This is when magic starts. All knowledge is accomulated and now agent can efficiently use it the existing precomputed embeddings as well as metadata - without even touching slow storage - all in from context.

```prompt
Find top 5 similar dogs to scottish_terrier_132.jpg but not Scottish Terrier that has bounding box
```

```
⏺ Bash(python3 -c "
      import datachain as dc…)
⏺ Results:

  ┌─────┬──────────────────────────┬────────────┬───────┬────────────────┬────────┐
  │ Ran │          Image           │   Breed    │ Pose  │      BBox      │ Distan │
  │  k  │                          │            │       │                │   ce   │
  ├─────┼──────────────────────────┼────────────┼───────┼────────────────┼────────┤
  │   1 │ wheaten_terrier_114.jpg  │ Wheaten    │ Front │ (272,93,347,16 │  0.237 │
  │     │                          │ Terrier    │ al    │ 6)             │        │
  ├─────┼──────────────────────────┼────────────┼───────┼────────────────┼────────┤
  │   2 │ wheaten_terrier_106.jpg  │ Wheaten    │ Front │ (84,40,313,262 │  0.238 │
  │     │                          │ Terrier    │ al    │ )              │        │

❯ how long did it take

⏺ Under 1 second — the embeddings were already saved in pet_embeddings, so it was just a
   metadata query (filter + cosine distance in SQLite). No images were read, no model
  inference ran.
```

Generated: `search_filtered.py`

No scripts were generated - agent just found the answer in second.



**2 seconds, not 4 minutes.** The agent found existing embeddings in the knowledge graph and reused them — no model loading, no reprocessing. Every `.save()` compounds. The agent gets smarter about your data with every run.

## 2. Incremental updates and checkpoints

### 2.000.

Prompt:

```prompt
Make a dataset from dir gs://datachain-demo/dogs-and-cats/ and include image sizes.
```

In the session:

```

```

Generated: `create_dogs_cats.py` ```python ```

### 2.1 New images arrive

Copy some images to simulate new data arriving:

```bash
cp data/images/Abyssinian_{1..100}.jpg data/images/new_{1..100}.jpg
python ingest.py
```

```
Saved oxford-pets@0.0.2  (100 new, 7,400 unchanged — skipped)  2s
```

No code change. DataChain tracked which files were already processed.

Update the embeddings:

```bash
python embed.py
```

```
Saved oxford-pets-emb@0.0.2  (100 new embeddings, 7,400 from cache)  5s
```

### 2.2 Crash recovery — LLM enrichment

LLM calls are expensive. A crash without checkpoints means paying twice.

```
Generate a one-sentence caption for every image using Claude. Store as a dataset.
```

Generated: `caption.py`

```python
import anthropic, base64
import datachain as dc

client = anthropic.Anthropic()

def caption(file: dc.ImageFile) -> str:
    data = base64.standard_b64encode(file.read()).decode("utf-8")
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": data
                }},
                {"type": "text", "text": "One sentence description."}
            ]
        }]
    )
    return msg.content[0].text

(
    dc.read_dataset("oxford-pets")
    .settings(parallel=4, cache=True)
    .map(caption=caption)
    .save("oxford-pets-caps")
)
```



Simulate a corrupted image and run:

```bash
touch data/images/Abyssinian_1.jpg   # empty file
python caption.py
```

```
Processing... [======>        ]  4,218/7,400
ERROR: Abyssinian_1.jpg — cannot identify image file
```

Fix it and re-run:

```bash
cp batch1/Abyssinian_2.jpg batch1/Abyssinian_1.jpg
python caption.py
```

```
Resuming from checkpoint  (4,218 already done)
Saved oxford-pets-caps@0.0.1  (3,182 processed, 4,218 from checkpoint)
```

Same script, same command. 3,182 LLM calls — not 7,400.

## 3. Physical AI: multi-sensor data

Same pattern, real-world complexity. An AV team has three sources on S3: camera frames (JPEG, Unix timestamp in filename), LiDAR scans (PCD with metadata headers), annotations (COCO JSON).

```
Set up datasets for our three AV data sources in s3://av-data/
Then build a fine-tuning dataset: match each LiDAR scan to its nearest
camera frame within 50ms, join with annotations.
```

Generated: `av_ingest.py`

```python
import datachain as dc


# Camera: timestamp from filename e.g. 1700000123456.jpg
def parse_camera_ts(file: dc.File) -> int:
    stem = file.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return int(stem)

(
    dc.read_storage("s3://av-data/camera/", update=True, delta=True)
    .settings(parallel=-1)
    .map(timestamp_ms=parse_camera_ts)
    .save("av-camera")             # schema: {file: File, timestamp_ms: int}
)


# LiDAR: structured metadata from PCD header
class LidarMeta(dc.DataModel):
    n_points: int
    timestamp_ms: int
    sensor_id: str

def parse_pcd_header(file: dc.File) -> LidarMeta:
    header = {}
    for line in file.read_text().splitlines():
        if line.startswith("DATA"):
            break
        k, _, v = line.partition(" ")
        if k:
            header[k] = v
    return LidarMeta(
        n_points=int(header.get("POINTS", 0)),
        timestamp_ms=int(float(header.get("TIMESTAMP", 0)) * 1000),
        sensor_id=header.get("VIEWPOINT", "unknown").split()[0],
    )

(
    dc.read_storage("s3://av-data/lidar/", update=True, delta=True)
    .settings(parallel=-1)
    .map(lidar=parse_pcd_header)
    .save("av-lidar")              # schema: {file: File, lidar: LidarMeta}
)


# Annotations
(
    dc.read_json("s3://av-data/annotations/*.json", jmespath="frames")
    .save("av-annotations")
)
```



After ingest, the knowledge graph has:

```
av-camera       v0.0.1   50K frames   {file: File, timestamp_ms: int}
av-lidar        v0.0.1   50K scans    {file: File, lidar: LidarMeta}
av-annotations  v0.0.1   12K labels   {frame_path: str, label: str, bbox: ...}
```

The agent sees `timestamp_ms` in both `av-camera` and `av-lidar`. It writes the join. Without the knowledge graph, it couldn't — the field names are only visible in the registered schemas.

Generated: `av_align.py`

```python
import datachain as dc

# bucket into 50ms windows for coarse alignment
lidar  = dc.read_dataset("av-lidar").mutate(
    time_bucket=dc.C("lidar.timestamp_ms") // 50
)
camera = dc.read_dataset("av-camera").mutate(
    time_bucket=dc.C("timestamp_ms") // 50
)

# within each window, take the nearest camera frame
w = dc.func.window(partition_by="time_bucket", order_by="timestamp_ms")

aligned = (
    lidar.merge(camera, on="time_bucket")
    .mutate(rank=dc.func.row_number().over(w))
    .filter(dc.C("rank") == 1)
)

(
    aligned
    .merge(dc.read_dataset("av-annotations"),
           on="file.path", right_on="frame_path")
    .save("av-finetune")
    # lineage: av-camera + av-lidar + av-annotations → av-finetune
)
```



```
Saved av-finetune@0.0.1
  schema:  {camera: File, lidar: LidarMeta, label: str, bbox: BBox}
  lineage: av-camera + av-lidar + av-annotations
  records: 48,312  elapsed: 6m 41s
```

New data in any source bucket? Re-run ingest (delta-aware), re-run alignment. The context layer tracks what changed across all three sources.

## How it works

```
┌─────────────────────────────────────────┐
│             your pipelines              │
└───────────────────┬─────────────────────┘
                    │ .save()
        ┌───────────▼──────────┐
        │   operational layer  │  .datachain/db
        │  • dataset registry  │
        │  • typed schemas     │
        │  • processing state  │
        │  • checkpoints       │
        │  • lineage graph     │
        └───────────┬──────────┘
                    │ derived
        ┌───────────▼──────────┐
        │   knowledge graph    │  datachain/graph/
        │  • agent-readable    │
        │  • dataset summaries │
        │  • schema + versions │
        │  • dependency map    │
        └──────────────────────┘
```

**Operational layer** — the ground truth. Every `.save()` records schema, processing state, and lineage. This is what makes incremental updates and crash recovery work.

**Knowledge graph** — derived from the operational layer, stored as structured markdown in `datachain/graph/`. This is what `datachain-graph` skill reads. Instead of guessing at folder structure, the agent reads the graph: what exists, what schema it has, what's already been computed.

```markdown
## oxford-pets
- version: 0.0.2
- schema: {file: File, meta: {breed, species, split, index}}
- source: ./batch1/
- records: 7,400

## oxford-pets-emb
- version: 0.0.2
- schema: {file: File, emb: list[float]}
- depends_on: oxford-pets

## oxford-pets-caps
- version: 0.0.1
- schema: {file: File, caption: str}
- depends_on: oxford-pets
```

Plain text. Agents read it. Humans can audit it. Lives in your repo alongside your code.

## Team and cloud: Studio

Context built locally stays local. DataChain Studio makes it shared.

```bash
datachain auth login
datachain job run --workers 20 --cluster gpu-pool caption.py
# ✓ Job submitted → studio.datachain.ai/jobs/1042
# Resuming from checkpoint (4,218 already done)...
# Saved oxford-pets-caps@0.0.1  (3,182 processed)
```

![DataChain Studio architecture](docs/assets/studio_architecture.svg)

Studio adds: shared dataset registry, access control, UI for video/DICOM/NIfTI/point clouds, lineage graphs, reproducible runs.

Bring Your Own Cloud — all data and compute stay in your infrastructure. AWS, GCP, Azure, on-prem Kubernetes.

→ [studio.datachain.ai](https://studio.datachain.ai)

## Contributing

Contributions are very welcome. To learn more, see the [Contributor Guide](https://docs.datachain.ai/contributing).

## Community and Support

- [Report an issue](https://github.com/datachain-ai/datachain/issues) if you encounter any problems
- [Docs](https://docs.datachain.ai/)
- [Email](mailto:support@datachain.ai)
- [Twitter](https://twitter.com/datachain_ai)