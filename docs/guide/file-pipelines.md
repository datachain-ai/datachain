---
title: File Pipelines in Python
---

# File Pipelines in Python

Curate robot demonstrations, inspect saved results and revise a training selection.
The examples below also cover MCAP logs, WebDataset samples, images and PyTorch.

## Robot demonstrations

Install `datachain[video]` in an empty project directory. Save the following as
`build_episodes.py` and run `python build_episodes.py`:

```python
import datachain as dc


def motion(camera: dc.VideoFragment) -> float:
    start = round(camera.start * 15)
    end = round(camera.end * 15) - 1
    first = camera.video.get_frame(start).get_np().astype("float32")
    later = camera.video.get_frame(min(start + 15, end)).get_np()
    return float(abs(later - first).mean())


episodes = dc.read_lerobot("s3://robolake/droid/droid_100/", anon=True)
scored = (
    episodes.settings(parallel=4, prefetch=0)
    .map(motion=motion, params=["videos.observation_images_wrist_image_left"])
    .save("robot_episodes")
)
training = scored.filter(dc.C("motion") >= 10).save("training_episodes")
training.to_parquet("training_manifest.parquet")
```

The public DROID sample contains 100 LeRobot v3 episodes at 15 fps. The reader attaches
all three cameras as `VideoFragment` values: the original file identity and the episode's
start/end times in the shared MP4. Only the wrist camera is decoded for this score.
The score is mean absolute RGB pixel change between the first frame and one second later,
or the last frame for shorter episodes. It measures visual change, not grasp success.

A reference run selected 63 episodes. The manifest keeps the scores, metadata and
camera references; creating it does not repack videos. Accessing pixels reads video bytes.
Source-store versioning and retention determine how long the original contents remain available.

## Inspect and revise a saved selection

Saved datasets have versions, schemas and recorded dependencies. Run named scripts to capture
the code that produced each version. A fresh process can inspect the saved result:

```bash
datachain dataset describe training_episodes --json
datachain show training_episodes --limit 5
```

Query a pinned version or change the selection rule using the saved scores:

```python
import datachain as dc

scored = dc.read_dataset("robot_episodes", version="1.0.0")
original = dc.read_dataset("training_episodes", version="1.0.0")
relaxed = scored.filter(dc.C("motion") >= 5).save("training_episodes")
print(original.count(), relaxed.count())  # 63 71
```

This creates a new selection version without decoding the cameras again. The previous version
remains readable. To hand the selection to another project, share `training_manifest.parquet`
and access to the source storage. Load a selected camera frame in that project:

```python
import datachain as dc

session = dc.Session(client_config={"anon": True})
selection = dc.read_parquet("training_manifest.parquet", session=session)
camera = selection.limit(1).to_values("videos.observation_images_wrist_image_left")[0]
frame = camera.video.get_frame(round(camera.start * 15)).get_np()
print(frame.shape)  # (180, 320, 3)
```

Dataset versions preserve rows and source identities, not credentials or a copy of the videos.
Reusing computed scores is separate from reading the selected contents for training.

## Select successful robot demonstrations

DROID's success labels live in its per-frame data, separately from the episode metadata.
For a label-based selection, read the needed data columns and aggregate by episode:

```python
import datachain as dc

labels = dc.read_parquet(
    "s3://robolake/droid/droid_10k/data/chunk-000/file-000.parquet",
    columns=["episode_index", "is_episode_successful", "language_instruction"],
    anon=True,
).distinct("episode_index")
training = labels.filter(dc.C("is_episode_successful")).save("successful_episodes")
print(labels.count(), training.count())  # 1074 901
```

This shard contains 1,074 episodes, 901 marked successful. It is a separate dataset from
`droid_100`; join labels to camera episodes from the same dataset on `episode_index`.
The copied `meta/` totals describe a larger dataset, so count the rows actually present.

## Public images

```bash
pip install datachain
```

Run this in an empty project directory. Read image dimensions from a public 300-image prefix,
save a training selection and export a manifest. No credentials or repository checkout are needed.

Save this as `build_images.py`:

```python
from PIL import Image
import datachain as dc


def image_info(file: dc.ImageFile) -> dc.Image:
    with file.open(block_size=64 * 1024) as stream:
        with Image.open(stream) as image:
            return dc.Image(width=image.width, height=image.height)


files = dc.read_storage(
    "s3://dc-readme/oxford-pets-micro/images/*.jpg", anon=True, type="image"
).settings(parallel=True, prefetch=0)
images = files.map(info=image_info).save("pet_images")
training = images.filter(dc.C("info.width") >= 224, dc.C("info.height") >= 224).save(
    "training_images"
)
training.to_parquet("training_manifest.parquet")
training.select("file.path", "info.width", "info.height").order_by("file.path").show(3)
```

Run it with `python build_images.py`. A reference run saved 300 rows in `pet_images` and
291 in `training_images`:

```text
                                          file  info   info
                                          path width height
0  oxford-pets-micro/images/Abyssinian_141.jpg   461    500
1  oxford-pets-micro/images/Abyssinian_157.jpg   500    375
2  oxford-pets-micro/images/Abyssinian_175.jpg   600    234
```

The reader returns typed columns for SQL filtering. It reads file headers over the network; buffering
may read a small image in full. Saved results stay in `.datachain/` after Python exits.

## Feed PyTorch

First run the public image example above. For its training handoff, install `datachain[torch]`.
This also works in a new Python process:

```python
import datachain as dc
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

session = dc.Session(client_config={"anon": True})
snapshot = dc.read_dataset("training_images", version="1.0.0", session=session).select(
    "file"
)
transform = v2.Compose(
    [v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
)
loader = DataLoader(snapshot.to_pytorch(transform=transform), batch_size=8)
(images,) = next(iter(loader))
print(images.shape)  # torch.Size([8, 3, 224, 224])
```

This step downloads image contents. A new session needs the public source's `anon=True` access setting;
saved dataset versions record rows and file identities, not storage credentials.
[Dataset versions](https://www.datachain.ai/docs/guide/datasets) · [PyTorch integration](https://www.datachain.ai/docs/references/torch)
For video (`datachain[video]`), [VideoFile.get_frames()](https://www.datachain.ai/docs/references/data-types/videofile/) yields `VideoFrame` objects with timestamps.

## Catalog MCAP recordings

`dc.read_mcap()` indexes recording summaries without scanning all messages.
This standalone local demo makes two tiny recordings, then saves the one containing joint states.
Run it in a separate project directory after installing `datachain`:

```python
from pathlib import Path
from mcap.writer import Writer
import datachain as dc

Path("recordings").mkdir(exist_ok=True)
for name, topics in {
    "complete": ["/camera/front", "/joint_states"],
    "missing_joints": ["/camera/front"],
}.items():
    writer = Writer(f"recordings/{name}.mcap")
    writer.start()
    for topic in topics:
        channel = writer.register_channel(topic, "json", schema_id=0)
        for second in (1, 2):
            stamp = second * 1_000_000_000
            writer.add_message(channel, stamp, b"{}", stamp)
    writer.finish()

logs = dc.read_mcap("recordings/*.mcap", topics="/joint_states")
selected = logs.filter(dc.C("summary.messages") > 0).save("training_recordings")
selected.to_parquet("recording_manifest.parquet")
print(selected.count())  # 1
```

Use a cloud URI instead of `recordings/*.mcap` for your own logs. The reader requires complete
MCAP summaries. Topic presence does not establish sensor validity or synchronization.
For message processing, `dc.read_mcap(path, topics="/joint_states", duration=10)` produces
`McapSlice` windows. Apply your Python reader with `.map()` to compute features from their messages.
`duration` is in seconds; message timestamps and slice boundaries are in nanoseconds.

## WebDataset samples

Index a public uncompressed TAR shard once, then save caption-based selections from its metadata.
`WDSLaion` gives the JSON metadata typed columns; sample images remain file references:

```python
import datachain as dc
from datachain.lib.webdataset_laion import WDSLaion

samples = dc.read_webdataset(
    "gs://datachain-demo/datacomp-small/shards/00000000.tar",
    spec=WDSLaion,
    anon=True,
    settings={"prefetch": 0},
).save("image_samples")
dogs = samples.filter(dc.C("wds.json.caption").ilike("%dog%")).save("dogs")
dogs.to_parquet("dog_manifest.parquet")
```

A new process can select cats from `dc.read_dataset("image_samples")` without scanning the TAR again.
The initial index reads the archive. Reading selected images fetches their bytes; it is separate from
metadata filtering. For many training epochs, repacking a small subset can reduce repeated reads.
See [measured workflow costs](../benchmarks.md#training-subsets-across-file-formats).

## Inventory across clouds

Configure [access for each source](https://www.datachain.ai/docs/guide/remotes), then use the same API across stores.
This template lists current objects without downloading their contents or saving a dataset:

```python
from collections import Counter
import datachain as dc

sources = [
    "s3://my-bucket/deliveries/",
    "gs://my-bucket/deliveries/",
    "az://my-container/deliveries/",
]
counts, sizes = Counter(), Counter()
for file in dc.list_storage(sources, versions=False):
    key = file.source, file.parent, file.get_file_ext()
    counts[key] += 1
    sizes[key] += file.size
for key in sorted(counts):
    print(*key, counts[key], sizes[key])
```

Keep the source with the path: identical paths can exist in different stores. Each iteration lists again;
use `read_storage` and `save` when you want to reuse the inventory.

## Update and resume

- Captured scripts do not bundle imported modules, environments or model weights. Source-store versioning
  and retention determine whether the original file contents remain available.
- For compatible file-delta pipelines, start with `read_storage(..., update=True, delta=True)` and rerun to
  update changed files and remove deleted ones. See [supported operations and migration](https://www.datachain.ai/docs/guide/delta).
- After a model or preprocessing change, rebuild with `DATACHAIN_IGNORE_CHECKPOINTS=1` and
  `update=True, delta=False`, then save a new version under the same name. Use another name when keeping parallel model variants; earlier versions remain readable.
- After interruption, rerun the same script to reuse written checkpoints. Work not yet checkpointed may repeat.
