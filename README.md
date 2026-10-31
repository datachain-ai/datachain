<!-- Release draft. Benchmarks and newer APIs target the integrated eval builds;
     see docs/benchmarks.md and the release checklist in the eval repository. -->

# ![DataChain](docs/assets/datachain.svg) DataChain

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain/)
[![Tests](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml)

**One fast Python API for files across clouds. Built for coding agents.**

DataChain is a Python library for finding, processing and selecting the files your models learn from.
Your images, video, audio and episode data can live in S3, GCS, Azure and local folders. Run Python or models
over those sources, save typed, versioned results, and hand a selected dataset to training or evaluation.
Your agent writes the task logic; DataChain handles listing, workers, download overlap, checkpoints and datasets.
You can use the same API directly from Python.

- **15.6–22.6× faster large S3 inventories** than a serial boto3 paginator in our listing benchmarks.
- **2.6× faster to a verified result with Claude Code using DataChain + skill** versus Claude without a skill on held-out
  heavy media tasks. No speed advantage over expert Python was established.

[Measurements, builds and methodology](docs/benchmarks.md). Listing numbers exclude persistence; the agent
measurement includes the skill's practices and an explicit library instruction. Task-specific Codex results appear below.

## What you can build

| Your work | A useful result |
|---|---|
| Robotics research | Catalog MCAP topics and per-recording log-time ranges, select recordings containing the required sensors, and export a manifest. [MCAP example](examples/robotics/mcap_inventory.py) |
| Physical AI and video datasets | Sample frames with timestamps, score or embed them with your model, and save a selection for training or evaluation. [Video recipe](src/datachain/skill/datachain/SKILL.md#video-frames-one-row-per-sampled-frame-with-source-video-and-timestamp) |
| ML and data engineering | Combine file inventories with annotations, run models over new deliveries, and query saved labels or metadata. [Join example](docs/getting-started/python.md#selecting-files-using-json-metadata) |

Use existing Python readers for your recording format through `map` or `gen`. The MCAP example uses the
standard [`mcap` reader](https://mcap.dev/docs/python/mcap-apidoc/mcap.reader); DataChain handles storage,
parallel execution and saved results. Payload decoding
and sensor synchronization remain in your reader.

**Start here:** [with your coding agent](#give-your-agent-the-primitives) or [directly from Python](#try-it-on-mcap-recordings).

## Give your agent the primitives

```bash
pip install datachain
datachain skill install --target claude
# Or: --target codex, cursor, copilot, pi
```

The Python library works without an agent. The skill installs instructions and recipes for your chosen agent;
it is not required by your pipeline. Start a new agent session after installing it, then describe the task:

```text
Inventory our deliveries across s3://my-bucket/deliveries/,
gs://my-bucket/deliveries/ and az://my-container/deliveries/.
Give me one report: files and bytes by storage source, folder and extension.
Don't download the files.
```

```text
Find MCAP recordings containing both /camera/front and /joint_states.
Save a versioned selection and a manifest with source paths and log-time ranges
for our evaluation pipeline.
```

```text
Make our experiment videos searchable. Sample one frame per second,
compute CLIP embeddings, and retain each frame's source and timestamp.
Save the index, then find the 10 moments closest to "a pedestrian crossing".
```

```text
Which model and selection rule produced September's report?
Answer from the saved datasets and code; don't rerun the pipeline.
```

Claude Code and Codex discovered the installed skill from task-only prompts in our
[discovery probes](docs/benchmarks.md#skill-discovery-without-naming-the-library). The timing benchmarks used
an explicit library instruction; those probes did not remeasure the speedup without it.

**Agents reading this:** start with the [DataChain skill](src/datachain/skill/datachain/SKILL.md).
It contains complete recipes for inventories, media metadata, batched inference, video sampling and updates.

## Try it on MCAP recordings

From a checkout of this repository:

```bash
pip install datachain mcap
python examples/robotics/mcap_inventory.py --demo
```

The demo creates two tiny synthetic MCAP files locally: one with `/camera/front` and `/joint_states`,
and one missing `/joint_states`. It catalogs their topics, message counts and log-time ranges, saves the
recording selection, and writes `training_manifest.parquet`. No cloud credentials, ROS installation, model
or GPU are needed. These are generated messages for trying the workflow, not a robotics dataset.

The [complete example](examples/robotics/mcap_inventory.py) contains `inspect_mcap`: an ordinary Python reader
that uses MCAP summary statistics when available and scans messages otherwise. Its pipeline is:

```python
import datachain as dc
from examples.robotics.mcap_inventory import inspect_mcap

recordings = (
    dc.read_storage("./mcap-demo/*.mcap")
    .settings(parallel=True, prefetch=0)
    .map(recording=inspect_mcap)
    .save("recordings")
)
recordings.select(
    "file.path",
    "recording.topics",
    "recording.message_count",
    "recording.first_log_time_ns",
    "recording.last_log_time_ns",
).show()
```

To use your own recordings, replace `--demo` with one or more URI patterns, for example:

```bash
python examples/robotics/mcap_inventory.py \
  's3://my-bucket/recordings/**/*.mcap' \
  'gs://my-bucket/recordings/**/*.mcap' \
  --required-topic /camera/front --required-topic /joint_states
```

The same reader works with local, S3, GCS and Azure sources. `parallel=True` uses worker processes according
to the container's CPU quota. `prefetch=0` leaves reads to your reader. It uses summary statistics when
complete, otherwise scans messages; bytes transferred depend on the file and storage buffering. For model inference, use a `dc.BatchMapper` to load the model once per
worker and `.batch_map(..., batch=32)` for batched calls. See the [inference recipe](src/datachain/skill/datachain/SKILL.md#model-inference-over-files-embeddings-detectors-classifiers).

### Select recordings and hand them to your existing tools

The script selects recordings with both required topics and exports source identities and per-recording
log-time ranges. Here is that query over the saved catalog; it does not reopen the MCAP files:

```python
import datachain as dc

selected = (
    dc.read_dataset("recordings")
    .filter(
        dc.func.array.contains("recording.topics", "/camera/front")
        & dc.func.array.contains("recording.topics", "/joint_states")
    )
    .save("training_recordings")
)
selected.select(
    "file.source",
    "file.path",
    "file.etag",
    "file.version",
    "recording.first_log_time_ns",
    "recording.last_log_time_ns",
).to_parquet("training_manifest.parquet")
```

Use this manifest with your existing MCAP reader or training pipeline. Topic presence is a selection rule;
it does not establish that the sensor messages are valid or synchronized. Provider version IDs are recorded
where available; unversioned sources may have empty IDs.

To copy selected recordings, call `selected.to_storage("./training_files")`; that step reads and transfers
file contents. For image datasets, there is also a [PyTorch integration](examples/get_started/torch-loader.py)
via `.to_pytorch()`. For table analytics, use the Parquet manifest with DuckDB or a filtered `.to_pandas()`.

## Keep the work after the session ends

A dataset has a name, versions, a typed schema and recorded dependencies. Run named Python scripts so DataChain
also captures the script that produced each saved version. A fresh agent can inspect the project:

```bash
datachain dataset ls --local
datachain dataset describe training_recordings --json
datachain show training_recordings --limit 5
```

Pin an explicit version when revisiting a selection:

```python
import datachain as dc

first_selection = dc.read_dataset("training_recordings", version="1.0.0")
first_selection.select("file.source", "file.path").show(5)
```

A dataset version preserves recorded rows and file identities, not an archive of the source bytes. Keeping
old bytes available depends on the source store's versioning and retention. Recorded provenance describes
what ran; it does not check external freshness or bundle imported code and model weights.

- **After a crash:** rerun the same named script to reuse checkpointed work. The unfinished checkpoint may repeat.
- **After files change:** enable `update=True, delta=True` in `read_storage` when you first build a compatible
  pipeline, then rerun that script. It processes new and changed files and removes results for deleted files. Fixed overhead
  means a small update is not always faster.
- **After a model or preprocessing change:** explicitly rebuild affected results. File delta does not detect
  model changes; use a new dataset name for the new configuration.

[Dataset versions](docs/guide/datasets.md) · [Checkpoints](docs/guide/checkpoints.md) ·
[Incremental updates and supported operations](docs/guide/delta.md) · [Recorded provenance](docs/commands/dataset-describe.md)

## One inventory across clouds

For counts, sizes and file types, `list_storage` returns object metadata without downloading file contents
or saving a dataset. Pass one URI or a list, configure [access for each source](docs/guide/remotes.md),
and aggregate in one pass:

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

Keep `file.source` with the path: the same path can exist in different clouds. `versions=False` requests
current objects without version IDs. Each iteration lists storage again. Use `read_storage` for a saved
inventory; both APIs accept sources from several clouds in one call.

## Choose the right operation

| Task | Start here |
|---|---|
| Count objects, sizes or file types | `dc.list_storage()`; aggregate in one pass |
| Reuse an inventory or process files | `dc.read_storage()`; filter paths before `map` |
| Run a model over images, audio or documents | `dc.BatchMapper`, `.settings(parallel=True)`, `.batch_map(..., batch=N)` |
| Catalog MCAP recordings | Your `mcap` reader in `.map()`; filter the saved topics and time ranges |
| Sample video | `dc.VideoFile.sample_frames(fps=1)`; retain source path and timestamp |
| Answer from computed results | `dc.read_dataset(name, version=...)` |
| Understand an existing project | `datachain dataset ls --local`, then `dataset describe` |
| Feed another tool | `.to_parquet()`, `.to_pytorch()`, `.to_storage()` or a filtered `.to_pandas()` |

DataChain pays off at the file boundary: large inventories, parallel I/O, decoding, inference and repeatable
pipelines. For a small question one shell command answers, use that command. For analytics over ready-made
tables, use DuckDB, pandas or your warehouse.

## Where things run and live

Local pipelines run on the machine where you launch Python, with the cloud access you configure. Dataset
rows and provenance live in `.datachain/db` (SQLite), and downloaded media use `.datachain/cache` by default.
Keep the project state for later queries. Cross-cloud reads transfer bytes to the workers and incur the
providers' normal request and data-transfer charges; the API does not eliminate those costs.

The library needs no DataChain account or server for local use. [DataChain Studio](https://studio.datachain.ai)
adds shared datasets, remote compute and access for agents over MCP. The local catalog is project state;
team access controls and shared remote execution belong to the Studio deployment.

[Storage and credentials](docs/guide/remotes.md) · [Dataset DB](docs/concepts/dataset-db.md) · [Apache-2.0 license](LICENSE)

## Performance evidence

### Fast across stores, with the same API

Reported listing times for 257K objects per store. DataChain includes saving the dataset:

| Tool | S3 | GCS | Azure |
|---|---:|---:|---:|
| **DataChain** | **15 s** | **7 s** | **26 s** |
| rclone | 23 s | 20 s | 95 s |
| obstore | 20 s | 15 s | 66 s |
| Lance | 21 s | 17 s | 65 s |
| s3p | 9 s | — | — |
| swath | 11 s | — | — |
| Dataflux | — | 3 s | — |

Specialists can win on their store. A dash means no reported measurement, not necessarily lack of support.
These engineering measurements compare separate per-store workloads, not end-to-end cross-cloud performance.
[Full tool table, larger S3 results, scoped listing and methodology](docs/benchmarks.md#comparison-with-dedicated-listing-tools).

### Coding agents on heavy media tasks

A 90-run held-out evaluation compared no skill, an expert plain-Python skill, and the DataChain skill.
The metric is time to a verified result, with failures charged the task budget. Ratios above 1 favor DataChain;
brackets are 95% confidence intervals.

| Comparison | Claude Code | Codex |
|---|---:|---:|
| No skill / DataChain skill | **2.69×** [1.49–5.83] | 1.02× [0.91–1.16] |
| Expert Python skill / DataChain skill | 1.41× [0.41–4.04] | 0.93× [0.82–1.06] |

The headline rounds the 2.69× estimate down to 2.6×, as specified before the run. It includes the skill's practices and does not
establish a speed advantage over expert Python. Codex showed no overall gain in this independent task set.
Cold questions over small saved tables were also slower with DataChain in a separate pilot.

Codex did improve on specific development tasks:

| Codex task | No skill, min | Expert Python, min | DataChain, min | No skill / DC | Expert / DC |
|---|---:|---:|---:|---:|---:|
| Search dashcam video | 14.5 | 9.4 | 8.2 | 1.77× | 1.15× |
| Process and update video clips | 20.6 | 10.6 | 10.1 | 2.05× | 1.05× |
| Inventory a large S3 prefix | 2.2 | 4.1 | 1.3 | 1.67× | 3.19× |

These tasks were used during tuning, with two repetitions per cell. Times are rounded; ratios use the
underlying results. They are task-specific observations.
[Builds, task details, limits and reproduction](docs/benchmarks.md).

## Learn more

[Documentation](https://www.datachain.ai/docs) ·
[Python quickstart](docs/getting-started/python.md) ·
[Agent skill](src/datachain/skill/datachain/SKILL.md) ·
[Examples](examples) ·
[Contributing](docs/contributing.md) ·
[Issues](https://github.com/datachain-ai/datachain/issues) ·
[Discord](https://discord.com/invite/datachain)
