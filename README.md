# ![DataChain](docs/assets/datachain.svg) DataChain

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain/)
[![Tests](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml)

**Fast file pipelines across S3, GCS, Azure and local storage — for you and your coding agent.**

Write the task logic. DataChain handles listing, parallel workers, download overlap and checkpoints.
Save results as typed, versioned datasets you can query in the next session.

- Inventory millions of objects using storage metadata, before downloading contents.
- Run Python readers or models over images, video, audio, documents and recordings.
- Reuse saved metadata, labels and selections for the next question or experiment.

**Measured speed:** faster than rclone, obstore and Lance in our 257K-object listing comparisons across
S3, GCS and Azure, including dataset persistence. **Claude Code + skill: 2.6× faster** to a verified result
on held-out media tasks versus Claude without a skill. [Measurements and scope](docs/benchmarks.md).

[Start with Python](#quickstart) · [Use your coding agent](#use-it-from-your-coding-agent)

## Quickstart

```bash
pip install datachain
```

Run this in an empty project directory. It catalogs a public 300-image prefix, saves a size-based
selection and exports a manifest. No credentials, repository checkout or image downloads are needed.

```python
import datachain as dc

images = dc.read_storage(
    "s3://dc-readme/oxford-pets-micro/images/*.jpg", anon=True, type="image"
).save("pet_images")
training = images.filter(dc.C("file.size") < 100_000).save("training_images")
training.to_parquet("training_manifest.parquet")
training.select("file.path", "file.size").show(5)
```

The selection is based on object size; it is a starting point for adding your own metadata or model scores.
Results live in the project's `.datachain/` directory and remain available after Python exits.

## Choose the right operation

| Task | Start here |
|---|---|
| Count objects, sizes or file types | `dc.list_storage()`; aggregate in one pass |
| Build a reusable inventory | `dc.read_storage(...).save(name)` |
| Extract metadata or run per-file Python | A typed function in `.map()`, with `.settings(parallel=True)` |
| Run batched model inference | `dc.BatchMapper` with `setup()`, then `.batch_map(..., batch=32)` |
| Read MCAP or another recording format | Your Python reader in `.map()` or `.gen()` |
| Answer from previous results | `dc.read_dataset(name, version=...)` |
| Inspect recorded provenance | `datachain dataset describe NAME --json` |
| Feed another tool | `.to_parquet()`, `.to_pytorch()` or `.to_storage()` |

Use DataChain to keep file discovery, Python processing and saved results together. You can hand a selection
to an existing fsspec, Ray Data, Daft or Hugging Face workflow through a manifest or Parquet.
For a small question one shell command answers, use that command; for analytics over ready-made tables,
use DuckDB, pandas or your warehouse.

## Use it from your coding agent

```bash
datachain skill install --target claude
# Or: --target codex, cursor, copilot, pi
```

Start a new agent session, then describe the task in your own words:

```text
Inventory our deliveries across S3, GCS and Azure.
Report files and bytes by storage source, folder and extension without downloading contents.
```

```text
Find MCAP recordings containing both /camera/front and /joint_states.
Save a versioned selection and a manifest for our evaluation pipeline.
```

```text
Which model and selection rule produced September's report?
Answer from the saved datasets and code; don't rerun the pipeline.
```

The skill provides recipes for inventories, media metadata, batched inference, saved-result queries and
updates. Claude Code and Codex discovered it from task-only prompts in our
[discovery probes](docs/benchmarks.md#skill-discovery-without-naming-the-library).

**Agents reading this:** start with the [skill](src/datachain/skill/datachain/SKILL.md).
[llms.txt](llms.txt) links to the API and documentation.

## Keep the work and reproduce a selection

Saved datasets have names, versions, typed schemas and recorded dependencies. Run named Python scripts to
capture the code that produced each version. A fresh session can inspect the quickstart's results:

```bash
datachain dataset ls --local
datachain dataset describe training_images --json
datachain show training_images --limit 5
```

Pin the version used for an experiment. After the quickstart, this reads its first saved selection:

```python
import datachain as dc

snapshot = dc.read_dataset("training_images", version="1.0.0")
snapshot.select("file.source", "file.path", "file.etag", "file.version").show(5)
```

Record the dataset name and version in your training configuration. The version preserves rows and file
identities; source-store versioning and retention determine whether the original bytes remain available.

For an image-training handoff, install `datachain[torch]` and run after the quickstart:

```python
import datachain as dc
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

session = dc.Session(client_config={"anon": True})  # the example source is public
snapshot = dc.read_dataset("training_images", version="1.0.0", session=session).select(
    "file"
)
transform = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
loader = DataLoader(snapshot.to_pytorch(transform=transform), batch_size=8)
(images,) = next(iter(loader))
print(images.shape)
```

This step downloads image contents. The manifest also works as an input to your existing training tools.
[Dataset versions](docs/guide/datasets.md) · [PyTorch integration](docs/references/torch.md)

## Robotics example: catalog MCAP recordings

The [MCAP example](examples/robotics/mcap_inventory.py) catalogs topics, message counts and recording
log-time ranges, selects recordings containing the required topics, and exports source identities to
`training_manifest.parquet`. Its reader uses summary statistics when complete and scans messages otherwise.

From a repository checkout:

```bash
pip install datachain mcap
python examples/robotics/mcap_inventory.py --demo
```

The demo generates two tiny local recordings with known topics. For your own data, replace `--demo` with
one or more local or cloud URI patterns:

```bash
python examples/robotics/mcap_inventory.py \
  's3://my-bucket/recordings/**/*.mcap' \
  'gs://my-bucket/recordings/**/*.mcap' \
  --required-topic /camera/front --required-topic /joint_states
```

DataChain supplies storage access, workers and saved datasets; the standard `mcap` package reads the format.
Topic presence is a selection rule, not a check of sensor validity or synchronization.
Use `.to_storage(destination)` to copy a saved selection's files; this transfers their contents.

## One inventory across clouds

Configure [access for each source](docs/guide/remotes.md), then use the same API across stores.
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

## Where it runs

Local pipelines run on your machine using your cloud credentials. Dataset rows and provenance live in
`.datachain/db` (SQLite); downloaded files use `.datachain/cache`. Keep that project state for later sessions.
Local use needs no DataChain account or server. [DataChain Studio](https://studio.datachain.ai) adds shared
datasets, remote compute and agent access through MCP for teams.

Cloud reads incur the providers' normal request and data-transfer charges. The library is
[Apache-2.0](LICENSE). [Storage configuration](docs/guide/remotes.md) · [Dataset DB](docs/concepts/dataset-db.md)

## Performance evidence

Reported times for **257K objects per store**. DataChain includes saving the dataset:

| Tool | S3 | GCS | Azure |
|---|---:|---:|---:|
| **DataChain** | **15 s** | **7 s** | **26 s** |
| rclone | 23 s | 20 s | 95 s |
| obstore | 20 s | 15 s | 66 s |
| Lance | 21 s | 17 s | 65 s |

These engineering measurements are separate per-store workloads. Store-specific tools can be faster;
the [full comparison](docs/benchmarks.md#comparison-with-dedicated-listing-tools) includes them.

A separate 90-run held-out evaluation measured Claude Code + DataChain skill at **2.6× faster** to a
verified result than Claude without a skill. Expert-Python superiority was not established.
On a video-processing/update development task, Codex took **10.1 min versus 20.6 min** without a skill
(two repetitions). Its independent held-out round showed no overall gain. [Agent results, task timings, builds and methodology](docs/benchmarks.md#codex-on-development-tasks).

## Limits and updates

- Recorded lineage and captured scripts describe what ran; they do not verify external freshness or bundle
  imported modules, environments and model weights.
- For compatible file-delta pipelines, start with `read_storage(..., update=True, delta=True)` and rerun to
  update changed files and remove deleted ones. See [supported operations and migration](docs/guide/delta.md).
- Model or preprocessing changes require an explicit rebuild. Preserve historical dataset versions for old reports.
- After interruption, rerun the same script to reuse written checkpoints. Work not yet checkpointed may repeat.

[Documentation](https://www.datachain.ai/docs) · [Examples](examples) ·
[Contributing](docs/contributing.md) · [Issues](https://github.com/datachain-ai/datachain/issues) ·
[Discord](https://discord.com/invite/datachain)
