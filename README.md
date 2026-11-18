# ![DataChain](https://raw.githubusercontent.com/datachain-ai/datachain/main/docs/assets/datachain.svg) Data preparation and exploration for Physical AI

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain/)
[![Tests](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml)

DataChain is a Python library for processing large file collections in S3, GCS, Azure and local storage.
Run your readers and models over images, video, audio and sensor recordings, then save the results as typed, versioned datasets.

Query saved metadata to find relevant files, episodes and clips. Build new training selections from existing
labels, embeddings and features without repeating the original processing. Your coding agent can inspect
the saved results and producing code, then continue the work in a new session.

- **Training and evaluation:** combine metadata and model outputs, select useful examples, and export datasets to training tools.
- **Physical AI and robotics:** select recordings by sensor topics, extract video frames, and curate data for robot learning.
- **Data exploration:** investigate model failures, find rare conditions and gaps in data coverage, and answer new questions from saved results.
- **Large inventories:** discover millions of objects using storage metadata, before reading their contents.

DataChain handles parallel workers, download overlap and checkpoints; your Python defines the processing.
`File` gives readers streaming access or a reusable local cache; delta updates process changed files and reuse saved results.
LeRobot episodes, MCAP time windows and WebDataset samples use the same Python API.

[Quickstart](#quickstart) · [Python API](#the-python-pipeline) · [Examples](#python-and-more-examples)

## Performance

### Object storage

DataChain was **~10× faster than rclone and s5cmd** when listing and saving **10M S3 object versions**.
Below: **257K objects per store**.

| Tool | S3 | GCS | Azure |
|---|---:|---:|---:|
| **DataChain** | **15 s** | **7 s** | **26 s** |
| rclone | 23 s (1.5×) | 20 s (2.9×) | 95 s (3.7×) |
| obstore | 20 s (1.3×) | 15 s (2.1×) | 66 s (2.5×) |
| AWS CLI / gcloud / azcopy | 66 s (4.4×) | 175 s (25×) | 94 s (3.6×) |

Times include saving. [Full comparison, including faster single-cloud tools](https://github.com/datachain-ai/datachain/blob/main/docs/benchmarks.md#comparison-with-dedicated-listing-tools).

### Training-data workflows

DataChain created and read these training selections **2.1–8.1× faster**:

| Workload | Compared with | Speedup |
|---|---|---:|
| LeRobot: 300 DROID episodes (S3) | LeRobot streaming (HF Hub) | **8.1×** |
| MCAP: 207 grasp clips | `mcap filter` | **2.1×** |

Development runs; DROID compares S3 with HF Hub. [Details](https://github.com/datachain-ai/datachain/blob/main/docs/benchmarks.md#training-subsets-across-file-formats).

### Coding agents

DataChain + skill delivered verified results **up to 2.7× faster** than no skill.

| Agent / LLM | Task | Speedup |
|---|---|---:|
| Claude Code / Opus 5.5 | Held-out media tasks | **2.69×** |
| Codex / GPT-6.1 Sol | Bucket inventory | **1.67×** |
| Codex / GPT-6.1 Sol | Video updates | **2.05×** |

Codex: two runs per development task; no overall gain in held-out tests. [Details](https://github.com/datachain-ai/datachain/blob/main/docs/benchmarks.md#codex-on-development-tasks).

## Quickstart

In an empty project directory, install DataChain:

```bash
pip install 'datachain[video]'
```

For a coding agent, also install the skill and start a new agent session in that directory:

```bash
datachain skill install --target claude --local
# Or: --target codex, cursor, copilot, pi
```

Ask your agent:

```text
Prepare a training selection from the robot demonstrations at
s3://robolake/droid/droid_100/
Score wrist-camera motion over the first second: mean absolute pixel change.
Keep scores of at least 10. Save the scores, camera references and training manifest.
Save and run build_episodes.py.
```

### The Python pipeline

Here is the same task in Python. Save it as `build_episodes.py` and run `python build_episodes.py`:

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

This [DROID](https://droid-dataset.github.io/) sample records at 15 fps. Each episode keeps the
camera files and time windows as references; no video repacking, model download or GPU is needed.
A reference run produced:

```text
robot_episodes@1.0.0      100 rows
training_episodes@1.0.0   63 rows
training_manifest.parquet
```

For MCAP sensor logs, create time windows with the same API:

```python
windows = dc.read_mcap("recordings/*.mcap", topics="/joint_states", duration=10)
windows.save("joint_state_windows")
```

The [standalone MCAP example](https://www.datachain.ai/docs/guide/file-pipelines/#catalog-mcap-recordings) includes local recordings to try.

### Continue in a new session

Keep the project directory: it contains `.datachain/`, the Python script and the manifest.
Start a new agent session in the same project and ask:

```text
Which episodes did we exclude? How many would pass at threshold 5? Use the saved scores.
```

Inspect the saved scores, camera windows and producing code:

```bash
datachain dataset describe training_episodes --json
datachain show training_episodes --limit 3
```

See the [walkthrough](https://www.datachain.ai/docs/guide/file-pipelines/) for versioned reads, robot demonstrations and an image-to-PyTorch example.

**Agents reading this:** start with the [skill](https://github.com/datachain-ai/datachain/blob/main/src/datachain/skill/datachain/SKILL.md).
[llms.txt](https://github.com/datachain-ai/datachain/blob/main/llms.txt) links to the API and documentation.

## Python and more examples

- [Robot learning: curate LeRobot episodes](https://www.datachain.ai/docs/guide/file-pipelines/#robot-demonstrations): process camera clips and reuse saved selections.
- [WebDataset: revise a saved selection](https://www.datachain.ai/docs/guide/file-pipelines/#webdataset-samples): filter samples inside TAR shards without repacking them.
- [Video datasets: extract frames and timestamps](https://www.datachain.ai/docs/references/data-types/videofile/): read video with `datachain[video]`.
- [Training: load a saved selection into PyTorch](https://www.datachain.ai/docs/guide/file-pipelines/#feed-pytorch): use a pinned version in a new process.
- [Public image pipeline](https://www.datachain.ai/docs/guide/file-pipelines/#public-images): typed `.map()`, selection and manifest.
- [Multi-cloud inventory](https://www.datachain.ai/docs/guide/file-pipelines/#inventory-across-clouds): aggregate S3, GCS and Azure listings.
- [Updates and resume](https://www.datachain.ai/docs/guide/file-pipelines/#update-and-resume): refresh saved results and continue interrupted work.

## Where it runs

Locally, dataset rows and provenance live in `.datachain/db` (SQLite); downloaded files use `.datachain/cache`.
Keep that project state for later sessions. Local use needs no DataChain account or server.

Optional [DataChain Studio](https://datachain.ai) adds shared datasets, versions and jobs, with AI-generated descriptions.
People and agents can reuse the same catalog through the UI and [MCP](https://www.datachain.ai/docs/studio/user-guide/agents/mcp/).
Jobs run with or without a cluster; clusters process large workloads across dozens or hundreds of machines.

Cloud reads incur normal provider request and transfer charges. The library is
[Apache-2.0](https://github.com/datachain-ai/datachain/blob/main/LICENSE). [Storage configuration](https://www.datachain.ai/docs/guide/remotes) · [Dataset DB](https://www.datachain.ai/docs/concepts/dataset-db)

[Documentation](https://www.datachain.ai/docs) · [Examples](https://github.com/datachain-ai/datachain/blob/main/examples) ·
[Contributing](https://www.datachain.ai/docs/contributing) · [Issues](https://github.com/datachain-ai/datachain/issues) ·
[Discord](https://discord.com/invite/datachain)
