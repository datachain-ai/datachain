# ![DataChain](https://raw.githubusercontent.com/datachain-ai/datachain/main/docs/assets/datachain.svg) DataChain: Data preparation and exploration for Physical AI and Robotics

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

A reported **2.03M-object S3 inventory took 44 seconds**, including saving the dataset:
**6.4× faster than rclone** (283 s) and **6.7× faster than s5cmd** (294 s).
For **257K objects per store**, reported times were:

| Tool | S3 | GCS | Azure |
|---|---:|---:|---:|
| **DataChain** | **15 s** | **7 s** | **26 s** |
| rclone | 23 s (1.5×) | 20 s (2.9×) | 95 s (3.7×) |
| obstore | 20 s (1.3×) | 15 s (2.1×) | 66 s (2.5×) |
| AWS CLI / gcloud / azcopy | 66 s (4.4×) | 175 s (25×) | 94 s (3.6×) |

Specialized single-cloud tools were faster on some stores; the full comparison includes them.
Ratios compare each tool’s time with DataChain. DataChain times include dataset persistence. [Workloads, tools and measurement details](https://github.com/datachain-ai/datachain/blob/main/docs/benchmarks.md#comparison-with-dedicated-listing-tools).

### Training-data workflows

Selection, saved handoff and a full read of the selected data:

| Workload | Compared with | Observed speedup |
|---|---|---:|
| DROID from S3: 300 LeRobot episodes, three cameras, 260K frames | LeRobot streaming from HF Hub | **8.1×** |
| MCAP: 207 grasp clips from 100 logs | `mcap filter` | **2.1×** |

Individual development runs. DROID uses different storage backends in the two arms.
[Builds, timings and full workflow costs](https://github.com/datachain-ai/datachain/blob/main/docs/benchmarks.md#training-subsets-across-file-formats).

### Coding agents

DataChain + skill versus the same agent without a skill, on held-out media tasks:

| Agent | Time ratio (95% CI) |
|---|---:|
| Claude Code | **2.69×** (1.49–5.83) |
| Codex | 1.02× (0.91–1.16) |

Above 1 favors DataChain. Codex showed no overall gain. General Python practices
accounted for much of the Claude improvement; an advantage over expert Python was not established.
[Results and methodology](https://github.com/datachain-ai/datachain/blob/main/docs/benchmarks.md#coding-agents-on-heavy-media-tasks).

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
