# Benchmarks: what is faster, and what we measured

DataChain has two distinct performance claims: faster file operations, and less time for a coding agent to
produce a verified result. The latter measures the library together with its skill and the agent's choices.
It is not a measurement of library overhead alone.

These results were collected on integrated evaluation builds in October 2026. They are not measurements of
an arbitrary PyPI release. These historical measurements are not a timing guarantee for every package version.

## Large S3 inventories

List a public S3 prefix, filter by path, size and modification time, and compute counts and bytes per day.
No object contents are downloaded. Answers must match exactly.

The baseline is a single boto3 `list_objects_v2` paginator with Python aggregation. For persistence, it also
writes the listing to Parquet with pandas. DataChain uses `list_storage(..., versions=False)` without
persistence, or `read_storage` and SQL aggregation with persistence.

| Objects | DataChain, no persistence | boto3, no persistence | Speedup | DataChain, persisted | boto3 + pandas, persisted | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| 1.68 million | 18.4 s | 288 s | 15.6× | 40.5 s | 327 s | 8.1× |
| 9.8 million | 73.6 s | 1,666 s | 22.6× | 185 s | 1,893 s | 10.2× |

Medians of three pairs. Each pair runs both arms consecutively in the same 8-CPU, 16-GiB container in US East;
execution order alternates. Sources: GOES-16 RadC/2023 and GOES-18 RadM/2022. These measurements used an October 2026 pre-release build.

This is a comparison against ordinary serial paginator code, not a claim to beat every listing tool or tuned
parallel implementation. For example, parallel `s5cmd` plus polars took 14.7 s and 101 s respectively: faster
than DataChain on the smaller prefix, slower on the larger one.

## Comparison with dedicated listing tools

A separate comparison in [PR #2027](https://github.com/datachain-ai/datachain/pull/2027#issuecomment-6128223196)
reports both a 257K-object comparison across three stores and larger S3 workloads. These are separate from
the GOES inventory/aggregation benchmark above.

### The same listing API across stores

| Tool | S3, 257K objects | GCS, 257K objects | Azure, 257K objects |
|---|---:|---:|---:|
| DataChain | 15 s | 7 s | 26 s |
| rclone | 23 s | 20 s | 95 s |
| obstore | 20 s | 15 s | 66 s |
| Lance | 21 s | 17 s | 65 s |
| s3p | 9 s | — | — |
| swath | 11 s | — | — |
| Dataflux | — | 3 s | — |
| s5cmd | 24 s | — | — |
| AWS CLI | 66 s | — | — |
| gcloud | — | 175 s | — |
| azcopy | — | — | 94 s |

DataChain times include writing the dataset. A dash means no result was reported for that tool/store pair;
it does not establish that the tool lacks support. Several tools here support multiple stores.

Each store column is its own workload. Matching object counts do not make this a comparison of cloud
providers, and this table does not measure a pipeline reading several clouds at once. The multi-source
example demonstrates the API; no end-to-end cross-cloud speedup has been measured here.

### Larger S3 workloads

The following table preserves the reported current-object and version-enumeration values:

| Tool | S3, 2.03M current objects | S3, 10.5M object versions |
|---|---:|---:|
| DataChain | 44 s | 228 s (3.8 min) |
| swath | 45 s | No version listing in this comparison |
| s3p | 94 s | No version listing in this comparison |
| s3ls-rs | 138 s | 1,182 s (19.7 min) |
| rclone | 283 s | 2,260 s (37.7 min) |
| s5cmd | 294 s | 2,338 s (39.0 min) |
| AWS CLI | 730 s | Not reported |

The DataChain operation is `read_storage`, including writing the dataset. From these reported times, the
current-object ratios are rclone/DataChain 6.43×, s5cmd/DataChain 6.68×, AWS CLI/DataChain 16.59× and
s3p/DataChain 2.14×. The version-listing ratios are rclone/DataChain 9.91×, s5cmd/DataChain 10.25× and
s3ls-rs/DataChain 5.18×. The near tie with swath is not evidence of a dependable speed advantage.

The external-tool source does not provide a complete per-tool command, version and repetition ledger.
Treat these as reported engineering measurements, not a fresh reproduced benchmark. The table is a
selection of the reported tools; the source also records tools that failed on particular object keys.

## Lightweight and scoped listing

[PR #2051's measurements](https://github.com/datachain-ai/datachain/pull/2051#issuecomment-6247914415)
compare saved `read_storage` snapshots with the lightweight `list_storage` iterator.

Each fresh process lists once, then answers totals, grouping, top 100 and modification-time questions.
`read_storage(update=True)` saves the snapshot and queries SQLite; the lightweight path collects rows once
and computes the answers in Python. Modal containers have 8 CPUs and 16 GB RAM, in the
bucket's region. There are two rounds in opposite orders; the reported values are medians, with matching
answers in every run.

| Workload | Saved `read_storage` path | Lightweight listing path |
|---|---:|---:|
| List 2.03M S3 objects | 52.7 s | 29.5 s |
| List 6.5M GCS objects | 2.3 min | 45.5 s |
| List and group 6.5M GCS objects | 4.5 min | 50.6 s |

These timings show the cost of snapshot persistence and the particular SQLite/Python grouping expressions.
They are not an additional multiplier to apply to the external-tool table or the agent speedups.
`list_storage` itself is iterable; the benchmark's `.to_list()` materializes all rows, while a one-pass
inventory can aggregate as it iterates.

The same comment measures scoped listing: reading one level or a fixed part of a glob instead of scanning
the full prefix. For the top level of `gs://gresearch/`, returning 28 objects, the lightweight listing falls
from 44.0 s to 1.4 s. Its approximately 31× figure compares the old and new lightweight paths; the saved path took 2.3 min.
For `s3://robolake-lab/199*/**`, it falls from 7.5 s to 2.3 s, returning 5,868 objects. These are historical
optimization measurements for particular URI shapes, not a promise that every glob is faster by 31×.

Current `list_storage` returns current objects; `versions` controls whether their version IDs are included.
The 10.5M-version workload above enumerates historical object versions and is a different operation.

## Codex on development tasks

The absence of an overall Codex speedup in the later held-out media round does not mean every Codex task
was unchanged. Earlier development runs found these improvements:

| Task | No skill, RMST min | Expert Python, RMST min | DataChain, RMST min | No skill / DC | Expert / DC |
|---|---:|---:|---:|---:|---:|
| `dashcam_search` | 14.5 | 9.4 | 8.2 | 1.77× | 1.15× |
| `ugc_clips_update` | 20.6 | 10.6 | 10.1 | 2.05× | 1.05× |
| `goes_inventory` | 2.2 | 4.1 | 1.3 | 1.67× | 3.19× |

Times are rounded; ratios use the underlying results. All listed cells passed 2/2. There were two repetitions
per cell, using tasks seen during tuning. The inventory result shows a gain over the expert plain-Python
skill as well as no skill; the video gains over the expert skill are much smaller.

These are exploratory observations on tasks used during tuning, with two repetitions per cell. They are
not pooled with the independent held-out round below. They do not establish a general Codex speedup.

## Coding agents on heavy media tasks

An earlier registered campaign on October 28, 2026 found no speed advantage over expert Python;
a correction to its image verifier did not change that conclusion. The results below are a separate
campaign on new task families and a newer evaluation build.

The pre-registered second held-out round used three previously unused task families: Amazon bin images,
COCO photo refresh and Sentinel-2 imagery. Each task ran five times with each of two agents and three arms:
90 scheduled runs, 8-CPU sandboxes, 120-minute budget per run.

- **Vanilla:** the agent and task, with the standard tool environment and no skill.
- **Expert:** a library-neutral best-practices skill; the agent uses plain Python.
- **DataChain:** the same general practices plus the DataChain skill and “Use DataChain for this.”

Claude used Opus 5.5; Codex used gpt-6.1-sol. DataChain used an October 2026 pre-release build, not a published package.
Prompts, truth and verifiers for the held-out families were built separately from the tuning session.

The metric is restricted mean time to verified success (RMST): a wrong result or unclean completion receives
the full time budget. Ratios are comparator / DataChain; above 1 favors DataChain. Confidence intervals use
a paired block bootstrap over task families and repetitions.

Each cell below shows verified successes out of 15 and RMST in minutes, including the failure penalty:

| Agent | No skill | Python practices skill | DataChain skill |
|---|---:|---:|---:|
| Claude Code | 11/15 · 47.9 min | 13/15 · 25.2 min | 14/15 · 17.8 min |
| Codex | 15/15 · 10.0 min | 15/15 · 9.2 min | 15/15 · 9.8 min |

The corresponding comparator/DataChain ratios are:

| Comparison | Claude Code | Codex |
|---|---:|---:|
| Vanilla / DataChain | 2.69 [1.49–5.83] | 1.02 [0.91–1.16] |
| Expert / DataChain | 1.41 [0.41–4.04] | 0.93 [0.82–1.06] |

Both comparisons were registered before the run. The speed criterion required a no-skill/DataChain
ratio of at least 1.5 and a confidence interval above 1; only Claude met it. The parity criterion required
an expert/DataChain lower confidence bound of at least 0.9 for both agents; it was not met. The general Python practices alone
accounted for about a 1.9× improvement over no skill. This does not isolate a library-only effect or
establish superiority or equivalence to expert Python. Codex showed no overall gain.

With only three task families, the pooled bootstrap interval is coarse. Per-family no-skill/DataChain ratios:

| Family | Claude Code | Codex |
|---|---:|---:|
| Amazon bin images | 3.59 | 1.11 |
| COCO photo refresh | 2.04 | 1.14 |
| Sentinel-2 imagery | 2.45 | 0.70 |

[Sanitized run timings and reported contrasts](assets/heldout-media-results.json) include all 90 results.
A failed or incorrect run is charged the full 120-minute budget. These data let readers check the point
estimates; they do not replace the task inputs, prompts and independent correctness verifiers.

Seven failures were audited independently and classified as genuine. Two runs that failed with provider-capacity errors were repeated; their original attempts remain archived. A post-hoc sensitivity
analysis accepting an alternative interpretation of the satellite conversion produces Claude vanilla/DC
2.49 [1.76–3.74]; it is not the primary result.

## Skill discovery without naming the library

The README prompts describe the task without an explicit library instruction. We checked this separately
from performance, using fresh Claude Code and Codex sessions with the current installed skill.

Four probes covered two tasks per agent: generating a large-bucket inventory script and generating a reusable
CLIP image-embedding pipeline. Neither prompt contained the library name, a skill invocation or a branded
bucket name. The normal installer placed the skill in the agent's project-local skill directory; no library instruction was added to the prompt.

| Agent | Task | Discovered skill | Generated DataChain code | Clean completion |
|---|---|---|---|---|
| Claude Code | Inventory | Native Skill call | Yes | Yes |
| Claude Code | Image embeddings | Native Skill call | Yes | Yes |
| Codex | Inventory | Read installed SKILL.md | Yes | Yes |
| Codex | Image embeddings | Read installed SKILL.md | Yes | Yes |

All four generated scripts passed Python syntax parsing. These were discovery and script-generation smoke
checks: no remote inventory or inference was requested, and full pipeline correctness and runtime were not
measured. They demonstrate automatic selection in these sessions, not a guaranteed activation rate across
all prompts. These probes used an October 2026 pre-release build.

The timed performance evaluation did include an explicit library instruction. These discovery probes
are separate checks and do not establish the same speedup for automatic discovery.

## Cold agents and saved results

A later development pilot gave fresh agents correct reference projects in three media families: images,
video and audio. Each project had six datasets, two published versions and two model branches. The main
Python baseline had scripts, configs, versioned Parquet and a comparable README, without a manually written
machine-readable lineage manifest. DataChain used automatically recorded dependencies and saved scripts.

The pilot ran 36 sessions: three families × two agents × three arms × separate question/update tracks.
The pilot used an October 2026 pre-release build. There was one repetition per family, so these are exploratory
point estimates with no useful within-family bootstrap uncertainty.

All 18 question sessions and all 18 update sessions succeeded after a correction to how the verifier
interpreted the requested output; the same correction was applied to every arm. All arms answered the historical provenance and model-impact questions
correctly without new media reads or inference. The DataChain question sessions were slower:

| Track | Agent | Vanilla, mean min | Expert, mean min | DataChain, mean min |
|---|---|---:|---:|---:|
| Saved-result questions | Claude | 0.727 | 0.883 | 1.113 |
| Saved-result questions | Codex | 0.652 | 0.694 | 0.877 |
| Updates | Claude | 3.830 | 3.491 | 3.313 |
| Updates | Codex | 2.976 | 2.223 | 2.388 |

This supports a concrete capability—an agent can inspect recorded provenance and answer from saved results.
It does not demonstrate a speed or accuracy advantage for cold questions. The two published versions use
the same model-A checkpoint; model-impact questions are prospective, and actual checkpoint changes are tested
separately in the update track for the image and audio families.

## Training subsets across file formats

Development probes on 2026-11-06 exercised a complete data workflow: select episodes,
clips or samples, persist a selection another process can open, then decode/read all selected
contents. The step called `epoch` is a data-reading pass, not model training.

These are development runs on combined unreleased builds, on Modal with 8 CPUs in AWS us-east-1;
the S3 data is in us-east-2. Some configurations were rerun; the figures below are individual runs, not medians or a general speed ranking.
[Sanitized per-run timings, received bytes and answer digests](assets/format-workflow-results.json)
include the earlier configurations and all three formats. The complete harness and version ledger
are not published here, so this is not an independently reproducible benchmark.

### LeRobot camera episodes

DROID's LeRobot v3 layout packs hundreds of episodes into shared MP4 files. Saved selections
retain each camera's file identity and episode time window instead of repacking videos.

| Selection | Episodes | Create selection | Decode selected frames | Total | Hand-written total |
|---|---:|---:|---:|---:|---:|
| Clustered episodes | 300 | 8.15 s | 40.01 s | 48.16 s | 33.48 s |
| Scattered drawer demonstrations | 437 | 12.69 s | 66.79 s | 79.48 s | 51.76 s |
| Two-second windows around gripper events | 1,030 source episodes | 13.78 s | 114.94 s | 128.72 s | 96.74 s |

Frame counts and pixel-checksum digests match the hand-written arms. The hand-written path
uses DuckDB and PyAV over range reads. DataChain adds saved datasets and camera references,
with measurable overhead. A later clustered DataChain run with all combined fixes took 48.80 s
(8.35 s creating the selection, 40.45 s decoding). The LeRobot library's Hub-backed streaming
run took 393.13 s with the same digest: an observed workflow time ratio of 8.1×. DataChain
reads S3 and LeRobot reads the Hub, so this is not an isolated reader-speed comparison.
The map-style LeRobot run returned a different digest and is not an equivalent result.

### MCAP windows

From 100 logs (11 GB), the pipeline selects 207 windows around gripper-close events,
then reads 16,133 messages from the selected topics. DataChain took 78.11 s to create the
selection and 54.26 s to read it (132.37 s total). `mcap filter` took 277.85 s to write new
clips and 3.26 s to read them (281.11 s total), with matching digests. The experiment report
also records about 52 s total for a specialized parallel range reader.

The materialized clips read about 200 MB per pass, versus 3.38 GB for the original windows:
MCAP chunks interleave topics. Source references make selection and revision convenient;
repacking can pay off over repeated epochs. Neither approach is always cheaper.

### WebDataset samples

Twenty uncompressed TAR shards (6.9 GB) contain roughly 179K samples. Caption and image-size
filters select 182 dog samples, then 1,008 cat samples. After a saved metadata index exists,
DataChain creates the selections in 0.23 s and 0.36 s in the later run without scanning the TARs again.

The full workflow also pays for indexing and image reads. In an earlier configuration, indexing
cost 137.49 s and the two image-reading passes cost 7.49 s and 19.78 s: 165.38 s total.
The streaming baseline took 113.20 s, a hand-written offset index 38.82 s, and repacking
subset shards 179.23 s. All returned the same per-selection digests. Earlier DataChain configurations took 165.38 s and 235.32 s and fetched substantially more data.
A later run after the tar-member prefetch fix took 82.49 s (55.03 s indexing, 7.40 s and
19.47 s reading). The per-run snapshot includes these changes; it does not establish a
release-wide speedup.

## Limits that matter in practice

- Small commands and short saved-table questions can be dominated by startup and inspection overhead.
- Large table queries and joins can be substantially faster in pandas or DuckDB. DataChain's strongest
  measured primitive is large storage listing; it is not a replacement for a table analytics engine.
- Incremental work has fixed startup and persistence costs. Fewer recomputed rows do not guarantee less elapsed time.
- File delta does not detect model or preprocessing changes. Recorded lineage is not automatic invalidation.
- Captured main scripts do not bundle imported modules, environments or model weights. Recorded sources may
  omit storage accessed inside arbitrary user functions. Metadata is not an external-freshness check.

## Evidence and reproduction

- [Dedicated listing tools: measurements and limitations](https://github.com/datachain-ai/datachain/pull/2027#issuecomment-6128223196).
- [Lightweight and scoped listing measurements](https://github.com/datachain-ai/datachain/pull/2051#issuecomment-6247914415).
- [Held-out media evaluation: sanitized run timings and contrasts](assets/heldout-media-results.json).

The full agent harness and raw traces are not publicly available. The published timings do not make
that evaluation independently reproducible; development and discovery observations above remain reported
results. They apply to the measured pre-release builds, not every package release.

For a small, runnable comparison of listing without persistence, run each alternative in a fresh process.
This example uses the same public prefix, filter and aggregation in both paths. It downloads no object
contents. Network location and provider load affect elapsed time. A short prefix is useful for a smoke
check; reproducing the large-table timings requires the original workload and environment.

```python
from collections import Counter
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from time import perf_counter

bucket, prefix = "noaa-goes16", "ABI-L1b-RadC/2023/"
cutoff = datetime(2023, 3, 1, tzinfo=timezone.utc)
counts, sizes = Counter(), Counter()


def add(path, size, modified):
    if size > 100_000 and modified >= cutoff and fnmatchcase(path, "*RadC-M6C13_*"):
        day = path.rsplit("/", 2)[0]
        counts[day] += 1
        sizes[day] += size


start = perf_counter()
# Choose one listing path below, then print the result.
```

Serial boto3 paginator (`pip install boto3`):

```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config

client = boto3.client(
    "s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED)
)
for page in client.get_paginator("list_objects_v2").paginate(
    Bucket=bucket, Prefix=prefix
):
    for item in page.get("Contents", []):
        add(item["Key"], item["Size"], item["LastModified"])
```

DataChain iterator (`pip install datachain`):

```python
import datachain as dc

for file in dc.list_storage(f"s3://{bucket}/{prefix}", anon=True, versions=False):
    add(file.path, file.size, file.last_modified)
```

Print elapsed time and the exact counts/bytes for comparison:

```python
print(perf_counter() - start)
print([(day, counts[day], sizes[day]) for day in sorted(counts)])
```
