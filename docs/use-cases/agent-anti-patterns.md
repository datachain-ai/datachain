---
title: Agent Anti-Patterns Over Object Storage
---

# Agent Anti-Patterns Over Object Storage

A coding agent given an S3 bucket and a Python interpreter writes naive file-handling code by default. The patterns it reaches for (`os.listdir`, `boto3 download_file`, JSON sidecars, sequential loops) work on a laptop folder of 100 files. They break on a bucket of 10 million. This page lists the anti-patterns agents fall into without the harness, and the chain pattern each one is replaced by.

## Re-listing the bucket every session

A bucket with 10 million objects takes minutes to list and costs real money in API calls. An agent that runs `boto3.list_objects_v2` at the top of every session pays that cost every time, then forgets the listing when the session ends.

DataChain caches the listing as its own dataset on the first read and reuses it on every subsequent read until the source changes.

```python
import datachain as dc

# First run lists the bucket and persists a typed listing dataset
chain = dc.read_storage("s3://acme-robots/runs/", anon=True)

# Subsequent runs in any session start from the cached listing
chain = dc.read_storage("s3://acme-robots/runs/", anon=True)  # no re-list
```

## Downloading then re-uploading

A naive script downloads files to local disk, runs Python on them, and uploads results back to a different prefix. With 10 TB of video, that is a network round trip that costs hours and breaks halfway through.

DataChain keeps file references inside the chain; data flows through `map` and `save` without local copies, and processing runs in the cloud when Studio is enabled.

```python
import datachain as dc

(
    dc.read_storage("s3://acme-robots/runs/", anon=True, type="video")
    .settings(parallel=8, prefetch=4)
    .map(detections=detect_obstacles)        # no local copy of the video
    .save("obstacle_detections")             # writes typed records, not files
)
```

## Folder-as-dataset

An agent treats `s3://bucket/training-data-v2/` as the dataset, then someone adds three files to that folder and the next training run silently changes. There is no version, no diff, no way to know the model was trained on a different input than the eval was run on.

DataChain's `.save("name")` produces an immutable typed dataset version. Adding new files to the source storage does not change the saved version; a new chain run produces a new version.

```python
import datachain as dc

# v1.0.0 captured at this moment
dc.read_storage("s3://acme-robots/runs/").save("training_runs")

# Files added to the bucket later do not mutate v1.0.0
# A new save creates v1.0.0 with full lineage of what was added
dc.read_storage("s3://acme-robots/runs/").save("training_runs")
```

## Filtering on sidecar JSON metadata

An agent that wants "videos longer than 30s with at least one obstacle" downloads every JSON sidecar, parses it locally, and filters in Python. With a million sidecars, that is a million small object reads against object storage, which is the slowest and most expensive thing the bucket does.

DataChain reads the sidecars once into typed columns the Query Engine indexes, then filters happen in vectorized SQL.

```python
import datachain as dc

meta = dc.read_json("s3://acme-robots/runs/**/*.json", column="meta", anon=True)
videos = (
    dc.read_storage("s3://acme-robots/runs/**/*.mp4", anon=True, type="video")
    .map(run_id=lambda file: file.path.split("/")[-1].split(".")[0])
)

videos.merge(meta, on="run_id", right_on="meta.run_id").save("runs")

# Subsequent filters run in vectorized SQL on the typed dataset
long_with_obstacles = (
    dc.read_dataset("runs")
    .filter(dc.C("meta.duration_s") > 30)
    .filter(dc.C("meta.obstacle_count") > 0)
)
```

## Long pipelines without checkpoints

An agent runs an LLM scoring job over 500,000 documents. Three hours in, the script raises a `KeyError` on a malformed record. Without checkpoints, the next run starts from row 0 and re-pays for every successful row.

DataChain checkpoints partial work automatically. Fix the bug, re-run, and the chain skips rows it already processed.

```text
$ python score.py
739 processed records
KeyError: 'lidar_points'

$ vi score.py    # fix the bug

$ python score.py
Checkpoint: skipping 739 records
500000 processed records
```

## What this enables

- **The agent stops paying object-storage tax.** Listing, downloading, and small-file scans turn into cached datasets and vectorized filters.
- **Mid-run failures stop being expensive.** Checkpoints make the next run resume; mid-pipeline crashes do not lose hours of LLM spend.
- **Inputs stop changing under the agent.** Saved datasets are immutable; an experiment run today reads the same rows it read last week.

## See also

- [Reading data](../guide/reading-data.md): the storage and listing layer in depth
- [Checkpoints](../guide/checkpoints.md): how partial work is recovered
- [Delta processing](../guide/delta.md): incremental runs over new files only
- [Datasets](../concepts/datasets.md): why the unit of work is a typed dataset, not a folder
