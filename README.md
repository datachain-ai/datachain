# ![DataChain](docs/assets/datachain.svg) DataChain

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/datachain-ai/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/datachain-ai/datachain)
[![Tests](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/datachain-ai/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/datachain-ai/datachain)


**Direcotry in object storage is not a dataset**

Built for ML engineers and researchers:
- Your data is multimodal — images, video, sensors, not rows and column
- Your agents generate inefficient, non-resumable code
- Your team keeps reprocessing the same data

Object storage gives you files. Nothing else.

**DataChain turns object storage into datasets — and builds a knowledge graph from them.**

```bash
pip install datachain
```

Works with S3, GCS, Azure Blob, and local filesystem.


## The Problem

Ask Claude Code or Cursor to process 500K sensor recordings in S3:

```python
for obj in s3.list_objects(...):
    file = download(obj["Key"])         # no delta: reprocesses all 500K every run
    embedding = run_model(file)         # no checkpoints: crash = start over
    save_result(obj["Key"], embedding)  # no lineage: result is unattributable
                                        # no parallelism: sequential downloads
```

This code:
- reprocesses everything
- breaks on failure
- produces results you can’t trace

Your team runs this again next week. And again. Nothing compounds.

**This is not a prompt problem. It’s an abstraction problem.**


## The fix: datasets

DataChain introduces a dataset abstraction on top of object storage.

A **dataset**:
- knows what files it contains (refereces, not data copy)
- tracks what changed
- remembers how it was produced (including code)

### Example

```python
import datachain as dc

(
    dc.read_storage("s3://my-bucket/recordings/**/*.jpg", update=True, delta=True)
    .settings(parallel=8)
    .map(embedding=embed_file)
    .save("embeddings")
)
```

What you get:
- only new files processed
- failed due to a bug? fix and re-run - it resumes
- versioned dataset with lineage
- automatically added to your knowledge graph

Processing 5,000 new files (495,000 unchanged)...
Done. embeddings @ v1.0.2


## The knowledge graph

Every dataset `.save()` automatically becomes part of a **knowledge graph** in `.datachain/graph/` directory:
```
.datachain/graph/
    index.md
    datasets/
        embeddings.md
        raw-recordings.md
```

Each dataset includes:
- schema
- all versions and what changed
- lineage (what produced it)
- source code

**Work turns data into knowledge. Knowledge improves future work.**

### Enable knowledge graph

```bash
datachain skills install dc-graph
```

It installs skill plugging for Claude Code, Codex and Cursor. You can limit by `--target claude`.

This is not metadata.

**This is a growing memory of your team data work.**


## Why this matters for agents

Without DataChain:
- agents see only file paths
- they rewrite pipelines from scratch
- they don’t know what data exists

With DataChain:
- agents read your dataset graph
- find the right dataset automatically
- generate correct code using it

You describe what you want, not how and where data lives:
```
 ▐▛███▜▌   Claude Code v2.1.76
▝▜█████▛▘  Sonnet 4.6 · Claude Pro
  ▘▘ ▝▝    ~/src/acme-sensors
────────────────────────────────────────────────────────────────────────────────────────
❯ Find outdoor scenes with high confidence and find top 20 similar to the scene anomalies.txt
────────────────────────────────────────────────────────────────────────────────────────
⏺ Reading anomalies.txt
⏺ Bash(python skills/dc-graph/scripts/graph.py --db-mtime)
⏺ Now find relevant outdor scenes dataset in knowledge graph
⏺ Found 12 datasets, finding the best one
⏺ Creating simularity search code
```

Agent:
- scans graph
- finds relevant dataset with schemas and code
- decides what dataset to reuse
- writes correct, efficient pipeline

**Your agents stop guessing. They start using knowledge.**

## 3. Studio (optional)

Run the same pipelines in the cloud with:
- scheduling
- data and lineage UI
- scalable compute: 100s of CPU/GPU
- access control

```bash
$ datachain auth login
$ datachain job run --workers 20 --cluster gpu-pool embed.py
✓ Job submitted → studio.datachain.ai/jobs/1042
Processing 5,000 new files (495,000 unchanged)...
Done. embeddings@v0.0.2
```

See [studio.datachain.ai](https://studio.datachain.ai)

## Contributing

Contributions are very welcome. To learn more, see the [Contributor Guide](https://docs.
datachain.ai/contributing).

## Community and Support

- [Report an issue](https://github.com/datachain-ai/datachain/issues) if you encounter any problems
- [Docs](https://docs.datachain.ai/)
- [Email](mailto:support@datachain.ai)
- [Twitter](https://twitter.com/datachain_ai)
