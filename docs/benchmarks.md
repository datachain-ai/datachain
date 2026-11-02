# Benchmarks: what is faster, and what we measured

DataChain has two distinct performance claims: faster file operations, and less time for a coding agent to
produce a verified result. The latter measures the library together with its skill and the agent's choices.
It is not a measurement of library overhead alone.

These results were collected on integrated evaluation builds in October 2026. They are not measurements of
an arbitrary PyPI release. Read the build named for each result: these figures are historical measurements,
not a new timing guarantee for every package version. The release must support every API shown in its README.

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
execution order alternates. Sources: GOES-16 RadC/2023 and GOES-18 RadM/2022. Build: `eval-ops`, `ded3447c`.

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
it does not establish that the tool lacks support. Several tools here support multiple stores. The README's
positioning combines one Python API for supported stores with processing and saved datasets, rather than
claiming DataChain is the only cross-cloud tool.

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

The same comment reports smaller 257K-object comparisons across S3, GCS and Azure. Specialized tools can
be faster: on S3, s3p 9 s and swath 11 s versus DataChain 15 s; on GCS, Dataflux 3 s versus DataChain 7 s.
The README does not claim to be the fastest tool on every store and workload.

The PR's main before/after table documents 8-CPU, 16-GB measurements with region and order controls, but the
external-tool comment does not give a complete per-tool command/version/repetition ledger. Treat these
values as reported engineering measurements, not confidence intervals or a fresh reproduced benchmark.
The main PR body also reports later 42-second and 3.6-minute DataChain values; we keep the 44-second and
228-second values from the actual external comparison rather than combining different revisions.

## Lightweight and scoped listing

[PR #2051's measurements](https://github.com/datachain-ai/datachain/pull/2051#issuecomment-6247914415)
compare two DataChain paths, rather than DataChain against an external tool. The historical name `dc.listing`
in that comment became the public `dc.list_storage` API.

Each fresh process lists once, then answers totals, grouping, top 100 and modification-time questions.
`read_storage(update=True)` saves the snapshot and queries SQLite; the lightweight path collects rows once
and computes the answers in Python. Modal containers have 8 CPUs and 16 GB RAM (32GB for the 10.5M row), in the
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
from 44.0 s to 1.4 s. Its 32× figure compares the old and new lightweight paths; the saved path took 2.3 min.
For `s3://robolake-lab/199*/**`, it falls from 7.5 s to 2.3 s, returning 5,868 objects. These are historical
optimization measurements for particular URI shapes, not a promise that every glob is faster by 32×.

The benchmark comment predates subsequent review fixes, including version-default semantics. Use the current
API's documented behavior: current objects are returned, and `versions` controls whether version IDs are
included. The10.5M-versions listing timings in these historical tables describe version enumeration, not
10.5M current rows from today's `list_storage()` iterator.

## Codex on development tasks

The absence of an overall Codex speedup in the later held-out media round does not mean every Codex task
was unchanged. Earlier development runs found these improvements:

| Task | Campaign | No skill, RMST min | Expert Python, RMST min | DataChain, RMST min | No skill / DC | Expert / DC |
|---|---|---:|---:|---:|---:|---:|
| `dashcam_search` | dev2 | 14.5 | 9.4 | 8.2 | 1.77× | 1.15× |
| `ugc_clips_update` | dev2 | 20.6 | 10.6 | 10.1 | 2.05× | 1.05× |
| `goes_inventory` | first final campaign, dev task | 2.2 | 4.1 | 1.3 | 1.67× | 3.19× |

Times are rounded; ratios use the underlying results. All listed cells passed 2/2. There were two repetitions
per cell, using tasks seen during tuning. The inventory result shows a gain over the expert plain-Python
skill as well as no skill; the video gains over the expert skill are much smaller.

These are task-specific development observations. They are not new held-out claims and are not pooled with
the later independent round. Source tables: `results/dev2.md`, `results/final-amended.md` and
`results/final-cells.txt` in the companion evaluation repository. The amended first-campaign inventory rows
are unaffected by the image verifier correction recorded in that campaign.

## Coding agents on heavy media tasks

The pre-registered second held-out round used three previously unused task families: Amazon bin images,
COCO photo refresh and Sentinel-2 imagery. Each task ran five times with each of two agents and three arms:
90 scheduled runs, 8-CPU sandboxes, 120-minute budget per run.

- **Vanilla:** the agent and task, with the standard tool environment and no skill.
- **Expert:** a library-neutral best-practices skill; the agent uses plain Python.
- **DataChain:** the same general practices plus the DataChain skill and “Use DataChain for this.”

Claude used Opus 5.5; Codex used gpt-6.1-sol. The evaluated DataChain build was `eval-main`, `3a0710ab`.
Prompts, truth and verifiers for the held-out families were built separately from the tuning session.

The metric is restricted mean time to verified success (RMST): a wrong result or unclean completion receives
the full time budget. Ratios are comparator / DataChain; above 1 favors DataChain. Confidence intervals use
a paired block bootstrap over task families and repetitions.

| Comparison | Claude Code | Codex |
|---|---:|---:|
| Vanilla / DataChain | 2.69 [1.49–5.83] | 1.02 [0.91–1.16] |
| Expert / DataChain | 1.41 [0.41–4.04] | 0.93 [0.82–1.06] |

The headline “2.6× faster” follows the pre-registered rounding rule and names Claude. It applies to this
heavy-media task set, not all coding agents or all data questions. Much of the benefit is available through
good practices in plain Python. The results do not establish superiority or statistical equivalence to
expert Python.

Seven failures were audited independently and classified as genuine. Two provider-capacity failures were
repeated under the infrastructure rule; their original attempts remain archived. A post-hoc sensitivity
analysis accepting an alternative interpretation of the satellite conversion produces Claude vanilla/DC
2.49 [1.76–3.74]; it is not the primary result.

## Skill discovery without naming the library

The README prompts describe the task without an explicit library instruction. We checked this separately
from performance, using fresh Claude Code and Codex sessions with the current installed skill.

Four probes covered two tasks per agent: generating a large-bucket inventory script and generating a reusable
CLIP image-embedding pipeline. Neither prompt contained the library name, a skill invocation or a branded
bucket name. The normal installer placed the skill in the agent's project-local skill directory; the harness
used its `dc-silent` arm, which adds no library instruction to the prompt.

| Agent | Task | Discovered skill | Generated DataChain code | Clean completion |
|---|---|---|---|---|
| Claude Code | Inventory | Native Skill call | Yes | Yes |
| Claude Code | Image embeddings | Native Skill call | Yes | Yes |
| Codex | Inventory | Read installed SKILL.md | Yes | Yes |
| Codex | Image embeddings | Read installed SKILL.md | Yes | Yes |

All four generated scripts passed Python syntax parsing. These were discovery and script-generation smoke
checks: no remote inventory or inference was requested, and full pipeline correctness and runtime were not
measured. They demonstrate automatic selection in these sessions, not a guaranteed activation rate across
all prompts. Build: `eval-state-saved-answers`, `322b3452`; skill bundle `ad482936978f`.

Earlier Claude `dc-silent` runs also completed real video search/refresh tasks without an explicit library
request. The historical adoption table reports use in10/12 runs; four prompts contained a branded bucket
name, so that table is weaker evidence for entirely name-free discovery than the fresh probes above.

The performance campaigns described above did include an explicit library instruction. Removing it from
the README examples does not turn those timing results into measurements of automatic discovery.
Evidence: `results/readme-discovery1.json`, `notes/probes/readme_discovery.py`, `results/raw/readme-discovery1/`
and the historical `results/r5-silent-table.md` in the companion repository.

## Cold agents and saved results

A later development pilot gave fresh agents correct reference projects in three media families: images,
video and audio. Each project had six datasets, two published versions and two model branches. The main
Python baseline had scripts, configs, versioned Parquet and a comparable README, without a manually written
machine-readable lineage manifest. DataChain used automatically recorded dependencies and saved scripts.

The pilot ran 36 sessions: three families × two agents × three arms × separate question/update tracks.
Build: `eval-state-saved-answers`, `322b3452`. There was one repetition per family, so these are exploratory
point estimates with no useful within-family bootstrap uncertainty.

All 18 question sessions and all 18 update sessions succeeded after uniform correction of one independently
audited verifier convention bug. All arms answered the historical provenance and model-impact questions
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

## Limits that matter in practice

- Small commands and short saved-table questions can be dominated by startup and inspection overhead.
- Large table queries and joins can be substantially faster in pandas or DuckDB. DataChain's strongest
  measured primitive is large storage listing; it is not a replacement for a table analytics engine.
- Incremental work has fixed costs. In a 100,000-file benchmark with 1% change, DataChain took 29.6 s versus
  22.5 s for the explicit Python update pipeline. Fewer recomputed rows do not guarantee less elapsed time.
- File delta does not detect model or preprocessing changes. Recorded lineage is not automatic invalidation.
- Captured main scripts do not bundle imported modules, environments or model weights. Recorded sources may
  omit storage accessed inside arbitrary user functions. Metadata is not an external-freshness check.

## Evidence and reproduction

The companion [datachain-agent-evals](https://github.com/datachain-ai/datachain-agent-evals) repository holds
the evaluation protocol, harness and evidence. Relevant paths, relative to that repository:

| Claim | Source |
|---|---|
| Task-only prompts and automatic skill discovery | `results/readme-discovery1.json`, `notes/probes/readme_discovery.py` |
| Dedicated listing tools | [PR #2027 tool comparison](https://github.com/datachain-ai/datachain/pull/2027#issuecomment-6128223196) |
| Lightweight and scoped listing | [PR #2051 measurements](https://github.com/datachain-ai/datachain/pull/2051#issuecomment-6247914415) |
| Codex task-specific dev gains | `results/dev2.md`, `results/final-amended.md`, `results/final-cells.txt` |
| Agent headline and confidence intervals | `results/HELDOUT2-REPORT.md`, `results/heldout2.json`, `PREREG2.yaml` |
| Listing pairs and baseline code | `bench/ops/README.md`, `bench/ops/results/`, `bench/ops/arms/m1_dc_nosave.py`, `bench/ops/arms/m1_boto_plain.py` |
| Cold-project questions and updates | `results/REF-PROJECT-PILOT1-REPORT.md`, `results/ref-project-pilot1-summary.json` |
| Cold-project construction and acceptance | `notes/state/REF-PROJECT-DEV-PROTOCOL.md`, `results/REF-PROJECT-ACCEPTANCE.md` |

The library-only inventory benchmarks can be rerun from the companion repository with configured Modal
access and the matching product worktree:

```bash
modal run bench/ops/ops_bench.py --plan m1_1m_v2 --pairs 3
modal run bench/ops/ops_bench.py --plan m1_10m_v2 --pairs 3
```

The harness stages inputs, runs agents and independently verifies their outputs. Run logs, original failed
attempts and audit snapshots are retained with the evaluation records. Before publishing this draft,
make the referenced evidence available, identify the minimum package version and validate the README
examples on the released build. A performance smoke can detect regressions; it does not reproduce the
agent-level statistical result. The release checklist in PR #2114 separates required APIs from follow-up optimizations.
