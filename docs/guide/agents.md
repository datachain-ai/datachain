# DataChain with coding agents

Install the Python library and the project-local skill, then start a new agent session:

```bash
pip install datachain
datachain skill install --target claude
# Or: --target codex, cursor, copilot, pi
```

Describe the data task in your own words. For example:

```text
Inventory our deliveries across S3, GCS and Azure.
Report file counts and bytes by source, folder and extension without downloading contents.
```

The default `datachain` skill includes recipes for inventories, media metadata,
batched model inference, video sampling, saved-result queries, updates and resume.
It can be discovered from task-only prompts; discovery is not guaranteed on every request.
The legacy `core` and `knowledge` skills remain available by explicit name.

| Task | Pattern |
|---|---|
| One-off inventory | `dc.list_storage(uris, versions=False)` and one aggregation pass |
| Media headers | A typed file's `get_info()` inside `.map()`, with `prefetch=0` |
| Model inference | `dc.BatchMapper` with `setup()`, then `.batch_map(..., batch=32)` |
| Video frames | `VideoFile.sample_frames(fps=1)` inside a generator; retain source identities |
| Saved results | `dataset ls --local`, `dataset describe`, then `read_dataset(name, version=...)` |
| File updates | A compatible pipeline with `read_storage(..., update=True, delta=True)` |
| Interrupted runs | Rerun the same script to reuse checkpointed work |

For saved-result and provenance questions, inspect recorded versions and code before
running a pipeline. Check external sources when current freshness is part of the request;
refresh confirmed changes or an explicitly requested rebuild. File delta does not detect
model or preprocessing changes: rebuild affected outputs and preserve historical results.

Use a small shell command for questions it can answer directly. For ready-made tables,
use DuckDB or pandas. DataChain supplies file listing, I/O, workers and reusable datasets;
its benefit depends on the task and agent. Timing evidence and limitations are documented
with the repository README.
