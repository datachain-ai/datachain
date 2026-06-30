# LLM Operations

`datachain.llm` provides named LLM operations over chain columns, parallel to
[`datachain.func`](func.md). Where `func` transpiles to SQL (the cheap, structural
tier), `llm` runs models in the compute engine (the expensive tier). The import
boundary is the cost boundary: when code reads `llm.`, an expensive model call
happens whose result is materialized, typed, cached, and tracked by lineage.

Each operation is used inside a cardinality-matching chain verb (`.map()` for
1:1, `.gen()` for 1:N) and produces a typed column with no `output=` needed.

## Usage

```python
import datachain as dc
from datachain import llm
from pydantic import BaseModel

class Scene(BaseModel):
    objects: list[str]
    risk: float

(
    dc.read_storage("s3://frames", type="image")
    .settings(llm="anthropic/claude-haiku-4-5")
    .map(topic=llm.classify("file", into=["accident", "normal"]))  # -> str
    .map(risk=llm.score("file", "accident risk 0..1"))             # -> float
    .map(scene=llm.complete("file", schema=Scene))                 # -> Scene
    .save("frames")
)
```

## Inputs

How a column is sent to the model is decided by its **type**, so read files with the
right `type=` and the encoding follows:

| Column type | Sent to the model as |
|---|---|
| `str`, Pydantic model | text (a model is serialized to JSON) |
| `TextFile` (`type="text"`) | the file's text, regardless of extension |
| `ImageFile` (`type="image"`), video frame | an inline image (needs a vision model) |
| `bytes` | an inline image if they are a known image format, else an error |
| `File` (no `type=`) | by extension: image types → image, otherwise read as text |

Two things to keep in mind:

- A binary `File` that isn't an image (e.g. a PDF read with no `type=`) raises a clear
  error rather than sending garbage; convert it first (extract frames, OCR, etc.).
- A `str` is sent **verbatim**, so a column holding a *path* sends the path, not the
  file's contents. Read it as a `File` (`read_storage(...)`) to send the content.

`embed` takes text only (`str`, `TextFile`, or a model); embedding an image raises.
Embed a text column or a caption you generated, then use it for vector search:

```python
.map(vec=llm.embed("caption"))  # -> list[float]
```

## Model selection

The model is chosen once with `.settings(llm="provider/model")` and inherited by
every operation below it. Resolution order:

1. A per-call `llm=` argument (override, rare).
2. `.settings(llm=...)`, the main set-once path.
3. The `DATACHAIN_AI_MODEL` environment variable (final fallback).

Routing is handled by [LiteLLM](https://docs.litellm.ai), so any provider-prefixed
string works: `anthropic/claude-haiku-4-5`, `openai/gpt-5-mini`,
`bedrock/anthropic.claude-3-5-sonnet-v1:0`, `vertex_ai/gemini-2.0-flash`, etc.
Credentials are read from the environment / cloud IAM by default; pass them
explicitly (including per-worker secret resolution) with `.settings(llm_params=...)`.

Install the optional dependency with `pip install 'datachain[llm]'`.

## Scaling and caching

A model call is the most expensive step, and each worker processes rows
sequentially, so throughput comes from more workers: `.settings(parallel=N)` for N
processes on one machine, `.settings(workers=N)` to distribute.

Reliability is layered:

- **File data**: DataChain prefetches inputs (overlapping downloads with in-flight
  calls), caches them, and retries transient fetch errors; tune with
  `.settings(prefetch=N)` (default 2).
- **Model calls**: guarded per call with `retries=` and `fallback=`.

Materialized `llm.*` columns are cached and versioned, so re-running a chain reads
the stored result instead of re-calling the model; the cache invalidates only when
the model, prompt, schema, or parameters change.

## No fused predicate

There is intentionally no `llm.if` / fused filter. Materialize a column with an
`llm.*` operation (cached and versioned), then filter on it with a plain
`.filter()`, a cheap recall, not a model rerun. Prefer a continuous score over a
hard boolean so re-thresholding stays free:

```python
.map(spoiler_score=llm.score("file", "likelihood this is a spoiler, 0..1"))
.filter("spoiler_score > 0.7")
```

## Functions

::: datachain.llm.complete

::: datachain.llm.extract

::: datachain.llm.classify

::: datachain.llm.score

::: datachain.llm.embed
