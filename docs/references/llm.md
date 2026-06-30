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

A column value is sent to the model as **text**, an **image**, or a **document**.
The column's **type** decides how it is encoded:

| Column type | Sent as |
|---|---|
| `str`, Pydantic model | text (a model → JSON) |
| `TextFile` (`type="text"`) | text |
| `ImageFile` (`type="image"`), video frame | image (needs a vision model) |
| `File` (untyped) | text (`read_text`; errors if the bytes are binary) |
| `bytes` | text (errors if not UTF-8) |
| `AudioFile` / `VideoFile` | error; decode first (extract frames or a transcript) |

For raw `bytes` or an untyped `File`, declare the modality with `media=`:

```python
.map(cap=llm.complete("frames", media="image"))     # bytes / File -> image
.map(ext=llm.complete("file",  media="document"))    # File / bytes -> PDF (document-capable model)
```

Rules to keep in mind:

- `media` is validated up front: `media="image"` on non-image bytes, or
  `media="document"` on a non-PDF, raises a clear error.
- A `str` is sent **verbatim**, so a column holding a *path* sends the path, not the
  file's contents; read it as a `File` (`read_storage(...)`) to send the content.
- `media="document"` covers "summarize/extract from this PDF"; heavy document work
  (chunk by section or clause, pull embedded figures) is better done by decoding
  first, then `llm.*`.

`embed` takes text only (`str`, `TextFile`, or a model); embedding an image or
document raises. Embed a text column or a caption you generated, then vector-search:

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

## Token usage

Pass `include_usage=True` to capture token counts for a call. The function then
returns a tuple `(value, dc.llm.Usage)`, so you name both columns with the
multi-output form (the value column keeps its plain type, no wrapper):

```python
.map(
    llm.complete("file", schema=Scene, include_usage=True),
    output={"res": Scene, "tok": dc.llm.Usage},
)
# res: Scene   (res.risk works),  tok: Usage
```

`Usage` has `input_tokens`, `output_tokens`, and `retries` (extra attempts beyond
the first; 0 on a first-try success). Aggregate or filter on it afterwards:

```python
chain.agg(in_tok=dc.func.sum("tok.input_tokens"))
chain.filter("tok.retries > 0")
```

Notes: `retries` counts `datachain.llm`'s own schema-validation retries, not
LiteLLM's internal transient retries; it is only ever above 0 for structured
output. Token counts reflect the successful call only (tokens spent on retried
attempts are not added in). Embeddings report no output tokens, so `output_tokens`
stays 0 there. In `.gen()` (1:N) the single call's usage is replicated onto every
emitted row.

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

::: datachain.llm.classify

::: datachain.llm.score

::: datachain.llm.embed
