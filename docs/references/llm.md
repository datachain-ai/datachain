# LLM Operations

`datachain.llm` provides named LLM operations over chain columns, parallel to
[`datachain.func`](func.md). Where `func` transpiles to SQL, `llm` calls a model:
each operation runs one (expensive) model call per row and materializes the result
as a typed, cached, lineage-tracked column.

Use each operation inside the matching verb (`.map()` for 1:1, `.gen()` for 1:N);
the output column is typed automatically, with no `output=` needed. A
`schema=list[Model]` returns many items: fan them out with `.gen()` (one row each)
or keep them as one `list[Model]` column with `.map()`.

## Quickstart

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
Its **type** decides the encoding automatically, so loading a column with the
matching `read_storage(type=...)` is usually all you need:

| Column type | Sent as |
|---|---|
| `str`, Pydantic model | text (a model → JSON) |
| `TextFile` (`type="text"`) | text |
| `ImageFile` (`type="image"`), video frame | image (needs a vision model) |
| `File` (untyped) | text (`read_text`; errors if the bytes are binary) |
| `bytes` | text (errors if not UTF-8) |
| `AudioFile` / `VideoFile` | error; decode first (extract frames or a transcript) |

```python
# ImageFile is encoded as an image automatically; no type= needed:
dc.read_storage("s3://bucket/imgs/", type="image").map(cap=llm.complete("file"))
```

`type=` is an override for when the type is ambiguous (raw `bytes` or an untyped
`File`) or for a PDF (there is no document file type). Its values line up with
`read_storage`'s `type` (`text`, `image`), plus `document` for PDF input:

```python
.map(cap=llm.complete("frames", type="image"))     # raw bytes / untyped File -> image
.map(ext=llm.complete("file",  type="document"))    # File / bytes -> PDF (document-capable model)
```

Rules to keep in mind:

- A `type` mismatch (`type="image"` on non-image bytes, `type="document"` on
  a non-PDF) raises a clear error when the row is processed.
- `type` is the send modality (aligned with `read_storage`'s `type`), distinct
  from an object's `content_type` (its exact MIME type, e.g. `image/png`).
- A `str` is sent **verbatim**, so a column holding a *path* sends the path, not the
  file's contents; read it as a `File` (`read_storage(...)`) to send the content.
- `type="document"` covers "summarize/extract from this PDF"; heavy document work
  (chunk by section or clause, pull embedded figures) is better done by decoding
  first, then `llm.*`.
- A `None` (missing) input skips the model call and yields `None`, so an `llm.*`
  output column is always optional. In `.gen()` a `None` input emits no rows.

`embed` takes text only (`str`, `TextFile`, or a model); embedding an image or
document raises. Embed a text column or a caption you generated, then vector-search:

```python
.map(vec=llm.embed("caption"))  # -> list[float]
```

## Model selection

The model is chosen once with `.settings(llm="provider/model")` and inherited by
every operation below it. A per-call `llm=` argument overrides it for that call.

Routing is handled by [LiteLLM](https://docs.litellm.ai), so any provider-prefixed
string works: `anthropic/claude-haiku-4-5`, `openai/gpt-5-mini`,
`bedrock/anthropic.claude-3-5-sonnet-v1:0`, `vertex_ai/gemini-2.0-flash`, etc.
Credentials are read from the environment / cloud IAM by default; pass them
explicitly with `.settings(llm_params=...)`. A dict of params is part of the cache
key; a callable (resolved per worker, e.g. for secrets) is not, so put
output-affecting params in the dict form or in per-call keyword arguments.

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

`Usage` has `input_tokens` and `output_tokens`. Aggregate on it afterwards:

```python
chain.agg(in_tok=dc.func.sum("tok.input_tokens"))
```

Notes:

- Embeddings report no output tokens, so `output_tokens` stays 0.
- In `.gen()` (1:N) a call's usage is attributed to one emitted row and zero on
  the rest, so summing it counts each call once (a naive `sum` is correct).
- Transient provider errors are retried by LiteLLM (with backoff); structured
  output relies on the provider's response_format and is not re-asked on a bad
  parse (that raises a clear error).

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
the stored result instead of re-calling the model; the cache invalidates when any
output-affecting input changes (model, prompt, schema, the input column, `type`,
params, ...).

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
