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

`classify`, `score`, and `complete` accept image and document files directly (encoded
as multimodal input). `embed` operates on text — embed a text column or a caption you
generated, then use it for vector search:

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
