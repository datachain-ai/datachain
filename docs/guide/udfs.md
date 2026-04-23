---
title: Python Operations (UDFs)
---

# Python Operations (UDFs)

Python operations run arbitrary Python code on each record in a chain. Use them when you need file content, ML models, or LLM calls. For everything else, use [data engine operations](operations.md).

## Three Types

### Mapper -- map()

Runs once per record. Returns metadata added to the same record.

```python
import datachain as dc
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

def analyze(file: dc.File, client) -> Analysis:
    resp = client.messages.create(model="claude-sonnet-4-20250514", ...)
    return Analysis.model_validate_json(resp.content[0].text)
```

### Generator -- gen()

Expands one record into multiple records. One video produces thousands of frames, one PDF yields many chunks.

```python
import datachain as dc
from pydantic import BaseModel
from typing import Iterator

class Chunk(BaseModel):
    text: str
    embeddings: list[float]

def split_pdf(file: dc.File) -> Iterator[Chunk]:
    for page in extract_pages(file):
        yield Chunk(text=page.text, embeddings=embed(page.text))

chain = dc.read_storage("s3://docs/*.pdf").gen(chunk=split_pdf).save("chunks")
```

### Aggregator -- agg()

Groups records by a key, then produces output records per group.

```python
import datachain as dc

def group_messages(text, sender) -> Iterator[Dialog]:
    yield Dialog(text="\n".join(text))

chain = dc.read_csv("s3://logs.csv").agg(dialog=group_messages, partition_by="session_id")
```

## Column Naming

The new column name is the keyword assigned in the call:

```python
chain = chain.map(embedding=clip_image_embedding)
```

Here, `embedding` becomes the new column name.

## Auto-Inference

DataChain infers parameters from the function signature and output types from return type hints. The input column is always `file` regardless of modality.

**Always type your UDFs.** Missing return type annotations default to `str` and crash at runtime. This is the single most common source of production failures.

```python
# GOOD: explicit return type
def classify(file: dc.File) -> str:
    return "positive"

# BAD: no return type -- defaults to str, crashes with non-string returns
def classify(file):
    return 0.95  # runtime error
```

Use `params=` only for nested column binding:

```python
chain.map(ext=lambda path: path.rsplit(".", 1)[-1], params=["file.path"])
```

## Setup for Expensive Resources

When a Python operation needs an expensive resource (model, tokenizer, API client), `.setup()` initializes it once per worker:

```python
from transformers import Pipeline, pipeline
import datachain as dc

def caption_image(file: dc.File, pipeline: Pipeline) -> str:
    image = file.read().convert("RGB")
    return pipeline(image)[0]["generated_text"]

chain = (
    dc.read_storage("gs://bucket/images/", type="image", anon=True)
    .limit(5)
    .settings(cache=True)
    .setup(pipeline=lambda: pipeline("image-to-text", model="Salesforce/blip-image-captioning-large"))
    .map(caption=caption_image)
)
```

The keyword argument name in `.setup()` must match the parameter name in the function. The lambda runs once per worker at startup.

Works the same for LLM API clients:

```python
import google.generativeai as genai
import datachain as dc

def classify(file: dc.File, model: genai.GenerativeModel) -> str:
    text = file.read_text()
    response = model.generate_content(f"Classify this document: {text}")
    return response.text

chain = (
    dc.read_storage("s3://docs/")
    .setup(model=lambda: genai.GenerativeModel("gemini-2.0-flash"))
    .settings(parallel=4)
    .map(category=classify)
)
```

## Class-Based Lifecycle

When `.setup()` is not enough -- you need `teardown()`, shared mutable state across records, or multi-resource initialization -- use `Mapper`, `Generator`, and `Aggregator` base classes:

```python
from datachain.lib.udf import Mapper

class ImageEncoder(Mapper):
    def setup(self):
        self.model = load_model("ViT-B-32")

    def process(self, file):
        return self.model.encode(file.read())

    def teardown(self):
        del self.model
```

Use class-based UDFs sparingly -- `.setup()` covers most cases.

## Execution and Scale

By default, execution is single-threaded. Add `.settings(parallel=True)` for expensive operations:

```python
import datachain as dc

chain = (
    dc.read_storage("s3://mybucket/dir1")
    .settings(parallel=8, workers=60)  # 8 threads on 60 machines
    .map(emb=image_to_embedding)
    .save("img_emb")
)
```

## One Signal Per Operation

Each `map()`, `gen()`, or `agg()` call produces exactly one output signal. Group multiple outputs in a Pydantic model:

```python
from pydantic import BaseModel
import datachain as dc

class Result(BaseModel):
    label: str
    confidence: float
    tokens: int

# GOOD: one Pydantic model
chain.map(result=classify)

# BAD: multiple outputs per call
# chain.map(label=..., confidence=..., tokens=...)  # not supported
```
