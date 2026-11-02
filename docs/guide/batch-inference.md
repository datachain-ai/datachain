---
title: Batched Inference
---

# Batched Inference

`batch_map()` calls a function on up to `batch` rows at once and stores one result
per input row, keeping the existing columns. Use it when a model or service
handles many inputs in one call, such as a GPU forward pass or an embeddings
request. For one row per call, use `map()`.

```python
import datachain as dc
import open_clip
import torch


class Embed(dc.BatchMapper):
    def setup(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.eval()

    def process(self, file: list[dc.ImageFile]) -> list[list[float]]:
        images = torch.stack([self.preprocess(f.read()) for f in file])
        with torch.inference_mode():
            return self.model.encode_image(images).tolist()


chain = (
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=4, prefetch=2)
    .batch_map(embedding=Embed(), batch=32)
    .save("image_embeddings")
)
```

A plain function works too, with resources from `setup()`:

```python
def encode(file: list[dc.ImageFile], model) -> list[list[float]]:
    return model.encode([f.read() for f in file]).tolist()


chain = (
    dc.read_storage("s3://bucket/images/", type="image")
    .setup(model=load_model)
    .batch_map(embedding=encode, batch=32)
    .save("image_embeddings")
)
```

## Inputs and outputs

- `batch` is required. Each column parameter receives a list of up to `batch`
  values, also with `batch=1`; the last batch can be shorter.
- Resources from `setup()` are passed unchanged, not as lists.
- The function returns `list[T]` or `Iterator[T]` with one result per input row,
  in input order. Any other number of results raises `UdfError`.
- The output column has type `T`, and an explicit `output=` describes one row.

| Operation | Column parameter | Return annotation | Output column |
|---|---|---|---|
| `map(embedding=encode_one)` | `dc.ImageFile` | `list[float]` | `list[float]` |
| `batch_map(embedding=encode_batch, batch=32)` | `list[dc.ImageFile]` | `list[list[float]]` | `list[float]` |

Use `dc.BatchMapper` for a class with `setup()` and `teardown()`; `dc.Mapper` is
for `map()`. Multiple named functions work as in `map()`, for example
`batch_map(embedding=encode, label=classify, batch=32)`. `dc.llm` operations call
the model once per row, so use them with `map()` and `settings(parallel=N)`.

## Row independence

Delta updates and checkpoint recovery process only the rows that need it, so
their batches can differ from the first run. Each row's result must not depend on
the other rows in its batch. A rerun keeps every completed row, also with a
different `batch`, for example a smaller one after running out of GPU memory.

## Parallel execution and prefetch

With `settings(parallel=N)`, each worker process runs its own copy of the
function, and `setup()` and `teardown()` run once per process. Batches are formed
before they are distributed, so only the last batch is short.

In `batch_map()`, `settings(prefetch=N)` counts batches: up to N batches download
at a time, with up to 20 files per batch downloading concurrently. Up to 2N+1
batches can be on local disk per process: the batch being processed and 2N ahead.
Budget about (2N+1) × `batch` × average file size of disk per process, and start with
`prefetch=1` or `2`.

`settings(batch_size=N)` sets database insert buffering; it does not change the
number of rows per call.
