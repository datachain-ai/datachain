---
title: Batched Inference
---

# Batched Inference

`batch_map()` calls a function on up to `batch` rows at once and stores one result per input row, keeping the existing columns. Use it when a model or service handles many inputs in one call, such as a GPU forward pass or an embeddings request. For one row per call, use `map()`.

```python
import datachain as dc
import open_clip
import torch


def load_encoder():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    return model, preprocess


def encode(file: list[dc.ImageFile], encoder) -> list[list[float]]:
    model, preprocess = encoder
    images = torch.stack([preprocess(image.read()) for image in file])
    with torch.inference_mode():
        return model.encode_image(images).tolist()


chain = (
    dc.read_storage("s3://bucket/images/", type="image")
    .setup(encoder=load_encoder)
    .batch_map(embedding=encode, batch=32)
    .save("image_embeddings")
)
```

`setup()` loads the model once per process and passes it to `encode()` for each batch. A runnable version is in [`examples/get_started/udfs/stateful.py`](https://github.com/datachain-ai/datachain/blob/main/examples/get_started/udfs/stateful.py).

## Inputs and outputs

- `batch` is required. Each column parameter receives a list of up to `batch` values, also with `batch=1`; the last batch can be shorter.
- Resources from `setup()` are passed unchanged, not as lists.
- The function returns `list[T]` or `Iterator[T]` with one result per input row, in input order. Any other number of results raises `UdfError`.
- The output column has type `T`, and an explicit `output=` describes one row.

| Operation | Column parameter | Return annotation | Output column |
|---|---|---|---|
| `map(embedding=encode_one)` | `dc.ImageFile` | `list[float]` | `list[float]` |
| `batch_map(embedding=encode_batch, batch=32)` | `list[dc.ImageFile]` | `list[list[float]]` | `list[float]` |

Use `dc.BatchMapper` for a class with `setup()` and `teardown()`; `dc.Mapper` is for `map()`. Multiple named functions work as in `map()`, for example `batch_map(embedding=encode, label=classify, batch=32)`. `dc.llm` operations call the model once per row, so use them with `map()` and `settings(parallel=N)`.

## Row independence

Delta updates and retries can form different batches. Each row's result must be independent of the other rows in its batch.

A rerun reuses rows already stored in checkpoints, even with a different batch size. For example, if a run fails because GPU memory runs out, lower `batch` and rerun the script.

## Prefetch

In `batch_map()`, `settings(prefetch=N)` counts batches: up to N batches download at a time, the files of each batch one after another. Up to 2N+1 batches can be on local disk per process: the batch being processed and 2N ahead. Budget about (2N+1) × `batch` × average file size of disk per process, and start with `prefetch=1` or `2`.

`settings(batch_size=N)` sets database insert buffering; it does not change the number of rows per call.
