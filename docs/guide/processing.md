# Data Processing Features

DataChain offers specialized data processing features that help optimize the handling of large and evolving datasets:

## Key Processing Features

- [Delta Processing](./delta.md): Efficiently process only new or changed records between dataset versions
- [Retry Processing](./retry.md): Automatically identify and reprocess records with errors or missing records

These features can be used independently or combined to create powerful data processing workflows.

## Batched model inference

Use `map(batch=N)` to run inference on up to N rows at once. Each input
parameter receives a list, and the function returns one result per row, in the
same order. The final batch can contain fewer than N rows.

```python
import datachain as dc
import open_clip
import torch


class Embed(dc.Mapper):
    def setup(self):
        torch.set_num_threads(2)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.eval()

    def process(self, file: list[dc.ImageFile]) -> list[list[float]]:
        images = torch.stack([self.preprocess(f.read()) for f in file])
        with torch.inference_mode():
            return self.model.encode_image(images).tolist()


chain = (
    dc.read_storage("./images", type="image")
    .settings(parallel=4, prefetch=2)
    .map(embedding=Embed(), batch=32)
    .save("image_embeddings")
)
```

Plain functions work too:

```python
def double(value: list[int]) -> list[int]:
    return [v * 2 for v in value]


chain = dc.read_values(value=[1, 2, 3]).map(doubled=double, batch=32)
```

A return annotation of `list[T]` or `Iterator[T]` creates an output column of
type `T`. For embeddings, `list[list[float]]` therefore creates a `list[float]`
column. An explicit `output=` describes the result for **one row**. Returning
a different number of results raises an error.

Each worker initializes its own mapper and processes batches independently.
File inputs are prefetched for the batch; `prefetch` counts batches in this
mode. `setup()` and `teardown()` run once per worker. Checkpoint recovery keeps
completed batches and retries incomplete batches. Delta processing batches only
the rows selected for processing, so a UDF should compute each row's result
independently of the other rows in its batch.

`batch=1` (the default) keeps the usual scalar input and output behavior.
`settings(batch_size=...)` separately controls database insert buffering.
