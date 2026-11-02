# Data Processing Features

DataChain offers specialized data processing features that help optimize the handling of large and evolving datasets:

## Key Processing Features

- [Delta Processing](./delta.md): Efficiently process only new or changed records between dataset versions
- [Retry Processing](./retry.md): Automatically identify and reprocess records with errors or missing records
- [Batched Inference](./batch-inference.md): Run a model on batches of records with `batch_map()`, keeping one result per record

These features can be used independently or combined to create powerful data processing workflows.
