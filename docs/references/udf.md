# Python Operations

Python functions run batch processing on a chain to generate new chain
values. A function takes fields from one or more rows
of the data and outputs new fields. Functions run at scale on multiple workers and
processes.

Any Python function works as an operation. The classes below are useful to implement a "stateful"
operation where a plain function is insufficient, such as when additional `setup()` or `teardown()`
steps need to happen before or after the processing function runs.

::: datachain.lib.udf.UDFBase

::: datachain.lib.udf.Aggregator

::: datachain.lib.udf.Generator

::: datachain.lib.udf.Mapper

`Mapper.process()` normally receives one row at a time. With
`chain.map(result=mapper, batch=N)` for N > 1, each parameter receives a list
of up to N row values and `process()` returns `list[T]` or `Iterator[T]`.
The output column has type T. Results must match the input count and order;
a count mismatch raises `UdfError`. The last partial batch is processed.

Both plain functions and `Mapper` subclasses support this interface, including
file parameters, prefetch, parallel workers, setup/teardown, delta processing,
and checkpoint recovery. A completed batch is the unit of checkpoint reuse;
an incomplete batch is discarded and retried. `batch=1` retains scalar behavior.
See [batched model inference](../guide/processing.md#batched-model-inference).
