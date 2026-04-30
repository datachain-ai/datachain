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
