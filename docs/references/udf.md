# UDF

User-defined functions (UDFs) can run batch processing on a chain to generate new chain
values. The UDF will take fields from one or more rows
of the data and output new fields. A UDF can run at scale on multiple workers and
processes.

A UDF can be any Python function. The classes below are useful to implement a "stateful"
UDF where a function is insufficient, such as when additional `setup()` or `teardown()`
steps need to happen before or after the processing function runs.

::: datachain.lib.udf.UDFBase

::: datachain.lib.udf.Aggregator

::: datachain.lib.udf.Generator

::: datachain.lib.udf.Mapper
