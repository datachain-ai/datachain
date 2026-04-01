# DataChain

The `DataChain` class creates a data chain, which is a sequence of data manipulation
steps such as reading data from storages, running AI or LLM models or calling external
services API to validate or enrich data. See [`DataChain`](#datachain.lib.dc.DataChain)
for examples of how to create a chain.

::: datachain.query.schema.C

::: datachain.lib.dc.database.ConnectionType


::: datachain.lib.dc.listings.listings

::: datachain.lib.dc.csv.read_csv

::: datachain.lib.dc.datasets.read_dataset

::: datachain.lib.dc.hf.read_hf

::: datachain.lib.dc.json.read_json

::: datachain.lib.dc.pandas.read_pandas

::: datachain.lib.dc.parquet.read_parquet

::: datachain.lib.dc.records.read_records

::: datachain.lib.dc.storage.read_storage

::: datachain.lib.dc.values.read_values

::: datachain.lib.dc.database.read_database

::: datachain.lib.dc.datasets.datasets

::: datachain.lib.dc.datasets.delete_dataset

::: datachain.lib.dc.datasets.move_dataset

::: datachain.lib.namespaces.delete_namespace


::: datachain.lib.dc.utils.is_studio

::: datachain.lib.dc.utils.is_local



::: datachain.query.schema.Column

### ColumnExpr

A column expression — either a [`Column`](#datachain.query.schema.Column) reference
like `C("file.size")` or any arithmetic or comparison expression built from columns.

```python
C("width") * C("height")
C("file.size") // 1024
(C("score") > 0.5) | (C("label") == "positive")
```


::: datachain.lib.dc.DataChain



::: datachain.lib.utils.DataChainError

::: datachain.query.session.Session

::: datachain.lib.dc.Sys
