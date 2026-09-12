# Data Types

Data types supported by `DataChain` must be of type
[`DataType`](#datachain.lib.data_model.DataType). `DataType` includes most Python types
supported in [Pydantic](https://docs.pydantic.dev) fields, as well as any class that
inherits from Pydantic `BaseModel`.

Pydantic models can group and nest multiple fields into one type. When reading a saved
dataset, DataChain reuses a matching model class that is already imported. If the class
is unavailable, DataChain rebuilds one from the stored schema. Models may alternatively
inherit from [`DataModel`](#datachain.lib.data_model.DataModel), a lightweight
`BaseModel` wrapper that registers subclasses automatically.

::: datachain.lib.data_model.DataModel

::: datachain.lib.data_model.DataType

::: datachain.lib.data_model.is_chain_type
