# Data Types

Data types supported by `DataChain` must be of type
[`DataType`](#datachain.lib.data_model.DataType). `DataType` includes most Python types
supported in [Pydantic](https://docs.pydantic.dev) fields, as well as any class that
inherits from Pydantic `BaseModel`.

Pydantic models can be used to group and nest multiple fields together into a single
type object.  Any Pydantic model must be
[registered](#datachain.lib.data_model.DataModel.register) so that the chain knows the
expected schema of the model. Alternatively, models may inherit from
[`DataModel`](#datachain.lib.data_model.DataModel), which is a lightweight wrapper
around Pydantic's `BaseModel` that automatically handles registering the model.

::: datachain.lib.data_model.DataModel

::: datachain.lib.data_model.DataType

::: datachain.lib.data_model.is_chain_type
