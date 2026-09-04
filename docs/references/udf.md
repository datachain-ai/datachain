# Python Operations

Python functions run batch processing on a chain to generate new chain
values. A function takes fields from one or more rows
of the data and outputs new fields. Functions run at scale on multiple workers and
processes.

Any Python function works as an operation. The classes below are useful to implement a "stateful"
operation where a plain function is insufficient, such as when additional `setup()` or `teardown()`
steps need to happen before or after the processing function runs.

## Cache identity

DataChain hashes class operation code, schemas, and constructor arguments. Primitive
values and nested built-in containers are handled automatically. Callables and custom
objects receive a unique identity instead, preventing incorrect checkpoint reuse.
Override `identity_hash()` when you can provide a stable identity for such arguments:

```python
import hashlib
import json
from datachain.lib.udf import Mapper

class Tokenize(Mapper):
    def __init__(self, tokenizer, tokenizer_id: str, max_length: int):
        self.tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id
        self.max_length = max_length

    def identity_hash(self) -> str:
        state = json.dumps(
            {"tokenizer_id": self.tokenizer_id, "max_length": self.max_length},
            sort_keys=True,
        )
        return hashlib.sha256(state.encode()).hexdigest()

    def process(self, text: str) -> list[str]:
        return self.tokenizer(text)[: self.max_length]
```

`identity_hash()` must return a SHA-256 hexadecimal string covering every constructor
input that affects output. Overriding it replaces automatic constructor-argument
hashing. UDF code and schemas are always included; an incomplete identity can reuse an
incorrect checkpoint result.

::: datachain.lib.udf.UDFBase

::: datachain.lib.udf.Aggregator

::: datachain.lib.udf.Generator

::: datachain.lib.udf.Mapper
