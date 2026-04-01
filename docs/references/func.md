# Functions

Use built-in functions for data manipulation and analysis to operate on the underlying database storing the chain data. These functions are useful for operations like [`DataChain.filter`](datachain.md#datachain.lib.dc.DataChain.filter) and [`DataChain.mutate`](datachain.md#datachain.lib.dc.DataChain.mutate).

Functions are organized by category and accessed through their respective modules. For example, string functions are accessed via `func.string.length()`, array functions via `func.array.contains()`, etc.

!!! note "Global Function Access"
    Only a subset of functions are available directly from `datachain.func` (e.g., `func.length`). Most functions should be accessed through their specific module namespace (e.g., `func.string.length`) to avoid naming conflicts.

## Function Categories

DataChain provides several categories of functions for different types of operations:

- **[Aggregate Functions](functions/aggregate.md)** - Functions for aggregating data like `sum`, `count`, `avg`, etc.
- **[Array Functions](functions/array.md)** - Functions for working with arrays and lists
- **[Conditional Functions](functions/conditional.md)** - Functions for conditional logic like `ifelse`, `case`, etc.
- **[Numeric Functions](functions/numeric.md)** - Functions for numeric operations and computations
- **[Path Functions](functions/path.md)** - Functions for working with file paths
- **[Random Functions](functions/random.md)** - Functions for generating random values
- **[String Functions](functions/string.md)** - Functions for string manipulation and processing
- **[Window Functions](functions/window.md)** - Functions for window operations

::: datachain.func.func.Func
    options:
      members: false

## Usage

```python
from datachain.func import aggregate, array, conditional, numeric, path, random, string, window

# Access functions through their module namespaces
dc.mutate(
    text_length=string.length("text_column"),
    contains_item=array.contains("array_column", "value"),
    file_extension=path.file_ext("file_path")
)

# Some commonly used functions are also available directly
from datachain.func import sum, count, length, ifelse
dc.mutate(total=sum("amount"))
```
