# String Functions

Functions for string manipulation, text processing, and string analysis.

## Usage

String functions are available under the `func.string` namespace to avoid name collisions with other functions:

```python
from datachain.func import string

# Use string functions with the string namespace
dc.mutate(
    str_len=string.length("text_column"),
    parts=string.split("text_column", ","),
    cleaned=string.replace("text_column", "old", "new"),
    regex_cleaned=string.regexp_replace("text_column", r"\d+", "X"),
    distance=string.byte_hamming_distance("col1", "col2"),
)
```

::: datachain.func.string
