---
title: Function Library
---

# Function Library

DataChain ships built-in functions in `dc.func` that run inside the Query Engine as SQL. They never touch Python at runtime and execute at warehouse speed.

All functions are accessed through `dc.func` after `import datachain as dc`.

## Distance Functions

Array distance functions for vector search and analytics:

```python
import datachain as dc

chain.mutate(
    cos_dist=dc.func.cosine_distance("embedding", target_embedding),
    euc_dist=dc.func.euclidean_distance("embedding", target_embedding),
)
```

- `cosine_distance(column, reference)`: cosine distance between two vectors
- `euclidean_distance(column, reference)`: Euclidean distance
- `l2_distance(column, reference)`: L2 (squared Euclidean) distance

## Aggregate Functions

Standard SQL aggregates, usable in `group_by`:

```python
import datachain as dc

chain.group_by(
    avg_size=dc.func.avg("file.size"),
    total_size=dc.func.sum("file.size"),
    count=dc.func.count(),
    partition_by="column.category",
)
```

- `count()`: number of rows
- `sum(col)`: total of a numeric column
- `avg(col)`: arithmetic mean
- `min(col)`, `max(col)`: minimum/maximum value
- `collect(col)`: gather values into a list
- `concat(col)`: concatenate string values
- `any_value(col)`: arbitrary value from the group

## Window Functions

Partitioned analytics without leaving the engine:

```python
import datachain as dc

w = dc.func.window(partition_by="category", order_by="created_at")
chain.mutate(
    row_num=dc.func.row_number().over(w),
    running_rank=dc.func.rank().over(w),
    first_path=dc.func.first("file.path").over(w),
)
```

- `window(partition_by=, order_by=)`: creates a window spec (both required)
- `rank()`: rank with gaps for ties
- `dense_rank()`: rank without gaps
- `row_number()`: sequential row number within partition
- `first(col)`: first value in the partition

## Path Functions

Work natively with storage paths via `dc.func.path.*`:

```python
import datachain as dc

chain.mutate(
    ext=dc.func.path.file_ext("file.path"),
    stem=dc.func.path.file_stem("file.path"),
    filename=dc.func.path.name("file.path"),
    parent=dc.func.path.parent("file.path"),
)
```

## Conditional Functions

SQL-style branching:

```python
import datachain as dc

chain.mutate(
    label=dc.func.case(
        (dc.C("score") > 0.9, "high"),
        (dc.C("score") > 0.5, "medium"),
        else_="low",
    ),
    status=dc.func.ifelse(dc.func.isnone("result"), "pending", "done"),
)
```

- `case((cond, val), ..., else_=)`: multi-branch conditional
- `ifelse(cond, true_val, false_val)`: two-branch conditional
- `isnone(col)`: null check

## String Functions

String operations via `dc.func.string.*`:

- `length(col)`: string length
- `split(col, sep)`: split on separator
- `replace(col, old, new)`: substring replacement
- `regexp_replace(col, pattern, replacement)`: regex-based replacement. Matching
  is case-sensitive by default — prepend the `(?i)` inline flag to the pattern
  for case-insensitive matching (e.g. `r"(?i)foo"`).

```python
import datachain as dc
from datachain.func import string

chain.mutate(
    # Replace any digit run with "X"
    masked=string.regexp_replace("text", r"\d+", "X"),
    # Case-insensitive replacement: matches "foo", "FOO", "Foo", ...
    normalized=string.regexp_replace("text", r"(?i)foo", "bar"),
)
```

## Numeric / Bit Functions

- `bit_and(col_a, col_b)`, `bit_or`, `bit_xor`: bitwise operations
- `bit_hamming_distance(col_a, col_b)`: Hamming distance between bit vectors

## Hash Functions (ClickHouse only)

- `sip_hash_64(col)`: SipHash-2-4 producing a 64-bit integer
- `int_hash_64(col)`: integer hash producing a 64-bit integer
