# Typed column codec (draft)

This document describes a prototype of the first, deliberately limited typed storage
codec. It is a design draft, not a finalized production migration plan or a promise
to support every Python or Pydantic type. The goal is to make one declared annotation
produce one physical representation and one inverse read conversion, regardless of
whether a value came from `read_records()`, `read_values()`, or a UDF.

New typed writes are currently opt-in with
`DATACHAIN_EXPERIMENTAL_TYPED_CODEC=1`. The flag controls only selection for a new
schema. A persisted codec marker always controls how an existing dataset is read, so
turning the flag off cannot reinterpret typed bytes as legacy bytes.

```py
import os
from datetime import datetime, timezone

from pydantic import model_serializer

import datachain as dc

os.environ["DATACHAIN_EXPERIMENTAL_TYPED_CODEC"] = "1"


class Event(dc.DataModel):
    windows: list[list[datetime] | None]
    count: int

    @model_serializer(mode="plain")
    def export(self) -> dict:
        return {"external_count": self.count}


now = datetime(2024, 1, 2, tzinfo=timezone.utc)
event = Event(windows=[None, [now], None], count=1)
saved = dc.read_values(event=[event], output={"event": Event}).save(
    "typed-codec-example"
)

assert saved.to_values("event")[0] == event
assert saved.to_values("event.windows") == [[None, [now], None]]
```

The nested datetime and structural null round-trip from the explicit annotation. The
model serializer remains available for export, but DataChain stores `count=1` as
declared field state instead of persisting `{"external_count": 1}`.

## Why a codec is needed

Historically, type handling was split across annotation-to-SQL mapping, model
flattening, value inspection in the warehouse writer, SQL read conversion, and model
hydration. A physical `JSON` column could consequently mean either an untyped JSON
document or a typed value whose native SQL shape was unavailable. Once both meanings
were collapsed to JSON, the reader had to guess whether strings represented strings,
datetimes, bytes, or encoded JSON.

The typed codec keeps that logical information through the database boundary. It is
native-first: values use queryable native SQL columns where the backend can represent
the annotation, and use JSON only as a carrier for a typed subtree that cannot be
represented natively.

## Contract and responsibilities

`compile_codec(annotation)` returns a `ColumnCodec` for the supported subset, or
`None` for an annotation that must remain on the legacy path. A compiled codec owns:

- the original annotation;
- a concrete physical `SQLType`;
- recursive child codecs and nullability;
- `encode(value)`, which produces a database-bindable value; and
- `decode(value)`, which restores the declared logical value after the backend's
  outer SQL read conversion.

The rest of the pipeline has narrower jobs:

- `SignalSchema` selects and caches codecs and maps flattened model leaves back to
  their declared annotations.
- Model flattening projects declared fields into columns. For codec-backed models it
  passes the field values through structurally and does not call `model_dump()`.
- The common output adjustment step calls the codec once for all ingestion paths.
- SQL types and backend adapters bind and retrieve the physical value. They do not
  infer Pydantic or nested Python types.
- Typed `to_values()`/iteration and UDF-parameter leaf reads use the same codec as
  whole-model reads. Whole models are assembled from already decoded field values.

This separation is important: a write change and its inverse read change are one
storage-format change and must ship together.

### Verified paths and paths not yet integrated

This first vertical slice covers explicitly annotated `read_records()`,
`read_values()`, and UDF outputs; typed `to_values()` and normal iteration; whole-model
hydration and selected model leaves; UDF parameters; and persisted schema reloads.

It is not yet the conversion boundary for every consumer. Internal/raw result paths
such as `_leaf_values()` and `results()`, and pandas and Arrow exporters, can still
consume physical values. Schema inference and top-level default handling are also
unchanged. Finally, the native physical mappings in this prototype have not yet been
qualified on non-SQLite backends. Until those paths and backends have their own
contract tests, this is a scoped vertical slice rather than a replacement for the
entire serialization stack.

## Physical shapes in `typed-v1`

The codec preserves native scalar and array types where possible. Representative
shapes are:

| Annotation | Physical shape |
| --- | --- |
| `datetime` | native datetime |
| `bytes` | native binary |
| `list[int]` | `Array(Int64)` |
| `list[int | None]` | `Array(Nullable(Int64))` |
| `tuple[int, ...]` | `Array(Int64)`; decoded back to a tuple |
| `list[int] | None` | nullable JSON carrier |
| `list[list[datetime] | None]` | `Array(Nullable(JSON))` |
| `list[Child | None]` | `Array(Nullable(JSON))` |
| `dict[str, T]` | JSON carrier with a codec for `T` |

Backends cannot represent every recursive nullability pattern natively. For example,
an optional array is stored in a nullable JSON carrier, while an array containing
optional arrays or models uses nullable JSON elements. Those JSON values are still
typed: a child codec restores each nested value rather than returning the physical
JSON representation.

Datetime and bytes remain native unless a containing node requires JSON. Inside a
JSON carrier, `typed-v1` uses ISO datetime strings and complete base64-encoded bytes.
The descriptor, rather than inspection of a neighboring value, determines how they
are decoded.

For example, all of these positions have the same declared representation:

```py
annotation = list[int | None]
values = [[], [None], [None, 1], [1, None], [None, None]]
```

Likewise, the annotation is sufficient even when a particular value contains no
`None`:

```py
annotation = list[list[datetime] | None]
value = [[created_at, updated_at], [finished_at]]
```

No first-element inference or full-array scan is part of the format.

## Models are field-state snapshots

The explicit model storage contract is a snapshot of the validated values of the
model's declared fields. DataChain does not use arbitrary Pydantic export output as
its storage format. In particular:

- `model_serializer` and `field_serializer` affect explicit Pydantic export, not
  DataChain storage;
- computed fields, private attributes, and export-only aliases are not persisted;
- nested models use the same declared-field rule as top-level models; and
- typed reads construct the model from decoded field state without rerunning
  validators.

When a supported ingestion API accepts a mapping for a declared model, it first
validates that mapping with the original model class and then snapshots the resulting
field state. That validation is an input normalization step, not a request to persist
the model's export serializer output.

This is intentional. Stored feature metadata contains field annotations and model
bases, but not the original class's serializer and validator methods. Treating
`model_dump()` as durable storage would create data that a reconstructed model cannot
necessarily invert. A Pydantic `TypeAdapter` also cannot make an arbitrary one-way
serializer reversible.

For example, if a validator changes `value=1` into validated state `value=11`, and a
field serializer exports it as `"export:11"`, DataChain stores the integer `11`.
Reading the dataset returns state `11`; it neither stores the export string nor runs
the validator a second time.

A future custom storage extension should be explicit and paired: a stable codec ID,
an encoder, and a decoder. It should not be inferred from Pydantic decorator metadata.

## Nulls and defaults

Nullability belongs to each recursive codec node. Thus `None` and `[]` remain distinct
for `list[int] | None`, and `None` elements are preserved for
`list[int | None]`. A `None` encountered inside an encoded non-nullable collection or
model node is rejected.

The prototype does not redesign top-level output defaults. The common output layer
still handles a missing column or a top-level explicit `None` before invoking the
codec, using the existing SQL type and backend default policy. Optional-model absence
also continues to use the existing model sentinel because a flattened model spans
more than one database column. Separating missing values from top-level explicit
`None` is future work, not a behavior change in `typed-v1`.

## Versioning and compatibility

Feature metadata records codec selection per signal:

```json
{
  "items": "list[Optional[int]]",
  "_storage_codecs": {"items": "typed-v1"}
}
```

Physical array types also carry `dc_codec: "typed-v1"`. This prevents the SQL reader
from applying the historical per-item JSON conversion before the logical codec sees
the value.

The per-signal map allows legacy inputs and newly produced typed columns to coexist in
one chain. A feature schema with no `_storage_codecs` entry is explicitly legacy; it
must not silently opt into the newest codec merely because its annotation is now
supported. Existing datasets therefore retain their historical reader. With the
experimental flag enabled, new writes use `typed-v1` for supported collection and
model signals. Reads obey stored metadata whether or not the flag is enabled.

The codec version is part of schema identity and checkpoint hashing. A future physical
change must use a new version and ship its writer and reader together. Operations that
compare or combine persisted physical values must not assume differently versioned
representations are byte-compatible; they should re-encode to a common version or
refuse the operation clearly.

The draft refuses a union when corresponding signals use different storage codec
versions. A join policy is still forthcoming. Even where a legacy and typed simple
shape happen to decode to equal logical values, that observation is not evidence that
their persisted bytes are compatible for SQL-level union, distinct, or comparison.

Before production rollout, the migration needs an explicit compatibility matrix for
each backend and operation, fixture datasets written by the legacy release, and a
decision about whether cross-version union, merge, and distinct normalize or refuse.
The metadata mechanism makes that migration possible; this draft does not claim that
the migration has been completed.

## Initial supported subset

`typed-v1` is intentionally small:

- `bool`, `int`, `float`, `str`, `bytes`, and `datetime`;
- `T | None` with exactly one non-`None` arm;
- `list[T]`, fixed tuples, and `tuple[T, ...]`;
- `dict`, `dict[str, Any]`, and `dict[str, T]`;
- non-recursive Pydantic models whose declared fields all compile; and
- `Annotated[T, ...]`, using the codec for `T`.

Not yet supported by the typed codec:

- general or ambiguous unions such as `int | str`;
- `Enum` and `Literal`;
- recursive model graphs;
- mappings with non-string keys;
- native arrays whose element shape contains `bytes`;
- arbitrary collection protocols or sets; and
- custom Pydantic export formats as storage formats.

Unsupported annotations return `None` from `compile_codec()` and stay on the legacy
path. Adding a type family requires paired encode/decode behavior and end-to-end tests
for every ingestion path, physical save/reload, full-model reads, and leaf reads. It
must not be enabled by a writer-only fallback or runtime inspection of sample values.

The codec also does not redesign schema inference. For example, `read_values()` without
an explicit `output` still uses the existing inference rules before codec compilation.
Callers that need the typed contract for ambiguous, optional, or all-null values should
provide the declared output annotation.
