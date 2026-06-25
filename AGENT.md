# AGENT.md

Durable, general insights about the DataChain **product and architecture** — what any
contributor (human or AI) should hold in mind before changing the core. Keep this file
general: task-specific findings and postmortems belong with the code or the PR, not here.

## Core model

- A signal is a typed column — a basic type (e.g. `int`, `str`) or a pydantic
  `DataModel`. A `DataModel`'s nested fields **flatten** into DB columns joined by `__`.
  The `SignalSchema` (the model tree) is serialized and travels with every dataset version.
- One user-facing API runs over several SQL **warehouse backends**, and user-visible
  behavior must be identical across all of them.
- Chain operations return a new chain and **never mutate the receiver** (its schema or
  query) in place — chains get reused, so mutating `self` corrupts later use.

## Two schemas, and the mapping between them

DataChain always holds two views of the data and constantly converts between them:

- **Logical** — the Python/pydantic model tree (`SignalSchema` is the source of truth).
- **Physical** — the flattened columns and their per-backend SQL types.

That mapping — *how a signal becomes columns and back* — is the heart of the system, and
where most bugs live, for one structural reason:

**There is no single chokepoint for the mapping.** It is re-implemented across many paths
that tend to drift apart:

- **ingestion** (UDFs, reading values, file/arrow readers),
- **export** (`to_pandas` / `to_parquet` / `to_csv` / `to_json` / `to_records` /
  `to_database` / `to_storage`),
- **each query operation** (filter / select / mutate / join / group_by / aggregate /
  union),
- **object hydration** vs **flat tabular export** — different read paths, different rules,
- **each backend.**

So a change to how signals map to storage must be **fanned out and verified across the
whole matrix**. One path passing tells you nothing about the others.

## Backends diverge — design for the set, not for one

Today: SQLite (local default) and ClickHouse (SaaS); BigQuery and Snowflake next;
Postgres and others later. They differ in **load-bearing** ways — null handling and its
cost, null ordering, aggregate-over-null semantics, supported types, even how an
expression evaluates over a join. Two consequences:

- **Keep the shared (logical) layer SQL-standard; push each backend's quirks into that
  backend's converter.** Never bake one backend's behavior into the shared layer.
- **The permissive default backend masks bugs** that only surface on stricter ones.
  Verify every change on all backends, not just the convenient one.

The canonical example is **nullability**: most warehouses are nullable-by-default and
cheap, but ClickHouse is NOT NULL by default with an explicit, costly `Nullable(T)`. Treat
nullability the standard way in the logical schema and make the NOT-NULL tradeoff a
ClickHouse-converter concern. A derived or composed expression over a nullable input is
itself nullable — propagate nullability through `Func` composition, not per leaf only.

The outlier is per-axis, not global: on the NaN axis SQLite is the odd one. It stores
`NaN` as `NULL`, so an `Optional[float]` `NaN` reads back as `None` and counts as null
there, while every other backend keeps them distinct (IEEE). `None` itself round-trips
consistently everywhere. Conform to the standard/majority; document the lone backend's
limit rather than contorting the others to match it.

## Serialization is a boundary

Both schema and data cross serialization boundaries — the `SignalSchema` and its SQL types
round-trip through the metastore with every dataset version, and exported files (parquet,
JSON, CSV) get read back later. These boundaries are **lossy unless you make them
otherwise, and the right answer differs per property.** A property that affects generated
DDL must survive the metastore round-trip, or a reload silently builds a different physical
schema. But internal/hidden columns are deliberately dropped from exports, and their
visibility is configurable. So decide, per property, whether it must round-trip — and test
the round-trip you depend on instead of assuming it.

## Not all of DataChain is in this repo

Some backend converters live in other repositories (e.g. the SaaS/Studio repo), and CI
couples them. A change to physical types may need a coordinated change there before it
works end-to-end — check before assuming the whole story is in this tree.

## Comments and docstrings

DataChain keeps comments **sparse**. Prefer code that reads clearly on its own; add a
comment only for the genuinely non-obvious (an unusual invariant, a subtle workaround).
Do not narrate *why a choice was made in the moment* or restate what the code already says.
Write every comment as a standalone, timeless fact about current behavior — a reader
months later on `main` has no notion of "this change". Watch for change-narrative tells:
*"now that", "without it", "previously", "used to", "we decided", "regression",
"this PR"*, and before/after framing. Git history records why it changed.

- **Tests:** name the test so its intent is obvious, and skip the docstring/comment — a
  good name makes it redundant.
- **Docstrings:** expected on public-facing APIs; rarely wanted on internal functions and
  helpers, where a clear name and signature should be enough.

## Working effectively (people and agents)

- **Run the full matrix.** Test on every backend and across the supported Python versions
  before calling something done; a single convenient setup hides divergences.
- **Test across feature boundaries**, not a feature in isolation — the surprising bugs
  live in compositions (two features used together, or one used as the only signal in a
  chain).
- **Separate a correctness bug from a design decision.** Fix the bugs; raise design and
  strategy questions — especially anything that would overfit the shared layer to one
  backend — with the maintainers instead of quietly encoding them.
