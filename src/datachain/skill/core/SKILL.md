---
name: datachain-core
description: Use ONLY for abstract DataChain SDK questions — API usage, method signatures, or code patterns — when no specific dataset or bucket is referenced. If the request mentions creating, saving, listing, exploring datasets or buckets, use datachain-knowledge instead.
---

You are now loaded with expert-level DataChain SDK context. Apply every rule below when generating DataChain Python code.

## Scope of this skill

This file is SDK mechanics — how to write DataChain code that runs correctly: API usage, UDF signatures, settings, delta semantics, materialization patterns, saving, exporting.

**It does not own methodology.** Decisions about *which* datasets to build, what scope, what shape (Container / Asset / Sense / Task), what fields to save, and when to dialogue with the user about layer choices — those are the CAST methodology, which lives in the **datachain-knowledge** skill at `{knowledge_skill_dir}/CAST.md`.

When knowledge is loaded, it is the orchestrator: it plans the layers (CAST §4), invokes the rules in this file to write the code, then runs the KB pipeline. When knowledge is *not* loaded (raw SDK use, no `dc-knowledge/` directory), this file is self-sufficient — CAST doctrine simply does not apply.

If you find yourself reasoning about "should I build a Sense layer here?" or "should this be scoped to the bucket or the directory?" from inside this file, stop — those questions belong upstream. Ask the user to load the knowledge skill, or fall through to a direct solve.

## Pre-Generation Checklist

- [ ] **Every UDF has a known output type.** Functions passed to `.map()`, `.gen()`, or `.agg()` must have their return type resolved. See §2 Rule 2 — the #1 runtime error.
- [ ] **No `from __future__ import annotations` in UDF modules.** It stringifies type hints; DataChain's signal-schema resolution then rejects the string-vs-class mismatch.
- [ ] **Bucket access: anonymous or authenticated?** Check `dc-knowledge/buckets/` for a `.md` file with `anon: true/false` in frontmatter. If none, run `datachain bucket status <uri>` to detect. If `denied` or `not found`, stop and ask the user.
- [ ] **Heavy-init resources load via `.setup()`**, not module-level lazy globals:
  ```python
  chain.setup(model=lambda: load_model()).map(result=run_model)
  ```
  Lazy globals leak across `parallel=N` workers and hide the dependency from the chain definition. See §2 Rule 20.
- [ ] **`.settings(parallel=N)` is the right tool only when the workload benefits.** See §2 Rule 6.

---

## Section 1 — Dataset Reuse (Highest Priority)

**Before writing any pipeline code, check what already exists.**

1. If `dc-knowledge/index.md` exists, read it **first**.
2. When the user's task overlaps with an existing dataset, read its `.md` under `dc-knowledge/datasets/` for schema, code patterns, and lineage.
3. **Reuse over rebuild.** Start from an existing dataset (`dc.read_dataset("name")`) whenever it covers the data the user needs — even partially. Filter, merge, or extend it instead of re-reading raw storage.

Only go to raw storage when no existing dataset covers the needed data, or the user explicitly asks to start fresh.

### Dataset-first reasoning

Datasets are the unit of reasoning. Chains that transform data through UDFs — or that produce a pipeline's final result — should be saved as named datasets.

**Core rule: always `.save()`, never just `.show()`.** A pipeline's terminal operation is `.save("descriptive_name")`, followed by `.show()` on the saved result for display. Two exceptions: (1) one-off exploratory queries where the user explicitly asks to "show me" or "print"; (2) Task-layer outputs per the CAST methodology — persist by exception, not by default. The always-save rule is absolute for C/A/S substrate layers.

**Critical anti-pattern: bypassing `.save()` by dumping in-memory rows to a file.** Reading the chain via `.to_list()` / `.to_values()` and writing to disk via `open()`, `json.dump`, `pandas.to_csv`, or any Python-side file handle is forbidden for UDF-bearing pipelines. The pipeline result must land as a saved dataset first via `.save()`. Once saved, exporting via `chain.to_csv()`, `chain.to_parquet()`, `chain.to_storage()` is fine.

**Not a bypass:** a UDF that materializes a payload to storage and returns a `dc.File` pointer. The dataset still lands via `.save()`; the file in storage is the row's payload, owned by DataChain via the pointer.

```python
# ✗ ANTI-PATTERN — UDF result pulled into Python and dumped to disk.
results = chain.map(emb=encode_image).to_list("file", "emb")
with open("similar_results.json", "w") as f:
    json.dump(results, f)

# ✓ Save the dataset first, then export from it if needed.
saved = chain.map(emb=encode_image).save("product_catalog_embeddings", attrs=[...])
saved.to_csv("similar_results.csv")
```

**What to save — the UDF rule:**

- **Any chain that runs a UDF (`.map()`, `.gen()`, `.agg()`)** must be saved with `.save("name")`. UDFs embody domain logic and produce structured output worth preserving.
- **Final pipeline results.** Rankings, filtered cohorts, evaluation outputs, aggregations — always `.save("name")`.
- **Chains with no UDFs** (`read_storage` + `filter`/`mutate`/`select` only) may remain transient — cheap to recompute, easy to read from the code.

**Prompt-trigger keywords for `.save()`.** When the user's task description contains "make available for downstream queries", "compute per-X aggregates", "build / extract / produce X", "store / persist / materialize / save", "process and save" — call `.save("name")` and print a short summary (name + row count or a few stats), not the full result set.

**`.persist()` is not `.save()`.** `.persist()` materializes a chain into an anonymous dataset — it prevents re-execution but creates no named dataset. When a chain should be saved per the rules above, use `.save("name")`.

### Code-level decomposition: one stage = one script

A multi-stage pipeline that produces multiple named datasets through expensive stages (LLM calls, embeddings, ML inference) belongs in MULTIPLE scripts — one per stage — not folded into one monolith. Each script reads from the previous stage's saved dataset via `dc.read_dataset(...)` and writes its own with `.save("name")`.

**Split when ANY** of these hold for the stage output: runs an LLM/VLM/embedding/inference call; will be reused by future questions; will exceed ~10 rows and the user might inspect or merge later; wall time >5 min; chain has 3+ distinct operations.

**Don't split** for: single filter/select/limit on an existing dataset; cheap metadata aggregation; a one-off query that displays rather than saves.

**Naming:** each script is named after the dataset it produces.

```
build_product_catalog_embeddings.py    →  l3_product_catalog_emb dataset
build_product_catalog_metadata.py      →  l1_product_catalog_meta dataset
similar_to_query.py                     →  products_similar_to_query (or ad-hoc)
```

Generate stage scripts up front, on the first pass — "I'll write one script for speed and refactor later" is the regression.

### Expensive compute: save full, filter downstream

When a UDF is expensive (ML inference, LLM calls), save the **full, unfiltered** result before any filtering. A downstream `.save()` after filtering only preserves a fraction of the rows; the rest of the compute is lost.

**Problem-specific filters belong DOWNSTREAM of the expensive `.save()`.** A filter is **problem-specific** if its criterion comes from the user's task description (a named exclusion, a threshold, a category) — it MUST go after the expensive `.save()`. A filter is **data-quality** if it would apply to ANY question over this dataset (corrupted file, mandatory field missing) — it MAY go before.

```python
# ✗ Pre-filtering with problem-specific criteria — embeddings useless for next question.
embeddings = (
    dc.read_storage("s3://product-catalog/images/")
    .filter(dc.C("condition") != "refurbished")     # ← problem-specific
    .filter(dc.C("width") > 400)                    # ← problem-specific
    .setup(model=lambda: clip)
    .map(emb=encode_image)
    .save("l3_product_catalog_clip")                # ← USELESS for next question
)

# ✓ Save embeddings over the WHOLE input, filter downstream as a Task.
embeddings = (
    dc.read_storage("s3://product-catalog/images/")
    .setup(model=lambda: clip)
    .map(emb=encode_image)
    .save(
        "l3_product_catalog_clip",
        attrs=["cast:sense", "scope:bucket", "source:product_catalog"],
        description="CLIP ViT-B-32 embeddings over the full product-catalog bucket.",
    )
)

ranked = (
    dc.read_dataset("l3_product_catalog_clip")
    .merge(dc.read_dataset("l1_product_catalog_meta"), on="file.stem")
    .filter(dc.C("condition") != "refurbished")     # ← problem-specific, downstream
    .filter(dc.C("width") > 400)
    .mutate(distance=dc.func.cosine_distance(dc.C("emb"), query_emb))
    .order_by("distance").limit(5)
    .save(
        "products_similar_to_query",
        attrs=["cast:task", "scope:onetime", "source:products_similar_to_query"],
        description="Top-5 catalog products visually closest to query.jpg under the filter set.",
    )
)
```

**Data-quality filters before the UDF are fine:**

```python
embeddings = (
    dc.read_storage("s3://b/")
    .filter(dc.C("file.size") > 0)                  # ← data-quality, OK
    .setup(model=lambda: clip)
    .map(emb=encode_image)
    .save("clip_embeddings")
)
```

### CAST quick reference

CAST is the four-layer pattern owned by the `knowledge` skill at `CAST.md`. The full doctrine (recall economics, layer-ladder walk, calibration, dialogue) lives there. This is just enough to recognize the layer names:

- **Container** — typed index of what each file IS without full decode (paths, headers, sidecars).
- **Asset** — raw extracted or mixed data in workable shape (decoded units, joined mixtures).
- **Sense** — what a model said about the data (embeddings, classifications, transcriptions).
- **Task** — task-specific composition on top of C/A/S. Persist by exception.

**Naming convention:**

```
l1_<source>_<descriptor>      # Container — listings, headers, sidecar metadata
l2_<source>_<descriptor>      # Asset — extracted/reshaped raw data
l3_<source>_<descriptor>      # Sense — model-derived signals
<descriptor>                  # Task — no prefix
```

The `l1_` / `l2_` / `l3_` prefix is enough; do NOT add layer-type infixes like `_container_`. Cap at 30 chars; the full doctrine is in `knowledge/CAST.md` §3.

**Tag every `.save()` with `attrs` and `description`** so the knowledge skill can resolve the layer:

```python
chain.save(
    "l3_product_catalog_clip",
    attrs=[
        "cast:sense",                               # container | asset | sense | task
        "scope:bucket",                             # bucket | directory | sample | onetime
        "source:product_catalog",
    ],
    description="CLIP ViT-B-32 embeddings over the full product-catalog bucket.",
)
```

Lineage is tracked automatically; do NOT add `parent:` attrs.

Never create or modify files under `dc-knowledge/` — that directory is owned by the `knowledge` skill.

---

## Section 2 — Critical Rules

```
0. TRAILING SLASH: Always add / to bucket/prefix paths.
   ✓ dc.read_storage("s3://bucket/images/")
   ✗ dc.read_storage("s3://bucket/images")  ← permission error on anon access

1. ANON FOR PUBLIC BUCKETS (auto-detected): When `anon` is not passed,
    `dc.read_storage()` probes the bucket anonymously first; if the probe
    succeeds, it transparently sets `anon=True`. No need to pass `anon=True`
    for public buckets — works whether or not cloud credentials are configured.
    Pass `anon=True` explicitly only to skip the probe round-trip in a
    latency-sensitive path. Pass `anon=False` to bypass the anonymous probe
    for private buckets.
    Applies ONLY to dc.read_storage() — NOT to File.at() or other APIs.
    ✓ dc.read_storage("gs://bucket/data/")                # auto-detected
    ✓ dc.read_storage("gs://bucket/data/", anon=True)     # explicit, skips probe
    ✗ dc.File.at("gs://bucket/file.txt", anon=True)        ← File.at() has no anon param

    **Anon does NOT propagate across `.save()` boundaries.** A downstream UDF
    that calls `file.open()` in a new process makes a fresh HeadObject without
    anon → 403. Fix: pass anon into the downstream session via `client_config`:
    ✓ session = dc.Session.get(client_config={"anon": True})
      (dc.read_dataset("l2_my_bucket_files", session=session)
         .map(emb=encode_image)
         .save("l3_my_bucket_emb"))

2. EVERY UDF MUST HAVE A KNOWN OUTPUT TYPE. A UDF passed to map/gen/agg without
   a resolved return type defaults to str and crashes at runtime for any non-str
   value. This is the #1 source of production errors — enforce strictly.

   Three ways to specify the output type, in priority order:

   (a) Named function with return type annotation — PREFERRED, use by default.
       ✓ def get_info(file: dc.ImageFile) -> dc.Image:
             return file.get_info()
         chain.map(info=get_info)
       ✓ def get_name(path: str) -> str:
             return path.split("/")[-1]
         chain.map(name=get_name, params=["file.path"])

   (b) Lambda — ONLY when return type is str (the default). If the return type
       is anything other than str, you MUST pair the lambda with output=.
       ✓ chain.map(name=lambda path: path.split("/")[-1], params=["file.path"])
       ✓ chain.map(sz=lambda size: size // 1024, params=["file.size"], output={"sz": int})
       ✗ chain.map(info=lambda file: file.get_info())  # no output= → crash; also downloads file
       ✗ chain.map(size=lambda file: file.size)         # no output= → crash; also downloads file

   (c) output= parameter — LAST RESORT for named functions, only when you cannot
       annotate the function (e.g., third-party callable you cannot modify).
       ✓ chain.map(emb=third_party_fn, output={"emb": list[float]})

   params= is allowed with any of the above to bind function parameters to specific
   columns (e.g., nested fields like "file.path"). Prefer matching function parameter
   names to column names when possible.

3. AVOID FILE OBJECT WHEN CONTENT NOT NEEDED: Passing a File object to a UDF
    downloads the full content, even if the UDF only reads metadata. This applies
    to File and ALL its subclasses.
    - Use params= to bind UDF args to nested columns like "file.path", "file.size".
    - For pure SQL path ops: use mutate() with func.path.* (no Python needed).
    - Pass File object ONLY when you need file content (.read(), .open(), etc.).
    ✓ chain.map(category=classify, params=["file.path"])     # no download
    ✓ chain.mutate(stem=dc.func.path.file_stem(dc.C("file.path")))   # pure SQL
    ✗ chain.map(category=lambda file: file.path.split("/")[-2])  # downloads entire file

4. COLUMN NAMING: keyword in map/gen/agg = new column name.
   chain.map(embedding=fn)  → column is named "embedding"

5. INPUT PARAM: The file column is always named "file" regardless of modality.
   Use params= when arg names don't match column names:
   chain.map(label=process, params=["file.path"])

6. PARALLEL WHEN NEEDED: Only use .settings(parallel=N) when the workload benefits.
   Use when per-row work is I/O-bound (file download, API calls), or CPU-bound
   AND the UDF doesn't already saturate cores via internal threading.
   Skip when the model saturates the device (single-GPU), the total wall is <30s
   (bootstrap dominates), or per-worker memory × N would OOM.
   Rough picks: I/O-bound → parallel=4-8; CPU-bound → parallel=2-4; external API → parallel=8-16.
   ✓ chain.settings(parallel=4).map(emb=model_fn)
   ✓ chain.map(label=classify)                       # lightweight → sequential
   `workers=N` is Studio-only distributed processing — guard with dc.is_studio():
   ✓ chain = chain.settings(parallel=4)
     if dc.is_studio():
         chain = chain.settings(workers=8)

7. PREFETCH FOR FILE-READING UDFs: Estimate avg file size and compute:
     prefetch = clamp(4MB / estimated_avg_size, 2, 128)
   Only add .settings(prefetch=N) if N > 4 (default is 2). Skip for UDFs that
   don't read file content. Skip if the user explicitly sets prefetch.

8. CACHE ONLY WHEN NEEDED: Do not add cache=True by default. Use only when the
   same files are read multiple times (multi-stage pipelines), or the user asks.

9. COLUMN-COLUMN ARITHMETIC: Use chain.column() instead of C() when combining
   two columns. C() does not carry type info → transpiler can't infer the result.
   ✓ chain.mutate(total=chain.column("price") * chain.column("qty"))
   ✓ chain.mutate(discounted=C("price") * 0.9)          # scalar → C() is fine
   ✗ chain.mutate(total=C("price") * C("qty"))           # no type → error

10. READ NOT FROM: Use dc.read_* module functions. `DataChain.from_*` methods
    were removed; they raise AttributeError.
    ✓ dc.read_csv("s3://data.csv")
    ✓ dc.read_dataset("name")
    ✗ DataChain.from_csv("s3://data.csv")        ← AttributeError
    `from datachain import DataChain` is itself a smell — never write it.
    Never assign to the name `dc`: `dc = DataChain.from_dataset("x")` shadows
    the package and breaks every subsequent `dc.read_*`, `dc.C`, `dc.func.*`.

11. GLOB IN PATH: When filtering by extension or name pattern, put the glob
    directly in the read_storage() path instead of a separate .filter() call.
    The type= parameter only sets the File subclass — it does NOT filter the listing.
    ✓ dc.read_storage("s3://bucket/**/*.{jpg,jpeg,png}", type="image")
    ✗ dc.read_storage("s3://bucket/", type="image")  ← lists ALL files

12. SINGLE FILE vs MULTI FILE: Use the right API.
    - One known file: dc.File.at() / dc.TextFile.at() / dc.ImageFile.at()
    - One known CSV/JSON/Parquet: dc.read_csv() / dc.read_json() / dc.read_parquet()
    - A small fixed set: one read_storage() with a glob pattern.
    - Many files in a directory: dc.read_storage()
    read_storage() is for directory listing — don't use it for a single known file.

13. ONE SIGNAL PER MAP/GEN/AGG: Each call accepts exactly one signal.
   For multiple columns, chain calls or return a Pydantic BaseModel.
   ✗ chain.map(a=fn1, b=fn2)              # ERROR: multiple signals

14. NO TUPLE RETURNS, NO DICT RETURNS: Always prefer Pydantic BaseModel classes to tuple
    or dict in map/gen/agg functions until user directly asks for tuple.
    ✓ def fn(file: dc.File) -> MyModel: ...   # named fields via BaseModel
    ✓ def fn(file: dc.File) -> int: ...       # single scalar
    ✗ def fn(file: dc.File) -> tuple[int, int]: ...  # → col_0, col_1
    ✗ def fn(file: dc.File) -> dict: ...      # dict keys become VALUES, not names → crash

    **Multi-column output: use BaseModel, NEVER tuple-via-output{}.** Pairing a
    tuple-returning UDF with `output={"width": int, "height": int}` works by
    positional accident; BaseModel is the canonical, named, addressable form:
    ✓ class Dims(BaseModel):
          width: int
          height: int
      def get_dims(file: dc.ImageFile) -> Dims:
          info = file.get_info()
          return Dims(width=info.width, height=info.height)
      chain.map(dims=get_dims)                                     # → dims.width, dims.height
    ✗ def get_dims(file: dc.ImageFile) -> tuple[int, int]: ...
      chain.map(dims=get_dims, output={"width": int, "height": int})  # ← anti-pattern

    **Scope.** This rule covers UDFs passed to `.map()`, `.gen()`, `.agg()` — their
    return values become chain signals (columns). Tuple returns force auto-generated
    names (`col_0`, `col_1`) that downstream `.merge()` / `.select_except()` can't
    address. The rule does NOT apply to:
      - Free helper functions called inside UDFs (return any Python shape).
      - `.setup(name=loader_fn)` loaders. The loader's return value is an opaque
        per-worker resource, not a chain signal. A tuple works; BaseModel is nicer
        when the loader produces multiple distinct resources.

15. MERGE NOT DICTS: When combining sources, read each as its own chain, parse
    inside map()/gen(), then merge(). Never build Python dicts outside the chain
    and close over them in map()/gen().
    ✓ annotations = dc.read_storage("./**/list.txt", type="text").gen(ann=parse_list)
      images = dc.read_storage("./images/", type="image")
      images.merge(annotations, on=dc.func.path.file_stem(dc.C("file.path")), right_on="ann.name")

16. SHARED LISTING PREFIX: When multiple read_storage() target the same tree,
    use the common parent prefix and glob from there. DataChain caches listings
    by prefix — shared prefix = one listing + cache hits.

17. LAZY CHAINS — NO DOUBLE EXECUTION: Chains are lazy — each terminal op
    re-executes the pipeline. Never call two terminal ops on the same chain.
    - After save(): use the returned chain (save() returns the saved dataset).
    - For multi-use chains: call .persist() to materialize once.
    ✓ saved = chain.save("my_data"); saved.show(5)
    ✓ materialized = chain.persist(); materialized.show(5); materialized.to_csv("out.csv")
    ✗ chain.save("my_data"); chain.show(5)               ← runs the pipeline twice

18. INLINE FUNC EXPRESSIONS: Pass func/C expressions directly to on=, right_on=,
    partition_by=, order_by=. Don't mutate() a throwaway column.
    For UDFs that need nested fields, use params= instead of mutate().
    ✓ chain.merge(other, on=dc.func.path.file_stem(dc.C("file.path")), ...)
    ✓ chain.map(label=classify, params=["file.path"])
    ✗ chain.mutate(stem=...).merge(other, on="stem")  ← unnecessary column

19. SELECT_EXCEPT AFTER MERGE: After merge(), use select_except() to drop duped
    columns. Never write a long select() list (>4 columns). When chaining multiple
    merges, do ALL merges first, then ONE select_except() at the end.

20. INLINE SETUP OVER UDF CLASS: Prefer chain.setup() over dc.Mapper/Generator
    classes. A plain function + .setup() achieves model/client initialization
    without introducing a class. Use dc.Mapper only when setup requires multiple
    self.* fields or custom __init__ args.

21. MATERIALIZE BEFORE MULTI-MERGE AND AFTER GROUP_BY: When merging 2+ right-side
    chains that contain UDFs, materialize each before the merge. Without it, the
    final terminal op may re-execute UDF pipelines multiple times during merge.
    Use .save("name") when the chain warrants a named dataset (has a UDF with reuse
    value). Use .persist() only when not warranting a name (group_by intermediates).
    Skip materialization when the right side has no UDFs (pure metadata/filter).
    Also persist() after group_by() when the result feeds further operations.
    ✓ a = chain_a.map(x=fn1).save("feature_a")
      b = chain_b.gen(y=fn2).save("feature_b")
      images.merge(a, ...).merge(b, ...).save("out")
    ✓ counts = chain.group_by(n=func.count(), partition_by="cat").persist()
      counts.filter(C("n") > 5).save("popular")
    ✗ a = chain_a.map(x=fn1)       # lazy → UDFs re-execute during merge
      b = chain_b.gen(y=fn2)
      images.merge(a, ...).merge(b, ...).save("out")
```

---

## Section 3 — Golden Rule

1. **Always use DataChain for data file access.** NEVER use stdlib (`os.walk`, `os.listdir`, `glob.glob`, `pathlib.Path.iterdir()`, `.glob()`, `.rglob()`) to discover or traverse DATA files. They lose lineage, skip prefetch/cache, and can't be materialized as a typed dataset.
   - Single known file: `dc.File.at()` / `dc.TextFile.at()` / `dc.ImageFile.at()`
   - Single CSV/JSON/Parquet: `dc.read_csv()` / `dc.read_json()` / `dc.read_parquet()`
   - Many files: `dc.read_storage()` (vectorised, preserves lineage)
   - Glob: use `dc.read_storage("path/**/*.ext")`, NOT `glob.glob`

   **Scope.** This rule governs DATA file access. Reading a handful of log/metadata files for one-off introspection during debugging is outside the rule.

2. **Prefer Data Memory over Compute Engine.** Data Memory ops — `filter()`, `mutate()`, `group_by()`, `order_by()`, `select()`, `merge()`, `union()`, `distinct()`, `limit()` — run as SQL at warehouse speed using `dc.C()` and `dc.func.*`. Compute Engine ops (`map`/`gen`/`agg`) run heavy Python in parallel workers and are expensive. Use Compute Engine ONLY for file content reads, model inference, LLM calls, external APIs.

3. **Extracting results.** Use `to_values()` for one column (returns flat list); `to_list()` for multiple columns (returns tuples). Never use `to_iter()` — it loses parallelism and lineage. For processing, use `map()` / `gen()` rather than extracting and looping.
   ```python
   files = chain.to_values("file")           # → [File(...), ...]
   rows = chain.to_list("file", "label")     # → [(File, "cat"), ...]
   ```

---

## Section 4 — Import Cheat Sheet

- ✓ `import datachain as dc` — the ONLY way to import the package.
- ✓ `from pydantic import BaseModel` — for custom schemas.
- ✓ `from datachain import model` — for annotation type imports (rare).
- ✗ `from datachain import File, C, func, ...` — NEVER. Use `dc.File`, `dc.C`, `dc.func`.
- ✗ `from datachain import DataChain` — NEVER. Use `dc.read_*` module functions.
- ✗ `dc = DataChain.from_dataset("x")` — NEVER assign to `dc`. It shadows the package.

```python
import datachain as dc
from pydantic import BaseModel
```

---

## Section 5 — Core API Reference

**Entry points.** `read_storage()` creates a cached listing keyed by prefix; subsequent calls with the same prefix reuse the cache.

```python
dc.read_storage("s3://bucket/prefix/", type="image")          # File / ImageFile etc.
dc.read_storage("s3://bucket/imgs/**/*.{jpg,png}", type="image")  # glob in path
dc.read_csv("s3://bucket/data.csv")
dc.read_json("s3://bucket/ann.json", jmespath="images")
dc.read_parquet("s3://bucket/data/*.parquet")
dc.read_hf("dataset-name", split="train")
dc.read_pandas(df); dc.read_values(scores=[1.2, 3.4]); dc.read_records([{"a": 1}, ...])
dc.read_database("SELECT * FROM t", "sqlite:///local.db")
dc.read_dataset("name")                    # latest version
dc.read_dataset("name", version="2.0.0")
```

**Data Memory (SQL, fast).** `C` = `dc.C`, `func` = `dc.func`.

```python
chain.filter(C("file.size") > 1000)
chain.filter((C("det.label") == "cat") & (C("det.conf") > 0.9))
chain.filter(C("file.path").glob("*.jpg"))
chain.filter(C("name").contains("alice"))     # also startswith, endswith, like, ilike, regexp
chain.filter(C("name").isnot(None))           # also is_(None)
chain.filter(C("price").between(10, 25))
chain.filter(C("name").in_(["alice", "bob"]))
# Combinators — always parenthesize: & (and), | (or), ~ (not)
chain.mutate(ext=func.path.file_ext(C("file.path")))
chain.mutate(dist=func.cosine_distance(C("emb"), reference))
chain.mutate(total=chain.column("price") * chain.column("qty"))      # column-column
chain.mutate(discounted=C("price") * 0.9)                            # scalar → C() is fine
chain.mutate(price_int=chain.column("price").cast(sa.Integer))       # import sqlalchemy as sa
chain.group_by(cnt=func.count(), total=func.sum(C("file.size")), partition_by="category")
chain.order_by("dist"); chain.order_by("score", descending=True)
chain.distinct("response.text")
chain.distinct(file_ext=func.path.file_ext(C("file.path")))          # expressions need names
chain.limit(100)
chain.select("file", "score"); chain.select("file", score_pct=C("score") * 100)
chain.select_except("internal_id")
chain.merge(other, on="id", right_on="meta.id")                      # left join (default)
chain.merge(other, on="id", inner=True)                              # inner
chain.merge(other, on="id", full=True)                               # full outer
chain.union(other); chain.subtract(other)
chain.diff(other, on="id", compare=["score"]); chain.file_diff(other)
```

`merge()` has NO `how=` parameter. Use `inner=True` or `full=True`.

**Compute Engine (Python workers, expensive):**

```python
chain.map(col_name=fn)              # 1 input → 1 output record
chain.gen(col_name=fn)              # 1 input → N output records
chain.agg(col_name=fn, partition_by="key")
```

**Setup and settings:**

```python
chain.setup(model=lambda: load_model()).map(fn)
chain.settings(parallel=4, cache=True, prefetch=10)
if dc.is_studio():
    chain = chain.settings(workers=50)
```

**Terminal operations.** `.save()` creates a named, versioned, KB-tracked dataset. `.persist()` materializes anonymously (calibration runs, intermediate materialization not entering KB).

```python
chain.save("dataset_name")                     # versioned, in dc.datasets() and KB
chain.save("ns.proj.name", update_version="minor")
chain.persist()                                # anonymous, NOT named, NOT in KB
chain.show(limit=10)
chain.to_values("col")                         # → flat list
chain.to_list("col1", "col2")                  # → list of tuples
chain.to_pandas(); chain.to_parquet("out.parquet"); chain.to_csv("out.csv")
chain.to_pytorch(transform=..., tokenizer=...)
chain.to_storage("s3://output/", signal="file", placement="filepath")
chain.count(); chain.sum("column"); chain.avg("column")
```

**Delta + incremental:**

```python
dc.read_storage("s3://bucket/", update=True, delta=True)
# Defaults: delta_on=("file.path", "file.etag", "file.version"); delta_compare=None.
# Override delta_compare="file.mtime" only when etag is unreliable (e.g. local FS).
```

---

## Section 6 — Type System

**Structured types — use Pydantic BaseModel:**

```python
from pydantic import BaseModel
from datachain import model

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: model.BBox
```

**File types (all inherit from `dc.File`):**

| Type | `type=` | `.read()` | Extra methods |
|---|---|---|---|
| `dc.File` | (default) | `bytes` | `.read_text()`, `.open()`, `.ensure_cached()` |
| `dc.TextFile` | `"text"` | `str` | `.read_text()` |
| `dc.ImageFile` | `"image"` | `PIL.Image` | `.get_info()` → `dc.Image` |
| `dc.VideoFile` | `"video"` | — | `.get_frame(frame, ...)`, `.get_frames(step=N, ...)`, `.get_fragments(duration)`, `.get_info()` → `dc.Video` |
| `dc.AudioFile` | `"audio"` | — | `.get_fragments(duration)`, `.get_info()` → `dc.Audio` |

`dc.Image`, `dc.Video`, `dc.Audio` are media metadata models in the `dc` namespace — NOT in `datachain.model`.

Sub-file units: `VideoFrame` (`.timestamp`, `.get_np()`, `.read_bytes(format)`, `.save(path)`), `VideoFragment` (`.save(path)`), `AudioFragment` (`.get_np()` → `(ndarray, sample_rate)`, `.save(path)`).

**Annotation types** (prefer these over custom BaseModels):

```python
from datachain import model      # import is mandatory; dc.model.BBox is not enough

model.BBox(title="car", coords=[x1,y1,x2,y2])              # PASCAL VOC
model.BBox.from_coco([x,y,w,h], title="car")
model.BBox.from_yolo([cx,cy,w,h], img_size=(640,480))
bbox.to_coco() / .to_yolo(img_size) / .to_voc(); bbox.point_inside(x, y)
model.OBBox(...)                                            # oriented bbox (4 corners)
model.Pose(x=[...], y=[...])                                # 2D keypoints
model.Pose3D(x=[...], y=[...], visible=[...])
model.Segment(title="road", x=[...], y=[...])               # segmentation polygon
```

**Column references:**

```python
dc.C("file.size")                  # top-level
dc.C("det.bbox.x1")                # nested
dc.C("file.path").glob("*.jpg")
chain.column("price")              # typed column for column-column arithmetic
```

---

## Section 7 — func Module

All run inside Data Memory (no Python, no deserialization). `C` = `dc.C`, `func` = `dc.func`.

```python
# Distance (vector search)
func.cosine_distance(C("emb"), reference); func.euclidean_distance(...); func.l2_distance(...)

# Aggregate (in group_by)
func.count(); func.sum(C("file.size")); func.avg(C("score")); func.min/max(C("val"))
func.collect(C("label")); func.first(C("path"))

# Path
func.path.file_ext(C("file.path"))    # → "jpg"
func.path.file_stem(C("file.path"))   # → "image01"
func.path.name(C("file.path"))        # → "image01.jpg"
func.path.parent(C("file.path"))      # → "folder/subfolder"

# Conditional
func.case((C("score") > 0.9, "high"), (C("score") > 0.5, "medium"), else_="low")
func.ifelse(func.isnone(C("result")), "pending", "done")

# String
func.string.length(C("text")); func.string.split(C("path"), "/")

# Window (both partition_by and order_by required)
w = func.window(partition_by="category", order_by="created_at")
chain.mutate(rank=func.rank().over(w), row_num=func.row_number().over(w))

# Ranking (in group_by): func.rank(), func.dense_rank(), func.row_number()

# Hashing / sampling (ClickHouse only — not on local SQLite)
func.sip_hash_64(C("file.path")); func.int_hash_64(C("file.path"))
```

---

## Section 8 — Common Pipeline Templates

**Basic: read → filter → map → save**

```python
import datachain as dc

def compute_embedding(file: dc.File) -> list[float]:
    return model.encode(file.read()).tolist()

(
    dc.read_storage("s3://bucket/data/")
    .filter(dc.C("file.size") > 1000)
    .settings(parallel=4)
    .map(emb=compute_embedding)
    .save("embeddings")
)
```

**Heavy-init via `.setup()`:**

```python
def encode(file: dc.ImageFile, model, preprocess) -> list[float]:
    img = preprocess(file.read()).unsqueeze(0)
    return model.encode_image(img)[0].tolist()

m, _, p = open_clip.create_model_and_transforms("ViT-B-32", "laion2b_s34b_b79k")

(
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=4)
    .setup(model=lambda: m, preprocess=lambda: p)
    .map(emb=encode)
    .save("image_embeddings")
)
```

**Multi-stage pipeline (one script per stage; this snippet shows the dataset graph):**

```python
# Stage 1 — build_chunks.py
dc.read_storage("s3://docs/*.pdf").gen(chunk=split_pdf).save("chunks")

# Stage 2 — build_chunk_embeddings.py
(dc.read_dataset("chunks")
   .setup(model=lambda: load_embedding_model())
   .settings(parallel=4)
   .map(emb=embed_chunk)
   .save("chunk_embeddings"))

# Stage 3 — classify_chunks.py
(dc.read_dataset("chunk_embeddings")
   .setup(client=lambda: create_llm_client())
   .settings(parallel=8)
   .map(category=classify)
   .save("classified_chunks"))
```

**Generator (1 input → N outputs):**

```python
from typing import Iterator

class Chunk(BaseModel):
    text: str
    start_offset: int

def split_doc(file: dc.File) -> Iterator[Chunk]:
    for offset, text in chunk(file.read_text()):
        yield Chunk(text=text, start_offset=offset)

(dc.read_storage("s3://docs/").settings(parallel=4).gen(chunk=split_doc).save("chunks"))
```

**Merge sidecar metadata:**

```python
items = dc.read_storage("gs://bucket/data/")
meta = dc.read_json("gs://bucket/annotations.json", jmespath="items")
annotated = items.merge(meta, on="file.path", right_on="items.file_name")
```

**Multi-source merge (shared prefix, inline func, select_except at end):**

```python
annotations = dc.read_storage("gs://b/**/*.txt", type="text").gen(ann=parse_list)
xmls = dc.read_storage("gs://b/**/*.xml").settings(prefetch=128).map(xml=parse_xml)
images = dc.read_storage("gs://b/**/*.jpg", type="image")

(
    images
    .merge(annotations, on=dc.func.path.file_stem(dc.C("file.path")), right_on="ann.name")
    .merge(xmls, on=dc.func.path.file_stem(dc.C("file.path")),
                 right_on=dc.func.path.file_stem(dc.C("file.path")))
    .select_except("right_file", "ann.name")
    .save("annotated_items")
)
```

**Vector similarity search:**

```python
(
    dc.read_dataset("embeddings")
    .mutate(dist=dc.func.cosine_distance(dc.C("emb"), query_embedding))
    .order_by("dist").limit(10)
    .show()
)
```

**LLM extraction with structured Pydantic output:**

```python
class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

def analyze(file: dc.File, client) -> Analysis:
    resp = client.messages.create(model="claude-sonnet-4-6", ...)
    return Analysis.model_validate_json(resp.content[0].text)

(dc.read_storage("s3://docs/")
   .setup(client=lambda: anthropic.Anthropic())
   .settings(parallel=8)
   .map(result=analyze)
   .save("analyzed"))
```

**Metadata analytics (no Python needed):**

```python
(
    dc.read_storage("gs://bucket/")
    .filter(dc.C("file.size") > 0)
    .group_by(
        count=dc.func.count(),
        total=dc.func.sum(dc.C("file.size")),
        partition_by=dc.func.path.file_ext(dc.C("file.path")),
    )
    .order_by("total", descending=True)
    .show()
)
```

**Delta updates (incremental):**

```python
(
    dc.read_storage("s3://bucket/data/", update=True, delta=True)
    .map(result=process_file)
    .save("processed_data")
)
```

**In-memory data joined with storage files:**

```python
labels = dc.read_records([{"name": "a.jpg", "cls": "cat"}, ...])
items = dc.read_storage("s3://bucket/data/")
combined = items.merge(labels, on="file.name", right_on="labels.name")
```
