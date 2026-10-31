# DataChain API reference (agent edition)

`import datachain as dc`; `from datachain import func, C`.

## Sources

| call | notes |
|---|---|
| `dc.list_storage(uri, versions=True, anon=None)` | Iterate rows of the objects (`f.path`, `f.size`, `f.etag`, `f.last_modified`, `f.name`, `f.parent`, `f.get_file_ext()`); nothing saved, each iteration lists again. `.to_list()`, `.to_objects(type=...)`, `.save()`. For one-off inventories and totals. |
| `dc.read_storage(uri, anon=None, type="binary", update=False, delta=False, recursive=True)` | `uri`: `s3://`, `gs://`, `az://`, local path; globs `**/*.jpg`; a list of URIs. `type`: `"image"`, `"video"`, `"audio"`, `"text"` gives typed `file` (ImageFile, VideoFile...). Listing is cached in the project DB; `update=True` re-lists. `anon=True` for public buckets. |
| `dc.read_dataset(name, version=None)` | a saved dataset |
| `dc.read_csv(path, delimiter=None, header=True, output=None, column="")`, `dc.read_parquet(path)`, `dc.read_json(path, format="json"/"jsonl")` | local or remote files, typed columns; `column="x"` nests columns under `x` |
| `dc.read_values(a=[...], b=[...])`, `dc.read_pandas(df)`, `dc.read_records([...], schema=...)` | in-memory data |
| `dc.datasets(column="dataset")` | chain of saved datasets (`dataset.name`, `dataset.version`) |

CLI: `datachain dataset ls --local`, `datachain dataset describe NAME[@VERSION] --json`, `datachain show <name> [--limit N] [--schema]`, `datachain ls <uri> [--anon]`,
`datachain du <uri> [--anon]`.

## File objects (`file` column)

`file.path`, `file.source` (bucket URI), `file.size`, `file.etag`, `file.version`, `file.last_modified`.
Methods inside UDFs: `file.read()` (bytes), `file.read_text()`, `file.open()` (file object, range reads when
prefetch=0), `file.get_local_path()` / `file.ensure_cached()`, `file.as_image_file()`. Full URI: `f"{file.source}/{file.path}"`
(there is no `get_uri()`).

- `dc.ImageFile.get_info() -> dc.Image(width, height, format)`; `.read()` returns a PIL image.
- `dc.VideoFile.get_info() -> dc.Video(width, height, fps, duration, frames, format, codec)` (display orientation,
  header bytes only when prefetch=0). `.get_frames(start=0, end=None, step=1)` yields `dc.VideoFrame(frame,
  timestamp)` with `.get_np()` (HxWx3 uint8, display orientation) and `.read_bytes("jpg")`.
  `.sample_frames(fps=1.0)` yields the first decoded frame at/after each t = 0, 1/fps, 2/fps... (VFR-safe, one
  sequential decode, `frame.timestamp` = actual PTS).
  `.get_frame(n)`, `.get_fragment(start_s, end_s)`, `.get_fragments(duration=...)`.
- `dc.AudioFile.get_info() -> dc.Audio(sample_rate, channels, duration, ...)`, `.get_fragment(start, end)`.

## Settings

`chain.settings(parallel=N, prefetch=K, cache=False, workers=None, batch_size=None, min_task_size=None)`

- `parallel`: processes for UDFs (`True` = all CPUs). Use it for any per-file Python work; for network-bound UDFs
  (header reads, small range reads, API calls) use many more processes than CPUs, e.g. `parallel=64, prefetch=0`.
- `prefetch`: input rows downloaded ahead per process; `batch_map` counts batches, not individual files.
  Use `prefetch=0` for header-only or custom partial reads. For full decoding, start with a small positive
  value such as 2, accounting for batch size and worker count when estimating memory and disk usage.
- `cache=True` keeps downloaded files on disk for reuse across runs.

## UDFs

- `chain.batch_map(new_col=fn, batch=N)`: with `batch=N` the UDF receives lists of N rows (`file: list[dc.File]`) and
  returns a list of N outputs (`-> list[T]`); use it for model inference.
- `chain.map(new_col=fn)` 1:1. `fn` params are column names (`file`, `info.width` -> `info`), return annotation
  gives the type: `-> float`, `-> list[float]`, `-> dc.Video`, `-> MyModel` (pydantic BaseModel), or `output=`.
- `chain.gen(new_col=fn)` 1:N, `fn` yields; annotate `-> Iterator[Model]`. Output rows contain only the yielded
  columns. Keep a top-level `file` output when using file delta; also retain source and timestamp for frame results.
- `chain.agg(new_col=fn, partition_by="col")` N:M per group; `fn` receives lists.
- Class UDFs: subclass `dc.Mapper` / `dc.BatchMapper` / `dc.Generator` / `dc.Aggregator` with `setup()` (once per process),
  `process(...)`, `teardown()`. Use `.map(emb=Embed())` for a scalar `Mapper`, or `.batch_map(emb=Embed(), batch=32)` for a `BatchMapper`.

## Query (SQL, no Python per row)

`filter(C("info.width") > 400, C("file.path").glob("*.jpg"))`, combine with `&`, `|`, `~`.
`mutate(new=expr)`, `select("a", "b.c")`, `select_except(...)`, `order_by("x", descending=True)`, `limit(n)`,
`distinct("col")`, `group_by(n=func.count(), s=func.sum("x"), partition_by=["a", "b"])`,
  aggregates: `count`, `sum`, `avg`, `min`, `max`, `any_value`, `collect`, `concat` (no `count_distinct`: `distinct()` then `count()`),
`merge(right, on="a", right_on="b", inner=True)`, `union(other)`, `subtract(other, on=...)`,
`diff(other, on=..., compare=..., status_col="status")`, `file_diff(other)`,
`similarity_search("emb", vector, k=10, score_column="dist", metric="cosine")`.

`func`: `count, sum, avg, min, max, collect, any_value, concat, first`; `func.path.{parent,name,file_stem,file_ext}`;
`func.string.{split,length,replace,regexp_replace,byte_hamming_distance}`; `func.array.{cosine_distance,
euclidean_distance,length,contains}`; `func.cosine_distance`; `func.case(...)`, `func.ifelse(cond, a, b)`,
`func.isnone(col)`; `func.greatest/least`; window functions `func.row_number().over(partition_by=..., order_by=...)`.
Scalars: `chain.count()`, `chain.sum("x")`, `chain.avg("x")`.

## Persist, export

`chain.save("name")` creates a new version (`name@1.0.0`, `1.0.1`, ...) with lineage. `chain.persist()` caches
an intermediate result in-session. Export: `to_pandas(flatten=True)` (flat `file.path` columns; the default is a MultiIndex), `to_parquet(path)`, `to_csv(path)`,
`to_json(path)`, `to_list(cols...)`, `to_iter(cols...)`, `to_records()`, `to_storage(dir)` (copy files),
`to_pytorch(transform=...)` (iterable dataset streaming from storage).

## Delta (incremental refresh)

`read_storage(uri, update=True, delta=True)` (or `read_dataset(name, delta=True)`) + UDFs + `.save(name)`:
on re-run only added and modified source files go through the UDFs. Identity is `file.path`; modification is
detected by `file.etag` / `file.version`. With the replacement/deletion fix, the saved version equals a full recompute: rows of modified files are
replaced, rows of deleted files removed.

## Checkpoints (resume)

Re-running a script that crashed or was killed reuses UDF results that were already written; only unfinished
rows are computed. Same script, same chain; edits inside the UDF body are allowed (bug fixes), changes to the
chain structure (filters, outputs) start over.

Delete a dataset with all its versions: `datachain dataset rm <name> --force` (without `--force` or
`--version N` it fails with "Missing dataset version").
