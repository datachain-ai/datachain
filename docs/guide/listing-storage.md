---
title: Listing Storage
---

# Listing Storage

`dc.list_storage()` lists the objects of a storage and gives them to Python as rows. It is not a chain, unlike the `read_*` functions: nothing is saved, and no query runs over the rows.

| | `dc.list_storage(uri)` | `dc.read_storage(uri)` |
|---|---|---|
| Returns | rows to iterate | a chain of `File` objects |
| Lists the storage | every call | once, then reuses the saved snapshot |
| Saves a dataset | only with `save()` | always, the snapshot |
| Use it for | totals, inventories, checks, a table of the objects | processing files: filter, map, join, save results |

Both take the same URIs: one or a list, with glob patterns (`*`, `**`, `?`, `{a,b}`, `{1..9}`) and `recursive=False` for the first level only. Always include a trailing slash in bucket and prefix URIs.

## Rows

```python
import datachain as dc

total = sum(f.size for f in dc.list_storage("s3://bucket/episodes/"))
images = dc.list_storage("s3://bucket/images/**/*.{jpg,png}").to_objects(type="image")
```

Rows are plain tuples with `File`'s columns (`f.path`, `f.size`, `f.etag`, `f.last_modified`, ...) and its path helpers (`f.name`, `f.parent`, `f.get_file_ext()`). Rows don't read files; `to_objects()` gives `File` objects that do.

Iterate a listing, or take it whole:

```python
rows = dc.list_storage(uri).to_list()
table = dc.list_storage(uri).to_arrow()  # pyarrow.Table; to_pandas() for a DataFrame
# ImageFile objects, which read the images
images = dc.list_storage(uri).to_objects(type="image")
chain = dc.list_storage(uri).save()  # saved for read_storage(uri) to reuse
```

Each use lists the storage again: iterating the same listing twice lists it twice. A listing reads only what its URI selects: with `recursive=False`, or a pattern with a fixed start such as `s3://bucket/logs/2024-*`, it reads that level or those keys, not the whole prefix (S3 and GCS). On S3, GCS and Azure, iterating holds a few pages at a time however large the storage is, while a local folder is read whole; `to_list()`, `to_arrow()`, `to_pandas()` and `to_objects()` hold every row.

## Versioned Buckets

On a bucket with versioning enabled (S3, GCS, Azure), `list_storage()` and `read_storage()` give the current files only: old versions are not included, so counts and sums are over current files. `versions` decides only whether the rows carry version ids:

- `versions=True`, the default: each file has its `version` id. The listing asks the storage for versions, so it is slower, and `save()` keeps old versions in the saved listing too.
- `versions=False`: no version ids; a faster listing.

```python
total = sum(
    f.size for f in dc.list_storage("s3://versioned-bucket/data/", versions=False)
)
```

`read_storage()` reuses a listing from `save()` made with the same `versions`, on S3, GCS and Azure even when the bucket has no versioning. Local files, Hugging Face and HTTP storages have one kind of listing, and `versions` changes nothing there.

## Credentials

`dc.list_storage()` takes `anon` and `client_config` as `read_storage()` does. With `anon=None`, the default, a bucket that needs no credentials is detected and read anonymously. Files from `to_objects()` read with the credentials of their listing:

```python
minio = {
    "endpoint_url": "https://minio.example.com",
    "key": "access-key",
    "secret": "secret-key",
}
lst = dc.list_storage("s3://my-bucket/data/", client_config=minio)
for img in lst.to_objects(type="image"):
    print(img.path, img.read().size)
```

## Sessions

Listing a storage and reading its files do not use a session, so a session matters only for `save()`. Without one, `save()` uses the default session. To save in a session of your own, open it with `with`, or pass it as `session=`, which also brings its credentials; `client_config`, if given, replaces them:

```python
with dc.Session("ingest", client_config=minio):
    chain = dc.list_storage("s3://my-bucket/data/").save()  # saved in "ingest"
    same = dc.read_storage("s3://my-bucket/data/")  # reuses that snapshot
```

Both functions distinguish `None` from `{}`: omit `client_config` (or pass `None`) to inherit the session's configuration. Passing `{}` replaces it with empty client options.
