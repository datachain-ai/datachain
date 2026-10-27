# File

`File` is a special [`DataModel`](index.md#datachain.lib.data_model.DataModel),
which is automatically generated when a `DataChain` is created from files,
such as in [`dc.read_storage`](../datachain.md#datachain.lib.dc.storage.read_storage):

```python
import datachain as dc

chain = dc.read_storage("gs://datachain-demo/dogs-and-cats")
print(chain.schema)
```

Output:

```
file: File@v1
    source: str
    path: str
    size: int
    version: str
    etag: str
    is_latest: bool
    last_modified: datetime
    location: Union[dict, list[dict], NoneType]
```

`File` classes include various metadata fields describing the underlying file,
along with methods to read and manipulate file contents.

### Metadata without downloading

With the default chain settings, `ImageFile`, `AudioFile`, and `VideoFile` inputs
to `map()` and `gen()` defer whole-file prefetch until content access. Their
`get_info()` methods stream the ranges needed for metadata, using 64 KiB
read-ahead, and use an existing cached copy when available. Ordinary content
reads and `get_local_path()` complete the deferred download on demand.

```python
metadata = (
    dc.read_storage("s3://bucket/videos/", type="video")
    .map(info=lambda file: file.get_info(), output=dc.Video)
)
```

An explicit positive `prefetch` setting (including `True`) opts into eager
whole-file downloads for media, preserving download overlap for content-heavy
UDFs. For metadata extraction on a chain with explicit prefetch, or on generic
`File` inputs converted to media inside a UDF, use
`.settings(prefetch=0, cache=False)` before the UDF.

`get_info()` uses `File.open(cache=False)` to avoid populating the whole-file
cache even with `cache=True`. This cannot undo an eager download already made
by explicit prefetch. `cache=False` still permits reads from existing cached
copies. Custom partial readers can use `file.open(cache=False, block_size=65536)`.

Metadata reads are format-dependent: video inspection may seek to container
indexes and decode the first frame to determine display orientation. Some formats
require much more than a header, potentially the entire file. HTTP(S) reads have
backend-specific range behavior and ignore `block_size`.

S3 range reads reuse known size and version/ETag from the listing to avoid an
extra HEAD request. Each range is pinned to the listed version and/or checked
with `If-Match`; changed unversioned objects fail rather than mixing versions.
Whole-file cache downloads retain their source validation.

::: datachain.lib.file.File

::: datachain.lib.file.FileError

::: datachain.lib.file.TarVFile
