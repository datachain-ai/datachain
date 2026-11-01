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

`get_info()` methods read the ranges needed for metadata with 64 KiB read-ahead,
using an existing cached copy when available. Set `prefetch=0` before the UDF
to avoid whole-file downloads:

```python
metadata = (
    dc.read_storage("s3://bucket/videos/", type="video")
    .settings(prefetch=0)
    .map(info=lambda file: file.get_info(), output=dc.Video)
)
```

The default prefetch is 2 for all file types, preserving download overlap
for UDFs that read content. `get_info()` uses `File.open(cache=False)` to avoid
populating the whole-file cache even with `cache=True`; this cannot undo a
prefetch download. Custom partial readers can use
`file.open(cache=False, block_size=65536)`.

Metadata reads are format-dependent: video inspection may seek to container
indexes and decode the first frame to determine display orientation. Some formats
require much more than a header, potentially the entire file. HTTP(S) reads have
backend-specific range behavior and ignore `block_size`.

S3 range reads reuse known size and version/ETag from the listing to avoid an
extra HEAD request. Each range is pinned to the listed version and/or checked
with `If-Match`; changed unversioned objects fail rather than mixing versions.
Whole-file cache downloads retain their source validation.

GCS uses generation-pinned range reads for listed objects in standard buckets,
including objects larger than 5 MiB. Small pinned objects also avoid metadata
requests during cache downloads. Unpinned objects and zonal buckets retain the
backend's normal metadata validation and reader.

::: datachain.lib.file.File

::: datachain.lib.file.FileError

::: datachain.lib.file.TarVFile
