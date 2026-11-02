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

Default prefetch downloads whole files for every UDF that receives a `File`.
Custom partial readers can use `file.open(cache=False, block_size=65536)`.

Transfer size depends on the format: video inspection may read indexes and decode a
frame, and some formats require the entire file. HTTP(S) reads ignore `block_size`.

::: datachain.lib.file.File

::: datachain.lib.file.FileError

::: datachain.lib.file.TarVFile
