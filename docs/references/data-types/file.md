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

### Read media metadata

Set `.settings(prefetch=0)` before a UDF that calls `get_info()` to avoid downloading entire files ahead of the UDF. Metadata reads reuse a cached copy when available:

```python
metadata = (
    dc.read_storage("s3://bucket/videos/", type="video")
    .settings(prefetch=0)
    .map(info=lambda file: file.get_info(), output=dc.Video)
)
```

The default `prefetch=2` downloads entire files for UDFs that receive a `File`. With `prefetch=0`, `get_info()` reads the ranges needed for the format: video inspection may read an index and decode a frame, and some image formats require the entire file.

::: datachain.lib.file.File

::: datachain.lib.file.FileError

::: datachain.lib.file.TarVFile
