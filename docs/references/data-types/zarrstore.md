# ZarrStore

`ZarrStore` is a [`DataModel`](index.md#datachain.lib.data_model.DataModel) that
points at a [Zarr](https://zarr.dev) store root and provides methods for
inspecting and reading its arrays.

Unlike [`File`](file.md), which represents a single byte stream, a Zarr store is
a *tree* of objects (metadata and chunks). `ZarrStore` rows are created when a
`DataChain` is initialized [from Zarr stores](../datachain.md#datachain.lib.dc.zarr.read_zarr),
which collapses every object under a store root into a single row:

```python
import datachain as dc

chain = dc.read_zarr("s3://bucket-name/data/")
for (store,) in chain.limit(1).to_iter("zarr"):
    print(store.get_info())
```

There are additional models for working with Zarr stores:

- [`ZarrInfo`](#datachain.lib.zarr.ZarrInfo) - summary metadata for a store
  (format, array paths, attributes).
- [`ZarrArray`](#datachain.lib.zarr.ZarrArray) - a single array within a store;
  exposes `shape`, `chunks`, `dtype`, and `attrs`, and reads data via `read()`
  or `select()`.
- [`ZarrSelection`](#datachain.lib.zarr.ZarrSelection) - a lazy, bounded region
  inside an array (e.g. one image frame) that can travel through a chain as a
  column and is materialized on demand via `read()` or rendered to image bytes
  via `read_bytes()`.

For a complete example of Zarr processing with DataChain, see
[Embedding Zarr image frames](https://github.com/datachain-ai/datachain/blob/main/examples/multimodal/zarr-robot-frames.py) -
a pipeline that reads RGB camera frames from a directory of Zarr stores and
encodes them with OpenCLIP.

::: datachain.lib.zarr.ZarrStore

::: datachain.lib.zarr.ZarrArray

::: datachain.lib.zarr.ZarrSelection

::: datachain.lib.zarr.ZarrInfo
