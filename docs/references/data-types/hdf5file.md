# Hdf5File

`Hdf5File` is a [`File`](file.md) subclass that points at an
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) file and provides methods for
inspecting its groups and datasets and reading dataset data.

Install the optional dependency with `pip install 'datachain[hdf5]'`.

An HDF5 file is a single byte stream, so `Hdf5File` rows are created by
[`read_storage`](../datachain.md#datachain.lib.dc.storage.read_storage) with
`type="hdf5"`. Data is read through the regular streaming file handle, so a
single dataset - or a single slice of one - is fetched without pulling the whole
file:

```python
import datachain as dc

chain = dc.read_storage("s3://bucket-name/trajectories/", type="hdf5")
for (file,) in chain.limit(1).to_iter("file"):
    print(file.get_info())
```

Paths follow the HDF5 convention and are absolute within the file (e.g.
`/robot/joint_positions`); the reader also accepts them without the leading
slash. A `File` obtained some other way can be converted with
`file.as_hdf5_file()`.

The models are not re-exported from the top-level `datachain` namespace, so that
`import datachain` never loads `h5py`. Import them directly when annotating a
UDF or building a model by hand:

```python
from datachain.lib.hdf5 import Hdf5Dataset, Hdf5File, Hdf5Selection
```

There are additional models for working with HDF5 files:

- [`Hdf5Info`](#datachain.lib.hdf5.Hdf5Info) - summary metadata for a file
  (attributes, dataset paths, group paths).
- [`Hdf5Dataset`](#datachain.lib.hdf5.Hdf5Dataset) - a single dataset within a
  file; exposes `shape`, `chunks`, `dtype`, and `attrs`, and reads data via
  `read()` or `select()`.
- [`Hdf5Selection`](#datachain.lib.hdf5.Hdf5Selection) - a lazy, bounded region
  inside a dataset (e.g. one image frame) that can travel through a chain as a
  column and is materialized on demand via `read()` or rendered to image bytes
  via `read_bytes()`.

Only the generic HDF5 group/dataset model is handled here. Conventions layered on
top of HDF5 - NetCDF4 dimensions and coordinates, LeRobot episode layouts - are
not interpreted, though such files still load as ordinary HDF5.

::: datachain.lib.hdf5.Hdf5File

::: datachain.lib.hdf5.Hdf5Dataset

::: datachain.lib.hdf5.Hdf5Selection

::: datachain.lib.hdf5.Hdf5Info
