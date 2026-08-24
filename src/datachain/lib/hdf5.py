"""HDF5 support for DataChain.

An HDF5 file is a single byte stream holding a tree of groups and datasets, so
:class:`Hdf5File` is a :class:`~datachain.lib.file.File` subclass and is read
through the regular streaming file handle: ``h5py`` needs only ``read``,
``seek`` and ``tell``, so a dataset (or a slice of one) is fetched without
pulling the whole file.  Higher-level conventions layered on top of HDF5
(NetCDF4 dimensions/coordinates, LeRobot episode layout, ...) are intentionally
*not* handled here.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, Literal

from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.lib.file import File

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]


def _require_h5py() -> Any:
    """Return the ``h5py`` module, raising a clear error if it is missing.

    h5py is an optional dependency, so importing this module must not require
    it; the hard failure is deferred to the moment HDF5 functionality is used.
    """
    if h5py is None:
        raise ImportError(
            "Missing dependencies for HDF5 support.\n"
            "To install run:\n\n  pip install 'datachain[hdf5]'\n"
        )
    return h5py


def _to_python(value: Any) -> Any:
    """Coerce an HDF5 attribute value to a JSON-serializable Python object.

    ``h5py`` returns attributes as NumPy objects (``ndarray``, ``int64``,
    ``bytes_``, ...), none of which survive the JSON column that holds them.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


def _attrs(node: Any) -> dict:
    return {name: _to_python(value) for name, value in node.attrs.items()}


class Hdf5Info(DataModel):
    """Summary metadata for an HDF5 file."""

    attrs: dict = Field(default_factory=dict)
    datasets: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)


class Hdf5File(File):
    """
    A data model for handling HDF5 files.

    This model inherits from the `File` model and provides additional
    functionality for inspecting an HDF5 file's groups and datasets and reading
    dataset data.

    Paths follow the HDF5 convention and are absolute within the file
    (e.g. ``/robot/joint_positions``); the reader also accepts them without the
    leading slash.
    """

    @contextmanager
    def _open_h5(self) -> Iterator[Any]:
        h5py = _require_h5py()
        with self.open("rb") as stream, h5py.File(stream, "r") as f:
            yield f

    def get_info(self) -> Hdf5Info:
        """Return summary metadata for the file."""
        h5py = _require_h5py()
        datasets: list[str] = []
        groups: list[str] = []

        def collect(_name: str, node: Any) -> None:
            target = datasets if isinstance(node, h5py.Dataset) else groups
            target.append(node.name)

        with self._open_h5() as f:
            f.visititems(collect)
            return Hdf5Info(attrs=_attrs(f), datasets=datasets, groups=groups)

    def get_datasets(self, group: str = "/") -> Iterator["Hdf5Dataset"]:
        """Yield every dataset under ``group`` (recursively)."""
        h5py = _require_h5py()
        found: list[Hdf5Dataset] = []

        def collect(_name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset):
                found.append(self._to_dataset(node))

        with self._open_h5() as f:
            node = f[group]
            if isinstance(node, h5py.Dataset):
                found.append(self._to_dataset(node))
            else:
                # visititems tracks visited objects, so a group reachable
                # through more than one hard link is walked only once.
                node.visititems(collect)

        yield from found

    def get_dataset(self, path: str) -> "Hdf5Dataset":
        """Return a single dataset by its path within the file."""
        h5py = _require_h5py()
        with self._open_h5() as f:
            node = f[path]
            if not isinstance(node, h5py.Dataset):
                raise ValueError(  # noqa: TRY004
                    f"'{path}' is not an HDF5 dataset in file {self.path!r}"
                )
            return self._to_dataset(node)

    def _to_dataset(self, node: Any) -> "Hdf5Dataset":
        chunks = list(node.chunks) if node.chunks is not None else None
        return Hdf5Dataset(
            file=self,
            path=node.name,
            shape=list(node.shape),
            chunks=chunks,
            dtype=str(node.dtype),
            attrs=_attrs(node),
        )


class Hdf5Dataset(DataModel):
    """A single dataset within an :class:`Hdf5File`.

    ``shape`` is the HDF5 shape as a list, so a scalar dataset has an empty
    ``shape`` while a zero-length one-dimensional dataset has ``[0]``.
    """

    file: Hdf5File
    path: str = Field(default="")
    shape: list[int] = Field(default_factory=list)
    chunks: list[int] | None = Field(default=None)
    dtype: str = Field(default="")
    attrs: dict = Field(default_factory=dict)

    _hidden_fields: ClassVar[list[str]] = ["attrs"]

    def read(self, selection: Any = None) -> Any:
        """Read dataset data, optionally restricted to a NumPy-style selection."""
        with self.file._open_h5() as f:
            node = f[self.path]
            if selection is None:
                return node[...]
            return node[selection]

    def select(
        self,
        index: "int | list[int]",
        media: "Literal['image', 'audio', 'video'] | None" = None,
    ) -> "Hdf5Selection":
        """Return a lazy :class:`Hdf5Selection` pointing at an item in this dataset.

        ``index`` addresses the leading axes (e.g. ``i`` or ``[i]`` for one
        frame of an ``(N, H, W, C)`` dataset).  The region is read on demand via
        :meth:`Hdf5Selection.read`, so the item can travel through a DataChain
        as a column without materializing its bytes.
        """
        idx = [index] if isinstance(index, int) else list(index)
        return Hdf5Selection(dataset=self, index=idx, media=media)


class Hdf5Selection(DataModel):
    """A lazy, bounded region inside an :class:`Hdf5Dataset`.

    Points at a single item (or block) inside a dataset without reading it,
    analogous to how :class:`~datachain.lib.file.File` points at a byte stream.
    ``index`` addresses the leading axes; :meth:`read` materializes the region.
    """

    dataset: Hdf5Dataset
    index: list[int] = Field(default_factory=list)
    media: Literal["image", "audio", "video"] | None = Field(default=None)

    def read(self) -> Any:
        """Read and return the selected region."""
        return self.dataset.read(tuple(self.index))

    def read_bytes(self, format: str = "PNG") -> bytes:
        """Render the selected region to encoded media bytes.

        Only ``media="image"`` is supported for now: the region is read and
        encoded with Pillow (e.g. PNG), so callers such as Studio can stream a
        preview without materializing the image into the row.
        """
        if self.media not in (None, "image"):
            raise ValueError(f"read_bytes() supports image media, not {self.media!r}")
        import io

        import numpy as np
        from PIL import Image

        # Normalize e.g. "jpg"/".png" to a registered Pillow format name, with a
        # plain upper-cased fallback.
        ext = format if format.startswith(".") else f".{format}"
        pil_format = Image.registered_extensions().get(ext.lower(), format.upper())

        arr = np.asarray(self.read())
        if arr.dtype != np.uint8:
            arr = arr.astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format=pil_format)
        return buf.getvalue()
