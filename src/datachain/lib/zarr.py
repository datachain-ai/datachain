"""Generic Zarr support for DataChain.

This module provides the generic Zarr "core": a :class:`ZarrStore` data model
that points at a Zarr store root and lets you inspect its arrays and read array
data.  Higher-level conventions (xarray dimensions/coordinates, OME-Zarr
multiscale preview, ...) are intentionally *not* handled here.
"""

import posixpath
from collections.abc import Iterator
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.lib.file import File

try:
    import zarr
except ImportError:
    zarr = None  # type: ignore[assignment]


def _require_zarr() -> Any:
    """Return the ``zarr`` module, raising a clear error if it is missing.

    Zarr is an optional dependency, so importing this module must not require
    it; the hard failure is deferred to the moment Zarr functionality is used.
    """
    if zarr is None:
        raise ImportError(
            "Missing dependencies for Zarr support.\n"
            "Zarr requires Python >= 3.11.  On a supported Python, run:\n\n"
            "  pip install 'datachain[zarr]'\n"
        )
    return zarr


# Names of the metadata objects that mark a Zarr store/array root.
#   - ``zarr.json``  : Zarr v3 group or array metadata
#   - ``.zgroup``    : Zarr v2 group metadata
#   - ``.zarray``    : Zarr v2 array metadata (array-only store)
ZARR_ROOT_MARKERS = frozenset({"zarr.json", ".zgroup", ".zarray"})
ZARR_SUFFIX = ".zarr"


class ZarrInfo(DataModel):
    """Summary metadata for a Zarr store."""

    zarr_format: int | None = Field(default=None)
    arrays: list[str] = Field(default_factory=list)
    attrs: dict = Field(default_factory=dict)


class ZarrStore(DataModel):
    """A Zarr store root.

    Unlike :class:`~datachain.lib.file.File`, a store is a *tree* of objects
    (metadata and chunks) rather than a single byte stream, so it is modeled as
    a plain ``DataModel``.  The nested ``file`` points at the store root prefix
    and carries the storage credentials/catalog needed to read the store.
    """

    file: File

    @property
    def source(self) -> str:
        return self.file.source

    @property
    def path(self) -> str:
        return self.file.path

    def _open(self, mode: Literal["r", "r+", "a", "w", "w-"] = "r") -> Any:
        zarr = _require_zarr()
        f = self.file
        url = f.get_fs_path()
        storage_options = None
        if f.source and not f.source.startswith("file://"):
            catalog = getattr(f, "_catalog", None)
            storage_options = getattr(catalog, "client_config", None) or None
        if storage_options:
            return zarr.open(url, mode=mode, storage_options=storage_options)
        return zarr.open(url, mode=mode)

    def get_info(self) -> ZarrInfo:
        """Return summary metadata for the store."""
        node = self._open()
        return ZarrInfo(
            zarr_format=int(node.metadata.zarr_format),
            arrays=[a.path for a in self._arrays(node)],
            attrs=dict(node.attrs),
        )

    def get_arrays(self) -> Iterator["ZarrArray"]:
        """Yield every array in the store (recursively)."""
        yield from self._arrays(self._open())

    def _arrays(self, node: Any) -> Iterator["ZarrArray"]:
        if isinstance(node, zarr.Array):
            yield self._to_array(node, "")
            return
        yield from self._walk_arrays(node, "")

    def _walk_arrays(self, group: Any, prefix: str) -> Iterator["ZarrArray"]:
        seen: set[str] = set()
        for name, child in group.members():
            # Some store backends (e.g. ``HfFileSystem``) list each member more
            # than once; dedup by name so an array is yielded only once and we
            # don't re-descend into the same subgroup repeatedly.
            if name in seen:
                continue
            seen.add(name)
            child_path = f"{prefix}/{name}".lstrip("/")
            if isinstance(child, zarr.Array):
                yield self._to_array(child, child_path)
            else:
                yield from self._walk_arrays(child, child_path)

    def get_array(self, path: str = "") -> "ZarrArray":
        """Return a single array by its path within the store."""
        node = self._open()
        arr = node[path] if path else node
        if not isinstance(arr, zarr.Array):
            raise ValueError(  # noqa: TRY004
                f"'{path}' is not a Zarr array in store {self.path!r}"
            )
        return self._to_array(arr, path)

    def _to_array(self, arr: Any, path: str) -> "ZarrArray":
        chunks = list(arr.chunks) if arr.chunks is not None else None
        return ZarrArray(
            store=self,
            path=path,
            shape=list(arr.shape),
            chunks=chunks,
            dtype=str(arr.dtype),
            attrs=dict(arr.attrs),
        )


class ZarrArray(DataModel):
    """A single array within a :class:`ZarrStore`."""

    store: ZarrStore
    path: str = Field(default="")
    shape: list[int] = Field(default_factory=list)
    chunks: list[int] | None = Field(default=None)
    dtype: str = Field(default="")
    attrs: dict = Field(default_factory=dict)

    _hidden_fields: ClassVar[list[str]] = ["attrs"]

    def read(self, selection: Any = None) -> Any:
        """Read array data, optionally restricted to a NumPy-style selection."""
        node = self.store._open()
        arr = node[self.path] if self.path else node
        if selection is None:
            return arr[...]
        return arr[selection]

    def select(
        self,
        index: "int | list[int]",
        media: "Literal['image', 'audio', 'video'] | None" = None,
    ) -> "ZarrSelection":
        """Return a lazy :class:`ZarrSelection` pointing at an item in this array.

        ``index`` addresses the leading axes (e.g. ``i`` or ``[i]`` for one
        frame of an ``(N, H, W, C)`` array).  The region is read on demand via
        :meth:`ZarrSelection.read`, so the item can travel through a DataChain
        as a column without materializing its bytes.
        """
        idx = [index] if isinstance(index, int) else list(index)
        return ZarrSelection(array=self, index=idx, media=media)


class ZarrSelection(DataModel):
    """A lazy, bounded region inside a :class:`ZarrArray`.

    Points at a single item (or block) inside an array without reading it,
    analogous to how :class:`~datachain.lib.file.File` points at a byte stream.
    ``index`` addresses the leading axes; :meth:`read` materializes the region.
    """

    array: ZarrArray
    index: list[int] = Field(default_factory=list)
    media: Literal["image", "audio", "video"] | None = Field(default=None)

    def read(self) -> Any:
        """Read and return the selected region."""
        return self.array.read(tuple(self.index))

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
        # plain upper-cased fallback (mirrors VideoFrame.read_bytes without the
        # optional video dependency).
        ext = format if format.startswith(".") else f".{format}"
        pil_format = Image.registered_extensions().get(ext.lower(), format.upper())

        arr = np.asarray(self.read())
        if arr.dtype != np.uint8:
            arr = arr.astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format=pil_format)
        return buf.getvalue()


def _store_root_split(file: File) -> tuple[str, str]:
    """Return ``(source, path)`` for the store root containing ``file``.

    ``file`` is a store-root metadata marker.  The store root is the directory
    that contains it.  When the marker sits directly under the listing source
    (so its relative path has no parent), the store root *is* the source: the
    last path segment is peeled off for a clean ``(source, name)`` pair, falling
    back to the whole source with an empty path when it has no path segment
    (e.g. a bare bucket like ``s3://bucket``).
    """
    parent = posixpath.dirname(file.path)
    if parent:
        return file.source, parent
    parts = urlsplit(file.source)
    root = parts.path.rstrip("/")
    if "/" in root:
        head, _, name = root.rpartition("/")
        base = urlunsplit((parts.scheme, parts.netloc, head, "", ""))
        return base, name
    return file.source.rstrip("/"), ""


def file_to_store(file: File) -> ZarrStore:
    """Build a :class:`ZarrStore` from a matched store-root metadata file."""
    source, path = _store_root_split(file)
    root = File(source=source, path=path)
    # Carry over the stream so a store built in-process keeps its credentials
    # (DataChain otherwise re-injects the catalog when materializing rows).
    if file._catalog is not None:
        root._set_stream(
            file._catalog,
            caching_enabled=file._caching_enabled,
            download_cb=file._download_cb,
        )
    return ZarrStore(file=root)
