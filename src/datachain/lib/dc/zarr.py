import os
from functools import reduce
from typing import TYPE_CHECKING

from datachain.query import Session

if TYPE_CHECKING:
    from .datachain import DataChain


def read_zarr(
    path: str | os.PathLike[str] | list[str] | list[os.PathLike[str]],
    column: str = "zarr",
    session: Session | None = None,
    settings: dict | None = None,
    in_memory: bool = False,
    **kwargs,
) -> "DataChain":
    """Generate a chain with one row per Zarr store.

    Unlike :func:`read_storage`, which emits one row per physical object, this
    reader collapses every object under a store root into a single
    :class:`~datachain.lib.zarr.ZarrStore` row.  Stores are discovered using the
    standard ``*.zarr`` naming convention.

    Parameters:
        path: Storage path(s) or URI(s). Can be a local path or start with a
            storage prefix like `s3://`, `gs://`, `az://` or `file://`.
            Supports glob patterns.
        column: Created column name. Defaults to ``"zarr"``.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        in_memory: If True, use an in-memory database.

    Example:
        ```py
        import datachain as dc
        chain = dc.read_zarr("s3://mybucket/data/")
        for (store,) in chain.limit(1).to_iter("zarr"):
            print(store.get_info())
        ```
    """
    from datachain.lib.zarr import (
        ZARR_ROOT_MARKERS,
        ZARR_SUFFIX,
        ZarrStore,
        file_to_store,
    )

    from .datachain import C
    from .storage import read_storage

    # ``read_storage`` lists the whole prefix recursively (markers *and*
    # chunks); we then keep only the store-root markers.
    chain = read_storage(
        path,
        session=session,
        settings=settings,
        in_memory=in_memory,
        **kwargs,
    )

    # Keep only store-root markers (drop chunks and nested array/group
    # metadata). A marker sits either directly under a ``*.zarr`` directory
    # (discovery) or at the listing root when the path itself is a concrete
    # store directory.
    conditions = []
    for marker in sorted(ZARR_ROOT_MARKERS):
        conditions.append(C("file.path").glob(f"*{ZARR_SUFFIX}/{marker}"))
        conditions.append(C("file.path") == marker)
    chain = chain.filter(reduce(lambda a, b: a | b, conditions))

    return chain.map(file_to_store, output={column: ZarrStore})
