import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from datachain.lib.data_model import DataType
from datachain.query import Session

if TYPE_CHECKING:
    from .datachain import DataChain


def read_lance(
    path: str | os.PathLike[str] | list[str] | list[os.PathLike[str]],
    output: dict[str, DataType] | None = None,
    column: str = "",
    model_name: str = "",
    source: bool = True,
    session: Session | None = None,
    settings: dict | None = None,
    columns: Sequence[str] | None = None,
    **kwargs: Any,
) -> "DataChain":
    """Generate chain from standalone Lance v2 data files (not Lance table directories).

    Supports column projection and filter pushdown like `read_parquet`.

    Parameters:
        path: Local path or storage URI (e.g. `s3://`, `gs://`, `az://`, `hf://`),
            or list of paths. Supports `*`, `**`, `?`, `{a,b}`, and `{1..9}` globs.
        output: Dictionary mapping result column names to Python types. Omit to
            infer from the file schema or restore a saved DataChain schema.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source file (`ArrowRow`).
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        columns: Input columns (or DataChain signal paths) to read, in order.
        kwargs: Parameters passed to `read_storage` (e.g. `update=True`).

    Example:
        ```py
        import datachain as dc

        chain = dc.read_lance("s3://mybucket/dir/**/*.lance")
        ```
    """
    from .storage import read_storage

    chain = read_storage(path, session=session, settings=settings, **kwargs)
    return chain.parse_tabular(
        output=output,
        column=column,
        model_name=model_name,
        source=source,
        columns=columns,
        format="lance",
    )
