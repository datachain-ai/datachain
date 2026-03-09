import os
from typing import TYPE_CHECKING, Any

from datachain.lib.data_model import DataType
from datachain.query import Session

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def read_parquet(
    path: str | os.PathLike[str] | list[str] | list[os.PathLike[str]],
    partitioning: Any = "hive",
    output: dict[str, DataType] | None = None,
    column: str = "",
    model_name: str = "",
    source: bool = True,
    session: Session | None = None,
    settings: dict | None = None,
    **kwargs,
) -> "DataChain":
    """Generate chain from parquet files.

    Parameters:
        path: Storage path(s) or URI(s). Can be a local path or start with a
            storage prefix like `s3://`, `gs://`, `az://`, `hf://` or "file:///".
            Supports glob patterns:
              - `*` : wildcard
              - `**` : recursive wildcard
              - `?` : single character
              - `{a,b}` : brace expansion list
              - `{1..9}` : brace numeric or alphabetic range
        partitioning: Any pyarrow partitioning schema.
        output: Dictionary defining column names and their corresponding types.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source file.
        session: Session to use for the chain.
        settings: Settings to use for the chain.

    Example:
        Reading a single file:
        ```py
        import datachain as dc
        dc.read_parquet("s3://mybucket/file.parquet")
        ```

        All files from a directory:
        ```py
        dc.read_parquet("s3://mybucket/dir/")
        ```

        Only parquet files from a directory, and all it's subdirectories:
        ```py
        dc.read_parquet("s3://mybucket/dir/**/*.parquet")
        ```

        Using filename patterns - numeric, list, starting with zeros:
        ```py
        dc.read_parquet("s3://mybucket/202{1..4}/{yellow,green}-{01..12}.parquet")
        ```
    """
    from .storage import read_storage

    chain = read_storage(path, session=session, settings=settings, **kwargs)
    return chain.parse_tabular(
        output=output,
        column=column,
        model_name=model_name,
        source=source,
        format="parquet",
        partitioning=partitioning,
    )
