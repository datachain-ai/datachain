import os
from collections.abc import Sequence
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
    columns: Sequence[str] | None = None,
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
        output: Result column names and Python types as an ordered dictionary.
            For ordinary Parquet files, fields are assigned by position in
            dictionary insertion order. Names do not select or match input
            columns; use `columns` to select them. There must be exactly one
            output field per column read (all columns by default, or those
            selected by `columns`). With no output, names and types are inferred.
            Parquet files exported by DataChain restore their saved schema.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source file.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        columns: Input columns to read, in this order; reads all if omitted.
            Accepts original names or DataChain-normalized names, e.g. "ZIP Code"
            or "zip_code". Unknown or repeated columns raise an error. Unsupported
            for Parquet files exported by DataChain; read them without `columns`
            and use `.select(...)` on the resulting chain.

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

        Selecting and renaming columns from an ordinary Parquet file with
        columns `a, b, c`:
        ```py
        chain = dc.read_parquet("s3://mybucket/file.parquet", columns=["c", "a"])
        chain = dc.read_parquet(
            "s3://mybucket/file.parquet",
            columns=["c", "a"],
            output={"x": float, "y": float},
        )  # x receives c; y receives a
        ```

        `output` alone renames every column by position. For columns `a, b, c`,
        the following puts a's values in c and c's values in a:
        ```py
        chain = dc.read_parquet(
            "s3://mybucket/file.parquet",
            output={"c": float, "b": float, "a": float},
        )
        ```
        `output={"c": float}` alone raises an error when the chain is executed:
        it supplies one field for three columns. Use `columns=["c"]` to read c.
    """
    from .storage import read_storage

    chain = read_storage(path, session=session, settings=settings, **kwargs)
    return chain.parse_tabular(
        output=output,
        column=column,
        model_name=model_name,
        source=source,
        columns=columns,
        format="parquet",
        partitioning=partitioning,
    )
