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
    **kwargs: Any,
) -> "DataChain":
    """Generate chain from Parquet files.

    Parameters:
        path: Local path or storage URI, e.g. `s3://`, `gs://`, `az://`, `hf://`.
            Supports `*`, `**`, `?`, `{a,b}`, and `{1..9}` glob patterns.
        partitioning: Any pyarrow partitioning schema.
        output: Result names and Python types as a dictionary, assigned by
            insertion order with exactly one field per column read. Use `columns`
            to select inputs by name. Omit to infer names and types. Parquet
            exported by DataChain restores its saved schema.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source file.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        columns: Input columns to read, in order; all if omitted. Accepts original
            or normalized names, e.g. "ZIP Code" or "zip_code". For Parquet exported
            by DataChain, omit this parameter and use `.select(...)` after reading.
        kwargs: Parameters to pass to `read_storage`. A directory or glob is
            listed once and the listing is reused for 4 hours, so files added or
            changed since are missed or fail to read. Pass `update=True` to list
            it again.

    Example:
        Read a file or files matching a pattern:
        ```py
        import datachain as dc

        chain = dc.read_parquet("s3://mybucket/file.parquet")
        chain = dc.read_parquet("s3://mybucket/dir/**/*.parquet")
        ```

        Select and rename columns from an ordinary Parquet file with `a, b, c`:
        ```py
        chain = dc.read_parquet(
            "s3://mybucket/file.parquet",
            columns=["c", "a"],
            output={"x": float, "y": float},
        )  # x receives c; y receives a
        ```
        `output={"c": float}` alone supplies one field for three columns and raises
        an error. Use `columns=["c"]` to select c.
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
