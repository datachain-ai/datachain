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

    Parameters:
        path: Local path or storage URI, e.g. `s3://`, `gs://`, `az://`, `hf://`.
            Supports `*`, `**`, `?`, `{a,b}`, and `{1..9}` glob patterns.
        output: Result names and Python types as a dictionary, assigned by
            insertion order with exactly one field per column read. Use `columns`
            to select inputs by name. Omit to infer names and types.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source file. It is named
            "source", or the first free "cN_source" if that name is taken.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        columns: Input columns to read, in order; all if omitted. Accepts original
            or normalized names, e.g. "ZIP Code" or "zip_code". When omitted (along
            with `output` and `column`), DataChain automatically reads only the
            columns used by `select()`, `group_by()`, `count()`, or `gen()` (even
            after `filter()`, `order_by()`, or `mutate()`), and applies `filter()`
            conditions while reading so only the needed rows and columns are loaded
            (`filter()`, `select()`, and `order_by()` can appear in any order).
        kwargs: Parameters to pass to `read_storage`. A directory or glob is
            listed once and the listing is reused for 4 hours, so files added or
            changed since are missed or fail to read. Pass `update=True` to list
            it again.

    Example:
        Read a file or files matching a pattern:
        ```py
        import datachain as dc

        chain = dc.read_lance("s3://mybucket/file.lance")
        chain = dc.read_lance("s3://mybucket/dir/**/*.lance")
        ```

        Select and rename columns from a Lance file with `a, b, c`:
        ```py
        chain = dc.read_lance(
            "s3://mybucket/file.lance",
            columns=["c", "a"],
            output={"x": float, "y": float},
        )  # x receives c; y receives a
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
