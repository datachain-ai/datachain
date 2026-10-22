import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from datachain.lib.dc.utils import DatasetPrepareError, OutputType
from datachain.lib.model_store import ModelStore
from datachain.query import Session

if TYPE_CHECKING:
    from pyarrow import DataType as ArrowDataType

    from .datachain import DataChain


def read_csv(
    path: str | os.PathLike[str] | list[str] | list[os.PathLike[str]],
    delimiter: str | None = None,
    header: bool = True,
    output: OutputType = None,
    column: str = "",
    model_name: str = "",
    source: bool = True,
    nrows: int | None = None,
    session: Session | None = None,
    settings: dict | None = None,
    column_types: dict[str, "str | ArrowDataType"] | None = None,
    parse_options: dict[str, str | bool | Callable] | None = None,
    columns: Sequence[str] | None = None,
    **kwargs,
) -> "DataChain":
    """Generate chain from csv files.

    Parameters:
        path: Storage URI with directory. URI must start with storage prefix such
            as `s3://`, `gs://`, `az://` or "file:///".
        delimiter: Character for delimiting columns. Takes precedence if also
            specified in `parse_options`. Defaults to ",".
        header: Whether the files include a header row.
        output: Result column names and Python types, given as a dictionary or
            Pydantic model class. A list of names infers the types. Fields are
            assigned by position: dictionary insertion order, model field order,
            or list order. Names do not select or match input columns; use
            `columns` to select them. There must be exactly one output field per
            column read (all columns by default, or those selected by `columns`).
            With no output, names and types are inferred. With `header=False`
            and no `columns`, output is required and supplies the file's column
            names. To control CSV parsing types, use `column_types`.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source file.
        nrows: Optional row limit.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        column_types: CSV parsing types keyed by input column name, before
            normalization or renaming with `output`. Values are Arrow type names
            (e.g. "string") or Arrow data types. Overrides inference for those
            columns without selecting or renaming them. Use this to preserve
            values such as ZIP codes with leading zeros.
        parse_options: Tells the parser how to process lines.
            See https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html
        columns: Input columns to read, in this order; reads all if omitted.
            Accepts original names or DataChain-normalized names, e.g. "ZIP Code"
            or "zip_code". Unknown or repeated columns raise an error. With
            `header=False`, select using generated names "f0", "f1", etc., and
            optionally rename them with `output`.

    Example:
        Reading a csv file:
        ```py
        import datachain as dc
        chain = dc.read_csv("s3://mybucket/file.csv")
        ```

        Reading csv files from a directory as a combined dataset:
        ```py
        import datachain as dc
        chain = dc.read_csv("s3://mybucket/dir")
        ```

        Selecting and renaming columns from a CSV with header `a,b,c`:
        ```py
        chain = dc.read_csv("s3://mybucket/file.csv", columns=["c", "a"])
        chain = dc.read_csv(
            "s3://mybucket/file.csv",
            columns=["c", "a"],
            output={"x": float, "y": float},
        )  # x receives c; y receives a
        ```

        `output` alone renames every column by position. For header `a,b,c`,
        the following puts a's values in c and c's values in a:
        ```py
        chain = dc.read_csv(
            "s3://mybucket/file.csv", output={"c": float, "b": float, "a": float}
        )
        ```
        `output={"c": float}` alone raises an error when the chain is executed:
        it supplies one field for three columns. Use `columns=["c"]` to read c.

        Preserving ZIP codes as strings while selecting and renaming a column:
        ```py
        chain = dc.read_csv(
            "s3://mybucket/customers.csv",
            columns=["zip_code"],
            column_types={"ZIP Code": "string"},
            output={"postal_code": str},
        )  # "02134" keeps its leading zero
        ```

        Selecting the third and first columns from a file without a header:
        ```py
        chain = dc.read_csv(
            "s3://mybucket/no-header.csv",
            header=False,
            columns=["f2", "f0"],
            output=["c", "a"],
        )
        ```
    """
    from pandas._libs.parsers import STR_NA_VALUES
    from pyarrow.csv import ConvertOptions, ParseOptions, ReadOptions
    from pyarrow.dataset import CsvFileFormat
    from pyarrow.lib import type_for_alias

    from .storage import read_storage

    parse_options = parse_options or {}
    if "delimiter" not in parse_options:
        parse_options["delimiter"] = ","
    if delimiter:
        parse_options["delimiter"] = delimiter

    if column_types:
        column_types = {
            name: type_for_alias(typ) if isinstance(typ, str) else typ
            for name, typ in column_types.items()
        }
    else:
        column_types = {}

    chain = read_storage(path, session=session, settings=settings, **kwargs)

    column_names = None
    # With columns=, a file without a header is read with generated names.
    if not header and columns is None:
        if not output:
            msg = "error parsing csv - provide output or columns if no header"
            raise DatasetPrepareError(chain.name, msg)
        if isinstance(output, Sequence):
            column_names = output  # type: ignore[assignment]
        elif isinstance(output, dict):
            column_names = list(output.keys())
        elif (fr := ModelStore.to_pydantic(output)) is not None:
            column_names = list(fr.model_fields.keys())
        else:
            msg = f"error parsing csv - incompatible output type {type(output)}"
            raise DatasetPrepareError(chain.name, msg)

    parse_options = ParseOptions(**parse_options)
    read_options = ReadOptions(
        column_names=column_names,
        autogenerate_column_names=not header and columns is not None,
    )
    convert_options = ConvertOptions(
        strings_can_be_null=True,
        null_values=STR_NA_VALUES,
        column_types=column_types,
    )
    format = CsvFileFormat(
        parse_options=parse_options,
        read_options=read_options,
        convert_options=convert_options,
    )
    return chain.parse_tabular(
        output=output,
        column=column,
        model_name=model_name,
        source=source,
        nrows=nrows,
        columns=columns,
        format=format,
        parse_options=parse_options,
    )
