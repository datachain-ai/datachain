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
    **kwargs,
) -> "DataChain":
    """Generate chain from csv files.

    Parameters:
        path: Storage URI with directory. URI must start with storage prefix such
            as `s3://`, `gs://`, `az://` or "file:///".
        delimiter: Character for delimiting columns. Takes precedence if also
            specified in `parse_options`. Defaults to ",".
        header: Whether the files include a header row.
        output: Dictionary or feature class defining column names and their
            corresponding types. List of column names is also accepted, in which
            case types will be inferred.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source file.
        nrows: Optional row limit.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        column_types: Dictionary of column names and their corresponding types.
            It is passed to CSV reader and for each column specified type auto
            inference is disabled.
        parse_options: Tells the parser how to process lines.
            See https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html

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
    if not header:
        if not output:
            msg = "error parsing csv - provide output if no header"
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
    read_options = ReadOptions(column_names=column_names)
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
        format=format,
        parse_options=parse_options,
    )
