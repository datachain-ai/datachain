import os
import re
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from datachain.lib.dc.utils import DatasetPrepareError, OutputType
from datachain.lib.model_store import ModelStore
from datachain.query import Session

if TYPE_CHECKING:
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
    column_types: dict[str, Any] | None = None,
    parse_options: dict[str, str | bool | Callable] | None = None,
    columns: Sequence[str] | None = None,
    column_names: Sequence[str] | None = None,
    **kwargs,
) -> "DataChain":
    """Generate chain from csv files.

    Columns are named by the header, cleaned up to be valid signal names: "Unit
    Price (USD)" becomes `unit_price_usd`. Wherever a column is named below, either
    form works. Files without a header get the names `f0`, `f1`, and so on.

    Parameters:
        path: Storage URI with directory. URI must start with storage prefix such
            as `s3://`, `gs://`, `az://` or "file:///".
        delimiter: Character for delimiting columns. Takes precedence if also
            specified in `parse_options`. Defaults to ",".
        header: Whether the files include a header row.
        output: Columns to read, by name. A dictionary or a model also gives
            their types; with a list of names, types are inferred.
        column: Created column name.
        model_name: Generated model name.
        source: Whether to include info about the source file.
        nrows: Optional row limit.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        column_types: Types for some columns, by name: a Python type, a pyarrow
            type or a type name. They are used while parsing, so a column read as
            `str` keeps values like "02134" as written. The other columns' types are
            inferred.
        parse_options: Tells the parser how to process lines.
            See https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html
        columns: Names of the columns to read, in this order. Can't be combined
            with `output`.
        column_names: Names for all columns, in file order. With a header, they
            replace the header's names.

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

        Reading two columns, one of them as text:
        ```py
        import datachain as dc
        chain = dc.read_csv(
            "s3://mybucket/file.csv",
            columns=["zip_code", "customer_name"],
            column_types={"zip_code": str},
        )
        ```
    """
    from pandas._libs.parsers import STR_NA_VALUES
    from pyarrow.csv import ConvertOptions, ParseOptions, ReadOptions
    from pyarrow.dataset import CsvFileFormat

    from .storage import read_storage

    parse_options = parse_options or {}
    if "delimiter" not in parse_options:
        parse_options["delimiter"] = ","
    if delimiter:
        parse_options["delimiter"] = delimiter

    chain = read_storage(path, session=session, settings=settings, **kwargs)

    if column_names is not None:
        read_options = ReadOptions(
            column_names=list(column_names), skip_rows=1 if header else 0
        )
    elif header:
        read_options = ReadOptions()
    elif output and not _generated_names(_output_names(chain, output)):
        read_options = ReadOptions(column_names=_output_names(chain, output))
        warnings.warn(
            "Naming the columns of a CSV file without a header through output is "
            "deprecated and will stop working. Pass column_names= instead.",
            FutureWarning,
            stacklevel=2,
        )
    else:
        read_options = ReadOptions(autogenerate_column_names=True)

    parse_options = ParseOptions(**parse_options)
    convert_options = ConvertOptions(
        strings_can_be_null=True,
        null_values=STR_NA_VALUES,
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
        column_types=column_types,
        format=format,
        parse_options=parse_options,
    )


def _generated_names(names: list[str]) -> bool:
    """Whether these are names pyarrow generates for a file without a header."""
    return all(re.fullmatch(r"f\d+", name) for name in names)


def _output_names(chain: "DataChain", output: OutputType) -> list[str]:
    if isinstance(output, Sequence):
        return list(output)
    if isinstance(output, dict):
        return list(output.keys())
    if (fr := ModelStore.to_pydantic(output)) is not None:
        return list(fr.model_fields.keys())
    msg = f"error parsing csv - incompatible output type {type(output)}"
    raise DatasetPrepareError(chain.name, msg)
