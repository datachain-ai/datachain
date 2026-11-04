from typing import TYPE_CHECKING

from datachain.query import Session

from .values import read_values

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def read_pandas(  # type: ignore[override]
    df: "pd.DataFrame",
    *,
    session: Session | None = None,
    settings: dict | None = None,
    in_memory: bool = False,
    column: str = "",
) -> "DataChain":
    """Generate chain from pandas data-frame.

    Example:
        ```py
        import pandas as pd
        import datachain as dc

        df = pd.DataFrame({"fib": [1, 2, 3, 5, 8]})
        dc.read_pandas(df)
        ```
    """
    from .utils import DatasetPrepareError

    def get_col_name(col):
        if isinstance(col, tuple):
            # Join tuple elements with underscore for MultiIndex columns
            return "_".join(map(str, col)).lower()
        # Handle regular string column names
        return str(col).lower()

    fr_map = {get_col_name(col): _column_values(df[col]) for col in df.columns}

    for c in fr_map:
        if not c.isidentifier():
            raise DatasetPrepareError(
                "",
                f"import from pandas error - '{c}' cannot be a column name",
            )

    return read_values(
        session=session,
        settings=settings,
        column=column,
        in_memory=in_memory,
        **fr_map,
    )


def _column_values(series: "pd.Series") -> list:
    import pandas as pd

    values = series.tolist()
    if series.dtype.kind == "f":
        # Float NaN is a value, so only NA becomes None.
        return [None if v is pd.NA else v for v in values]
    # Other columns mark missing values with NaN, NaT or NA.
    return [None if m else v for v, m in zip(values, series.isna(), strict=True)]
