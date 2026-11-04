from datetime import datetime
from typing import TYPE_CHECKING

from datachain.lib.convert.values_to_tuples import infer_type_from_sequence
from datachain.query import Session

from .values import read_values

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import ParamSpec

    from datachain.lib.data_model import DataType

    from .datachain import DataChain

    P = ParamSpec("P")


def read_pandas(  # type: ignore[override]
    df: "pd.DataFrame",
    name: str = "",
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

    columns = {get_col_name(col): df[col] for col in df.columns}

    for c in columns:
        if not c.isidentifier():
            raise DatasetPrepareError(
                name,
                f"import from pandas error - '{c}' cannot be a column name",
            )

    fr_map = {c: _column_values(s) for c, s in columns.items()}
    output = {c: _column_type(s, fr_map[c], c, name) for c, s in columns.items()}

    return read_values(
        name,
        session,
        settings=settings,
        column=column,
        in_memory=in_memory,
        output=output,
        **fr_map,
    )


def _column_values(series: "pd.Series") -> list:
    import pandas as pd

    # NaT and NA can't be stored, so they become None. Float NaN is kept.
    return [None if v is pd.NaT or v is pd.NA else v for v in series.tolist()]


def _column_type(
    series: "pd.Series", values: list, signal_name: str, ds_name: str
) -> "DataType":
    typ = _KIND_TYPES.get(series.dtype.kind)
    if typ is None:
        return infer_type_from_sequence(values, signal_name, ds_name)
    # Float NaN is a value, while NaT and NA are stored as None.
    return typ | None if None in values else typ  # type: ignore[return-value]


_KIND_TYPES = {"b": bool, "i": int, "u": int, "f": float, "M": datetime}
