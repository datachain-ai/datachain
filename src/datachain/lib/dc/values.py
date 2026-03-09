from collections.abc import Iterator
from typing import TYPE_CHECKING

from datachain.lib.convert.values_to_tuples import values_to_tuples
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.dc.records import read_records
from datachain.lib.dc.utils import OutputType
from datachain.query import Session

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def read_values(
    ds_name: str = "",
    session: Session | None = None,
    settings: dict | None = None,
    in_memory: bool = False,
    output: OutputType = None,
    column: str = "",
    **fr_map,
) -> "DataChain":
    """Generate chain from list of values.

    Example:
        ```py
        import datachain as dc
        dc.read_values(fib=[1, 2, 3, 5, 8])
        ```
    """
    tuple_type, output, tuples = values_to_tuples(ds_name, output, **fr_map)

    def _func_fr() -> Iterator[tuple_type]:  # type: ignore[valid-type]
        yield from tuples

    _func_fr.__name__ = "read_values"

    # Start with a single dummy record so .gen() has one row to iterate over.
    # The actual data comes from the generator function.
    chain = read_records(
        [{"seed": 0}],
        schema={"seed": int},
        session=session,
        settings=settings,
        in_memory=in_memory,
    )
    if column:
        output = {column: dict_to_data_model(column, output)}  # type: ignore[arg-type]
    return chain.gen(_func_fr, output=output)
