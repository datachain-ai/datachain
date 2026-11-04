from collections.abc import Iterator
from typing import TYPE_CHECKING

from pydantic import BaseModel

from datachain.lib.convert.values_to_tuples import values_to_tuples
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.dc.records import create_records_dataset
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
    if column:
        model = dict_to_data_model(column, output)  # type: ignore[arg-type]
        rows = tuples if len(fr_map) > 1 else [(v,) for v in tuples]
        # Each row becomes the column's model, so it is flattened by its schema.
        tuples = [_model_row(model, r) for r in rows]
        output = {column: model}

    def _func_fr() -> Iterator[tuple_type]:  # type: ignore[valid-type]
        yield from tuples

    _func_fr.__name__ = "read_values"

    # Seed for .gen() iteration. content_hash=None because hash_callable
    # doesn't capture _func_fr's closure (which holds `tuples`) — auto-hashing
    # the seed would collide every read_values() call saved under the same
    # name. Drop this once hash_callable becomes closure-aware.
    chain = create_records_dataset(
        [{"seed": 0}],
        schema={"seed": int},
        content_hash=None,
        session=session,
        settings=settings,
        in_memory=in_memory,
    )
    return chain.gen(_func_fr, output=output)


def _model_row(model: type[BaseModel], values: tuple) -> BaseModel:
    # Setting fields one by one skips validation, like rows without column, and
    # allows a signal named "cls", which model_construct(**fields) rejects.
    row = model.model_construct()
    for name, value in zip(model.model_fields, values, strict=True):
        setattr(row, name, value)
    return row
