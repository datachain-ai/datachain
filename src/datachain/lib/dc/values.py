from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from datachain.lib.convert.flatten import flatten_value
from datachain.lib.convert.values_to_tuples import values_to_tuples
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.dc.records import create_records_dataset
from datachain.lib.dc.utils import OutputType
from datachain.query import Session

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from datachain.lib.data_model import DataType

    from .datachain import DataChain

    P = ParamSpec("P")


def read_values(
    *,
    session: Session | None = None,
    settings: dict | None = None,
    in_memory: bool = False,
    output: OutputType = None,
    column: str = "",
    **fr_map: Sequence,
) -> "DataChain":
    """Generate chain from lists of values, one list per signal.

    All values are held in memory. For large data, use `read_records()`.

    Parameters:
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        in_memory: If True, creates an in-memory session. Defaults to False.
        output: Signal types as a dictionary of names to types, covering every
            signal. Omit to infer each type from the values: a `None` among them
            makes the type optional, e.g. `float | None`.
        column: Created column name. Signals are nested under it, e.g. `fib`
            becomes `row.fib` with `column="row"`.
        fr_map: Signals as keyword arguments, each a list of values. All lists
            must have the same length. Values can be str, int, float, bool, bytes,
            datetime, lists, dicts, DataModel objects or `None`.

    Example:
        Create a chain with one signal:
        ```py
        import datachain as dc

        chain = dc.read_values(fib=[1, 2, 3, 5, 8])
        ```

        Create several signals, with one value per row in each list:
        ```py
        chain = dc.read_values(name=["cat.jpg", "dog.jpg"], size=[1024, 2048])
        ```

        `None` marks a missing value:
        ```py
        chain = dc.read_values(score=[0.5, None])  # score: float | None
        ```

        Declare the types when the values can't show them, e.g. all missing:
        ```py
        from datetime import datetime

        chain = dc.read_values(
            id=[1, 2],
            seen=[None, None],
            output={"id": int, "seen": datetime | None},
        )
        ```

        Store DataModel objects, with each field in its own column:
        ```py
        class Point(dc.DataModel):
            x: int
            y: int

        chain = dc.read_values(point=[Point(x=1, y=2), Point(x=3, y=4)])
        chain.to_values("point.x")  # [1, 3]
        ```

        Nest all signals under one column:
        ```py
        chain = dc.read_values(fib=[1, 2, 3], odd=[True, False, True], column="row")
        chain.to_values("row.fib")  # [1, 2, 3]
        ```
    """
    tuple_type, output, tuples = values_to_tuples(output, **fr_map)
    if column:
        rows = tuples if len(fr_map) > 1 else [(v,) for v in tuples]
        # The UDF stores tuples as they are, so rows are flattened here.
        tuples = [_flat_row(r, output) for r in rows]  # type: ignore[arg-type]
        output = {column: dict_to_data_model(column, output)}  # type: ignore[arg-type]

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


def _flat_row(values: tuple, output: dict[str, "DataType"]) -> tuple:
    return tuple(
        v
        for value, typ in zip(values, output.values(), strict=True)
        for v in flatten_value(value, typ)
    )
