from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING, TypeVar, cast

import sqlalchemy
from sqlalchemy.sql.elements import BindParameter
from sqlalchemy.sql.visitors import iterate, replacement_traverse

from datachain.func.base import Function
from datachain.lib.data_model import DataModel, DataType
from datachain.lib.utils import DataChainParamsError
from datachain.query.schema import DEFAULT_DELIMITER, ColumnExpr
from datachain.utils import getenv_bool

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Concatenate

    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")

D = TypeVar("D", bound="DataChain")


def is_studio() -> bool:
    """Check if the runtime environment is Studio (not local)."""
    return getenv_bool("DATACHAIN_IS_STUDIO", default=False)


def is_local() -> bool:
    """Check if the runtime environment is local (not Studio)."""
    return not is_studio()


def resolve_columns(
    method: "Callable[Concatenate[D, P], D]",
) -> "Callable[Concatenate[D, P], D]":
    """Normalize positional column inputs against the current schema.

    This is mainly used by APIs like filter() and order_by() that accept a mix of
    string column names, DataChain Function objects, and already-built SQLAlchemy
    expressions.

    In particular:
    - string names are resolved to DB column names, including nested signals like
      file.path -> file__path
    - DataChain Function objects are converted to SQL expressions with the current
      schema
    - existing SQL expressions are type-enriched, and are only traversed when they
      still contain Function-valued bind parameters that must be converted to SQL
    """

    @wraps(method)
    def _inner(self: D, *args: "P.args", **kwargs: "P.kwargs") -> D:
        resolved_args: list[object] = []

        def resolve_expr(expr: ColumnExpr) -> ColumnExpr:
            plain_bind_params: list[BindParameter] = []
            func_bind_replacements: dict[int, ColumnExpr] = {}

            for element in iterate(expr):
                if not isinstance(element, BindParameter):
                    continue
                if isinstance(element.value, Function):
                    func_bind_replacements[id(element)] = element.value.get_column(
                        self.signals_schema
                    )
                else:
                    plain_bind_params.append(element)

            if func_bind_replacements:
                # replacement_traverse clones visited bind params by default.
                # Preserve ordinary binds and only replace the Function-valued ones
                # we collected from the original expression tree.
                expr = replacement_traverse(
                    expr,
                    {"stop_on": plain_bind_params},
                    lambda element, **_kwargs: func_bind_replacements.get(id(element)),
                )

            return self.signals_schema.enrich_expr_types(expr)

        for arg in args:
            if isinstance(arg, Function):
                resolved_args.append(arg.get_column(self.signals_schema))
            elif isinstance(arg, ColumnExpr):
                resolved_args.append(resolve_expr(arg))
            else:
                resolved_args.extend(
                    cast(
                        "list[str]",
                        self.signals_schema.resolve(cast("str", arg)).db_signals(),
                    )
                )

        return method(self, *resolved_args, **kwargs)  # type: ignore[arg-type,misc]

    return _inner


class DatasetPrepareError(DataChainParamsError):
    def __init__(self, name, msg, output=None):
        name = f" '{name}'" if name else ""
        output = f" output '{output}'" if output else ""
        super().__init__(f"Dataset{name}{output} processing prepare error: {msg}")


class DatasetFromValuesError(DataChainParamsError):
    def __init__(self, name, msg):
        name = f" '{name}'" if name else ""
        super().__init__(f"Dataset{name} from values error: {msg}")


MergeColType = str | Function | ColumnExpr


def _validate_merge_on(
    on: MergeColType | Sequence[MergeColType],
    ds: "DataChain",
) -> Sequence[MergeColType]:
    if isinstance(on, (str, ColumnExpr)):
        return [on]
    if isinstance(on, Function):
        return [on.get_column(ds.signals_schema, table=ds._query.table)]
    if isinstance(on, Sequence):
        return [
            c.get_column(ds.signals_schema, table=ds._query.table)
            if isinstance(c, Function)
            else c
            for c in on
        ]


def _get_merge_error_str(col: MergeColType) -> str:
    if isinstance(col, str):
        return col
    if isinstance(col, Function):
        return f"{col.name}()"
    if isinstance(col, sqlalchemy.Column):
        return col.name.replace(DEFAULT_DELIMITER, ".")
    if isinstance(col, ColumnExpr) and hasattr(col, "name"):
        return f"{col.name} expression"
    return str(col)


class DatasetMergeError(DataChainParamsError):
    def __init__(
        self,
        on: MergeColType | Sequence[MergeColType],
        right_on: MergeColType | Sequence[MergeColType] | None,
        msg: str,
    ):
        def _get_str(
            on: MergeColType | Sequence[MergeColType],
        ) -> str:
            if isinstance(on, (str, Function, ColumnExpr)) or not isinstance(
                on, Sequence
            ):
                return _get_merge_error_str(on)
            return ", ".join([_get_merge_error_str(col) for col in on])

        on_str = _get_str(on)
        right_on_str = (
            ", right_on='" + _get_str(right_on) + "'" if right_on is not None else ""
        )
        super().__init__(f"Merge error on='{on_str}'{right_on_str}: {msg}")


OutputType = DataType | Sequence[str] | dict[str, DataType] | None


class Sys(DataModel):
    """Model for internal DataChain signals `id` and `rand`."""

    id: int
    rand: int
