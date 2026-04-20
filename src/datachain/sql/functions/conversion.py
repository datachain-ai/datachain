from sqlalchemy import case
from sqlalchemy import cast as sa_cast
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import func as sa_func
from sqlalchemy.sql.functions import GenericFunction

from datachain.sql.types import String


class datetime_to_string(GenericFunction):  # noqa: N801
    type = String()
    package = "conversion"
    name = "datetime_to_string"
    inherit_cache = True


@compiles(datetime_to_string)
def compile_datetime_to_string(element, compiler, **kwargs):
    (value,) = element.clauses.clauses
    casted = sa_cast(value, String())
    has_zero_fraction = (
        sa_func.substr(casted, sa_func.length(casted) - 6, 7) == ".000000"
    )
    expr = case(
        (
            has_zero_fraction,
            sa_func.substr(casted, 1, sa_func.length(casted) - 7),
        ),
        else_=casted,
    )
    return compiler.process(expr, **kwargs)
