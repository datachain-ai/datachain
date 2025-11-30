import logging
import re
import sqlite3
import warnings
from collections.abc import Callable, Iterable
from contextlib import closing
from datetime import MAXYEAR, MINYEAR, datetime, timezone
from functools import cache
from types import MappingProxyType

import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.elements import literal
from sqlalchemy.sql.expression import case
from sqlalchemy.sql.functions import func

from datachain import json
from datachain.sql.functions import (
    aggregate,
    array,
    conditional,
    numeric,
    random,
    string,
)
from datachain.sql.functions import path as sql_path
from datachain.sql.selectable import Values, base_values_compiler
from datachain.sql.sqlite.types import (
    SQLiteTypeConverter,
    SQLiteTypeReadConverter,
    register_type_converters,
)
from datachain.sql.types import (
    DBDefaults,
    TypeDefaults,
    register_backend_types,
    register_db_defaults,
    register_type_defaults,
    register_type_read_converters,
)

logger = logging.getLogger("datachain")

_registered_function_creators: dict[str, Callable[[sqlite3.Connection], None]] = {}
registered_function_creators = MappingProxyType(_registered_function_creators)

_compiler_hooks: dict[str, Callable] = {}

sqlite_dialect = sqlite.dialect(paramstyle="named")

setup_is_complete: bool = False

slash = literal("/")
empty_str = literal("")
dot = literal(".")

MAX_INT64 = 2**64 - 1


def setup():
    global setup_is_complete  # noqa: PLW0603
    if setup_is_complete:
        return

    # sqlite 3.31.1 is the earliest version tested in CI
    if sqlite3.sqlite_version_info < (3, 31, 1):
        logger.warning(
            "Possible sqlite incompatibility. The earliest tested version of "
            "sqlite is 3.31.1 but you have %s",
            sqlite3.sqlite_version,
        )

    # We want to show tracebacks for user-defined functions
    sqlite3.enable_callback_tracebacks(True)
    sqlite3.register_adapter(datetime, adapt_datetime)
    sqlite3.register_converter("datetime", convert_datetime)

    register_type_converters()
    register_backend_types("sqlite", SQLiteTypeConverter())
    register_type_read_converters("sqlite", SQLiteTypeReadConverter())
    register_type_defaults("sqlite", TypeDefaults())
    register_db_defaults("sqlite", DBDefaults())

    compiles(sql_path.parent, "sqlite")(compile_path_parent)
    compiles(sql_path.name, "sqlite")(compile_path_name)
    compiles(sql_path.file_stem, "sqlite")(compile_path_file_stem)
    compiles(sql_path.file_ext, "sqlite")(compile_path_file_ext)
    compiles(array.length, "sqlite")(compile_array_length)
    compiles(array.contains, "sqlite")(compile_array_contains)
    compiles(array.slice, "sqlite")(compile_array_slice)
    compiles(array.join, "sqlite")(compile_array_join)
    compiles(array.get_element, "sqlite")(compile_array_get_element)
    compiles(string.length, "sqlite")(compile_string_length)
    compiles(string.split, "sqlite")(compile_string_split)
    compiles(string.regexp_replace, "sqlite")(compile_string_regexp_replace)
    compiles(string.replace, "sqlite")(compile_string_replace)
    compiles(string.byte_hamming_distance, "sqlite")(compile_byte_hamming_distance)
    compiles(conditional.greatest, "sqlite")(compile_greatest)
    compiles(conditional.least, "sqlite")(compile_least)
    compiles(Values, "sqlite")(compile_values)
    compiles(random.rand, "sqlite")(compile_rand)
    compiles(aggregate.avg, "sqlite")(compile_avg)
    compiles(aggregate.group_concat, "sqlite")(compile_group_concat)
    compiles(aggregate.any_value, "sqlite")(compile_any_value)
    compiles(aggregate.collect, "sqlite")(compile_collect)
    compiles(numeric.bit_and, "sqlite")(compile_bitwise_and)
    compiles(numeric.bit_or, "sqlite")(compile_bitwise_or)
    compiles(numeric.bit_xor, "sqlite")(compile_bitwise_xor)
    compiles(numeric.bit_rshift, "sqlite")(compile_bitwise_rshift)
    compiles(numeric.bit_lshift, "sqlite")(compile_bitwise_lshift)
    compiles(numeric.int_hash_64, "sqlite")(compile_int_hash_64)
    compiles(numeric.bit_hamming_distance, "sqlite")(compile_bit_hamming_distance)

    with closing(sqlite3.connect(":memory:")) as _usearch_conn:
        usearch_available = load_usearch_extension(_usearch_conn)

    if usearch_available:
        compiles(array.cosine_distance, "sqlite")(compile_cosine_distance_ext)
        compiles(array.euclidean_distance, "sqlite")(compile_euclidean_distance_ext)
    else:
        compiles(array.cosine_distance, "sqlite")(compile_cosine_distance)
        compiles(array.euclidean_distance, "sqlite")(compile_euclidean_distance)

    register_user_defined_sql_functions()
    setup_is_complete = True


def run_compiler_hook(name):
    try:
        hook = _compiler_hooks[name]
    except KeyError:
        return
    hook()


def functions_exist(
    names: Iterable[str], connection: sqlite3.Connection | None = None
) -> bool:
    """
    Returns True if all function names are defined for the given connection.
    """

    names = list(names)
    for n in names:
        if not isinstance(n, str):
            raise TypeError(
                "functions_exist(): names argument must contain str values. "
                f"Found value of type {type(n).__name__}: {n!r}"
            )

    close_connection = False
    if connection is None:
        connection = sqlite3.connect(":memory:")
        close_connection = True

    try:
        if not names:
            return True
        column1 = sa.column("column1", sa.String)
        func_name_query = column1.not_in(
            sa.select(sa.column("name", sa.String)).select_from(
                func.pragma_function_list()
            )
        )
        query = (
            sa.select(func.count() == 0)
            .select_from(sa.values(column1).data([(n,) for n in names]))
            .where(func_name_query)
        )
        comp = query.compile(dialect=sqlite_dialect)
        if comp.params:
            result = connection.execute(comp.string, comp.params)
        else:
            result = connection.execute(comp.string)
        return bool(result.fetchone()[0])
    finally:
        if close_connection:
            connection.close()


def create_user_defined_sql_functions(connection):
    for function_creator in registered_function_creators.values():
        function_creator(connection)


def missing_vector_function(name, exc):
    def unavailable_func(*args):
        raise ImportError(
            f"Missing dependencies for SQL vector function, {name}\n"
            "To install run:\n\n"
            "  pip install 'datachain[vector]'\n"
        ) from exc

    return unavailable_func


def sqlite_string_split(string: str, sep: str, maxsplit: int = -1) -> str:
    return json.dumps(string.split(sep, maxsplit), ensure_ascii=False)


def sqlite_int_hash_64(x: int) -> int:
    """IntHash64 implementation from ClickHouse."""
    x ^= 0x4CF2D2BAAE6DA887
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & MAX_INT64
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & MAX_INT64
    x ^= x >> 33
    # SQLite does not support unsigned 64-bit integers,
    # so we need to convert to signed 64-bit
    return x if x < 1 << 63 else (x & MAX_INT64) - (1 << 64)


def sqlite_bit_hamming_distance(a: int, b: int) -> int:
    """Calculate the Hamming distance between two integers."""
    diff = (a & MAX_INT64) ^ (b & MAX_INT64)
    return diff.bit_count()


def sqlite_byte_hamming_distance(a: str, b: str) -> int:
    """Calculate the Hamming distance between two strings."""
    diff = 0
    if len(a) < len(b):
        diff = len(b) - len(a)
        b = b[: len(a)]
    elif len(b) < len(a):
        diff = len(a) - len(b)
        a = a[: len(b)]
    return diff + sum(c1 != c2 for c1, c2 in zip(a, b, strict=False))


def register_user_defined_sql_functions() -> None:
    # Register optional functions if we have the necessary dependencies
    # and otherwise register functions that will raise an exception with
    # installation instructions
    try:
        from .vector import cosine_distance, euclidean_distance
    except ImportError as exc:
        # We want to throw an exception when trying to compile these
        # functions and also if the functions are called using raw SQL.
        cosine_distance = missing_vector_function("cosine_distance", exc)
        euclidean_distance = missing_vector_function("euclidean_distance", exc)
        _compiler_hooks["cosine_distance"] = cosine_distance
        _compiler_hooks["euclidean_distance"] = euclidean_distance

    def create_vector_functions(conn):
        conn.create_function("cosine_distance", 2, cosine_distance, deterministic=True)
        conn.create_function(
            "euclidean_distance", 2, euclidean_distance, deterministic=True
        )

    _registered_function_creators["vector_functions"] = create_vector_functions

    def create_numeric_functions(conn):
        conn.create_function("divide", 2, lambda a, b: a / b, deterministic=True)
        conn.create_function("bitwise_and", 2, lambda a, b: a & b, deterministic=True)
        conn.create_function("bitwise_or", 2, lambda a, b: a | b, deterministic=True)
        conn.create_function("bitwise_xor", 2, lambda a, b: a ^ b, deterministic=True)
        conn.create_function(
            "bitwise_rshift", 2, lambda a, b: a >> b, deterministic=True
        )
        conn.create_function(
            "bitwise_lshift", 2, lambda a, b: a << b, deterministic=True
        )
        conn.create_function("int_hash_64", 1, sqlite_int_hash_64, deterministic=True)
        conn.create_function(
            "bit_hamming_distance", 2, sqlite_bit_hamming_distance, deterministic=True
        )

    _registered_function_creators["numeric_functions"] = create_numeric_functions

    def sqlite_regexp_replace(string: str, pattern: str, replacement: str) -> str:
        return re.sub(pattern, replacement, string)

    def create_string_functions(conn):
        conn.create_function("split", 2, sqlite_string_split, deterministic=True)
        conn.create_function("split", 3, sqlite_string_split, deterministic=True)
        conn.create_function(
            "regexp_replace", 3, sqlite_regexp_replace, deterministic=True
        )
        conn.create_function(
            "byte_hamming_distance", 2, sqlite_byte_hamming_distance, deterministic=True
        )

    _registered_function_creators["string_functions"] = create_string_functions

    def create_array_functions(conn):
        conn.create_function(
            "json_array_get_element", 2, py_json_array_get_element, deterministic=True
        )
        conn.create_function(
            "json_array_slice", 2, py_json_array_slice, deterministic=True
        )
        conn.create_function(
            "json_array_slice", 3, py_json_array_slice, deterministic=True
        )
        conn.create_function(
            "json_array_join", 2, py_json_array_join, deterministic=True
        )

    _registered_function_creators["array_functions"] = create_array_functions

    has_json_extension = functions_exist(["json_array_length", "json_array_contains"])
    if not has_json_extension:

        def create_json_functions(conn):
            conn.create_function(
                "json_array_length", 1, py_json_array_length, deterministic=True
            )
            conn.create_function(
                "json_array_contains", 3, py_json_array_contains, deterministic=True
            )

        _registered_function_creators["json_functions"] = create_json_functions


def adapt_datetime(val: datetime) -> str:
    is_utc_check = val.tzinfo is timezone.utc
    tzname_check = val.tzname() == "UTC"
    combined_check = is_utc_check or tzname_check

    if not combined_check:
        try:
            val = val.astimezone(timezone.utc)
        except (OverflowError, ValueError, OSError):
            if val.year == MAXYEAR:
                val = datetime.max.replace(tzinfo=timezone.utc)
            elif val.year == MINYEAR:
                val = datetime.min.replace(tzinfo=timezone.utc)
            else:
                raise

    return val.replace(tzinfo=None).isoformat(" ")


def convert_datetime(val: bytes) -> datetime:
    return datetime.fromisoformat(val.decode()).replace(tzinfo=timezone.utc)


def path_parent(path):
    return func.rtrim(func.rtrim(path, func.replace(path, slash, empty_str)), slash)


def path_name(path):
    return func.ltrim(func.substr(path, func.length(path_parent(path)) + 1), slash)


def name_file_ext_length(name):
    expr = func.length(name) - func.length(
        func.rtrim(name, func.replace(name, dot, empty_str))
    )
    return case((func.instr(name, dot) == 0, 0), else_=expr)


def path_file_ext_length(path):
    name = path_name(path)
    return name_file_ext_length(name)


def path_file_stem(path):
    path_length = func.length(path)
    parent_length = func.length(path_parent(path))

    name_expr = func.rtrim(
        func.substr(
            path,
            1,
            path_length - name_file_ext_length(path),
        ),
        dot,
    )

    full_path_expr = func.ltrim(
        func.rtrim(
            func.substr(
                path,
                parent_length + 1,
                path_length - parent_length - path_file_ext_length(path),
            ),
            dot,
        ),
        slash,
    )

    return case((func.instr(path, slash) == 0, name_expr), else_=full_path_expr)


def path_file_ext(path):
    return func.substr(path, func.length(path) - path_file_ext_length(path) + 1)


def compile_path_parent(element, compiler, **kwargs):
    return compiler.process(path_parent(*element.clauses.clauses), **kwargs)


def compile_path_name(element, compiler, **kwargs):
    return compiler.process(path_name(*element.clauses.clauses), **kwargs)


def compile_path_file_stem(element, compiler, **kwargs):
    return compiler.process(path_file_stem(*element.clauses.clauses), **kwargs)


def compile_path_file_ext(element, compiler, **kwargs):
    return compiler.process(path_file_ext(*element.clauses.clauses), **kwargs)


def compile_cosine_distance_ext(element, compiler, **kwargs):
    run_compiler_hook("cosine_distance")
    return f"distance_cosine_f32({compiler.process(element.clauses, **kwargs)})"


def compile_cosine_distance(element, compiler, **kwargs):
    run_compiler_hook("cosine_distance")
    return f"cosine_distance({compiler.process(element.clauses, **kwargs)})"


def compile_euclidean_distance_ext(element, compiler, **kwargs):
    run_compiler_hook("euclidean_distance")
    return (
        f"sqrt(distance_sqeuclidean_f32({compiler.process(element.clauses, **kwargs)}))"
    )


def compile_euclidean_distance(element, compiler, **kwargs):
    run_compiler_hook("euclidean_distance")
    return f"euclidean_distance({compiler.process(element.clauses, **kwargs)})"


def compile_bitwise_and(element, compiler, **kwargs):
    return compiler.process(func.bitwise_and(*element.clauses.clauses), **kwargs)


def compile_bitwise_or(element, compiler, **kwargs):
    return compiler.process(func.bitwise_or(*element.clauses.clauses), **kwargs)


def compile_bitwise_xor(element, compiler, **kwargs):
    return compiler.process(func.bitwise_xor(*element.clauses.clauses), **kwargs)


def compile_bitwise_rshift(element, compiler, **kwargs):
    return compiler.process(func.bitwise_rshift(*element.clauses.clauses), **kwargs)


def compile_bitwise_lshift(element, compiler, **kwargs):
    return compiler.process(func.bitwise_lshift(*element.clauses.clauses), **kwargs)


def compile_int_hash_64(element, compiler, **kwargs):
    return compiler.process(func.int_hash_64(*element.clauses.clauses), **kwargs)


def compile_bit_hamming_distance(element, compiler, **kwargs):
    return compiler.process(
        func.bit_hamming_distance(*element.clauses.clauses), **kwargs
    )


def compile_byte_hamming_distance(element, compiler, **kwargs):
    return compiler.process(
        func.byte_hamming_distance(*element.clauses.clauses), **kwargs
    )


def py_json_array_length(arr):
    return len(json.loads(arr))


def py_json_array_contains(arr, value, is_json):
    if is_json:
        value = json.loads(value)
    return value in json.loads(arr)


def py_json_array_get_element(val, idx):
    arr = json.loads(val)
    try:
        return arr[idx]
    except IndexError:
        return None


def py_json_array_slice(val, offset: int, length: int | None = None):
    arr = json.loads(val)
    try:
        return json.dumps(
            list(arr[offset : offset + length] if length is not None else arr[offset:]),
            ensure_ascii=False,
        )
    except IndexError:
        return None


def py_json_array_join(val, sep: str):
    return sep.join(json.loads(val))


def compile_array_get_element(element, compiler, **kwargs):
    return compiler.process(
        func.json_array_get_element(*element.clauses.clauses), **kwargs
    )


def compile_array_length(element, compiler, **kwargs):
    return compiler.process(func.json_array_length(*element.clauses.clauses), **kwargs)


def compile_array_contains(element, compiler, **kwargs):
    return compiler.process(
        func.json_array_contains(*element.clauses.clauses), **kwargs
    )


def compile_array_slice(element, compiler, **kwargs):
    return compiler.process(func.json_array_slice(*element.clauses.clauses), **kwargs)


def compile_array_join(element, compiler, **kwargs):
    return compiler.process(func.json_array_join(*element.clauses.clauses), **kwargs)


def compile_string_length(element, compiler, **kwargs):
    return compiler.process(func.length(*element.clauses.clauses), **kwargs)


def compile_string_split(element, compiler, **kwargs):
    return compiler.process(func.split(*element.clauses.clauses), **kwargs)


def compile_string_regexp_replace(element, compiler, **kwargs):
    return f"regexp_replace({compiler.process(element.clauses, **kwargs)})"


def compile_string_replace(element, compiler, **kwargs):
    return compiler.process(func.replace(*element.clauses.clauses), **kwargs)


def compile_greatest(element, compiler, **kwargs):
    """
    Compiles a sql function for `greatest(*args)` taking 1 or more args

    Compiles to:
      - `max(arg1, arg2...)` for 2 or more args
      - `arg1` for 1 arg

    sqlite's max() is a simple function when it has 2 or more
    arguments but operates as an aggregate function if given only a
    single argument
    See https://www.sqlite.org/lang_corefunc.html#max_scalar
    """
    args = element.clauses.clauses
    nargs = len(args)
    if nargs < 1:
        raise TypeError(
            f"conditional.greatest requires at least 1 argument ({nargs} found)"
        )
    if nargs == 1:
        expr = args[0]
    else:
        expr = func.max(*args)
    return compiler.process(expr, **kwargs)


def compile_least(element, compiler, **kwargs):
    """
    Compiles a sql function for `least(*args)` taking 1 or more args

    Compiles to:
      - `min(arg1, arg2...)` for 2 or more args
      - `arg1` for 1 arg

    sqlite's min() is a simple function when it has 2 or more
    arguments but operates as an aggregate function if given only a
    single argument
    See https://www.sqlite.org/lang_corefunc.html#min_scalar
    """
    args = element.clauses.clauses
    nargs = len(args)
    if nargs < 1:
        raise TypeError(
            f"conditional.least requires at least 1 argument ({nargs} found)"
        )
    if nargs == 1:
        expr = args[0]
    else:
        expr = func.min(*args)
    return compiler.process(expr, **kwargs)


def compile_values(element, compiler, **kwargs):
    return base_values_compiler(lambda i: f"column{i}", element, compiler, **kwargs)


def compile_rand(element, compiler, **kwargs):
    return compiler.process(func.random(), **kwargs)


def compile_avg(element, compiler, **kwargs):
    return compiler.process(func.avg(*element.clauses.clauses), **kwargs)


def compile_group_concat(element, compiler, **kwargs):
    return compiler.process(func.aggregate_strings(*element.clauses.clauses), **kwargs)


def compile_any_value(element, compiler, **kwargs):
    # use bare column to return any value from the group,
    # this is documented behavior for sqlite,
    # see https://www.sqlite.org/lang_select.html#bare_columns_in_an_aggregate_query
    return compiler.process(*element.clauses.clauses, **kwargs)


def compile_collect(element, compiler, **kwargs):
    return compiler.process(func.json_group_array(*element.clauses.clauses), **kwargs)


@cache
def usearch_sqlite_path() -> str | None:
    try:
        import usearch
    except ImportError:
        return None

    with warnings.catch_warnings():
        # usearch binary is not available for Windows, see: https://github.com/unum-cloud/usearch/issues/427.
        # and, sometimes fail to download the binary in other platforms
        # triggering UserWarning.

        warnings.filterwarnings("ignore", category=UserWarning, module="usearch")

        try:
            return usearch.sqlite_path()
        except FileNotFoundError:
            return None


def load_usearch_extension(conn: sqlite3.Connection) -> bool:
    # usearch is part of the vector optional dependencies
    # we use the extension's cosine and euclidean distance functions
    ext_path = usearch_sqlite_path()
    if ext_path is None:
        return False

    try:
        conn.enable_load_extension(True)
    except AttributeError:
        # sqlite3 module is not built with loadable extension support by default.
        return False

    try:
        conn.load_extension(ext_path)
    except sqlite3.OperationalError:
        return False
    else:
        return True
    finally:
        conn.enable_load_extension(False)
