import random
import sqlite3

import pytest
import sqlalchemy as sa
from sqlalchemy import case, func

from datachain.sql.sqlite import base

slash = sa.literal("/")
dot = sa.literal(".")
empty_str = sa.literal("")


# Former SQL-expression implementations, kept as the reference semantics.
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


CASES = [
    "",
    "a",
    "a.b",
    ".a",
    "a.",
    "a/b",
    "a/b/",
    "/a",
    "a//b",
    "a/b.c.d",
    "a/.b",
    "a/b./",
    "x/y.tar.gz",
    "..",
    "./",
    "/",
]
_rnd = random.Random(0)  # noqa: S311
CASES += [
    "".join(_rnd.choice("ab./_é") for _ in range(_rnd.randint(0, 12)))
    for _ in range(3000)
]


@pytest.mark.parametrize(
    "reference,native",
    [
        (path_parent, base.sqlite_path_parent),
        (path_name, base.sqlite_path_name),
        (path_file_stem, base.sqlite_path_file_stem),
        (path_file_ext, base.sqlite_path_file_ext),
    ],
)
def test_native_path_functions_match_sql_reference(reference, native):
    conn = sqlite3.connect(":memory:")
    expr = reference(sa.literal_column("p"))
    sql = str(sa.select(expr).compile(compile_kwargs={"literal_binds": True}))
    for path in CASES:
        (expected,) = conn.execute(f"{sql} FROM (SELECT ? AS p)", (path,)).fetchone()
        assert native(path) == expected, path
