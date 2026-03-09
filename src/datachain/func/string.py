from typing import get_origin

from sqlalchemy import literal

from datachain.sql.functions import string

from .func import ColT, Func

__all__ = [
    "byte_hamming_distance",
    "length",
    "regexp_replace",
    "replace",
    "split",
]


def length(col: ColT) -> Func:
    """
    Returns the length of the string.

    Args:
        col (str | Column | Func | literal): String to compute the length of.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a string.
            If a literal is provided, it is assumed to be a string literal.

    Returns:
        Func: A `Func` object that represents the string length function.

    Example:
        ```py
        dc.mutate(
            len1=func.string.length("file.path"),
            len2=func.string.length(dc.C("file.path")),
            len3=func.string.length(dc.func.literal("Random string")),
        )
        ```

    Notes:
        - The result column will always be of type int.
    """
    return Func("length", inner=string.length, cols=[col], result_type=int)


def split(col: ColT, sep: str, limit: int | None = None) -> Func:
    """
    Takes a column and split character and returns an array of the parts.

    Args:
        col (str | Column | Func | literal): Column to split.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a string.
            If a literal is provided, it is assumed to be a string literal.
        sep (str): Separator to split the string.
        limit (int, optional): Maximum number of splits to perform.

    Returns:
        Func: A `Func` object that represents the split function.

    Example:
        ```py
        dc.mutate(
            path_parts=func.string.split("file.path", "/"),
            signal_values=func.string.split(dc.C("signal.value"), ","),
            str_words=func.string.split(dc.func.literal("Random string"), " "),
        )
        ```

    Notes:
        - The result column will always be of type array of strings.
    """

    def inner(arg):
        if limit is not None:
            return string.split(arg, sep, limit)
        return string.split(arg, sep)

    if get_origin(col) is literal:
        cols = None
        args = [col]
    else:
        cols = [col]
        args = None

    return Func("split", inner=inner, cols=cols, args=args, result_type=list[str])


def replace(col: ColT, pattern: str, replacement: str) -> Func:
    """
    Replaces substring with another string.

    Args:
        col (str | Column | Func | literal): Column to perform replacement on.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a string.
            If a literal is provided, it is assumed to be a string literal.
        pattern (str): Pattern to replace.
        replacement (str): Replacement string.

    Returns:
        Func: A `Func` object that represents the replace function.

    Example:
        ```py
        dc.mutate(
            s1=func.string.replace("signal.name", "pattern", "replacement"),
            s2=func.string.replace(dc.C("signal.name"), "pattern", "replacement"),
            s3=func.string.replace(dc.func.literal("Random string"), "Random", "New"),
        )
        ```

    Notes:
        - The result column will always be of type string.
    """

    def inner(arg):
        return string.replace(arg, pattern, replacement)

    if get_origin(col) is literal:
        cols = None
        args = [col]
    else:
        cols = [col]
        args = None

    return Func("replace", inner=inner, cols=cols, args=args, result_type=str)


def regexp_replace(col: ColT, regex: str, replacement: str) -> Func:
    r"""
    Replaces substring that match a regular expression.

    Args:
        col (str | Column | Func | literal): Column to perform replacement on.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a string.
            If a literal is provided, it is assumed to be a string literal.
        regex (str): Regular expression pattern to replace.
        replacement (str): Replacement string.

    Returns:
        Func: A `Func` object that represents the regexp_replace function.

    Example:
        ```py
        dc.mutate(
            s1=func.string.regexp_replace("signal.name", r"\d+", "X"),
            s2=func.string.regexp_replace(dc.C("signal.name"), r"\d+", "X"),
            s3=func.string.regexp_replace(
                dc.func.literal("Random string"),
                r"\s+",
                "_",
            ),
        )
        ```

    Notes:
        - The result column will always be of type string.
    """

    def inner(arg):
        return string.regexp_replace(arg, regex, replacement)

    if get_origin(col) is literal:
        cols = None
        args = [col]
    else:
        cols = [col]
        args = None

    return Func("regexp_replace", inner=inner, cols=cols, args=args, result_type=str)


def byte_hamming_distance(*args: ColT) -> Func:
    """
    Computes the Hamming distance between two strings.

    The Hamming distance is the number of positions at which the corresponding
    characters are different. This function returns the dissimilarity between
    the strings, where 0 indicates identical strings and values closer to the length
    of the strings indicate higher dissimilarity.

    Args:
        args (str | Column | Func | literal): Two strings to compute
            the Hamming distance between.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column in the dataset.
            If a Func is provided, it is assumed to be a function returning a string.
            If a literal is provided, it is assumed to be a string literal.

    Returns:
        Func: A `Func` object that represents the Hamming distance function.

    Example:
        ```py
        dc.mutate(
            hd1=func.byte_hamming_distance("file.phash", literal("hello")),
            hd2=func.byte_hamming_distance(dc.C("file.phash"), "hello"),
            hd3=func.byte_hamming_distance(
                dc.func.literal("hi"),
                dc.func.literal("hello"),
            ),
        )
        ```

    Notes:
        - The result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if get_origin(arg) is literal:
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("byte_hamming_distance() requires exactly two arguments")

    return Func(
        "byte_hamming_distance",
        inner=string.byte_hamming_distance,
        cols=cols,
        args=func_args,
        result_type=int,
    )
