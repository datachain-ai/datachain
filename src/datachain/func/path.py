from datachain.sql.functions import path

from .func import ColT, Func


def parent(col: ColT) -> Func:
    """
    Returns the directory component of a posix-style path.
    Empty segments are preserved: `parent("d//e.txt")` returns `"d/"`.
    `File.parent` uses the same rule for nested paths.

    Args:
        col (str | Column | Func | literal): String to compute the path parent of.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column object.
            If a Func is provided, it is assumed to be a function returning a string.
            If a literal is provided, it is assumed to be a string literal.

    Returns:
        Func: A `Func` object that represents the path parent function.

    Example:
        ```py
        dc.mutate(
            parent1=func.path.parent("file.path"),
            parent2=func.path.parent(dc.C("file.path")),
            parent3=func.path.parent(dc.func.literal("/path/to/file.txt")),
        )
        ```

    Note:
        - The result column will always be of type string.
    """
    return Func("parent", inner=path.parent, cols=[col], result_type=str)


def name(col: ColT) -> Func:
    """
    Returns the final component of a posix-style path.

    Args:
        col (str | Column | Func | literal): String to compute the path name of.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column object.
            If a Func is provided, it is assumed to be a function returning a string.
            If a literal is provided, it is assumed to be a string literal.

    Returns:
        Func: A `Func` object that represents the path name function.

    Example:
        ```py
        dc.mutate(
            filename1=func.path.name("file.path"),
            filename2=func.path.name(dc.C("file.path")),
            filename3=func.path.name(dc.func.literal("/path/to/file.txt")
        )
        ```

    Note:
        - The result column will always be of type string.
    """

    return Func("name", inner=path.name, cols=[col], result_type=str)


def file_stem(col: ColT) -> Func:
    """
    Returns the path without the extension.

    Args:
        col (str | Column | Func | literal): String to compute the file stem of.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column object.
            If a Func is provided, it is assumed to be a function returning a string.
            If a literal is provided, it is assumed to be a string literal.

    Returns:
        Func: A `Func` object that represents the file stem function.

    Example:
        ```py
        dc.mutate(
            filestem1=func.path.file_stem("file.path"),
            filestem2=func.path.file_stem(dc.C("file.path")),
            filestem3=func.path.file_stem(dc.func.literal("/path/to/file.txt")
        )
        ```

    Note:
        - The result column will always be of type string.
    """

    return Func("file_stem", inner=path.file_stem, cols=[col], result_type=str)


def file_ext(col: ColT) -> Func:
    """
    Returns the extension of the given path.

    Args:
        col (str | Column | Func | literal): String to compute the file extension of.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column object.
            If a Func is provided, it is assumed to be a function returning a string.
            If a literal is provided, it is assumed to be a string literal.

    Returns:
        Func: A `Func` object that represents the file extension function.

    Example:
        ```py
        dc.mutate(
            filestem1=func.path.file_ext("file.path"),
            filestem2=func.path.file_ext(dc.C("file.path")),
            filestem3=func.path.file_ext(dc.func.literal("/path/to/file.txt")
        )
        ```

    Note:
        - The result column will always be of type string.
    """

    return Func("file_ext", inner=path.file_ext, cols=[col], result_type=str)
