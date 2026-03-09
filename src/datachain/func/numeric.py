from datachain.query.schema import Column
from datachain.sql.functions import numeric

from .func import Func


def bit_and(*args: str | Column | Func | int) -> Func:
    """
    Returns a function that computes the bitwise AND operation between two values.

    Args:
        args (str | Column | Func | int): Two values to compute
            the bitwise AND operation between.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column.
            If a Func is provided, it is assumed to be a function returning an int.
            If an integer is provided, it is assumed to be a constant value.

    Returns:
        Func: A `Func` object that represents the bitwise AND function.

    Example:
        ```py
        dc.mutate(
            and1=func.bit_and("signal.value", 0x0F),
            and2=func.bit_and(dc.C("signal.value1"), "signal.value2"),
        )
        ```

    Notes:
        - The result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, int):
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("bit_and() requires exactly two arguments")

    return Func(
        "bit_and",
        inner=numeric.bit_and,
        cols=cols,
        args=func_args,
        result_type=int,
    )


def bit_or(*args: str | Column | Func | int) -> Func:
    """
    Returns a function that computes the bitwise OR operation between two values.

    Args:
        args (str | Column | Func | int): Two values to compute
            the bitwise OR operation between.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column.
            If a Func is provided, it is assumed to be a function returning an int.
            If an integer is provided, it is assumed to be a constant value.

    Returns:
        Func: A `Func` object that represents the bitwise OR function.

    Example:
        ```py
        dc.mutate(
            or1=func.bit_or("signal.value", 0x0F),
            or2=func.bit_or(dc.C("signal.value1"), "signal.value2"),
        )
        ```

    Notes:
        - The result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, int):
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("bit_or() requires exactly two arguments")

    return Func(
        "bit_or",
        inner=numeric.bit_or,
        cols=cols,
        args=func_args,
        result_type=int,
    )


def bit_xor(*args: str | Column | Func | int) -> Func:
    """
    Returns a function that computes the bitwise XOR operation between two values.

    Args:
        args (str | Column | Func | int): Two values to compute
            the bitwise XOR operation between.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column.
            If a Func is provided, it is assumed to be a function returning an int.
            If an integer is provided, it is assumed to be a constant value.

    Returns:
        Func: A `Func` object that represents the bitwise XOR function.

    Example:
        ```py
        dc.mutate(
            xor1=func.bit_xor("signal.value", 0x0F),
            xor2=func.bit_xor(dc.C("signal.value1"), "signal.value2"),
        )
        ```

    Notes:
        - The result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, int):
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("bit_xor() requires exactly two arguments")

    return Func(
        "bit_xor",
        inner=numeric.bit_xor,
        cols=cols,
        args=func_args,
        result_type=int,
    )


def int_hash_64(col: str | Column | Func | int) -> Func:
    """
    Returns a function that computes the 64-bit hash of an integer.

    Args:
        col (str | Column | Func | int): Integer to compute the hash of.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column.
            If a Func is provided, it is assumed to be a function returning an int.
            If an int is provided, it is assumed to be an int literal.

    Returns:
        Func: A `Func` object that represents the 64-bit hash function.

    Example:
        ```py
        dc.mutate(
            val_hash=func.int_hash_64("val"),
            val_hash2=func.int_hash_64(dc.C("val2")),
        )
        ```

    Notes:
        - The result column will always be of type int.
    """
    cols, args = [], []
    if isinstance(col, int):
        args.append(col)
    else:
        cols.append(col)

    return Func(
        "int_hash_64", inner=numeric.int_hash_64, cols=cols, args=args, result_type=int
    )


def bit_hamming_distance(*args: str | Column | Func | int) -> Func:
    """
    Returns a function that computes the Hamming distance between two integers.

    The Hamming distance is the number of positions at which the corresponding bits
    are different. This function returns the dissimilarity between the integers,
    where 0 indicates identical integers and values closer to the number of bits
    in the integer indicate higher dissimilarity.

    Args:
        args (str | Column | Func | int): Two integers to compute
            the Hamming distance between.
            If a string is provided, it is assumed to be the name of the column.
            If a Column is provided, it is assumed to be a column.
            If a Func is provided, it is assumed to be a function returning an int.
            If an int is provided, it is assumed to be an integer literal.

    Returns:
        Func: A `Func` object that represents the Hamming distance function.

    Example:
        ```py
        dc.mutate(
            hd1=func.bit_hamming_distance("signal.value1", "signal.value2"),
            hd2=func.bit_hamming_distance(dc.C("signal.value1"), 0x0F),
        )
        ```

    Notes:
        - The result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, int):
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("bit_hamming_distance() requires exactly two arguments")

    return Func(
        "bit_hamming_distance",
        inner=numeric.bit_hamming_distance,
        cols=cols,
        args=func_args,
        result_type=int,
    )
