from collections.abc import Sequence
from typing import Any

from datachain.query.schema import Column
from datachain.sql.functions import array

from .func import Func


def cosine_distance(*args: str | Column | Func | Sequence) -> Func:
    """
    Returns the cosine distance between two vectors.

    The cosine distance is derived from the cosine similarity, which measures the angle
    between two vectors. This function returns the dissimilarity between the vectors,
    where 0 indicates identical vectors and values closer to 1
    indicate higher dissimilarity.

    Args:
        args (str | Column | Func | Sequence): Two vectors to compute the cosine
            distance between.
            If a string is provided, it is assumed to be the name of the column vector.
            If a Column is provided, it is assumed to be an array column.
            If a Func is provided, it is assumed to be a function returning an array.
            If a sequence is provided, it is assumed to be a vector of values.

    Returns:
        Func: A `Func` object that represents the cosine_distance function.

    Example:
        ```py
        target_embedding = [0.1, 0.2, 0.3]
        dc.mutate(
            cos_dist1=func.cosine_distance("embedding", target_embedding),
            cos_dist2=func.cosine_distance(dc.C("emb1"), "emb2"),
            cos_dist3=func.cosine_distance(target_embedding, [0.4, 0.5, 0.6]),
        )
        ```

    Notes:
        - Ensure both vectors have the same number of elements.
        - The result column will always be of type float.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, (str, Column, Func)):
            cols.append(arg)
        else:
            func_args.append(list(arg))

    if len(cols) + len(func_args) != 2:
        raise ValueError("cosine_distance() requires exactly two arguments")
    if not cols and len(func_args[0]) != len(func_args[1]):
        raise ValueError("cosine_distance() requires vectors of the same length")

    return Func(
        "cosine_distance",
        inner=array.cosine_distance,
        cols=cols,
        args=func_args,
        result_type=float,
    )


def euclidean_distance(*args: str | Column | Func | Sequence) -> Func:
    """
    Returns the Euclidean distance between two vectors.

    The Euclidean distance is the straight-line distance between two points
    in Euclidean space. This function returns the distance between the two vectors.

    Args:
        args (str | Column | Func | Sequence): Two vectors to compute the Euclidean
            distance between.
            If a string is provided, it is assumed to be the name of the column vector.
            If a Column is provided, it is assumed to be an array column.
            If a Func is provided, it is assumed to be a function returning an array.
            If a sequence is provided, it is assumed to be a vector of values.

    Returns:
        Func: A `Func` object that represents the euclidean_distance function.

    Example:
        ```py
        target_embedding = [0.1, 0.2, 0.3]
        dc.mutate(
            eu_dist1=func.euclidean_distance("embedding", target_embedding),
            eu_dist2=func.euclidean_distance(dc.C("emb1"), "emb2"),
            eu_dist3=func.euclidean_distance(target_embedding, [0.4, 0.5, 0.6]),
        )
        ```

    Notes:
        - Ensure both vectors have the same number of elements.
        - The result column will always be of type float.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, (str, Column, Func)):
            cols.append(arg)
        else:
            func_args.append(list(arg))

    if len(cols) + len(func_args) != 2:
        raise ValueError("euclidean_distance() requires exactly two arguments")
    if not cols and len(func_args[0]) != len(func_args[1]):
        raise ValueError("euclidean_distance() requires vectors of the same length")

    return Func(
        "euclidean_distance",
        inner=array.euclidean_distance,
        cols=cols,
        args=func_args,
        result_type=float,
    )


def length(arg: str | Column | Func | Sequence) -> Func:
    """
    Returns the length of the array.

    Args:
        arg (str | Column | Func | Sequence): Array to compute the length of.
            If a string is provided, it is assumed to be the name of the array column.
            If a Column is provided, it is assumed to be an array column.
            If a Func is provided, it is assumed to be a function returning an array.
            If a sequence is provided, it is assumed to be an array of values.

    Returns:
        Func: A `Func` object that represents the array length function.

    Example:
        ```py
        dc.mutate(
            len1=func.array.length("signal.values"),
            len2=func.array.length(dc.C("signal.values")),
            len3=func.array.length([1, 2, 3, 4, 5]),
        )
        ```

    Notes:
        - The result column will always be of type int.
    """
    if isinstance(arg, (str, Column, Func)):
        cols = [arg]
        args = None
    else:
        cols = None
        args = [arg]

    return Func("length", inner=array.length, cols=cols, args=args, result_type=int)


def contains(arr: str | Column | Func | Sequence, elem: Any) -> Func:
    """
    Checks whether the array contains the specified element.

    Args:
        arr (str | Column | Func | Sequence): Array to check for the element.
            If a string is provided, it is assumed to be the name of the array column.
            If a Column is provided, it is assumed to be an array column.
            If a Func is provided, it is assumed to be a function returning an array.
            If a sequence is provided, it is assumed to be an array of values.
        elem (Any): Element to check for in the array.

    Returns:
        Func: A `Func` object that represents the contains function. Result of the
            function will be `1` if the element is present in the array,
            and `0` otherwise.

    Example:
        ```py
        dc.mutate(
            contains1=func.array.contains("signal.values", 3),
            contains2=func.array.contains(dc.C("signal.values"), 7),
            contains3=func.array.contains([1, 2, 3, 4, 5], 7),
        )
        ```

    Notes:
        - The result column will always be of type int.
    """

    def inner(arg):
        is_json = type(elem) in [list, dict]
        return array.contains(arg, elem, is_json)

    if isinstance(arr, (str, Column, Func)):
        cols = [arr]
        args = None
    else:
        cols = None
        args = [arr]

    return Func("contains", inner=inner, cols=cols, args=args, result_type=int)


def slice(
    arr: str | Column | Func | Sequence,
    offset: int,
    length: int | None = None,
) -> Func:
    """
    Returns a slice of the array starting from the specified offset.

    Args:
        arr (str | Column | Func | Sequence): Array to slice.
            If a string is provided, it is assumed to be the name of the array column.
            If a Column is provided, it is assumed to be an array column.
            If a Func is provided, it is assumed to be a function returning an array.
            If a sequence is provided, it is assumed to be an array of values.
        offset (int): Starting position of the slice (0-based).
        length (int, optional): Number of elements to include in the slice.
            If not provided, returns all elements from offset to the end.

    Returns:
        Func: A `Func` object that represents the slice function.

    Example:
        ```py
        dc.mutate(
            slice1=func.array.slice("signal.values", 1, 3),
            slice2=func.array.slice(dc.C("signal.values"), 2),
            slice3=func.array.slice([1, 2, 3, 4, 5], 1, 2),
        )
        ```

    Notes:
        - The result column will be of type array with the same element type
            as the input.
    """

    def inner(arg):
        if length is not None:
            return array.slice(arg, offset, length)
        return array.slice(arg, offset)

    def element_type(el):
        if isinstance(el, list):
            try:
                return list[element_type(el[0])]
            except IndexError:
                # if the array is empty, return list[str] as default type
                return list[str]
        return type(el)

    def type_from_args(arr, *_):
        if isinstance(arr, list):
            try:
                return list[element_type(arr[0])]
            except IndexError:
                pass
        # if not an array or array is empty, return list[str] as default type
        return list[str]

    if isinstance(arr, (str, Column, Func)):
        cols = [arr]
        args = None
    else:
        cols = None
        args = [arr]

    return Func(
        "slice",
        inner=inner,
        cols=cols,
        args=args,
        from_array=True,
        is_array=True,
        type_from_args=type_from_args,
    )


def join(
    arr: str | Column | Func | Sequence,
    sep: str = "",
) -> Func:
    """
    Returns a string that is the concatenation of the elements of the array.

    Args:
        arr (str | Column | Func | Sequence): Array to join.
            If a string is provided, it is assumed to be the name of the array column.
            If a Column is provided, it is assumed to be an array column.
            If a Func is provided, it is assumed to be a function returning an array.
            If a sequence is provided, it is assumed to be an array of values.
        sep (str): Separator to use for the concatenation. Default is an empty string.

    Returns:
        Func: A `Func` object that represents the join function.

    Example:
        ```py
        dc.mutate(
            join1=func.array.join("signal.values", ":"),
            join2=func.array.join(dc.C("signal.values"), ","),
            join3=func.array.join(["1", "2", "3", "4", "5"], "/"),
        )
        ```

    Notes:
        - The result column will always be of type string.
    """

    def inner(arg):
        return array.join(arg, sep)

    if isinstance(arr, (str, Column, Func)):
        cols = [arr]
        args = None
    else:
        cols = None
        args = [arr]

    return Func(
        "join",
        inner=inner,
        cols=cols,
        args=args,
        from_array=True,
        result_type=str,
    )


def get_element(arg: str | Column | Func | Sequence, index: int) -> Func:
    """
    Returns the element at the given index from the array.

    Uses 0-based indexing. Negative indices count from the end of the array
    (e.g., -1 is the last element, -2 is second-to-last). If the index is
    out of bounds in either direction, returns the column's default value
    (typically None).

    Args:
        arg (str | Column | Func | Sequence): Array to get the element from.
            If a string is provided, it is assumed to be the name of the array column.
            If a Column is provided, it is assumed to be an array column.
            If a Func is provided, it is assumed to be a function returning an array.
            If a sequence is provided, it is assumed to be an array of values.
        index (int): 0-based index of the element to get from the array.
            Negative values index from the end (-1 = last, -2 = second-to-last).

    Returns:
        Func: A `Func` object that represents the array get_element function.

    Example:
        ```py
        dc.mutate(
            first_el=func.array.get_element("signal.values", 0),
            second_el=func.array.get_element(dc.C("signal.values"), 1),
            third_el=func.array.get_element([1, 2, 3, 4, 5], 2),
            last_el=func.array.get_element("signal.values", -1),
            second_last=func.array.get_element("signal.values", -2),
        )
        ```

    Notes:
        - The result column will always be the same type as the elements of the array.
        - Out-of-bounds access (e.g., index 10 on a 3-element array, or -4 on a
          3-element array) returns the column's default value rather than raising
          an error.
    """

    def type_from_args(arr, _):
        if isinstance(arr, list):
            try:
                return type(arr[0])
            except IndexError:
                return str  # if the array is empty, return str as default type
        return None

    cols: str | Column | Func | Sequence | None
    args: str | Column | Func | Sequence | int

    if isinstance(arg, (str, Column, Func)):
        cols = [arg]
        args = [index]
    else:
        cols = None
        args = [arg, index]

    return Func(
        "get_element",
        inner=array.get_element,
        cols=cols,
        args=args,
        from_array=True,
        type_from_args=type_from_args,
    )


def sip_hash_64(arg: str | Column | Func | Sequence) -> Func:
    """
    Returns the SipHash-64 hash of the array.

    Args:
        arg (str | Column | Func | Sequence): Array to compute the SipHash-64 hash of.
            If a string is provided, it is assumed to be the name of the array column.
            If a Column is provided, it is assumed to be an array column.
            If a Func is provided, it is assumed to be a function returning an array.
            If a sequence is provided, it is assumed to be an array of values.

    Returns:
        Func: A `Func` object that represents the sip_hash_64 function.

    Example:
        ```py
        dc.mutate(
            hash1=func.sip_hash_64("signal.values"),
            hash2=func.sip_hash_64(dc.C("signal.values")),
            hash3=func.sip_hash_64([1, 2, 3, 4, 5]),
        )
        ```

    Note:
        - This function is only available for the ClickHouse warehouse.
        - The result column will always be of type int.
    """
    if isinstance(arg, (str, Column, Func)):
        cols = [arg]
        args = None
    else:
        cols = None
        args = [arg]

    return Func(
        "sip_hash_64", inner=array.sip_hash_64, cols=cols, args=args, result_type=int
    )
