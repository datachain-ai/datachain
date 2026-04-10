from sqlalchemy.sql.functions import GenericFunction

from datachain.sql.types import Array, Int64, String
from datachain.sql.utils import compiler_not_implemented


class length(GenericFunction):  # noqa: N801
    """
    Returns the length of the string.
    """

    type = Int64()
    package = "string"
    name = "length"
    inherit_cache = True


class split(GenericFunction):  # noqa: N801
    """
    Takes a column and split character and returns an array of the parts.
    """

    type = Array(String())
    package = "string"
    name = "split"
    inherit_cache = True


class regexp_replace(GenericFunction):  # noqa: N801
    """
    Replaces substring that match a regular expression.
    """

    type = String()
    package = "string"
    name = "regexp_replace"
    inherit_cache = True


class replace(GenericFunction):  # noqa: N801
    """
    Replaces substring with another string.
    """

    type = String()
    package = "string"
    name = "replace"
    inherit_cache = True


class byte_hamming_distance(GenericFunction):  # noqa: N801
    """
    Returns the Hamming distance between two strings.
    """

    type = Int64()
    package = "string"
    name = "hamming_distance"
    inherit_cache = True


class string_hash(GenericFunction):  # noqa: N801
    """
    Computes a 64-bit hash of one or more string arguments.
    """

    type = Int64()
    package = "hash"
    name = "string_hash"
    inherit_cache = True


compiler_not_implemented(length)
compiler_not_implemented(split)
compiler_not_implemented(regexp_replace)
compiler_not_implemented(replace)
compiler_not_implemented(byte_hamming_distance)
compiler_not_implemented(string_hash)
