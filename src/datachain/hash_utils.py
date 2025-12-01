import hashlib
import inspect
import textwrap
from collections.abc import Sequence
from typing import TypeAlias, TypeVar

from sqlalchemy.sql.elements import ClauseElement, ColumnElement

from datachain import json

T = TypeVar("T", bound=ColumnElement)
ColumnLike: TypeAlias = str | T


def _serialize_value(val):  # noqa: PLR0911
    """Helper to serialize arbitrary values recursively."""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, ClauseElement):
        return serialize_column_element(val)
    if isinstance(val, dict):
        # Sort dict keys for deterministic serialization
        return {k: _serialize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if callable(val):
        return val.__name__ if hasattr(val, "__name__") else str(val)
    return str(val)


def serialize_column_element(expr: str | ColumnElement) -> dict:
    """
    Recursively serialize a SQLAlchemy ColumnElement into a deterministic structure.
    Uses SQLAlchemy's _traverse_internals to automatically handle all expression types.
    """
    from sqlalchemy.sql.elements import BindParameter

    # Special case: BindParameter has non-deterministic 'key' attribute, only use value
    if isinstance(expr, BindParameter):
        return {"type": "bind", "value": _serialize_value(expr.value)}

    # Generic handling for all ClauseElement types using SQLAlchemy's internals
    if isinstance(expr, ClauseElement):
        # All standard SQLAlchemy types have _traverse_internals
        if hasattr(expr, "_traverse_internals"):
            result = {"type": expr.__class__.__name__}
            for attr_name, _ in expr._traverse_internals:
                # Skip 'table' attribute - table names can be auto-generated/random
                # and are not semantically important for hashing
                if attr_name == "table":
                    continue
                if hasattr(expr, attr_name):
                    val = getattr(expr, attr_name)
                    result[attr_name] = _serialize_value(val)
            return result
        # Rare case: custom user-defined ClauseElement without _traverse_internals
        # We don't know its structure, so just stringify it
        return {"type": expr.__class__.__name__, "repr": str(expr)}

    # Absolute fallback: stringify completely unknown types
    return {"type": "other", "repr": str(expr)}


def hash_column_elements(columns: ColumnLike | Sequence[ColumnLike]) -> str:
    """
    Hash a list of ColumnElements deterministically, dialect agnostic.
    Only accepts ordered iterables (like list or tuple).
    """
    # Handle case where a single ColumnElement is passed instead of a sequence
    if isinstance(columns, (ColumnElement, str)):
        columns = (columns,)

    serialized = [serialize_column_element(c) for c in columns]
    json_str = json.dumps(
        serialized, sort_keys=True, separators=(", ", ": ")
    )  # stable JSON
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def hash_callable(func):
    """
    Calculate a hash from a callable.
    Rules:
    - Named functions (def) → use source code for stable, cross-version hashing
    - Lambdas → use bytecode (deterministic in same Python runtime)
    """
    if not callable(func):
        raise TypeError("Expected a callable")

    # Determine if it is a lambda
    is_lambda = func.__name__ == "<lambda>"

    if not is_lambda:
        # Try to get exact source of named function
        try:
            lines, _ = inspect.getsourcelines(func)
            payload = textwrap.dedent("".join(lines)).strip()
        except (OSError, TypeError):
            # Fallback: bytecode if source not available
            payload = func.__code__.co_code
    else:
        # For lambdas, fall back directly to bytecode
        payload = func.__code__.co_code

    # Normalize annotations
    annotations = {
        k: getattr(v, "__name__", str(v)) for k, v in func.__annotations__.items()
    }

    # Extras to distinguish functions with same code but different metadata
    extras = {
        "name": func.__name__,
        "defaults": func.__defaults__,
        "annotations": annotations,
    }

    # Compute SHA256
    h = hashlib.sha256()
    h.update(str(payload).encode() if isinstance(payload, str) else payload)
    h.update(str(extras).encode())
    return h.hexdigest()
