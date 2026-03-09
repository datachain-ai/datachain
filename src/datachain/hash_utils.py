import hashlib
import inspect
import logging
import textwrap
from collections.abc import Sequence
from typing import TypeAlias, TypeVar
from uuid import uuid4

from sqlalchemy.sql.elements import ClauseElement, ColumnElement

from datachain import json

logger = logging.getLogger("datachain")

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
    Calculate a deterministic hash from a callable.

    Hashing Strategy:
    - **Named functions** (def): Uses source code via inspect.getsourcelines()
      → Produces stable hashes across Python versions and sessions
    - **Lambdas**: Uses bytecode (func.__code__.co_code)
      → Stable within same Python runtime, may differ across Python versions
    - **Callable objects** (with __call__): Extracts and hashes the __call__ method

    Supported Callables:
    - Regular Python functions defined with 'def'
    - Lambda functions
    - Classes/instances with __call__ method (uses __call__ method's code)
    - Methods (both bound and unbound)

    Limitations and Edge Cases:
    - **Mock objects**: Cannot reliably hash Mock(side_effect=...) because the
      side_effect is not discoverable via inspection. Use regular functions instead.
    - **Built-in functions** (len, str, etc.): Cannot access __code__ attribute.
      Returns a random hash that changes on each call.
    - **C extensions**: Cannot access source or bytecode. Returns a random hash
      that changes on each call.
    - **Dynamically generated callables**: If __call__ is created via exec/eval
      or the behavior depends on runtime state, the hash won't reflect changes
      in behavior. Only the method's code is hashed, not captured state.

    Args:
        func: A callable object (function, lambda, method, or object with __call__)

    Returns:
        str: SHA256 hexdigest of the callable's code and metadata. For unhashable
        callables (C extensions, built-ins), returns a hash of a random UUID that
        changes on each invocation.

    Raises:
        TypeError: If func is not callable

    Examples:
        >>> def my_func(x): return x * 2
        >>> hash_callable(my_func)  # Uses source code
        'abc123...'

        >>> hash_callable(lambda x: x * 2)  # Uses bytecode
        'def456...'

        >>> class MyCallable:
        ...     def __call__(self, x): return x * 2
        >>> hash_callable(MyCallable())  # Hashes __call__ method
        'ghi789...'
    """
    if not callable(func):
        raise TypeError("Expected a callable")

    # Handle callable objects (instances with __call__)
    # If it's not a function or method, it must be a callable object
    if not inspect.isfunction(func) and not inspect.ismethod(func):
        # For callable objects, hash the __call__ method instead
        func = func.__call__

    # Determine if it is a lambda
    try:
        is_lambda = func.__name__ == "<lambda>"
    except AttributeError:
        # Some callables (like Mock objects) may not have __name__
        is_lambda = False

    if not is_lambda:
        # Try to get exact source of named function
        try:
            lines, _ = inspect.getsourcelines(func)
            payload = textwrap.dedent("".join(lines)).strip()
        except (OSError, TypeError):
            # Fallback: bytecode if source not available
            try:
                payload = func.__code__.co_code
            except AttributeError:
                # C extensions, built-ins - use random UUID
                # Returns different hash on each call to avoid caching unhashable
                # functions
                logger.warning(
                    "Cannot hash callable %r (likely C extension or built-in). "
                    "Returning random hash.",
                    func,
                )
                payload = f"unhashable-{uuid4()}"
    else:
        # For lambdas, fall back directly to bytecode
        try:
            payload = func.__code__.co_code
        except AttributeError:
            # Unlikely for lambdas, but handle it just in case
            logger.warning("Cannot hash lambda %r. Returning random hash.", func)
            payload = f"unhashable-{uuid4()}"

    # Normalize annotations (may not exist for built-ins/C extensions)
    raw_annotations = getattr(func, "__annotations__", {})
    annotations = {
        k: getattr(v, "__name__", str(v)) for k, v in raw_annotations.items()
    }

    # Extras to distinguish functions with same code but different metadata
    extras = {
        "name": getattr(func, "__name__", ""),
        "defaults": getattr(func, "__defaults__", None),
        "annotations": annotations,
    }

    # Compute SHA256
    h = hashlib.sha256()
    h.update(str(payload).encode() if isinstance(payload, str) else payload)
    h.update(str(extras).encode())
    return h.hexdigest()
