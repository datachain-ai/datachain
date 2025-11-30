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


def hash_callable(func, _visited=None):
    """
    Calculate a hash from a callable, including its dependencies.
    Rules:
    - Named functions (def) → use source code for stable, cross-version hashing
    - Lambdas → use bytecode (deterministic in same Python runtime)
    - Recursively hashes helper functions from the same module
    """
    if not callable(func):
        raise TypeError("Expected a callable")

    # Track visited functions to avoid infinite recursion
    if _visited is None:
        _visited = set()

    # Use id(func) to track which functions we've visited
    func_id = id(func)
    if func_id in _visited:
        return hashlib.sha256(f"recursive:{func.__name__}".encode()).hexdigest()
    _visited.add(func_id)

    # Determine if it is a lambda
    is_lambda = func.__name__ == "<lambda>"

    if not is_lambda:
        # Try to get exact source of named function
        try:
            lines, _ = inspect.getsourcelines(func)
            payload = textwrap.dedent("".join(lines)).strip()
        except (OSError, TypeError):
            # Fallback: bytecode + constants if source not available
            code = func.__code__
            payload = (code.co_code, code.co_consts, code.co_names, code.co_varnames)
    else:
        # For lambdas, use bytecode + constants
        code = func.__code__
        payload = (code.co_code, code.co_consts, code.co_names, code.co_varnames)

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

    # Find helper functions that this function depends on
    dependencies = {}
    if hasattr(func, "__code__") and hasattr(func, "__globals__"):
        # Get all names referenced in the function's code
        referenced_names = func.__code__.co_names
        func_module = inspect.getmodule(func)

        for name in referenced_names:
            # Look up the name in the function's global namespace
            if name in func.__globals__:
                obj = func.__globals__[name]

                # Only hash user-defined functions from the same module
                # Skip built-ins, imported functions from other modules, and classes
                if (
                    callable(obj)
                    and hasattr(obj, "__module__")
                    and func_module is not None
                    and obj.__module__ == func_module.__name__
                    and not inspect.isclass(obj)
                    and not inspect.isbuiltin(obj)
                ):
                    # Recursively hash the dependency
                    try:
                        dependencies[name] = hash_callable(obj, _visited)
                    except (TypeError, OSError):
                        # If we can't hash it, skip it
                        pass

    # Compute SHA256
    h = hashlib.sha256()
    if isinstance(payload, str):
        h.update(payload.encode())
    else:
        # payload is a tuple of (bytecode, consts, names, varnames)
        h.update(str(payload).encode())
    h.update(str(extras).encode())
    # Include dependency hashes in sorted order for determinism
    if dependencies:
        deps_str = json.dumps(dependencies, sort_keys=True)
        h.update(deps_str.encode())
    return h.hexdigest()
