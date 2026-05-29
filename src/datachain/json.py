"""DataChain JSON utilities.

This module wraps :mod:`ujson` so we can guarantee consistent handling
of values that the encoder does not support out of the box (for example
``datetime`` objects or ``bytes``).
All code inside DataChain should import this module instead of using
:mod:`ujson` directly.
"""

import datetime as _dt
import json as _json
import sys as _sys
import uuid as _uuid
from collections.abc import Callable
from typing import Any

import ujson as _ujson

__all__ = [
    "JSONDecodeError",
    "dump",
    "dumps",
    "load",
    "loads",
]

JSONDecodeError = (_ujson.JSONDecodeError, _json.JSONDecodeError)

_SENTINEL = object()
_Default = Callable[[Any], Any]
DEFAULT_PREVIEW_BYTES = 1024


# To make it looks like Pydantic's ISO format with 'Z' for UTC
# It is minor but nice to have consistency
def _format_datetime(value: _dt.datetime) -> str:
    iso = value.isoformat()

    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        return iso

    if offset == _dt.timedelta(0) and iso.endswith(("+00:00", "-00:00")):
        return iso[:-6] + "Z"

    return iso


def _format_time(value: _dt.time) -> str:
    iso = value.isoformat()

    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        return iso

    if offset == _dt.timedelta(0) and iso.endswith(("+00:00", "-00:00")):
        return iso[:-6] + "Z"

    return iso


def _coerce(value: Any, serialize_bytes: bool, serialize_numpy: bool) -> Any:
    """Return a JSON-serializable representation for supported extra types."""

    converted = _SENTINEL
    if isinstance(value, _dt.datetime):
        converted = _format_datetime(value)
    elif isinstance(value, _dt.date):
        converted = value.isoformat()
    elif isinstance(value, _dt.time):
        converted = _format_time(value)
    elif isinstance(value, _uuid.UUID):
        converted = str(value)

    if converted is _SENTINEL and serialize_numpy:
        converted = _coerce_numpy(value)
    if (
        converted is _SENTINEL
        and serialize_bytes
        and isinstance(value, (bytes, bytearray))
    ):
        converted = list(bytes(value)[:DEFAULT_PREVIEW_BYTES])

    return converted


def _coerce_numpy(value: Any) -> Any:
    if "numpy" not in _sys.modules:
        return _SENTINEL

    import numpy as np

    if isinstance(value, (np.ndarray, np.generic)):
        return _numpy_to_python(value, np)
    return _SENTINEL


def _numpy_to_python(value: Any, numpy_module) -> Any:
    if isinstance(value, numpy_module.ndarray):
        converted = value.tolist()
        if value.dtype != object:
            return converted
        return _numpy_to_python(converted, numpy_module)
    if isinstance(value, numpy_module.generic):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [_numpy_to_python(item, numpy_module) for item in value]
    if isinstance(value, dict):
        return {
            _numpy_to_python(key, numpy_module): _numpy_to_python(item, numpy_module)
            for key, item in value.items()
        }
    return value


def _base_default(value: Any, serialize_bytes: bool, serialize_numpy: bool) -> Any:
    converted = _coerce(value, serialize_bytes, serialize_numpy)
    if converted is not _SENTINEL:
        return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _build_default(
    user_default: _Default | None, serialize_bytes: bool, serialize_numpy: bool
) -> _Default:
    if user_default is None:
        return lambda value: _base_default(value, serialize_bytes, serialize_numpy)

    def combined(value: Any) -> Any:
        converted = _coerce(value, serialize_bytes, serialize_numpy)
        if converted is not _SENTINEL:
            return converted
        return user_default(value)

    return combined


def dumps(
    obj: Any,
    *,
    default: _Default | None = None,
    serialize_bytes: bool = False,
    serialize_numpy: bool = False,
    **kwargs: Any,
) -> str:
    """Serialize *obj* to a JSON-formatted ``str``."""

    if serialize_bytes:
        return _json.dumps(
            obj, default=_build_default(default, True, serialize_numpy), **kwargs
        )

    return _ujson.dumps(
        obj, default=_build_default(default, False, serialize_numpy), **kwargs
    )


def dump(
    obj: Any,
    fp,
    *,
    default: _Default | None = None,
    serialize_bytes: bool = False,
    serialize_numpy: bool = False,
    **kwargs: Any,
) -> None:
    """Serialize *obj* as a JSON formatted stream to *fp*."""

    if serialize_bytes:
        _json.dump(
            obj, fp, default=_build_default(default, True, serialize_numpy), **kwargs
        )
        return

    _ujson.dump(
        obj, fp, default=_build_default(default, False, serialize_numpy), **kwargs
    )


def loads(s: str | bytes | bytearray, **kwargs: Any) -> Any:
    """Deserialize *s* to a Python object."""

    return _ujson.loads(s, **kwargs)


def load(fp, **kwargs: Any) -> Any:
    """Deserialize JSON content from *fp* to a Python object."""

    return loads(fp.read(), **kwargs)
