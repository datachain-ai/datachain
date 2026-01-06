"""DataChain JSON utilities.

This module wraps :mod:`ujson` so we can guarantee consistent handling
of values that the encoder does not support out of the box (for example
``datetime`` objects or ``bytes``).
All code inside DataChain should import this module instead of using
:mod:`ujson` directly.
"""

import datetime as _dt
import json as _json
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


def _coerce(value: Any, serialize_bytes: bool) -> Any:
    """Return a JSON-serializable representation for supported extra types."""

    if isinstance(value, _dt.datetime):
        return _format_datetime(value)
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, _dt.time):
        return _format_time(value)
    if isinstance(value, _uuid.UUID):
        return str(value)
    if serialize_bytes and isinstance(value, (bytes, bytearray)):
        return list(bytes(value)[:DEFAULT_PREVIEW_BYTES])
    return _SENTINEL


def _base_default(value: Any, serialize_bytes: bool) -> Any:
    converted = _coerce(value, serialize_bytes)
    if converted is not _SENTINEL:
        return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _build_default(user_default: _Default | None, serialize_bytes: bool) -> _Default:
    if user_default is None:
        return lambda value: _base_default(value, serialize_bytes)

    def combined(value: Any) -> Any:
        converted = _coerce(value, serialize_bytes)
        if converted is not _SENTINEL:
            return converted
        return user_default(value)

    return combined


def dumps(
    obj: Any,
    *,
    default: _Default | None = None,
    serialize_bytes: bool = False,
    **kwargs: Any,
) -> str:
    """Serialize *obj* to a JSON-formatted ``str``."""

    if serialize_bytes:
        return _json.dumps(obj, default=_build_default(default, True), **kwargs)

    return _ujson.dumps(obj, default=_build_default(default, False), **kwargs)


def dump(
    obj: Any,
    fp,
    *,
    default: _Default | None = None,
    serialize_bytes: bool = False,
    **kwargs: Any,
) -> None:
    """Serialize *obj* as a JSON formatted stream to *fp*."""

    if serialize_bytes:
        _json.dump(obj, fp, default=_build_default(default, True), **kwargs)
        return

    _ujson.dump(obj, fp, default=_build_default(default, False), **kwargs)


def loads(s: str | bytes | bytearray, **kwargs: Any) -> Any:
    """Deserialize *s* to a Python object."""

    return _ujson.loads(s, **kwargs)


def load(fp, **kwargs: Any) -> Any:
    """Deserialize JSON content from *fp* to a Python object."""

    return loads(fp.read(), **kwargs)
