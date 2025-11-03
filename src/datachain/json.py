"""DataChain JSON utilities.

This module wraps :mod:`ujson` so we can guarantee consistent handling
of values that the encoder does not support out of the box (for example
``datetime`` objects or ``bytes``).
All code inside DataChain should import this module instead of using
:mod:`ujson` directly.
"""

import datetime as _dt
import uuid as _uuid
from collections.abc import Callable
from typing import Any

import ujson as _ujson

__all__ = [
    "DEFAULT_PREVIEW_BYTES",
    "JSONDecodeError",
    "default",
    "dump",
    "dumps",
    "load",
    "loads",
]

JSONDecodeError = _ujson.JSONDecodeError

_SENTINEL = object()
_Default = Callable[[Any], Any]
DEFAULT_PREVIEW_BYTES = 1024


def _coerce(value: Any, preview_bytes: int | None) -> Any:
    """Return a JSON-serializable representation for supported extra types."""

    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        # ``datetime`` family classes expose ``isoformat`` with timezone support
        # when available, so no additional handling is required here.
        return value.isoformat()
    if isinstance(value, _uuid.UUID):
        return str(value)
    if preview_bytes is not None and isinstance(value, (bytes, bytearray)):
        return list(bytes(value)[:preview_bytes])
    return _SENTINEL


def _base_default(value: Any, preview_bytes: int | None) -> Any:
    converted = _coerce(value, preview_bytes)
    if converted is not _SENTINEL:
        return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _build_default(
    user_default: _Default | None, preview_bytes: int | None
) -> _Default:
    if user_default is None:
        return lambda value: _base_default(value, preview_bytes)

    def combined(value: Any) -> Any:
        converted = _coerce(value, preview_bytes)
        if converted is not _SENTINEL:
            return converted
        return user_default(value)

    return combined


def default(value: Any, *, preview_bytes: int | None = None) -> Any:
    """Return DataChain's default JSON representation for *value*.

    This mirrors the behavior applied automatically by :func:`dumps` and
    raises :class:`TypeError` when *value* cannot be serialized.
    """

    return _base_default(value, preview_bytes)


def dumps(
    obj: Any,
    *,
    default: _Default | None = None,
    preview_bytes: int | None = None,
    **kwargs: Any,
) -> str:
    """Serialize *obj* to a JSON-formatted ``str``.

    The default handler automatically converts :class:`datetime.datetime`,
    :class:`datetime.date`, and :class:`datetime.time` instances to ISO 8601
    strings.  Pass ``default=...`` to add additional behavior; it is only
    called for objects that are not handled by DataChain's default rules.
    """

    return _ujson.dumps(obj, default=_build_default(default, preview_bytes), **kwargs)


def dump(
    obj: Any,
    fp,
    *,
    default: _Default | None = None,
    preview_bytes: int | None = None,
    **kwargs: Any,
) -> None:
    """Serialize *obj* as a JSON formatted stream to *fp*."""

    _ujson.dump(obj, fp, default=_build_default(default, preview_bytes), **kwargs)


def loads(s: str | bytes | bytearray, **kwargs: Any) -> Any:
    """Deserialize *s* to a Python object."""

    return _ujson.loads(s, **kwargs)


def load(fp, **kwargs: Any) -> Any:
    """Deserialize JSON content from *fp* to a Python object."""

    return loads(fp.read(), **kwargs)
