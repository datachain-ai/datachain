"""Unified deprecation and warning helpers for DataChain."""

from __future__ import annotations

import threading
import warnings

# Thread-safe set to track warnings emitted with once=True
_WARNED: set[tuple[str, type[Warning]]] = set()
_WARNED_LOCK = threading.Lock()


def warn_deprecated(
    what: str,
    *,
    instead: str | None = None,
    removal: str | None = None,
    once: bool = False,
    category: type[Warning] | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a standardized deprecation warning.

    Deprecation Policy & Categories:
        - Default category is `FutureWarning` (visible by default in Python).
          Used for public APIs, CLI commands, and serialized data formats
          that end-users need to act on before removal.
        - `DeprecationWarning` (ignored by default outside `__main__` in Python)
          should be passed explicitly for internal developer-facing deprecations.
        - Deprecated features should be scheduled for removal no sooner than
          two minor versions (or one major release) after the warning was introduced.

    Message Format:
        Single-line standardized shape:
        - "X is deprecated; use Y instead."
        - "X is deprecated and will be removed in Z; use Y instead."
        - "X is deprecated and will be removed in Z."
        - "X is deprecated."

    Args:
        what: The deprecated feature, class, function, or data format.
        instead: The suggested replacement or alternative.
        removal: Scheduled removal milestone (e.g. version or release).
        once: If True, suppress duplicate emissions of this warning within
            the same process (useful for hot paths like row readers).
        category: Warning category class. Defaults to `FutureWarning`.
        stacklevel: Stack level for `warnings.warn`. Defaults to 2.
    """
    if category is None:
        category = FutureWarning

    parts = [f"{what} is deprecated"]
    if removal:
        parts.append(f" and will be removed in {removal}")
    if instead:
        parts.append(f"; use {instead} instead.")
    else:
        parts.append(".")

    msg = "".join(parts)

    if once:
        key = (msg, category)
        with _WARNED_LOCK:
            if key in _WARNED:
                return
            _WARNED.add(key)

    warnings.warn(msg, category, stacklevel=stacklevel)


def _reset_warned() -> None:
    """Reset the deduplication cache (primarily for test isolation)."""
    with _WARNED_LOCK:
        _WARNED.clear()


__all__ = ["warn_deprecated"]
