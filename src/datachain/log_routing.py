"""Route framework (datachain/datachain_saas/compute) logs to a Studio-provided fd.

Inside a SaaS job, framework log records go to the fd from
DATACHAIN_INTERNAL_LOG_FD as one JSON line each (message, time, level, logger)
instead of the captured stdout/stderr, so the parent worker can tell framework
logs from user output.
"""

import json
import logging
import os
from contextlib import suppress
from datetime import datetime, timezone
from io import TextIOWrapper

INTERNAL_LOG_FD_ENV = "DATACHAIN_INTERNAL_LOG_FD"
INTERNAL_LOG_LEVEL_ENV = "DATACHAIN_INTERNAL_LOG_LEVEL"

_FRAMEWORK_LOGGERS = ("datachain", "datachain_saas", "compute")

_routed = False


def setup_internal_log_routing() -> bool:
    """Route framework loggers to the parent fd; True if routing is active."""
    global _routed  # noqa: PLW0603
    if _routed:
        return True

    fd = _fd_from_env()
    if fd is None:
        return False

    try:
        pipe = os.fdopen(fd, "w", buffering=1)
    except OSError:
        return False

    level = logging.getLevelName(os.getenv(INTERNAL_LOG_LEVEL_ENV, "INFO").upper())
    if not isinstance(level, int):
        level = logging.INFO

    handler = _FdJsonHandler(pipe)
    for name in _FRAMEWORK_LOGGERS:
        framework_logger = logging.getLogger(name)
        framework_logger.handlers[:] = [handler]
        framework_logger.setLevel(level)
        framework_logger.propagate = False
    _routed = True

    return True


def internal_log_fds() -> tuple[int, ...]:
    """Return the fds that framework logs are routed to, or empty if none."""
    fd = _fd_from_env()
    if fd is None:
        return ()

    try:
        os.fstat(fd)
    except OSError:
        return ()

    return (fd,)


def _fd_from_env() -> int | None:
    fd_raw = os.getenv(INTERNAL_LOG_FD_ENV)
    if not fd_raw:
        return None

    try:
        return int(fd_raw)
    except ValueError:
        return None


class _FdJsonHandler(logging.Handler):
    stream: TextIOWrapper | None

    def __init__(self, stream: TextIOWrapper) -> None:
        super().__init__()
        self.stream = stream

    def close(self) -> None:
        if self.stream is not None:
            with suppress(OSError, ValueError):
                self.stream.close()
            self.stream = None
        super().close()

    def emit(self, record: logging.LogRecord) -> None:
        if self.stream is None:
            return

        try:
            entry = {
                "message": self.format(record),
                "time": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname.lower(),
                "logger": record.name,
            }
            self.stream.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            # Dead fd: disable quietly; handleError would dump a traceback
            # into the captured stream (the job log) for every record.
            with suppress(OSError, ValueError):
                self.stream.close()
            self.stream = None
        except Exception:  # noqa: BLE001
            self.handleError(record)
