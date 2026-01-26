"""Buffer utilities for batch database operations."""

import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    import sqlalchemy as sa

T = TypeVar("T")


class InsertBuffer(Generic[T]):
    """
    Inserts groups of entries in blocks rather than one or a few at a time.

    Supports two flushing strategies:
    1. Size-based: Flush when buffer reaches buffer_size
    2. Time-based: Flush when flush_interval seconds have elapsed since last flush
    """

    def __init__(
        self,
        table: "sa.Table",
        execute_callback: Callable[..., None],
        buffer_size: int,
        flush_interval: float | None = None,
        cursor: Any = None,
    ) -> None:
        """
        Initialize the insert buffer.

        Args:
            table: SQLAlchemy table to insert into
            execute_callback: Callback function to execute inserts.
                Signature: (table, entries, final=False, cursor=None) -> None
            buffer_size: Number of entries to accumulate before flushing
            flush_interval: Optional time in seconds between flushes.
                If set, buffer will flush when this interval elapses,
                even if buffer_size hasn't been reached.
            cursor: Optional database cursor for the callback
        """
        self.table = table
        self.execute_callback = execute_callback
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer: list[dict[str, Any]] = []
        self.cursor = cursor
        self._last_flush_time: float = time.monotonic()

    def insert(self, entry: dict[str, Any]) -> None:
        self.buffer.append(entry)
        self._process_blocks()

    def insert_many(self, entries: Iterable[dict[str, Any]]) -> None:
        """Add many entries to the insert buffer (lazy iteration)."""
        for entry in entries:
            self.buffer.append(entry)
            # Process blocks when buffer is full to maintain lazy evaluation
            if len(self.buffer) >= self.buffer_size:
                self._process_blocks()

    def _should_flush_by_time(self) -> bool:
        if self.flush_interval is None:
            return False
        elapsed = time.monotonic() - self._last_flush_time
        return elapsed >= self.flush_interval

    def _do_flush(self, entries: list[dict[str, Any]], final: bool = False) -> None:
        if not entries:
            return
        self.execute_callback(self.table, entries, final=final, cursor=self.cursor)
        self._last_flush_time = time.monotonic()

    def _process_blocks(self) -> None:
        # Size-based flushing: flush full blocks
        while len(self.buffer) >= self.buffer_size:
            self._do_flush(self.buffer[: self.buffer_size])
            self.buffer = self.buffer[self.buffer_size :]

        # Time-based flushing: flush whatever we have if interval elapsed
        if self.buffer and self._should_flush_by_time():
            self._do_flush(self.buffer)
            self.buffer = []

    def flush(self) -> None:
        self._process_blocks()
        if self.buffer:
            # Process any remaining entries
            self._do_flush(self.buffer, final=True)
            self.buffer = []

    def close(self) -> None:
        if self.cursor:
            self.cursor.close()

    def __len__(self) -> int:
        return len(self.buffer)
