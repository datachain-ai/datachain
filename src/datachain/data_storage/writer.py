"""Writer: the only way UDF output rows enter a table.

Ids are ``writer_no << ID_BITS | seq`` from 1; rows that carry their sys__id (map)
keep it. A gen/agg table's ledger ``<table>_parts`` holds a claim row per writer and
``(part_key, writer_no, lo, hi)`` per finished part, written with or after the
part's rows, so a writer's committed rows are its ids up to ``max(hi)``.
"""

import logging
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pyarrow as pa
import sqlalchemy as sa

from datachain.data_storage.schema import committed_part_keys, parts_table_name
from datachain.sql.types import Int64

if TYPE_CHECKING:
    from datachain.data_storage.warehouse import AbstractWarehouse

logger = logging.getLogger("datachain")

ID_BITS = 40
MAX_SEQ = (1 << ID_BITS) - 1
# SQLite ids are signed 64-bit; the top writer_no is reserved for Studio
MAX_WRITER_NO = (1 << 23) - 2
MAX_ID = (1 << 63) - 1
LEDGER_COLUMNS = ("part_key", "writer_no", "lo", "hi")


def parts_columns() -> list[sa.Column]:
    return [sa.Column(name, Int64) for name in LEDGER_COLUMNS]


class Writer:
    def __init__(
        self,
        warehouse: "AbstractWarehouse",
        table: sa.Table,
        columns: Iterable[str],
        writer_no: int = 0,
        parts: sa.Table | None = None,
        batch_size: int | None = None,
    ) -> None:
        if not 0 <= writer_no <= MAX_WRITER_NO:
            raise ValueError(f"writer_no {writer_no} is out of range")
        self.warehouse = warehouse
        self.table = table
        self.columns = list(columns)
        self.assign_ids = "sys__id" not in self.columns
        if self.assign_ids:
            self.columns.append("sys__id")
        self.writer_no = writer_no
        self.parts = parts
        self.batch_size = batch_size or warehouse.INSERT_BATCH_SIZE
        self.flush_interval = warehouse.INSERT_FLUSH_INTERVAL
        self.rows: list[tuple] = []
        self.ledger: list[tuple] = []
        self.next_id = (writer_no << ID_BITS) + 1  # id of self.rows[0]
        self.part_lo = self.next_id
        self.flush_at = time.monotonic() + self.flush_interval

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.flush()
        except Exception:
            if exc_type is None:
                raise
            # the error in flight says more than this one
            logger.exception("Could not flush rows into %s", self.table)

    def _reserve_ids(self, count: int) -> int:
        hi = self.next_id + count
        if self.assign_ids and hi - 1 - (self.writer_no << ID_BITS) > MAX_SEQ:
            raise RuntimeError(f"Writer {self.writer_no} ran out of ids")
        return hi

    def extend(self, rows: "Iterable[tuple | pa.RecordBatch]") -> None:
        buf, size = self.rows, self.batch_size
        for row in rows:
            if type(row) is not tuple:
                self.append_arrow(row)
                continue
            buf.append(row)
            if len(buf) >= size or time.monotonic() >= self.flush_at:
                self.flush()

    def append_arrow(self, batch: "pa.RecordBatch") -> None:
        """Append stored Arrow columns, assigning ids without Python row objects.
        Source constants, when present, are already columns in the batch."""
        if self.rows:
            self.flush()
        if not batch.num_rows:
            return
        hi = self._reserve_ids(batch.num_rows)
        columns = list(batch.columns)
        if self.assign_ids:
            columns.append(pa.array(np.arange(self.next_id, hi, dtype=np.int64)))
        batch = pa.RecordBatch.from_arrays(columns, names=self.columns)
        self.warehouse.insert_arrow(self.table, batch)
        self.next_id = hi

    def commit(self, part_key: Any) -> None:
        """The part's input is exhausted: its rows, possibly none, are final."""
        hi = self.next_id + len(self.rows) - 1
        if self.parts is not None:
            self.ledger.append((part_key, self.writer_no, self.part_lo, hi))
        self.part_lo = hi + 1
        if len(self.ledger) >= self.batch_size or time.monotonic() >= self.flush_at:
            self.flush()

    def flush(self) -> None:
        rows = self.rows
        block: list[tuple] = rows
        if rows and self.assign_ids:
            hi = self._reserve_ids(len(rows))
            # a list, not an iterator: a retry after a lock error re-reads it
            block = list(map(tuple.__add__, rows, zip(range(self.next_id, hi))))
        if rows or self.ledger:
            self.warehouse.insert_block(
                self.table, self.columns, block, self.parts, self.ledger
            )
        self.next_id += len(rows)
        rows.clear()
        self.ledger = []
        self.flush_at = time.monotonic() + self.flush_interval


__all__ = [
    "ID_BITS",
    "LEDGER_COLUMNS",
    "MAX_ID",
    "MAX_SEQ",
    "MAX_WRITER_NO",
    "Writer",
    "committed_part_keys",
    "parts_columns",
    "parts_table_name",
]
