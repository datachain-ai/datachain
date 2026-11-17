"""A native SQLite writer isolated from the parent's sqlite3 library.

One subprocess owns one ADBC connection for the Writer's lifetime. Arrow IPC
travels over pipes; the only database written is the destination database.
Each acknowledgement follows a committed batch, before the parent can publish
its part ledger. EOF closes the worker, including when the parent is killed.
"""

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa


class SQLiteArrowWriter:
    def __init__(self, path: str, table: str):
        self.process = subprocess.Popen(  # noqa: S603
            [sys.executable, "-P", "-m", __name__, path, table, str(os.getpid())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        self.stream: pa.ipc.RecordBatchStreamWriter | None = None
        self.ready = False

    def write(self, batch: "pa.RecordBatch") -> None:
        import pyarrow as pa

        process = self.process
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        try:
            if self.stream is None:
                self.stream = pa.ipc.new_stream(process.stdin, batch.schema)
            self.stream.write_batch(batch)
            process.stdin.flush()
            response = process.stdout.readline()
        except BrokenPipeError:
            response = b""
        if not response:
            raise sqlite3.OperationalError(
                "Arrow writer exited before acknowledging its batch: "
                + process.stderr.read().decode()
            ) from None
        if error := json.loads(response):
            exc = (
                sqlite3.IntegrityError
                if "constraint failed" in error
                else sqlite3.OperationalError
            )
            raise exc(error)
        self.ready = True

    def close(self) -> None:
        process = self.process
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        try:
            with suppress(OSError):
                if self.stream is not None:
                    self.stream.close()
                process.stdin.close()
            process.wait(timeout=60)
        finally:
            if process.poll() is None:  # pragma: no cover
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()


def _run(path: str, table: str, parent: str) -> None:  # pragma: no cover
    # Executed in a child process (`python -P -m datachain.data_storage.sqlite_arrow`).
    import adbc_driver_sqlite.dbapi as adbc
    import pyarrow as pa

    with adbc.connect(path, autocommit=True) as connection, connection.cursor() as cur:
        cur.execute("PRAGMA busy_timeout=60000")
        cur.execute("PRAGMA synchronous=NORMAL")
        for batch in pa.ipc.open_stream(sys.stdin.buffer):
            error = None
            try:
                cur.execute("BEGIN IMMEDIATE")
                # Take the lock before checking: an orphan's last commit must
                # precede a rerun's cleanup, never arrive after it.
                if os.getppid() != int(parent):
                    cur.execute("ROLLBACK")
                    return
                cur.adbc_ingest(table, batch, mode="append")
                cur.execute("COMMIT")
            except adbc.Error as exc:
                # SQLite can already have rolled back after an I/O/full error.
                with suppress(adbc.Error):
                    cur.execute("ROLLBACK")
                error = str(exc)
            print(json.dumps(error), flush=True)


if __name__ == "__main__":
    _run(*sys.argv[1:])
