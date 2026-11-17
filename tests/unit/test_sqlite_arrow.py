import sqlite3
import subprocess
import sys
from contextlib import closing

import pyarrow as pa
import pytest
import sqlalchemy as sa

from datachain.data_storage.codec import RowEncodeError
from datachain.data_storage.sqlite import _HAS_ADBC, SQLiteWarehouse
from datachain.data_storage.sqlite_arrow import SQLiteArrowWriter
from datachain.data_storage.writer import MAX_SEQ
from datachain.sql.types import Array, DateTime, Int, String
from tests.utils import skip_if_not_sqlite

pytestmark = [
    skip_if_not_sqlite,
    pytest.mark.skipif(not _HAS_ADBC, reason="adbc-driver-sqlite is not installed"),
]


@pytest.fixture(autouse=True)
def lower_arrow_writer_threshold(monkeypatch):
    monkeypatch.setattr(SQLiteWarehouse, "ARROW_WRITER_THRESHOLD", 1024)


def test_one_worker_commits_batches_without_staging(warehouse_tmpfile, mocker):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("native_direct", [sa.Column("x", Int)])
    started = mocker.spy(SQLiteArrowWriter, "__init__")
    try:
        with warehouse.writer(table, ["x"]) as writer:
            writer.append_arrow(pa.record_batch([list(range(2048))], names=["x"]))
            writer.append_arrow(pa.record_batch([[30]], names=["x"]))
            # Acknowledged rows are already visible from the parent's connection.
            assert warehouse.table_rows_count(table) == 2049
        assert started.call_count == 1
        process = started.call_args.args[0].process
        assert process.returncode == 0
        assert list(warehouse.db.execute(sa.select(table.c.sys__id, table.c.x))) == [
            *[(i + 1, i) for i in range(2048)],
            (2049, 30),
        ]
        assert warehouse.db.execute_str("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        warehouse.db.drop_table(table)


def test_worker_rejects_a_stale_owner_before_inserting(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("stale_owner", [sa.Column("x", Int)])
    batch = pa.record_batch([[1], [10]], names=["sys__id", "x"])
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_stream(buffer, batch.schema) as stream:
        stream.write_batch(batch)
    data = buffer.getvalue().to_pybytes()
    try:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "datachain.data_storage.sqlite_arrow",
                warehouse.db.db_file,
                table.name,
                "-1",
            ],
            input=data,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert result.stdout == b""  # no acknowledgement of an uncommitted batch
        assert warehouse.table_rows_count(table) == 0
    finally:
        warehouse.db.drop_table(table)


def test_worker_survives_an_automatic_sqlite_rollback(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("auto_rollback", [sa.Column("x", Int)])
    warehouse.db.execute_str(
        "CREATE TRIGGER fail_batch BEFORE INSERT ON auto_rollback WHEN NEW.x = 99 "
        "BEGIN SELECT RAISE(ROLLBACK, 'forced rollback'); END"
    )
    try:
        with warehouse.writer(table, ["x"]) as writer:
            writer.append_arrow(pa.record_batch([[10] * 2048], names=["x"]))
            with pytest.raises(sqlite3.Error, match="forced rollback"):
                writer.append_arrow(pa.record_batch([[20, 99]], names=["x"]))
            writer.append_arrow(pa.record_batch([[30]], names=["x"]))
        assert list(warehouse.db.execute(sa.select(table.c.x))) == [(10,)] * 2048 + [
            (30,)
        ]
    finally:
        warehouse.db.drop_table(table)


def test_small_batches_eventually_use_native_writer(warehouse_tmpfile, mocker):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("small_batches", [sa.Column("x", Int)])
    native = mocker.spy(SQLiteArrowWriter, "write")
    try:
        with warehouse.writer(table, ["x"]) as writer:
            for i in range(3):
                writer.append_arrow(pa.record_batch([[i] * 512], names=["x"]))
                assert native.call_count == i
        assert warehouse.table_rows_count(table) == 1536
    finally:
        warehouse.db.drop_table(table)


def test_dead_child_preserves_the_write_error(warehouse_tmpfile):
    with pytest.raises(sqlite3.OperationalError, match="exited before acknowledging"):
        with closing(
            SQLiteArrowWriter(warehouse_tmpfile.db.db_file, "unused")
        ) as writer:
            writer.process.kill()
            writer.process.wait()
            writer.write(pa.record_batch([[1]], names=["x"]))


def test_broken_worker_startup_falls_back_to_row_writer(warehouse_tmpfile, mocker):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("startup_fb", [sa.Column("x", Int)])
    mocker.patch.object(
        SQLiteArrowWriter,
        "write",
        side_effect=sqlite3.OperationalError(
            "Arrow writer exited before acknowledging its batch: dlopen failed"
        ),
    )
    try:
        with warehouse.writer(table, ["x"]) as writer:
            writer.append_arrow(pa.record_batch([list(range(1024))], names=["x"]))
            writer.append_arrow(pa.record_batch([[1024]], names=["x"]))
        assert warehouse.table_rows_count(table) == 1025
    finally:
        warehouse.db.drop_table(table)


def test_writer_exit_preserves_in_flight_exception_on_cleanup_failure(
    warehouse_tmpfile, mocker
):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("exc_keep", [sa.Column("x", Int)])
    close = SQLiteArrowWriter.close

    def fail_after_closing(self):
        close(self)
        raise RuntimeError("close boom")

    mocker.patch.object(SQLiteArrowWriter, "close", fail_after_closing)
    try:
        with pytest.raises(KeyboardInterrupt):
            with warehouse.writer(table, ["x"]) as writer:
                writer.append_arrow(pa.record_batch([[]], names=["x"]))
                writer.append_arrow(pa.record_batch([list(range(2048))], names=["x"]))
                raise KeyboardInterrupt
    finally:
        warehouse.db.drop_table(table)


def test_transaction_started_after_native_writer_opens_rolls_back(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("tx_after_open", [sa.Column("x", Int)])
    try:
        with warehouse.writer(table, ["x"]) as writer:
            writer.append_arrow(pa.record_batch([list(range(1024))], names=["x"]))
            with pytest.raises(RuntimeError, match="rollback tx"):
                with warehouse.db.transaction():
                    writer.append_arrow(pa.record_batch([[1024]], names=["x"]))
                    raise RuntimeError("rollback tx")
            assert warehouse.table_rows_count(table) == 1024
            with warehouse.db.transaction():
                writer.append_arrow(pa.record_batch([[1025]], names=["x"]))
            assert warehouse.table_rows_count(table) == 1025
    finally:
        warehouse.db.drop_table(table)


def test_encode_arrow_fills_missing_trailing_columns(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    batch = pa.record_batch([[1, 2]], names=["x"])

    rc_arrow = warehouse.row_codec([sa.Column("x", Int()), sa.Column("ts", DateTime())])
    enc_arrow = rc_arrow.encode_arrow(batch)
    assert isinstance(enc_arrow, pa.RecordBatch)
    assert enc_arrow.column(1).to_pylist() == [None, None]

    rc_tuples = warehouse.row_codec(
        [sa.Column("x", Int()), sa.Column("arr", Array(Int()))]
    )
    assert rc_tuples.encode_arrow(batch) == [(1, None), (2, None)]


def test_encode_arrow_falls_back_for_unsupported_or_out_of_range_values(
    warehouse_tmpfile,
):
    warehouse = warehouse_tmpfile
    dt_codec = warehouse.row_codec([sa.Column("ts", DateTime())])
    int_codec = warehouse.row_codec([sa.Column("x", Int())])

    # Naive timestamp before year 1 falls back to tuple path; aware timestamp clamps.
    far_past = pa.array([-70_000_000_000_000_000], type=pa.int64())
    with pytest.raises(OverflowError):
        dt_codec.encode_arrow(
            pa.record_batch([far_past.view(pa.timestamp("us"))], names=["ts"])
        )
    clamped = dt_codec.encode_arrow(
        pa.record_batch([far_past.view(pa.timestamp("us", "UTC"))], names=["ts"])
    )
    assert isinstance(clamped, pa.RecordBatch)
    assert clamped.column(0).to_pylist() == ["0001-01-01 00:00:00"]

    # Non-timestamp input for DateTime and non-scalar / overflowing uint64 for Int
    # fall back to the tuple path.
    with pytest.raises(RowEncodeError):
        dt_codec.encode_arrow(pa.record_batch([pa.array([1, 2])], names=["ts"]))
    with pytest.raises(RowEncodeError):
        int_codec.encode_arrow(
            pa.record_batch([pa.array([[1]], type=pa.list_(pa.int64()))], names=["x"])
        )
    assert int_codec.encode_arrow(
        pa.record_batch([pa.array([2**63], type=pa.uint64())], names=["x"])
    ) == [(2**63,)]


def test_encode_arrow_rejects_invalid_utf8_strings(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    str_codec = warehouse.row_codec([sa.Column("s", String())])
    invalid = pa.array([b"\xff"], type=pa.binary()).view(pa.string())
    with pytest.raises(UnicodeDecodeError):
        str_codec.encode_arrow(pa.record_batch([invalid], names=["s"]))


def test_append_arrow_raises_when_writer_runs_out_of_ids(warehouse_tmpfile):
    warehouse = warehouse_tmpfile
    table = warehouse.create_dataset_rows_table("overflow_id", [sa.Column("x", Int)])
    try:
        with warehouse.writer(table, ["x"]) as writer:
            writer.next_id = MAX_SEQ + 1
            with pytest.raises(RuntimeError, match="ran out of ids"):
                writer.append_arrow(pa.record_batch([[1]], names=["x"]))
    finally:
        warehouse.db.drop_table(table)
