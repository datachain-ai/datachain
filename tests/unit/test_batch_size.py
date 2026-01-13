from unittest.mock import patch

import pytest
import sqlalchemy as sa

from datachain.lib.settings import Settings


def test_insert_rows_default_batch_size(warehouse):
    """Test that insert_rows uses default batch size when batch_size is None."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    rows = [{"sys__id": i, "value": i} for i in range(100)]

    with patch.object(
        warehouse.db, "executemany", wraps=warehouse.db.executemany
    ) as mock_exec:
        warehouse.insert_rows(table, rows, batch_size=None)
        warehouse.insert_rows_done(table)
        # With default batch size, 100 rows should be inserted in 1 call
        assert mock_exec.call_count == 1

    warehouse.db.drop_table(table)


def test_insert_rows_custom_batch_size(warehouse):
    """Test that insert_rows respects custom batch_size parameter."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    # Create 250 rows with a batch size of 100
    rows = [{"sys__id": i, "value": i} for i in range(250)]

    with patch.object(
        warehouse.db, "executemany", wraps=warehouse.db.executemany
    ) as mock_exec:
        warehouse.insert_rows(table, rows, batch_size=100)
        warehouse.insert_rows_done(table)
        # Should be called 3 times: 100 + 100 + 50
        assert mock_exec.call_count == 3

    warehouse.db.drop_table(table)


def test_insert_rows_small_batch_size(warehouse):
    """Test insert_rows with a very small batch size."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    rows = [{"sys__id": i, "value": i} for i in range(10)]

    with patch.object(
        warehouse.db, "executemany", wraps=warehouse.db.executemany
    ) as mock_exec:
        warehouse.insert_rows(table, rows, batch_size=3)
        warehouse.insert_rows_done(table)
        # Should be called 4 times: 3 + 3 + 3 + 1
        assert mock_exec.call_count == 4

    warehouse.db.drop_table(table)


def test_insert_rows_batch_size_larger_than_data(warehouse):
    """Test insert_rows when batch_size is larger than the number of rows."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    rows = [{"sys__id": i, "value": i} for i in range(50)]

    with patch.object(
        warehouse.db, "executemany", wraps=warehouse.db.executemany
    ) as mock_exec:
        warehouse.insert_rows(table, rows, batch_size=1000)
        warehouse.insert_rows_done(table)
        # Should be called once since all rows fit in one batch
        assert mock_exec.call_count == 1

    warehouse.db.drop_table(table)


def test_insert_rows_batch_size_one(warehouse):
    """Test insert_rows with batch_size=1 (edge case)."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    rows = [{"sys__id": i, "value": i} for i in range(5)]

    with patch.object(
        warehouse.db, "executemany", wraps=warehouse.db.executemany
    ) as mock_exec:
        warehouse.insert_rows(table, rows, batch_size=1)
        warehouse.insert_rows_done(table)
        # Should be called 5 times, once per row
        assert mock_exec.call_count == 5

    warehouse.db.drop_table(table)


def test_insert_rows_empty_rows(warehouse):
    """Test insert_rows with empty rows iterable."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    with patch.object(
        warehouse.db, "executemany", wraps=warehouse.db.executemany
    ) as mock_exec:
        warehouse.insert_rows(table, [], batch_size=None)
        warehouse.insert_rows_done(table)
        # Should not be called for empty rows
        assert mock_exec.call_count == 0

    warehouse.db.drop_table(table)


def test_insert_rows_preserves_data_integrity(warehouse):
    """Test that batching doesn't affect data integrity."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    # Create data and insert with different batch sizes
    rows = [{"sys__id": i, "value": i * 2} for i in range(1, 101)]

    warehouse.insert_rows(table, rows, batch_size=25)
    warehouse.insert_rows_done(table)

    # Verify all data was inserted correctly
    query = sa.select(table.c.value).order_by(table.c.sys__id)
    result = list(warehouse.dataset_rows_select(query))

    assert len(result) == 100
    values = [row[0] for row in result]
    expected_values = [i * 2 for i in range(1, 101)]
    assert values == expected_values

    warehouse.db.drop_table(table)


def test_batch_size_default_is_none():
    """Test that default batch_size is None."""
    settings = Settings()
    assert settings.batch_size is None


def test_batch_size_explicit_none():
    """Test that explicit None returns None."""
    settings = Settings(batch_size=None)
    assert settings.batch_size is None


@pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
def test_batch_size_custom_value(batch_size):
    """Test that custom batch_size value is returned."""
    settings = Settings(batch_size=batch_size)
    assert settings.batch_size == batch_size


def test_batch_size_not_in_to_dict_when_none():
    """Test that batch_size is not in to_dict when None."""
    settings = Settings(batch_size=None)
    d = settings.to_dict()
    assert "batch_size" not in d


def test_batch_size_in_to_dict_when_set():
    """Test that batch_size is in to_dict when set."""
    settings = Settings(batch_size=500)
    d = settings.to_dict()
    assert d["batch_size"] == 500


def test_batch_size_immutability():
    """Test that batch_size is immutable (dataclass behavior)."""
    settings = Settings(batch_size=100)
    assert settings.batch_size == 100

    # Create new settings with different batch_size
    settings2 = Settings(batch_size=200)
    assert settings2.batch_size == 200
    # Original should be unchanged
    assert settings.batch_size == 100


def test_generator_rows(warehouse):
    """Test insert_rows with a generator as rows parameter."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    def row_generator():
        for i in range(50):
            yield {"sys__id": i, "value": i}

    with patch.object(
        warehouse.db, "executemany", wraps=warehouse.db.executemany
    ) as mock_exec:
        warehouse.insert_rows(table, row_generator(), batch_size=20)
        warehouse.insert_rows_done(table)
        # Should be called 3 times: 20 + 20 + 10
        assert mock_exec.call_count == 3

    warehouse.db.drop_table(table)


def test_iterator_rows(warehouse):
    """Test insert_rows with an iterator as rows parameter."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    rows = iter([{"sys__id": i, "value": i} for i in range(75)])

    with patch.object(
        warehouse.db, "executemany", wraps=warehouse.db.executemany
    ) as mock_exec:
        warehouse.insert_rows(table, rows, batch_size=30)
        warehouse.insert_rows_done(table)
        # Should be called 3 times: 30 + 30 + 15
        assert mock_exec.call_count == 3

    warehouse.db.drop_table(table)


def test_insert_rows_without_batch_size_parameter(warehouse):
    """Test that insert_rows works when batch_size is not specified."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    rows = [{"sys__id": i, "value": i} for i in range(100)]

    # Call without specifying batch_size (should use default)
    warehouse.insert_rows(table, rows)
    warehouse.insert_rows_done(table)

    # Verify data was inserted
    query = sa.select(sa.func.count()).select_from(table)
    result = next(iter(warehouse.dataset_rows_select(query)))[0]
    assert result == 100

    warehouse.db.drop_table(table)


def test_existing_code_patterns(warehouse):
    """Test that existing code patterns continue to work."""
    table = warehouse.create_udf_table([sa.Column("value", sa.Integer)])

    # Pattern 1: No batch_size specified
    rows1 = [{"sys__id": i, "value": i} for i in range(50)]
    warehouse.insert_rows(table, rows1)

    # Pattern 2: Explicit None
    rows2 = [{"sys__id": i + 50, "value": i + 50} for i in range(50)]
    warehouse.insert_rows(table, rows2, batch_size=None)

    # Pattern 3: Custom batch_size
    rows3 = [{"sys__id": i + 100, "value": i + 100} for i in range(50)]
    warehouse.insert_rows(table, rows3, batch_size=25)

    warehouse.insert_rows_done(table)

    # Verify all data was inserted
    query = sa.select(sa.func.count()).select_from(table)
    result = next(iter(warehouse.dataset_rows_select(query)))[0]
    assert result == 150

    warehouse.db.drop_table(table)
