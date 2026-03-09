from collections.abc import Iterator
from unittest.mock import patch

import pytest

import datachain as dc
from datachain.query.dataset import process_udf_outputs


def test_process_udf_outputs_signature_accepts_none():
    """Test that process_udf_outputs signature accepts None for batch_size."""
    import inspect

    sig = inspect.signature(process_udf_outputs)
    params = sig.parameters

    assert "batch_size" in params
    # Check that batch_size parameter has default value of None
    assert params["batch_size"].default is None


def test_map_with_default_batch_size(test_session, warehouse):
    """Test map UDF with default (None) batch_size."""

    def simple_udf(value: int) -> int:
        return value * 2

    with patch.object(
        warehouse,
        attribute="insert_rows",
        wraps=warehouse.insert_rows,
    ) as mock_insert_rows:
        dc.read_values(value=list(range(100)), session=test_session).map(
            result=simple_udf
        ).save("map_default")

        assert mock_insert_rows.call_count >= 1


def test_map_with_custom_batch_size(test_session, warehouse):
    """Test map UDF with custom batch_size."""

    def simple_udf(value: int) -> int:
        return value * 2

    with patch.object(
        warehouse.db,
        attribute="executemany",
        wraps=warehouse.db.executemany,
    ) as mock_executemany:
        chain = (
            dc.read_values(value=list(range(100)), session=test_session)
            .settings(batch_size=25)
            .map(result=simple_udf)
            .save("map_custom")
        )

        # Should have multiple calls due to batching
        assert mock_executemany.call_count >= 4

        # Verify the results are correct
        results = chain.to_values("result")
        assert set(results) == set(range(0, 200, 2))


def test_gen_with_default_batch_size(test_session, warehouse):
    """Test generator UDF with default batch_size."""

    def gen_udf(value: int) -> Iterator[int]:
        yield value
        yield value + 100

    with patch.object(
        warehouse,
        attribute="insert_rows",
        wraps=warehouse.insert_rows,
    ) as mock_insert_rows:
        dc.read_values(value=list(range(50)), session=test_session).gen(
            result=gen_udf
        ).save("gen_default")

        assert mock_insert_rows.call_count >= 1


def test_gen_with_custom_batch_size(test_session, warehouse):
    """Test generator UDF with custom batch_size."""

    def gen_udf(value: int) -> Iterator[int]:
        yield value
        yield value + 100

    with patch.object(
        warehouse.db,
        attribute="executemany",
        wraps=warehouse.db.executemany,
    ) as mock_executemany:
        chain = (
            dc.read_values(value=list(range(50)), session=test_session)
            .settings(batch_size=10)
            .gen(result=gen_udf)
            .save("gen_custom")
        )

        # Should have multiple calls due to batching
        assert mock_executemany.call_count >= 5

        # Verify the results are correct (should have 100 results: 50 * 2)
        results = list(chain.to_values("result"))
        assert len(results) == 100


def test_batch_settings_are_preserved(test_session):
    """Test that batch_size settings are preserved through operations."""

    def simple_udf(value: int) -> int:
        return value * 2

    chain = (
        dc.read_values(value=list(range(100)), session=test_session)
        .settings(batch_size=20)
        .map(result=simple_udf)
        .save("batch_settings")
    )

    # Verify results
    results = list(chain.to_values("result"))
    assert len(results) == 100
    assert set(results) == set(range(0, 200, 2))


def test_different_batch_sizes_in_sequence(test_session):
    """Test chaining operations with different batch sizes."""

    def udf1(value: int) -> int:
        return value * 2

    def udf2(result: int) -> int:
        return result + 10

    # First operation with batch_size=10
    (
        dc.read_values(value=list(range(50)), session=test_session)
        .settings(batch_size=10)
        .map(result=udf1)
        .save("step1")
    )

    # Second operation with batch_size=25
    chain2 = (
        dc.read_dataset("step1", session=test_session)
        .settings(batch_size=25)
        .map(final=udf2)
        .save("step2")
    )

    # Verify results
    results = list(chain2.to_values("final"))
    expected = [i * 2 + 10 for i in range(50)]
    assert set(results) == set(expected)


def test_invalid_batch_size_type(test_session):
    """Test that invalid batch_size types are handled appropriately."""
    from datachain.lib.settings import SettingsError

    def simple_udf(value: int) -> int:
        return value

    # This should raise an error or be caught during validation
    with pytest.raises(SettingsError):
        (
            dc.read_values(value=list(range(10)), session=test_session)
            .settings(batch_size="invalid")  # type: ignore[arg-type]
            .map(result=simple_udf)
            .save("invalid_batch")
        )


def test_negative_batch_size(test_session):
    """Test that negative batch_size is handled appropriately."""
    from datachain.lib.settings import SettingsError

    def simple_udf(value: int) -> int:
        return value

    # This should raise an error
    with pytest.raises(SettingsError):
        (
            dc.read_values(value=list(range(10)), session=test_session)
            .settings(batch_size=-1)
            .map(result=simple_udf)
            .save("negative_batch")
        )


def test_zero_batch_size(test_session):
    """Test that zero batch_size is handled appropriately."""
    from datachain.lib.settings import SettingsError

    def simple_udf(value: int) -> int:
        return value

    # This should raise an error
    with pytest.raises(SettingsError):
        (
            dc.read_values(value=list(range(10)), session=test_session)
            .settings(batch_size=0)
            .map(result=simple_udf)
            .save("zero_batch")
        )


def test_small_batch_size_memory_pattern(test_session, warehouse):
    """Test that small batch_size leads to more frequent inserts."""

    def simple_udf(value: int) -> int:
        return value

    with patch.object(
        warehouse.db,
        attribute="executemany",
        wraps=warehouse.db.executemany,
    ) as mock_executemany:
        dc.read_values(value=list(range(50)), session=test_session).settings(
            batch_size=5
        ).map(result=simple_udf).save("small_memory")

        assert mock_executemany.call_count >= 10


def test_large_batch_size_memory_pattern(test_session, warehouse):
    """Test that large batch_size leads to fewer inserts."""

    def simple_udf(value: int) -> int:
        return value

    with patch.object(
        warehouse.db,
        attribute="executemany",
        wraps=warehouse.db.executemany,
    ) as mock_executemany:
        dc.read_values(value=list(range(50)), session=test_session).settings(
            batch_size=50
        ).map(result=simple_udf).save("large_memory")

        assert mock_executemany.call_count >= 1
