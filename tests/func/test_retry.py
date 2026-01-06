from collections.abc import Iterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

import pytest

import datachain as dc
from datachain import C, DataModel
from datachain.error import DatasetNotFoundError

if TYPE_CHECKING:
    from datachain import DataChain


class ProcessingResult(DataModel):
    """Data model for processing results in retry tests."""

    processed_content: str
    processed_at: str
    error: str
    attempt: int


def _process_with_errors(id: int, content: str, attempt: int) -> ProcessingResult:
    error = id % 2 == 1 and attempt == 1
    return ProcessingResult(
        processed_content="" if error else content.upper(),
        processed_at=datetime.now(tz=timezone.utc).isoformat(),
        error=f"Processing error for item {id}" if error else "",
        attempt=attempt,
    )


def _create_sample_data(test_session, ids=None, contents=None):
    """Helper function to create sample data for retry tests."""
    ids = ids or [1, 2, 3, 4]
    contents = contents or ["first item", "second item", "third item", "fourth item"]
    dc.read_values(id=ids, content=contents, session=test_session).save("sample_data")


def _simple_process(id: int, content: str, attempt: int = 1) -> ProcessingResult:
    """Helper function for simple processing in retry tests."""
    return ProcessingResult(
        processed_content=content.upper(),
        processed_at=datetime.now(tz=timezone.utc).isoformat(),
        error="",
        attempt=attempt,
    )


def test_retry_with_error_records(test_session):
    """Test retry functionality with records that have errors."""

    def _run_processing(attempt: int) -> "DataChain":
        return (
            dc.read_dataset(
                "sample_data",
                delta=True,
                delta_on="id",
                delta_retry="result.error",
                session=test_session,
            )
            .setup(attempt=lambda: attempt)
            .map(result=_process_with_errors)
            .save("processed_data")
        )

    # First processing pass - some records will fail
    _create_sample_data(test_session)
    first_pass = _run_processing(1)

    # Check that some records failed
    error_count = first_pass.filter(C("result.error") != "").count()
    success_count = first_pass.filter(C("result.error") == "").count()

    assert error_count == 2
    assert success_count == 2

    # Retry processing - only failed records should be reprocessed
    _run_processing(2)

    final_result = dc.read_dataset("processed_data", session=test_session)
    results = final_result.to_list("id", "result.attempt", "result.error")
    assert set(results) == {(2, 1, ""), (4, 1, ""), (1, 2, ""), (3, 2, "")}


def test_retry_with_missing_records(test_session):
    """Test retry functionality with missing records."""
    _create_sample_data(test_session)

    # Process only first 2 records
    # Create partial result dataset (missing id=3)
    partial_result = (
        dc.read_dataset("sample_data", session=test_session)
        .setup(attempt=lambda: 1)
        .filter(C("id") < 3)
        .map(result=_simple_process)
        .save("partial_result")
    )

    assert partial_result.count() == 2

    # Use retry with delta_retry=True to process missing records
    retry_chain = (
        dc.read_dataset(
            "sample_data",
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry=True,
        )
        .setup(attempt=lambda: 2)
        .map(result=_simple_process)
        .save("partial_result")
    )

    # Should now have all 4 records
    assert retry_chain.count() == 4

    # Verify all records are present
    ids = set(retry_chain.to_values("id"))
    assert ids == {1, 2, 3, 4}

    final_first_attempts_count = retry_chain.filter(C("result.attempt") == 1).count()
    final_missing_attempts_count = retry_chain.filter(C("result.attempt") == 2).count()

    # Only missing records should have attempt 2
    assert final_missing_attempts_count == 2
    assert final_first_attempts_count == 2


def test_retry_with_missing_and_new_records(test_session):
    """Test retry functionality with missing records (e.g. ignored
    in first pass since they failed). Also we add new records to the source
    to test that retry and delta don't pick records twice."""
    _create_sample_data(test_session)

    # Process only first 2 records
    # Create partial result dataset (missing id=3)
    partial_result = (
        dc.read_dataset("sample_data", session=test_session)
        .setup(attempt=lambda: 1)
        .filter(C("id") < 3)
        .map(result=_simple_process)
        .save("partial_result")
    )

    assert partial_result.count() == 2

    ids = [1, 2, 3, 4, 5]
    contents = ["first item", "second item", "third item", "fourth item", "fifth item"]
    _create_sample_data(test_session, ids, contents)

    # Use retry with delta_retry=True to process missing records
    retry_chain = (
        dc.read_dataset(
            "sample_data",
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry=True,
        )
        .setup(attempt=lambda: 2)
        .map(result=_simple_process)
        .save("partial_result")
    )

    # Should now have all 3 records
    assert retry_chain.count() == 5

    # Verify all records are present
    ids = set(retry_chain.to_values("id"))
    assert ids == {1, 2, 3, 4, 5}

    final_first_attempts_count = retry_chain.filter(C("result.attempt") == 1).count()
    final_missing_attempts_count = retry_chain.filter(C("result.attempt") == 2).count()

    # Only missing records should have attempt 2
    assert final_missing_attempts_count == 3
    assert final_first_attempts_count == 2


def test_retry_no_records_to_retry(test_session):
    """Test retry when no records need to be retried."""
    _create_sample_data(test_session, ids=[1, 2], contents=["first", "second"])

    def successful_process(id: int, content: str) -> ProcessingResult:
        return ProcessingResult(
            processed_content=content.upper(),
            processed_at=datetime.now(tz=timezone.utc).isoformat(),
            error="",  # No errors
            attempt=1,
        )

    # First pass - all succeed
    first_pass = (
        dc.read_dataset("sample_data", session=test_session)
        .map(result=successful_process)
        .save("successful_data")
    )

    assert first_pass.count() == 2
    assert first_pass.filter(C("result.error") != "").count() == 0

    # Retry - should not create a new version since no records need retry
    (
        dc.read_dataset(
            "sample_data",
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry="result.error",
        )
        .map(result=successful_process)
        .save("successful_data")
    )

    # Should not create version 1.0.1 since no retry was needed
    with pytest.raises(DatasetNotFoundError):
        dc.read_dataset("successful_data", version="1.0.1", session=test_session)


def test_retry_first_dataset_creation(test_session):
    """Test retry when dataset doesn't exist yet (first creation)."""
    _create_sample_data(test_session, ids=[1, 2], contents=["first", "second"])

    # First run with retry enabled on non-existent dataset
    # Should process all records
    retry_chain = (
        dc.read_dataset(
            "sample_data",
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry="result.error",
        )
        .setup(attempt=lambda: 1)
        .map(result=_simple_process)
        .save("new_dataset")
    )

    assert retry_chain.count() == 2
    assert retry_chain.filter(C("result.error") != "").count() == 0


def test_retry_with_multiple_match_fields(test_session):
    """Test retry functionality with multiple fields for matching."""

    def process_with_compound_key(
        category: str, id: int, content: str, attempt: int
    ) -> ProcessingResult:
        error = category == "A" and id % 2 == 1 and attempt == 1
        return ProcessingResult(
            processed_content="" if error else f"{category}-{content}".upper(),
            processed_at=datetime.now(tz=timezone.utc).isoformat(),
            error=f"Error for {category}-{id}" if error else "",
            attempt=attempt,
        )

    # Create dataset with compound keys
    categories = ["A", "A", "B", "B"]
    ids = [1, 2, 1, 2]
    contents = ["first", "second", "third", "fourth"]

    dc.read_values(
        category=categories, id=ids, content=contents, session=test_session
    ).save("compound_source")

    first_pass = (
        dc.read_dataset("compound_source", session=test_session)
        .setup(attempt=lambda: 1)
        .map(result=process_with_compound_key)
        .save("compound_result")
    )

    # Only A-1 should fail (category="A" and odd id)
    results = list(first_pass.filter(C("result.error") != "").to_list("category", "id"))
    assert results == [("A", 1)]

    # Retry with multiple match fields
    (
        dc.read_dataset(
            "compound_source",
            session=test_session,
            delta=True,
            delta_on=["category", "id"],
            delta_retry="result.error",
        )
        .setup(attempt=lambda: 2)
        .map(result=process_with_compound_key)
        .save("compound_result")
    )

    final_result = dc.read_dataset("compound_result", session=test_session)
    results = final_result.to_list("category", "id", "result.attempt", "result.error")
    assert set(results) == {
        ("A", 2, 1, ""),
        ("B", 1, 1, ""),
        ("B", 2, 1, ""),
        ("A", 1, 2, ""),
    }


def test_retry_with_delta_functionality(test_session):
    """Test that retry and delta can work together."""

    def _run_processing(attempt: int) -> "DataChain":
        return (
            dc.read_dataset(
                "delta_source_v1",
                delta=True,
                delta_on="id",
                delta_retry="result.error",
                session=test_session,
            )
            .setup(attempt=lambda: attempt)
            .map(result=_process_with_errors)
            .save("delta_retry_result")
        )

    dc.read_values(id=[1, 2], content=["first", "second"], session=test_session).save(
        "delta_source_v1"
    )

    result = _run_processing(1)
    assert set(result.to_iter("id", "result.error")) == {
        (1, "Processing error for item 1"),
        (2, ""),
    }

    # Add more data
    # Process again - should only process new records and retry errors
    extended_ids = [1, 2, 3]
    extended_contents = ["first", "second", "third"]
    dc.read_values(
        id=extended_ids, content=extended_contents, session=test_session
    ).save("delta_source_v1")

    result_v2 = _run_processing(2)
    assert set(result_v2.to_iter("id", "result.error", "result.attempt")) == {
        (1, "", 2),
        (2, "", 1),
        (3, "", 2),
    }


def test_delta_and_delta_retry_no_duplicates(test_session):
    """Test that delta and delta_retry work together without creating duplicates
    when the same records are picked up for different reasons:
    - delta_retry=True picks up unprocessed records missing from result dataset
    - delta=True picks up modified records from source dataset
    """
    _create_sample_data(test_session)

    # First pass - process only records 1 and 2
    partial_result = (
        dc.read_dataset("sample_data", session=test_session)
        .setup(attempt=lambda: 1)
        .filter(C("id") < 3)  # Only process id=1,2, leaving id=3,4 unprocessed
        .map(result=_simple_process)
        .save("delta_retry_combined_result")
    )

    assert partial_result.count() == 2
    initial_results = set(partial_result.to_iter("id", "result.attempt"))
    assert initial_results == {(1, 1), (2, 1)}

    # Modify the source data - update content for records 3 and 4
    # This will make delta=True pick them up as "changed"
    # But delta_retry=True will also pick them up as "missing from result"
    modified_ids = [1, 2, 3, 4]
    modified_contents = [
        "first item",  # unchanged
        "second item",  # unchanged
        "MODIFIED third item",  # modified - delta will pick this up
        "MODIFIED fourth item",  # modified - delta will pick this up
    ]
    _create_sample_data(test_session, modified_ids, modified_contents)

    # Second pass with both delta=True and delta_retry=True
    # Records 3,4 should be picked up by BOTH:
    # - delta_retry=True (because they're missing from result dataset)
    # - delta=True (because their content was modified in source)
    # But they should only be processed ONCE (no duplicates)
    combined_result = (
        dc.read_dataset(
            "sample_data",
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry=True,
        )
        .setup(attempt=lambda: 2)
        .map(result=_simple_process)
        .save("delta_retry_combined_result")
    )

    # Should have 4 total records: 2 from first pass + 2 newly processed
    assert combined_result.count() == 4

    # Get all results and verify no duplicates
    all_results = set(
        combined_result.to_iter("id", "result.attempt", "result.processed_content")
    )

    # Records 1,2 should have attempt=1 (from first pass)
    # Records 3,4 should have attempt=2 (from second pass) and MODIFIED content
    expected_results = {
        (1, 1, "FIRST ITEM"),
        (2, 1, "SECOND ITEM"),
        (3, 2, "MODIFIED THIRD ITEM"),
        (4, 2, "MODIFIED FOURTH ITEM"),
    }

    assert all_results == expected_results

    # Verify counts by attempt
    first_attempt_count = combined_result.filter(C("result.attempt") == 1).count()
    second_attempt_count = combined_result.filter(C("result.attempt") == 2).count()

    assert first_attempt_count == 2  # Records 1,2 from first pass
    assert second_attempt_count == 2  # Records 3,4 from second pass (no duplicates)

    # Verify that each id appears exactly once
    ids_in_result = list(combined_result.to_values("id"))
    assert len(ids_in_result) == 4
    assert len(set(ids_in_result)) == 4  # No duplicate IDs
    assert set(ids_in_result) == {1, 2, 3, 4}


def test_repeating_errors(test_session):
    def run_delta():
        def func(id) -> Iterator[tuple[int, str, str]]:
            yield id, "name1", "error"
            yield id, "name2", "error"

        return (
            dc.read_dataset(
                "sample_data",
                delta=True,
                delta_on="id",
                delta_result_on="id",
                delta_retry="error",
                session=test_session,
            )
            .gen(func, output={"id": int, "name": str, "error": str})
            .save("processed_data")
        )

    def sorted_ids(chain: "DataChain") -> list[int]:
        ids = (cast("int", row[0]) for row in chain.to_list("id"))
        return sorted(ids)

    _create_sample_data(
        test_session, ids=list(range(1)), contents=[str(i) for i in range(1)]
    )
    ch1 = run_delta()
    assert sorted_ids(ch1) == [0, 0]

    _create_sample_data(
        test_session, ids=list(range(2)), contents=[str(i) for i in range(2)]
    )
    ch2 = run_delta()
    assert sorted_ids(ch2) == [0, 0, 1, 1]

    _create_sample_data(
        test_session, ids=list(range(3)), contents=[str(i) for i in range(3)]
    )
    ch3 = run_delta()
    assert sorted_ids(ch3) == [0, 0, 1, 1, 2, 2]
