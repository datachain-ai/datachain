#!/usr/bin/env python
"""
Retry Processing Example with DataChain

This script demonstrates DataChain's retry functionality by:
1. Creating an initial in-memory dataset with sample data
2. Processing records with simulated errors
3. Using retry flags to reprocess failed records in a second pass
"""

from datetime import datetime, timezone

import datachain as dc
from datachain import C, DataModel


class ProcessingResult(DataModel):
    """Data model for processing results."""

    processed_content: str
    processed_at: str
    error: str
    attempt: int


def process_data(item_id: int, content: str, attempt: int) -> ProcessingResult:
    """
    Process a data record - initially fails for odd IDs, but succeeds on retry.
    In a real-world application, this could be data validation,
    transformation, or an ML model inference step.
    """
    # Simulate an error for odd IDs on first attempt only
    # This will succeed on retry to demonstrate the retry functionality
    error = item_id % 2 == 1 and attempt == 1
    return ProcessingResult(
        processed_content="" if error else content.upper(),
        processed_at=datetime.now(tz=timezone.utc).isoformat(),
        error=f"Processing error for item {item_id} (attempt {attempt})"
        if error
        else "",
        attempt=attempt,
    )


def retry_processing_example():
    """
    Demonstrates retry processing using DataChain.
    """

    # Step 1: Create initial dataset with sample data
    print("Step 1: Creating initial dataset with sample data...")
    sample_ids = [1, 2, 3, 4, 5]
    sample_contents = [
        "first item",
        "second item",
        "third item",
        "fourth item",
        "fifth item",
    ]

    initial_chain = dc.read_values(
        item_id=sample_ids, content=sample_contents, in_memory=True
    ).save(name="sample_data")
    print(f"Created dataset with {initial_chain.count()} records\n")

    # Step 2: First processing pass - some records will fail
    # We enable delta, initially it won't do anything, but it means
    # that code stays the same across all attempts
    print("Step 2: First processing pass (some records will fail)...")
    first_pass = (
        dc.read_dataset(
            "sample_data",
            delta=True,
            delta_on="item_id",
            delta_retry="result.error",
            in_memory=True,
        )
        # Set attempt number for processing, this is specific to this example only
        .setup(attempt=lambda: 1)
        .map(result=process_data)
        .save(name="processed_data")
    )

    print(f"First pass completed. Total records: {first_pass.count()}")

    # Show results of first pass
    error_count = first_pass.filter(C("result.error") != "").count()
    success_count = first_pass.filter(C("result.error") == "").count()
    print(f"Successful: {success_count}, Failed: {error_count}")

    print("\nFirst pass results:")
    first_pass.show()

    # Step 3: Retry processing - only failed records will be reprocessed
    # Note how we use exactly the same DataChain code as above
    print("\nStep 3: Retry processing (failed records will be reprocessed)...")
    retry_chain = (
        dc.read_dataset(
            "sample_data",
            delta=True,
            delta_on="item_id",
            # Retry records where result.error field is not empty
            delta_retry="result.error",
            in_memory=True,
        )
        .setup(attempt=lambda: 2)  # Set attempt number for processing
        .map(result=process_data)
        .save(name="processed_data")
    )

    print(f"Retry pass completed. Total records: {retry_chain.count()}")

    # Show final results
    final_error_count = retry_chain.filter(C("result.error") != "").count()
    final_success_count = retry_chain.filter(C("result.error") == "").count()
    print(f"Final - Successful: {final_success_count}, Failed: {final_error_count}")

    print("\nFinal results after retry:")
    retry_chain.show()

    if final_error_count > 0:
        print(f"\nNote: {final_error_count} records still have errors after retry.")
        print(
            "In a real scenario, you could run this again or implement "
            "different retry logic."
        )
    else:
        print("\nAll records processed successfully after retry!")


if __name__ == "__main__":
    retry_processing_example()
