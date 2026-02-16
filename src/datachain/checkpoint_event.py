import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CheckpointEventType(str, Enum):
    """Types of checkpoint events."""

    # UDF events
    UDF_SKIPPED = "UDF_SKIPPED"
    UDF_CONTINUED = "UDF_CONTINUED"
    UDF_FROM_SCRATCH = "UDF_FROM_SCRATCH"

    # Dataset save events
    DATASET_SAVE_SKIPPED = "DATASET_SAVE_SKIPPED"
    DATASET_SAVE_COMPLETED = "DATASET_SAVE_COMPLETED"


class CheckpointStepType(str, Enum):
    """Types of checkpoint steps."""

    UDF_MAP = "UDF_MAP"
    UDF_GEN = "UDF_GEN"
    DATASET_SAVE = "DATASET_SAVE"


@dataclass
class CheckpointEvent:
    """
    Represents a checkpoint event for debugging and visibility.

    Checkpoint events are logged during job execution to track checkpoint
    decisions (skip, continue, run from scratch) and provide visibility
    into what happened during script execution.
    """

    id: str
    job_id: str
    run_group_id: str | None
    timestamp: datetime
    event_type: CheckpointEventType
    step_type: CheckpointStepType
    udf_name: str | None = None
    dataset_name: str | None = None
    checkpoint_hash: str | None = None
    hash_partial: str | None = None
    hash_input: str | None = None
    hash_output: str | None = None
    rows_input: int | None = None
    rows_processed: int | None = None
    rows_output: int | None = None
    rows_input_reused: int | None = None
    rows_output_reused: int | None = None
    rerun_from_job_id: str | None = None
    details: dict | None = None

    @classmethod
    def parse(  # noqa: PLR0913
        cls,
        id: str | uuid.UUID,
        job_id: str,
        run_group_id: str | None,
        timestamp: datetime,
        event_type: str,
        step_type: str,
        udf_name: str | None,
        dataset_name: str | None,
        checkpoint_hash: str | None,
        hash_partial: str | None,
        hash_input: str | None,
        hash_output: str | None,
        rows_input: int | None,
        rows_processed: int | None,
        rows_output: int | None,
        rows_input_reused: int | None,
        rows_output_reused: int | None,
        rerun_from_job_id: str | None,
        details: dict | None,
    ) -> "CheckpointEvent":
        return cls(
            id=str(id),
            job_id=job_id,
            run_group_id=run_group_id,
            timestamp=timestamp,
            event_type=CheckpointEventType(event_type),
            step_type=CheckpointStepType(step_type),
            udf_name=udf_name,
            dataset_name=dataset_name,
            checkpoint_hash=checkpoint_hash,
            hash_partial=hash_partial,
            hash_input=hash_input,
            hash_output=hash_output,
            rows_input=rows_input,
            rows_processed=rows_processed,
            rows_output=rows_output,
            rows_input_reused=rows_input_reused,
            rows_output_reused=rows_output_reused,
            rerun_from_job_id=rerun_from_job_id,
            details=details,
        )
