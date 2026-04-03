import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum


class CheckpointStatus(IntEnum):
    ACTIVE = 0
    EXPIRED = 1
    DELETED = 2


@dataclass
class Checkpoint:
    """
    Represents a checkpoint within a job run.

    A checkpoint marks a successfully completed stage of execution. In the event
    of a failure, the job can resume from the most recent checkpoint rather than
    starting over from the beginning.

    Checkpoints can also be created in a "partial" mode, which indicates that the
    work at this stage was only partially completed. For example, if a failure
    occurs halfway through running a UDF, already computed results can still be
    saved, allowing the job to resume from that partially completed state on
    restart.
    """

    id: str
    job_id: str
    hash: str
    partial: bool
    created_at: datetime
    status: int = CheckpointStatus.ACTIVE

    @staticmethod
    def output_table_name(job_id: str, _hash: str) -> str:
        """Final UDF output table. Job-specific, created when UDF completes."""
        return f"udf_{job_id}_{_hash}_output"

    @staticmethod
    def partial_output_table_name(job_id: str, _hash: str) -> str:
        """Partial UDF output table. Temporary, renamed to final on completion."""
        return f"udf_{job_id}_{_hash}_output_partial"

    @staticmethod
    def input_table_name(group_id: str, _hash: str) -> str:
        """Shared UDF input table. Scoped to run group, reused across jobs."""
        return f"udf_{group_id}_{_hash}_input"

    @staticmethod
    def partition_table_name(job_id: str, _hash: str) -> str:
        """Partition mapping table. Job-specific, maps sys__id to partition_id."""
        return f"udf_{job_id}_{_hash}_partition"

    @staticmethod
    def input_table_pattern(group_id: str) -> str:
        """LIKE pattern for finding all input tables in a run group."""
        return f"udf_{group_id}_%_input"

    @staticmethod
    def partition_table_pattern(job_id: str = "") -> str:
        """LIKE pattern for finding partition tables. If job_id given, scoped to job."""
        prefix = f"udf_{job_id}" if job_id else "udf_"
        return f"{prefix}%_partition"

    @property
    def table_name(self) -> str:
        """UDF output table name associated with this checkpoint."""
        if self.partial:
            return self.partial_output_table_name(self.job_id, self.hash)
        return self.output_table_name(self.job_id, self.hash)

    @classmethod
    def parse(
        cls,
        id: str | uuid.UUID,
        job_id: str,
        _hash: str,
        partial: bool,
        created_at: datetime,
        status: int = CheckpointStatus.ACTIVE,
    ) -> "Checkpoint":
        return cls(
            str(id),
            job_id,
            _hash,
            bool(partial),
            created_at,
            int(status),
        )
