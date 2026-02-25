import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum


class CheckpointStatus(IntEnum):
    ACTIVE = 0
    DELETED = 1


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
