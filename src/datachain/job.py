import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from datachain import json

J = TypeVar("J", bound="Job")


@dataclass
class Job:
    id: str
    name: str
    status: int
    created_at: datetime
    query: str
    query_type: int
    workers: int
    params: dict[str, str]
    metrics: dict[str, Any]
    finished_at: datetime | None = None
    python_version: str | None = None
    error_message: str = ""
    error_stack: str = ""
    parent_job_id: str | None = None
    rerun_from_job_id: str | None = None
    run_group_id: str | None = None
    is_remote_execution: bool = False

    @classmethod
    def create_ephemeral(cls) -> "Job":
        """Create an in-memory Job."""
        job_id = str(uuid.uuid4())
        return cls(
            id=job_id,
            name="",
            status=0,
            created_at=datetime.now(timezone.utc),
            query="",
            query_type=0,
            workers=1,
            params={},
            metrics={},
            run_group_id=job_id,
        )

    @classmethod
    def parse(  # noqa: PLR0913
        cls,
        id: str | uuid.UUID,
        name: str,
        status: int,
        created_at: datetime,
        finished_at: datetime | None,
        query: str,
        query_type: int,
        workers: int,
        python_version: str | None,
        error_message: str,
        error_stack: str,
        params: str,
        metrics: str,
        parent_job_id: str | None,
        rerun_from_job_id: str | None,
        run_group_id: str | None,
        is_remote_execution: bool = False,
    ) -> "Job":
        return cls(
            str(id),
            name,
            status,
            created_at,
            query,
            query_type,
            workers,
            json.loads(params),
            json.loads(metrics),
            finished_at,
            python_version,
            error_message,
            error_stack,
            str(parent_job_id) if parent_job_id else None,
            str(rerun_from_job_id) if rerun_from_job_id else None,
            str(run_group_id) if run_group_id else None,
            is_remote_execution,
        )
