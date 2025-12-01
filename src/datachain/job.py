import uuid
from dataclasses import dataclass
from datetime import datetime
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

    @classmethod
    def parse(
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
        )
