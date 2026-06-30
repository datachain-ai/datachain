import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

from datachain import json

J = TypeVar("J", bound="Job")


@dataclass
class JobIdentity:
    id: str
    name: str
    status: int


@dataclass
class JobExecution:
    created_at: datetime
    finished_at: datetime | None = None
    python_version: str | None = None
    is_remote_execution: bool = False


@dataclass
class JobQuery:
    query: str
    query_type: int
    workers: int
    params: dict[str, str]
    metrics: dict[str, Any]


@dataclass
class JobRelations:
    parent_job_id: str | None = None
    rerun_from_job_id: str | None = None
    run_group_id: str | None = None


@dataclass
class JobError:
    error_message: str = ""
    error_stack: str = ""


@dataclass
class Job:
    identity: JobIdentity
    execution: JobExecution
    query_data: JobQuery
    relations: JobRelations
    error: JobError

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
            identity=JobIdentity(
                id=str(id),
                name=name,
                status=status,
            ),
            execution=JobExecution(
                created_at=created_at,
                finished_at=finished_at,
                python_version=python_version,
                is_remote_execution=is_remote_execution,
            ),
            query_data=JobQuery(
                query=query,
                query_type=query_type,
                workers=workers,
                params=json.loads(params),
                metrics=json.loads(metrics),
            ),
            relations=JobRelations(
                parent_job_id=str(parent_job_id) if parent_job_id else None,
                rerun_from_job_id=str(rerun_from_job_id) if rerun_from_job_id else None,
                run_group_id=str(run_group_id) if run_group_id else None,
            ),
            error=JobError(
                error_message=error_message,
                error_stack=error_stack,
            ),
        )
