import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

from datachain import json

J = TypeVar("J", bound="Job")


def _as_dict(value: "str | dict | None") -> dict:
    """Normalize a JSON column value to a dict.

    A JSON column can hand back either a ``str`` or an already-decoded
    ``dict`` depending on the backend and how the row was written:

    - SQLite always returns the raw string.
    - PostgreSQL/JSONB returns a ``dict`` for rows stored as a JSON object,
      but a ``str`` for legacy rows that were double-encoded into a JSON
      string scalar (e.g. ``"{}"``) by a previous ``json.dumps``-on-write.

    Decoding only when we still have a string keeps both the current and the
    legacy on-disk formats readable, so no data migration is required.
    """
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return value


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
        params: "str | dict[str, str] | None",
        metrics: "str | dict[str, Any] | None",
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
            _as_dict(params),
            _as_dict(metrics),
            finished_at,
            python_version,
            error_message,
            error_stack,
            str(parent_job_id) if parent_job_id else None,
            str(rerun_from_job_id) if rerun_from_job_id else None,
            str(run_group_id) if run_group_id else None,
            is_remote_execution,
        )
