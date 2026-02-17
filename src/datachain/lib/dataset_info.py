from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import Field, field_validator

from datachain import json
from datachain.dataset import (
    DEFAULT_DATASET_VERSION,
    DatasetListRecord,
    DatasetListVersion,
    DatasetStatus,
)
from datachain.job import Job
from datachain.lib.data_model import DataModel
from datachain.query.session import Session
from datachain.utils import TIME_ZERO

if TYPE_CHECKING:
    from typing_extensions import Self


class DatasetInfo(DataModel):
    name: str
    namespace: str
    project: str
    uuid: str = Field(default=str(uuid4()))
    version: str = Field(default=DEFAULT_DATASET_VERSION)
    status: int = Field(default=DatasetStatus.CREATED)
    created_at: datetime = Field(default=TIME_ZERO)
    finished_at: datetime | None = Field(default=None)
    num_objects: int | None = Field(default=None)
    size: int | None = Field(default=None)
    params: dict[str, str] = Field(default={})
    metrics: dict[str, Any] = Field(default={})
    error_message: str = Field(default="")
    error_stack: str = Field(default="")
    attrs: list[str] = Field(default=[])

    @property
    def is_temp(self) -> bool:
        return Session.is_temp_dataset(self.name)

    def has_attr(self, attr: str) -> bool:
        s = attr.split("=")
        if len(s) == 1:
            return attr in self.attrs

        name = s[0]
        value = s[1]
        for a in self.attrs:
            s = a.split("=")
            if value == "*" and s[0] == name:
                return True
            if len(s) == 2 and s[0] == name and s[1] == value:
                return True

        return False

    @staticmethod
    def _validate_dict(
        v: str | dict | None,
    ) -> dict:
        if v is None or v == "":
            return {}
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    f"Unable to convert string '{v}' to dict for Dataset feature: {e}"
                ) from None
        return v

    # Workaround for empty JSONs converted to empty strings in some DBs.
    @field_validator("params", mode="before")
    @classmethod
    def validate_location(cls, v):
        return cls._validate_dict(v)

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, v):
        return cls._validate_dict(v)

    @classmethod
    def from_models(
        cls,
        dataset: DatasetListRecord,
        version: DatasetListVersion,
        job: Job | None,
    ) -> "Self":
        return cls(
            uuid=version.uuid,
            name=dataset.name,
            namespace=dataset.project.namespace.name,
            project=dataset.project.name,
            version=version.version,
            status=version.status,
            created_at=version.created_at,
            finished_at=version.finished_at,
            num_objects=version.num_objects,
            size=version.size,
            params=job.params if job else {},
            metrics=job.metrics if job else {},
            error_message=version.error_message,
            error_stack=version.error_stack,
            attrs=dataset.attrs,
        )
